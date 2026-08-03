//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_gltf_extract.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_gltf_extract.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <conduit_fmt/conduit_fmt.h>

#include <viskores/cont/ColorTable.h>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_mpi_utils.hpp>
#include <ascent_runtime_conduit_to_viskores_parsing.hpp>
#include <ascent_runtime_param_check.hpp>
#include <ascent_runtime_utils.hpp>
#include <ascent_string_utils.hpp>
#include <png_utils/ascent_png_encoder.hpp>

#include <flow_workspace.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

using namespace std;
using namespace conduit;
using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters::detail --
//-----------------------------------------------------------------------------
namespace detail
{

// glTF 2.0 spec constants
constexpr int GLTF_MODE_POINTS            = 0;
constexpr int GLTF_MODE_LINES             = 1;
constexpr int GLTF_MODE_TRIANGLES         = 4;
constexpr int GLTF_COMPONENT_UNSIGNED_INT = 5125;
constexpr int GLTF_COMPONENT_FLOAT        = 5126;
constexpr int GLTF_ARRAY_BUFFER           = 34962;
constexpr int GLTF_ELEMENT_ARRAY_BUFFER   = 34963;
constexpr int GLTF_FILTER_NEAREST         = 9728;
constexpr int GLTF_FILTER_LINEAR          = 9729;
constexpr int GLTF_WRAP_CLAMP_TO_EDGE     = 33071;

//-----------------------------------------------------------------------------
// mirrors detail::color_table_schema in ascent_runtime_rendering_filters.cpp,
// minus the render-only annotation property
void
gltf_color_table_schema(conduit::Node &param_schema)
{
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/name"]);
    bool_schema(param_schema["properties/reverse"]);
    string_schema(param_schema["properties/discrete"]);

    // --- Control Points ---
    {
        conduit::Node &control_points_schema = param_schema["properties/control_points"];

        conduit::Node &cp_compressed_schema = control_points_schema["oneOf"].append();
        cp_compressed_schema["type"] = "object";
        cp_compressed_schema["additionalProperties"] = false;
        ignore_schema(cp_compressed_schema["properties/r"]);
        ignore_schema(cp_compressed_schema["properties/g"]);
        ignore_schema(cp_compressed_schema["properties/b"]);
        ignore_schema(cp_compressed_schema["properties/a"]);
        ignore_schema(cp_compressed_schema["properties/position"]);
        cp_compressed_schema["constraints/forbid"].append() = "type";
        cp_compressed_schema["constraints/forbid"].append() = "alpha";
        cp_compressed_schema["constraints/forbid"].append() = "color";

        conduit::Node cp_list_item_schema;
        cp_list_item_schema["type"] = "object";
        cp_list_item_schema["additionalProperties"] = false;
        ignore_schema(cp_list_item_schema["properties/type"]);
        ignore_schema(cp_list_item_schema["properties/alpha"]);
        ignore_schema(cp_list_item_schema["properties/color"]);
        ignore_schema(cp_list_item_schema["properties/position"]);
        cp_list_item_schema["constraints/forbid"].append() = "r";
        cp_list_item_schema["constraints/forbid"].append() = "g";
        cp_list_item_schema["constraints/forbid"].append() = "b";
        cp_list_item_schema["constraints/forbid"].append() = "a";

        array_schema(control_points_schema["oneOf"].append(), cp_list_item_schema);
    }
}

//-----------------------------------------------------------------------------
struct DomainData
{
    index_t                    id = -1;
    int                        mode = -1;
    std::string                topology;
    std::string                dtype;
    conduit::Node              state;
    std::vector<float>         positions;
    std::vector<float>         texcoords;
    std::vector<std::uint32_t> indices;
    std::vector<double>        field_values;
    float                      bounds_min[3];
    float                      bounds_max[3];
    uint64                     invalid_count = 0;
};

//-----------------------------------------------------------------------------
struct PaletteData
{
    std::vector<unsigned char> png;
    conduit::Node              metadata;
    bool                       discrete = false;
    bool                       transparent = false;
};

//-----------------------------------------------------------------------------
std::string
domain_file_name(index_t domain_id)
{
    return conduit_fmt::format("domain_{:08d}.glb", domain_id);
}

//-----------------------------------------------------------------------------
std::uint32_t
align4(std::uint64_t value)
{
    if(value > std::numeric_limits<std::uint32_t>::max() - 3)
    {
        ASCENT_ERROR("gltf extract: GLB exceeds the 32-bit container limit");
    }
    return (std::uint32_t)((value + 3) & ~std::uint64_t(3));
}

//-----------------------------------------------------------------------------
std::uint32_t
checked_u32(std::uint64_t value, const char *description)
{
    if(value > std::numeric_limits<std::uint32_t>::max())
    {
        ASCENT_ERROR("gltf extract: GLB " << description <<
                     " exceeds the 32-bit container limit");
    }
    return (std::uint32_t)value;
}

//-----------------------------------------------------------------------------
void
write_u32(std::ostream &out, std::uint32_t value)
{
    out.write(reinterpret_cast<const char *>(&value), 4);
}

//-----------------------------------------------------------------------------
void
write_padding(std::ostream &out, std::size_t bytes)
{
    static const char padding[4] = { 0, 0, 0, 0 };
    out.write(padding, (std::streamsize)bytes);
}

//-----------------------------------------------------------------------------
template<typename T>
void
write_buffer(std::ostream &out, const std::vector<T> &values)
{
    out.write(reinterpret_cast<const char *>(values.data()),
              (std::streamsize)(values.size() * sizeof(T)));
}

//-----------------------------------------------------------------------------
std::vector<double>
numeric_values(const conduit::Node &n_vals)
{
    if(!n_vals.dtype().is_number())
    {
        ASCENT_ERROR("gltf extract requires numeric coordinate, connectivity, "
                     "and field arrays");
    }
    Node n_conv;
    n_vals.to_float64_array(n_conv);
    float64_array vals = n_conv.value();
    std::vector<double> res(vals.number_of_elements());
    for(index_t i = 0; i < vals.number_of_elements(); i++)
    {
        res[i] = vals[i];
    }
    return res;
}

//-----------------------------------------------------------------------------
std::string
resolve_topology(const conduit::Node &dom,
                 const conduit::Node &params,
                 const std::string &field)
{
    std::string topo_name;
    if(!field.empty())
    {
        if(!dom.has_path("fields/" + field))
        {
            ASCENT_ERROR("gltf extract: field '" << field <<
                         "' is missing from a domain");
        }
        if(!dom.has_path("fields/" + field + "/topology"))
        {
            ASCENT_ERROR("gltf extract: field '" << field <<
                         "' does not reference a topology");
        }
        topo_name = dom["fields/" + field + "/topology"].as_string();
    }

    if(params.has_child("topology"))
    {
        std::string requested = params["topology"].as_string();
        if(!topo_name.empty() && topo_name != requested)
        {
            ASCENT_ERROR("gltf extract: params/topology does not match the "
                         "selected field's topology");
        }
        topo_name = requested;
    }

    if(topo_name.empty())
    {
        if(!dom.has_child("topologies"))
        {
            ASCENT_ERROR("gltf extract: input domain has no topologies");
        }
        const Node &topos = dom["topologies"];
        if(topos.number_of_children() != 1)
        {
            ASCENT_ERROR("gltf extract: topology is ambiguous, "
                         "please specify params/topology");
        }
        topo_name = topos.child_names()[0];
    }

    if(!dom.has_path("topologies/" + topo_name))
    {
        ASCENT_ERROR("gltf extract: topology '" << topo_name << "' does not exist");
    }
    return topo_name;
}

//-----------------------------------------------------------------------------
void
read_positions(const conduit::Node &n_coords, DomainData &data)
{
    Node n_explicit;
    conduit::blueprint::mesh::coordset::to_explicit(n_coords, n_explicit);
    const Node &n_values = n_explicit["values"];
    std::vector<double> x = numeric_values(n_values["x"]);
    std::vector<double> y = numeric_values(n_values["y"]);
    std::vector<double> z;
    if(n_values.has_child("z"))
    {
        z = numeric_values(n_values["z"]);
    }
    else
    {
        z.resize(x.size(), 0.0);
    }

    if(x.size() != y.size() || x.size() != z.size())
    {
        ASCENT_ERROR("gltf extract: coordinate components have different lengths");
    }
    if(x.empty())
    {
        ASCENT_ERROR("gltf extract cannot export a domain with no vertices");
    }

    for(int axis = 0; axis < 3; axis++)
    {
        data.bounds_min[axis] =  std::numeric_limits<float>::max();
        data.bounds_max[axis] = -std::numeric_limits<float>::max();
    }

    data.positions.reserve(x.size() * 3);
    for(size_t i = 0; i < x.size(); i++)
    {
        float pos[3] = { (float)x[i], (float)y[i], (float)z[i] };
        for(int axis = 0; axis < 3; axis++)
        {
            data.positions.push_back(pos[axis]);
            data.bounds_min[axis] = std::min(data.bounds_min[axis], pos[axis]);
            data.bounds_max[axis] = std::max(data.bounds_max[axis], pos[axis]);
        }
    }
}

//-----------------------------------------------------------------------------
void
read_connectivity(const conduit::Node &n_topo, DomainData &data)
{
    if(n_topo["type"].as_string() != "unstructured")
    {
        ASCENT_ERROR("gltf extract supports surface points, lines, and triangles; "
                     "use external_surfaces, contour, or slice followed by "
                     "triangulate for structured or volume meshes");
    }

    const std::string shape = n_topo["elements/shape"].as_string();
    int stride = 0;
    if(shape == "point")
    {
        data.mode = GLTF_MODE_POINTS;
        stride = 1;
    }
    else if(shape == "line")
    {
        data.mode = GLTF_MODE_LINES;
        stride = 2;
    }
    else if(shape == "tri")
    {
        data.mode = GLTF_MODE_TRIANGLES;
        stride = 3;
    }
    else if(shape == "quad")
    {
        ASCENT_ERROR("gltf extract does not support quads; "
                     "add a triangulate pipeline filter");
    }
    else
    {
        ASCENT_ERROR("gltf extract does not support shape '" << shape <<
                     "'; use external_surfaces, contour, or slice followed "
                     "by triangulate");
    }

    const Node &n_conn = n_topo["elements/connectivity"];
    if(!n_conn.dtype().is_integer())
    {
        ASCENT_ERROR("gltf extract: connectivity must use an integer array");
    }

    const index_t conn_size = n_conn.dtype().number_of_elements();
    if(conn_size == 0)
    {
        ASCENT_ERROR("gltf extract cannot export a domain with no primitives");
    }
    if(conn_size % stride != 0)
    {
        ASCENT_ERROR("gltf extract: connectivity has an invalid primitive count");
    }

    const std::uint64_t num_verts = data.positions.size() / 3;
    Node n_conv;
    n_conn.to_int64_array(n_conv);
    int64_array conn_vals = n_conv.value();
    data.indices.reserve(conn_size);
    for(index_t i = 0; i < conn_vals.number_of_elements(); i++)
    {
        const int64 value = conn_vals[i];
        if(value < 0 ||
           (std::uint64_t)value >= num_verts ||
           (std::uint64_t)value > std::numeric_limits<std::uint32_t>::max())
        {
            ASCENT_ERROR("gltf extract: connectivity contains an out-of-range index");
        }
        data.indices.push_back((std::uint32_t)value);
    }
}

//-----------------------------------------------------------------------------
void
make_texcoords(const std::vector<double> &values,
               double min_value,
               double max_value,
               DomainData &data)
{
    const double range = max_value - min_value;
    data.texcoords.reserve(values.size() * 2);
    for(double value : values)
    {
        if(std::isfinite(value))
        {
            double u = 0.5;
            if(range != 0.0)
            {
                u = (value - min_value) / range;
            }
            data.texcoords.push_back((float)u);
            data.texcoords.push_back(0.25f);
        }
        else
        {
            // nan and inf sample the invalid-value row of the palette
            data.texcoords.push_back(0.5f);
            data.texcoords.push_back(0.75f);
        }
    }
}

//-----------------------------------------------------------------------------
PaletteData
make_palette(const conduit::Node &params)
{
    Node n_empty;
    const Node &n_color_table = params.has_child("color_table") ?
                                params["color_table"] : n_empty;
    viskores::cont::ColorTable table = parse_color_table(n_color_table);
    viskores::cont::ArrayHandle<viskores::Vec4ui_8> samples;
    if(!table.Sample(256, samples))
    {
        ASCENT_ERROR("gltf extract failed to sample the color table");
    }
    auto portal = samples.ReadPortal();

    PaletteData res;
    if(params.has_child("color_table"))
    {
        res.metadata.set(params["color_table"]);
    }
    res.metadata["name"] = table.GetName();
    res.metadata["samples"] = 256;
    res.discrete = params.has_path("color_table/discrete") &&
                   params["color_table/discrete"].as_string() == "true";
    res.metadata["discrete"] = res.discrete ? "true" : "false";

    // 256 x 2 rgba image, top row color samples, bottom row the nan color.
    // PNGEncoder flips rows vertically, so fill the nan row first.
    std::vector<unsigned char> pixels(256 * 2 * 4);
    const auto nan_color = table.GetNaNColor();
    for(int i = 0; i < 256; i++)
    {
        for(int c = 0; c < 3; c++)
        {
            pixels[4 * i + c] = (unsigned char)(255 * nan_color[c] + 0.5f);
        }
        pixels[4 * i + 3] = 255;
    }
    for(viskores::Id i = 0; i < 256; i++)
    {
        const auto color = portal.Get(i);
        for(int c = 0; c < 4; c++)
        {
            pixels[4 * (256 + i) + c] = color[c];
        }
        res.transparent = res.transparent || color[3] != 255;
    }

    PNGEncoder encoder;
    encoder.Encode(pixels.data(), 256, 2);
    if(encoder.PngBufferSize() == 0)
    {
        ASCENT_ERROR("gltf extract failed to encode the color table png");
    }
    const unsigned char *png = (const unsigned char *)encoder.PngBuffer();
    res.png.assign(png, png + encoder.PngBufferSize());
    return res;
}

//-----------------------------------------------------------------------------
conduit::Node
domain_record(const DomainData &data, const std::string &field)
{
    Node record;
    record["domain_id"] = data.id;
    record["uri"] = "domains/" + domain_file_name(data.id);
    record["primitive_mode"] = data.mode;
    record["topology"] = data.topology;
    if(!field.empty())
    {
        record["source_dtype"] = data.dtype;
    }
    if(data.state.has_child("cycle"))
    {
        record["cycle"].set(data.state["cycle"]);
    }
    if(data.state.has_child("time"))
    {
        record["time"].set(data.state["time"]);
    }
    record["vertex_count"] = data.positions.size() / 3;
    record["index_count"] = data.indices.size();
    for(int axis = 0; axis < 3; axis++)
    {
        record["bounds/min"].append() = data.bounds_min[axis];
        record["bounds/max"].append() = data.bounds_max[axis];
    }
    record["invalid_value_count"] = data.invalid_count;
    return record;
}

//-----------------------------------------------------------------------------
void
write_glb(const std::string &output_file,
          const DomainData &data,
          const std::string &field,
          const PaletteData &palette,
          double range_min,
          double range_max)
{
    const bool has_field = !field.empty();

    const std::uint32_t index_length =
        checked_u32(data.indices.size() * std::uint64_t(4), "index buffer");
    const std::uint32_t position_length =
        checked_u32(data.positions.size() * std::uint64_t(4), "position buffer");
    const std::uint32_t texcoord_length =
        checked_u32(data.texcoords.size() * std::uint64_t(4),
                    "texture-coordinate buffer");
    const std::uint32_t png_length = checked_u32(palette.png.size(), "png buffer");
    const std::uint32_t index_offset    = 0;
    const std::uint32_t position_offset = align4(std::uint64_t(index_offset) +
                                                 index_length);
    const std::uint32_t uv_offset       = align4(std::uint64_t(position_offset) +
                                                 position_length);
    const std::uint32_t png_offset      = align4(std::uint64_t(uv_offset) +
                                                 texcoord_length);
    const std::uint32_t binary_length   = align4(std::uint64_t(png_offset) +
                                                 png_length);

    Node root;
    root["asset/version"] = "2.0";
    root["asset/generator"] = "Ascent GLTF extract";
    root["extensionsUsed"].append() = "KHR_materials_unlit";
    root["buffers"].append()["byteLength"] = binary_length;

    Node &iview = root["bufferViews"].append();
    iview["buffer"] = 0;
    iview["byteOffset"] = index_offset;
    iview["byteLength"] = index_length;
    iview["target"] = GLTF_ELEMENT_ARRAY_BUFFER;
    Node &pview = root["bufferViews"].append();
    pview["buffer"] = 0;
    pview["byteOffset"] = position_offset;
    pview["byteLength"] = position_length;
    pview["target"] = GLTF_ARRAY_BUFFER;
    if(has_field)
    {
        Node &tview = root["bufferViews"].append();
        tview["buffer"] = 0;
        tview["byteOffset"] = uv_offset;
        tview["byteLength"] = texcoord_length;
        tview["target"] = GLTF_ARRAY_BUFFER;
        Node &png_view = root["bufferViews"].append();
        png_view["buffer"] = 0;
        png_view["byteOffset"] = png_offset;
        png_view["byteLength"] = png_length;
    }

    Node &iaccess = root["accessors"].append();
    iaccess["bufferView"] = 0;
    iaccess["componentType"] = GLTF_COMPONENT_UNSIGNED_INT;
    iaccess["count"] = data.indices.size();
    iaccess["type"] = "SCALAR";
    Node &paccess = root["accessors"].append();
    paccess["bufferView"] = 1;
    paccess["componentType"] = GLTF_COMPONENT_FLOAT;
    paccess["count"] = data.positions.size() / 3;
    paccess["type"] = "VEC3";
    for(int axis = 0; axis < 3; axis++)
    {
        paccess["min"].append() = data.bounds_min[axis];
        paccess["max"].append() = data.bounds_max[axis];
    }
    if(has_field)
    {
        Node &taccess = root["accessors"].append();
        taccess["bufferView"] = 2;
        taccess["componentType"] = GLTF_COMPONENT_FLOAT;
        taccess["count"] = data.texcoords.size() / 2;
        taccess["type"] = "VEC2";
        root["images"].append()["bufferView"] = 3;
        root["images"][0]["mimeType"] = "image/png";
        Node &sampler = root["samplers"].append();
        sampler["wrapS"] = GLTF_WRAP_CLAMP_TO_EDGE;
        sampler["wrapT"] = GLTF_WRAP_CLAMP_TO_EDGE;
        sampler["magFilter"] = palette.discrete ? GLTF_FILTER_NEAREST
                                                : GLTF_FILTER_LINEAR;
        sampler["minFilter"] = palette.discrete ? GLTF_FILTER_NEAREST
                                                : GLTF_FILTER_LINEAR;
        root["textures"].append()["sampler"] = 0;
        root["textures"][0]["source"] = 0;
    }

    Node &material = root["materials"].append();
    material["doubleSided"] = true;
    material["extensions/KHR_materials_unlit"].set(DataType::object());
    if(has_field)
    {
        material["pbrMetallicRoughness/baseColorTexture/index"] = 0;
    }
    else
    {
        double neutral[4] = { 0.65, 0.65, 0.65, 1.0 };
        material["pbrMetallicRoughness/baseColorFactor"].set(neutral, 4);
    }
    material["pbrMetallicRoughness/metallicFactor"] = 0.0;
    material["pbrMetallicRoughness/roughnessFactor"] = 1.0;
    if(palette.transparent)
    {
        material["alphaMode"] = "BLEND";
    }

    Node &primitive = root["meshes"].append()["primitives"].append();
    primitive["indices"] = 0;
    primitive["attributes/POSITION"] = 1;
    if(has_field)
    {
        primitive["attributes/TEXCOORD_0"] = 2;
    }
    primitive["material"] = 0;
    primitive["mode"] = data.mode;
    if(has_field)
    {
        primitive["extras/ascent/field"] = field;
        primitive["extras/ascent/source_dtype"] = data.dtype;
        primitive["extras/ascent/range/min"] = range_min;
        primitive["extras/ascent/range/max"] = range_max;
        primitive["extras/ascent/normalization"] =
            "u=(value-min)/(max-min); value=min+u*(max-min)";
        primitive["extras/ascent/invalid_value_count"] = data.invalid_count;
        primitive["extras/ascent/color_table"].set(palette.metadata);
    }
    root["nodes"].append()["mesh"] = 0;
    root["scenes"].append()["nodes"].append() = 0;
    root["scene"] = 0;

    // conduit has no json boolean type, doubleSided serializes as 1;
    // patch it so the json chunk satisfies the glTF schema
    std::string json = root.to_json();
    const std::string conduit_boolean = "\"doubleSided\": 1";
    const std::size_t boolean_pos = json.find(conduit_boolean);
    if(boolean_pos == std::string::npos)
    {
        ASCENT_ERROR("gltf extract failed to encode the doubleSided "
                     "material property");
    }
    json.replace(boolean_pos, conduit_boolean.size(), "\"doubleSided\": true");
    while(json.size() % 4 != 0)
    {
        json.push_back(' ');
    }

    const std::uint32_t total = align4(12ull + 8 + json.size() + 8 + binary_length);
    std::ofstream ofs(output_file.c_str(), std::ios::binary);
    if(!ofs)
    {
        ASCENT_ERROR("gltf extract failed to open output file '"
                     << output_file << "'");
    }
    ofs.write("glTF", 4);
    write_u32(ofs, 2);
    write_u32(ofs, total);
    write_u32(ofs, checked_u32(json.size(), "json chunk"));
    write_u32(ofs, 0x4e4f534a); // "JSON"
    ofs.write(json.data(), (std::streamsize)json.size());
    write_u32(ofs, binary_length);
    write_u32(ofs, 0x004e4942); // "BIN"
    write_buffer(ofs, data.indices);
    write_padding(ofs, position_offset - index_length);
    write_buffer(ofs, data.positions);
    write_padding(ofs, uv_offset - position_offset - position_length);
    if(has_field)
    {
        write_buffer(ofs, data.texcoords);
        write_padding(ofs, png_offset - uv_offset - texcoord_length);
        write_buffer(ofs, palette.png);
        write_padding(ofs, binary_length - png_offset - png_length);
    }
    if(!ofs)
    {
        ASCENT_ERROR("gltf extract failed while writing output file '"
                     << output_file << "'");
    }
}

};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters::detail --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GltfExtract::GltfExtract()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
GltfExtract::~GltfExtract()
{
// empty
}

//-----------------------------------------------------------------------------
void
GltfExtract::declare_interface(Node &i)
{
    i["type_name"]   = "gltf_extract";
    i["port_names"].append() = "in";
    i["output_port"] = "false";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/path"]);
    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/topology"]);
    number_schema(param_schema["properties/min_value"]);
    number_schema(param_schema["properties/max_value"]);
    detail::gltf_color_table_schema(param_schema["properties/color_table"]);

    param_schema["required"].append() = "path";
}

//-----------------------------------------------------------------------------
void
GltfExtract::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("gltf extract requires a DataObject input");
    }

    // GLB stores all binary payload data little endian
    if(conduit::Endianness::machine_default() != conduit::Endianness::LITTLE_ID)
    {
        ASCENT_ERROR("gltf extract requires a little endian machine");
    }

    DataObject *data_object = input<DataObject>(0);
    // skip on all ranks if any rank lacks valid data
    if(!global_agreement(data_object->is_valid()))
    {
        return;
    }

    std::shared_ptr<Node> n_mesh = data_object->as_low_order_bp();

    std::string field;
    if(params().has_child("field"))
    {
        field = params()["field"].as_string();
    }
    if(field.empty() && (params().has_child("min_value") ||
                         params().has_child("max_value") ||
                         params().has_child("color_table")))
    {
        ASCENT_ERROR("gltf extract: min_value, max_value, and color_table "
                     "require params/field");
    }

    int par_rank = 0;
    int mpi_comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    mpi_comm_id = Workspace::default_mpi_comm();
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
    MPI_Comm_rank(mpi_comm, &par_rank);
#endif

    // we create
    // directory: package_base/
    // files:     package_base/domains/domain_%08d.glb
    // file:      package_base/manifest.json
    std::string output_base = expand_path_special_variables(
                                  params()["path"].as_string(),
                                  "",
                                  mpi_comm_id);
    std::string output_domains_dir = conduit::utils::join_file_path(output_base,
                                                                    "domains");
    if(par_rank == 0)
    {
        if(!conduit::utils::is_directory(output_base))
        {
            conduit::utils::create_directory(output_base);
        }
        if(!conduit::utils::is_directory(output_domains_dir))
        {
            conduit::utils::create_directory(output_domains_dir);
        }
    }
#ifdef ASCENT_MPI_ENABLED
    // all ranks write into the output directories, wait until they exist
    MPI_Barrier(mpi_comm);
#endif

    // read all local domains, tracking the local scalar range
    std::vector<detail::DomainData> domains;
    double local_min =  std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    for(index_t i = 0; i < n_mesh->number_of_children(); i++)
    {
        const Node &dom = n_mesh->child(i);
        detail::DomainData data;
        if(!dom.has_path("state/domain_id"))
        {
            // domain ids name the output files, so they must exist and be
            // globally unique
            ASCENT_ERROR("gltf extract requires state/domain_id in every domain");
        }
        data.id = dom["state/domain_id"].to_index_t();
        if(data.id < 0)
        {
            ASCENT_ERROR("gltf extract: domain ids must be non-negative");
        }
        if(dom.has_path("state/cycle"))
        {
            data.state["cycle"].set(dom["state/cycle"]);
        }
        if(dom.has_path("state/time"))
        {
            data.state["time"].set(dom["state/time"]);
        }

        data.topology = detail::resolve_topology(dom, params(), field);
        const Node &n_topo = dom["topologies/" + data.topology];
        if(!n_topo.has_child("coordset") ||
           !dom.has_path("coordsets/" + n_topo["coordset"].as_string()))
        {
            ASCENT_ERROR("gltf extract: topology '" << data.topology <<
                         "' references a missing coordset");
        }
        detail::read_positions(dom["coordsets/" + n_topo["coordset"].as_string()],
                               data);
        detail::read_connectivity(n_topo, data);

        if(!field.empty())
        {
            const Node &n_field = dom["fields/" + field];
            if(!n_field.has_child("association") ||
               n_field["association"].as_string() != "vertex")
            {
                ASCENT_ERROR("gltf extract requires a vertex field; "
                             "use recenter for element fields");
            }
            if(!n_field.has_child("values") ||
               !n_field["values"].dtype().is_number())
            {
                ASCENT_ERROR("gltf extract requires a numeric scalar field; "
                             "use vector_component or vector_magnitude for "
                             "vector fields");
            }
            data.dtype = n_field["values"].dtype().name();
            data.field_values = detail::numeric_values(n_field["values"]);
            if(data.field_values.size() != data.positions.size() / 3)
            {
                ASCENT_ERROR("gltf extract: vertex field length does not match "
                             "the vertex count");
            }
            for(double value : data.field_values)
            {
                if(std::isfinite(value))
                {
                    local_min = std::min(local_min, value);
                    local_max = std::max(local_max, value);
                }
                else
                {
                    data.invalid_count++;
                }
            }
        }
        domains.push_back(std::move(data));
    }

    // find the global scalar range
    double range_min = local_min;
    double range_max = local_max;
#ifdef ASCENT_MPI_ENABLED
    Node n_local, n_global;
    n_local = local_min;
    conduit::relay::mpi::min_all_reduce(n_local, n_global, mpi_comm);
    range_min = n_global.to_float64();
    n_local = local_max;
    conduit::relay::mpi::max_all_reduce(n_local, n_global, mpi_comm);
    range_max = n_global.to_float64();
#endif
    if(params().has_child("min_value"))
    {
        range_min = params()["min_value"].to_float64();
    }
    if(params().has_child("max_value"))
    {
        range_max = params()["max_value"].to_float64();
    }
    if(!field.empty() && (!std::isfinite(range_min) ||
                          !std::isfinite(range_max) ||
                          range_min > range_max))
    {
        ASCENT_ERROR("gltf extract: scalar range is empty, non-finite, "
                     "or reversed");
    }

    detail::PaletteData palette;
    if(!field.empty())
    {
        palette = detail::make_palette(params());
    }

    // write one glb file per local domain
    Node n_local_records;
    for(detail::DomainData &data : domains)
    {
        if(!field.empty())
        {
            detail::make_texcoords(data.field_values, range_min, range_max, data);
        }
        n_local_records.append().set(detail::domain_record(data, field));
        std::string output_file = conduit::utils::join_file_path(
                                      output_domains_dir,
                                      detail::domain_file_name(data.id));
        detail::write_glb(output_file, data, field, palette,
                          range_min, range_max);
    }

    // gather all domain records on rank 0 for the manifest
    Node n_records;
#ifdef ASCENT_MPI_ENABLED
    Node n_recv;
    conduit::relay::mpi::gather_using_schema(n_local_records, n_recv, 0, mpi_comm);
    if(par_rank == 0)
    {
        for(index_t r = 0; r < n_recv.number_of_children(); r++)
        {
            const Node &n_rank_records = n_recv.child(r);
            for(index_t i = 0; i < n_rank_records.number_of_children(); i++)
            {
                n_records.append().set(n_rank_records.child(i));
            }
        }
    }
#else
    n_records.set_external(n_local_records);
#endif

    // write the manifest on rank 0
    if(par_rank == 0)
    {
        std::vector<const Node *> sorted;
        for(index_t i = 0; i < n_records.number_of_children(); i++)
        {
            sorted.push_back(&n_records.child(i));
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const Node *a, const Node *b)
                  {
                      return (*a)["domain_id"].to_index_t() <
                             (*b)["domain_id"].to_index_t();
                  });
        if(sorted.empty())
        {
            ASCENT_ERROR("gltf extract cannot publish a package with no domains");
        }

        Node manifest;
        manifest["protocol"] = "ascent-gltf";
        manifest["version"] = 1;
        if(!field.empty())
        {
            manifest["field"] = field;
            manifest["range/min"] = range_min;
            manifest["range/max"] = range_max;
            manifest["normalization"] =
                "u=(value-min)/(max-min); value=min+u*(max-min)";
            manifest["color_table"].set(palette.metadata);
        }
        const Node &first = *sorted.front();
        manifest["topology"] = first["topology"].as_string();
        if(first.has_child("cycle"))
        {
            manifest["cycle"].set(first["cycle"]);
        }
        if(first.has_child("time"))
        {
            manifest["time"].set(first["time"]);
        }

        uint64 invalid_count = 0;
        std::string source_dtype = field.empty() ? ""
                                   : first["source_dtype"].as_string();
        for(size_t i = 0; i < sorted.size(); i++)
        {
            const Node &record = *sorted[i];
            if(i > 0 && (*sorted[i - 1])["domain_id"].to_index_t() ==
                        record["domain_id"].to_index_t())
            {
                ASCENT_ERROR("gltf extract: domain ids must be globally unique");
            }
            if(record["topology"].as_string() !=
               manifest["topology"].as_string())
            {
                ASCENT_ERROR("gltf extract: topology selection must be "
                             "consistent across all domains");
            }
            if(record.has_child("cycle") != manifest.has_child("cycle") ||
               (record.has_child("cycle") &&
                record["cycle"].to_index_t() != manifest["cycle"].to_index_t()) ||
               record.has_child("time") != manifest.has_child("time") ||
               (record.has_child("time") &&
                record["time"].to_float64() != manifest["time"].to_float64()))
            {
                ASCENT_ERROR("gltf extract: cycle and time must be consistent "
                             "across all domains");
            }
            invalid_count += record["invalid_value_count"].to_uint64();
            if(!field.empty() && record["source_dtype"].as_string() != source_dtype)
            {
                source_dtype = "mixed";
            }
            manifest["domains"].append().set(record);
        }
        manifest["invalid_value_count"] = invalid_count;
        if(!field.empty())
        {
            manifest["source_dtype"] = source_dtype;
        }

        conduit::relay::io::save(manifest,
                                 conduit::utils::join_file_path(output_base,
                                                                "manifest.json"),
                                 "json");
    }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
