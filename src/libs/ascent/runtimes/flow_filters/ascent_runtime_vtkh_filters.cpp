//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_vtkh_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_vtkh_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_fmt/conduit_fmt.h>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_metadata.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_param_check.hpp>
#include <ascent_runtime_utils.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>
#include <ascent_data_object.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#endif

#if defined(ASCENT_VISKORES_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/rendering/ScalarRenderer.hpp>
#include <vtkh/rendering/AutoCamera.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/CleanGrid.hpp>
#include <vtkh/filters/CompositeVector.hpp>
#include <vtkh/filters/ExternalSurfaces.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/filters/Log.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/filters/Sample.hpp>
#include <vtkh/filters/Slice.hpp>
#include <vtkh/filters/Statistics.hpp>
#include <vtkh/filters/Streamline.hpp>
#include <vtkh/filters/WarpXStreamline.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/filters/Triangulate.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>
#include <vtkh/filters/VectorComponent.hpp>
#include <vtkh/filters/Histogram.hpp>
#include <vtkh/filters/HistSampling.hpp>
#include <vtkh/filters/PointTransform.hpp>
#include <vtkh/filters/MIR.hpp>
#include <viskores/Bounds.h>
#include <viskores/cont/DataSet.h>
#include <viskores/io/VTKDataSetWriter.h>
#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_viskores_parsing.hpp>
#include <ascent_runtime_vtkh_utils.hpp>
#include <ascent_expression_eval.hpp>

#endif

#include <stdio.h>
#include <random>

using namespace conduit;
using namespace std;

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


VTKHMarchingCubes::VTKHMarchingCubes()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHMarchingCubes::~VTKHMarchingCubes()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHMarchingCubes::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_marchingcubes";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/levels"]);
    number_schema(param_schema["properties/iso_values"]);
    string_schema(param_schema["properties/use_contour_tree"]);

    param_schema["required"].append() = "field";
    param_schema["anyOf"].append() = "levels";
    param_schema["anyOf"].append() = "iso_values";
}

//-----------------------------------------------------------------------------
void
VTKHMarchingCubes::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_marchingcubes input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();

    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkh::MarchingCubes marcher;

    marcher.SetInput(&data);
    marcher.SetField(field_name);

    if(params().has_path("iso_values"))
    {
      const Node &n_iso_vals = params()["iso_values"];

      // convert to contig doubles
      Node n_iso_vals_dbls;
      n_iso_vals.to_float64_array(n_iso_vals_dbls);

      marcher.SetIsoValues(n_iso_vals_dbls.as_double_ptr(),
                           n_iso_vals_dbls.dtype().number_of_elements());
    }
    else
    {
      marcher.SetLevels(params()["levels"].to_int32());
      if(params().has_path("use_contour_tree"))
      {
        std::string use = params()["use_contour_tree"].as_string();
        if(use == "true")
        {
          marcher.SetUseContourTree(true);
        }
      }
    }

    marcher.Update();

    vtkh::DataSet *iso_output = marcher.GetOutput();
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*iso_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete iso_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHExternalSurfaces::VTKHExternalSurfaces()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHExternalSurfaces::~VTKHExternalSurfaces()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHExternalSurfaces::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_external_surfaces";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/topology"]);
}

//-----------------------------------------------------------------------------
void
VTKHExternalSurfaces::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHExternalSurfaces input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = true;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
        // this creates a data object with an invalid soource
        set_output<DataObject>(new DataObject());
        return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::ExternalSurfaces ext_surf;
    ext_surf.SetInput(&data);

    ext_surf.Update();

    vtkh::DataSet *ext_surf_output = ext_surf.GetOutput();

    VTKHCollection *new_coll = new VTKHCollection();
    new_coll->add(*ext_surf_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete ext_surf_output;
    set_output<DataObject>(res);
}


//-----------------------------------------------------------------------------
VTKHVectorMagnitude::VTKHVectorMagnitude()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVectorMagnitude::~VTKHVectorMagnitude()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHVectorMagnitude::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_vector_magnitude";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHVectorMagnitude::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_vector_magnitude input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);

    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::VectorMagnitude mag;

    mag.SetInput(&data);
    mag.SetField(field_name);
    if(params().has_path("output_name"))
    {
      std::string output_name = params()["output_name"].as_string();
      mag.SetResultName(output_name);
    }

    mag.Update();

    vtkh::DataSet *mag_output = mag.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*mag_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete mag_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKH3Slice::VTKH3Slice()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKH3Slice::~VTKH3Slice()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKH3Slice::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_3slice";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/topology"]);
    number_schema(param_schema["properties/x_offset"], true);
    number_schema(param_schema["properties/y_offset"], true);
    number_schema(param_schema["properties/z_offset"], true);
}

//-----------------------------------------------------------------------------
void
VTKH3Slice::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKH3Slice input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = false;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
      // this creates a data object with an invalid source
      set_output<DataObject>(new DataObject());
      return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Slice slicer;

    slicer.SetInput(&data);

    using Vec3f = viskores::Vec<viskores::Float32,3>;
    viskores::Bounds bounds = data.GetGlobalBounds();
    Vec3f center = bounds.Center();
    Vec3f x_point = center;
    Vec3f y_point = center;
    Vec3f z_point = center;

    //
    // We look for offsets for each slice plane.
    // Offset values are between -1 and 1 where -1 pushes the plane
    // to the min extent on the bounds and 1 pushes the plane to
    // the max extent
    //

    const float eps = 1e-5; // ensure that the slice is always inside the data set
    if(params().has_path("x_offset"))
    {
      float offset = get_float32(params()["x_offset"], data_object);
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      x_point[0] = bounds.X.Min + t * (bounds.X.Max - bounds.X.Min);
    }

    if(params().has_path("y_offset"))
    {
      float offset = get_float32(params()["y_offset"], data_object);
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      y_point[1] = bounds.Y.Min + t * (bounds.Y.Max - bounds.Y.Min);
    }

    if(params().has_path("z_offset"))
    {
      float offset = get_float32(params()["z_offset"], data_object);
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      z_point[2] = bounds.Z.Min + t * (bounds.Z.Max - bounds.Z.Min);
    }

    Vec3f x_normal(1.f, 0.f, 0.f);
    Vec3f y_normal(0.f, 1.f, 0.f);
    Vec3f z_normal(0.f, 0.f, 1.f);


    slicer.AddPlane(x_point, x_normal);
    slicer.AddPlane(y_point, y_normal);
    slicer.AddPlane(z_point, z_normal);
    slicer.Update();

    vtkh::DataSet *slice_output = slicer.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*slice_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete slice_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHTriangulate::VTKHTriangulate()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHTriangulate::~VTKHTriangulate()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHTriangulate::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_triangulate";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/topology"]);
}

//-----------------------------------------------------------------------------
void
VTKHTriangulate::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHTriangulate input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = false;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkh::Triangulate tri;

    tri.SetInput(&data);

    tri.Update();

    vtkh::DataSet *tri_output = tri.GetOutput();

    VTKHCollection *new_coll = new VTKHCollection();
    new_coll->add(*tri_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete tri_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHCleanGrid::VTKHCleanGrid()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHCleanGrid::~VTKHCleanGrid()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHCleanGrid::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_clean";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    // optional
    string_schema(param_schema["properties/topology"]);
}

//-----------------------------------------------------------------------------
void
VTKHCleanGrid::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHCleanGrid input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = false;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
      // this creates a data object with an invalid source
      set_output<DataObject>(new DataObject());
      return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkh::CleanGrid cleaner;

    cleaner.SetInput(&data);

    cleaner.Update();

    vtkh::DataSet *clean_output = cleaner.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*clean_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete clean_output;
    set_output<DataObject>(res);

}

//-----------------------------------------------------------------------------
VTKHSlice::VTKHSlice()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHSlice::~VTKHSlice()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHSlice::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_slice";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;
    param_schema["constraints/exclusiveChildren"].append() = "sphere";
    param_schema["constraints/exclusiveChildren"].append() = "cylinder";
    param_schema["constraints/exclusiveChildren"].append() = "box";
    param_schema["constraints/exclusiveChildren"].append() = "plane";
    param_schema["constraints/exclusiveChildren"].append() = "point";
    param_schema["constraints/allowNoneInExclusiveGroup"] = false;

    // optional
    string_schema(param_schema["properties/topology"]);

    // --- sphere ---
    conduit::Node &sphere_schema = param_schema["properties/sphere"];
    sphere_schema["type"] = "object";
    sphere_schema["additionalProperties"] = false;
    vec3_schema(sphere_schema["properties/center"], true);
    number_schema(sphere_schema["properties/radius"], true);
    sphere_schema["required"].append() = "center";
    sphere_schema["required"].append() = "radius";

    // --- cylinder ---
    conduit::Node &cylinder_schema = param_schema["properties/cylinder"];
    cylinder_schema["type"] = "object";
    cylinder_schema["additionalProperties"] = false;
    vec3_schema(cylinder_schema["properties/center"], true);
    vec3_schema(cylinder_schema["properties/axis"], true);
    number_schema(cylinder_schema["properties/radius"], true);
    cylinder_schema["required"].append() = "center";
    cylinder_schema["required"].append() = "axis";
    cylinder_schema["required"].append() = "radius";

    // --- box ---
    conduit::Node &box_schema = param_schema["properties/box"];
    box_schema["type"] = "object";
    box_schema["additionalProperties"] = false;
    vec3_schema(box_schema["properties/min"], true);
    vec3_schema(box_schema["properties/max"], true);
    box_schema["required"].append() = "min";
    box_schema["required"].append() = "max";

    // --- plane ---
    conduit::Node &plane_schema = param_schema["properties/plane"];
    plane_schema["type"] = "object";
    plane_schema["additionalProperties"] = false;
    vec3_schema(plane_schema["properties/point"], true);
    vec3_schema(plane_schema["properties/normal"], true);
    plane_schema["required"].append() = "point";
    plane_schema["required"].append() = "normal";

    // --- old point style
    conduit::Node &point_schema = param_schema["properties/point"];
    point_schema["type"] = "object";
    point_schema["additionalProperties"] = false;

    number_schema(point_schema["properties/x"], true);
    number_schema(point_schema["properties/y"], true);
    number_schema(point_schema["properties/z"], true);
    number_schema(point_schema["properties/x_offset"], true);
    number_schema(point_schema["properties/y_offset"], true);
    number_schema(point_schema["properties/z_offset"], true);

    // Option A: explicit
    conduit::Node &option_1_explicit = point_schema["oneOf"].append();
    option_1_explicit["type"] = "object";
    option_1_explicit["required"].append() = "x";
    option_1_explicit["required"].append() = "y";
    option_1_explicit["required"].append() = "z";
    option_1_explicit["constraints/forbid"].append() = "x_offset";
    option_1_explicit["constraints/forbid"].append() = "y_offset";
    option_1_explicit["constraints/forbid"].append() = "z_offset";

    // Option B: offset
    conduit::Node &option_2_offset = point_schema["oneOf"].append();
    option_2_offset["type"] = "object";
    option_2_offset["required"].append() = "x_offset";
    option_2_offset["required"].append() = "y_offset";
    option_2_offset["required"].append() = "z_offset";
    option_2_offset["constraints/forbid"].append() = "x";
    option_2_offset["constraints/forbid"].append() = "y";
    option_2_offset["constraints/forbid"].append() = "z";

    vec3_schema(param_schema["properties/normal"], true);
    param_schema["constraints/dependencies/normal"].append() = "point";
}

//-----------------------------------------------------------------------------
void
VTKHSlice::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHSlice input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = false;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkh::DataSet *slice_output = nullptr;

    // original implementation
    if(params().has_child("point"))
    {
        vtkh::Slice slicer;

        slicer.SetInput(&data);

        const Node &n_point = params()["point"];
        const Node &n_normal = params()["normal"];

        using Vec3f = viskores::Vec<viskores::Float32,3>;
        viskores::Bounds bounds = data.GetGlobalBounds();
        Vec3f point;

        const float eps = 1e-5; // ensure that the slice is always inside the data set

        if(n_point.has_path("x_offset"))
        {
          float offset = get_float32(n_point["x_offset"], data_object);
          float t = (offset + 1.f) / 2.f;
          t = std::max(0.f + eps, std::min(1.f - eps, t));
          point[0] = bounds.X.Min + t * (bounds.X.Max - bounds.X.Min);

          offset = get_float32(n_point["y_offset"], data_object);
          t = (offset + 1.f) / 2.f;
          t = std::max(0.f + eps, std::min(1.f - eps, t));
          point[1] = bounds.Y.Min + t * (bounds.Y.Max - bounds.Y.Min);

          offset = get_float32(n_point["z_offset"], data_object);
          t = (offset + 1.f) / 2.f;
          t = std::max(0.f + eps, std::min(1.f - eps, t));
          point[2] = bounds.Z.Min + t * (bounds.Z.Max - bounds.Z.Min);
        }
        else
        {
          point[0] = get_float32(n_point["x"], data_object);
          point[1] = get_float32(n_point["y"], data_object);
          point[2] = get_float32(n_point["z"], data_object);
        }

        Vec3f v_normal;
        v_normal[0] = get_float32(n_normal["x"], data_object);
        v_normal[1] = get_float32(n_normal["y"], data_object);
        v_normal[2] = get_float32(n_normal["z"], data_object);

        slicer.AddPlane(point, v_normal);

        slicer.Update();

        slice_output = slicer.GetOutput();
    }
    else
    {
        // implicit func slice cases
        vtkh::SliceImplicit slicer;
        slicer.SetInput(&data);

        if(params().has_path("sphere"))
        {
          const Node &sphere = params()["sphere"];
          double center[3];

          center[0] = get_float64(sphere["center/x"], data_object);
          center[1] = get_float64(sphere["center/y"], data_object);
          center[2] = get_float64(sphere["center/z"], data_object);
          double radius = get_float64(sphere["radius"], data_object);
          slicer.SetSphereSlice(center, radius);
        }
        else if(params().has_path("cylinder"))
        {
          const Node &cylinder = params()["cylinder"];
          double center[3];
          double axis[3];

          center[0] = get_float64(cylinder["center/x"], data_object);
          center[1] = get_float64(cylinder["center/y"], data_object);
          center[2] = get_float64(cylinder["center/z"], data_object);

          axis[0] = get_float64(cylinder["axis/x"], data_object);
          axis[1] = get_float64(cylinder["axis/y"], data_object);
          axis[2] = get_float64(cylinder["axis/z"], data_object);

          double radius = get_float64(cylinder["radius"], data_object);
          slicer.SetCylinderSlice(center, axis, radius);
        }
        else if(params().has_path("box"))
        {
          const Node &box = params()["box"];
          viskores::Bounds bounds;
          bounds.X.Min= get_float64(box["min/x"], data_object);
          bounds.Y.Min= get_float64(box["min/y"], data_object);
          bounds.Z.Min= get_float64(box["min/z"], data_object);
          bounds.X.Max = get_float64(box["max/x"], data_object);
          bounds.Y.Max = get_float64(box["max/y"], data_object);
          bounds.Z.Max = get_float64(box["max/z"], data_object);
          slicer.SetBoxSlice(bounds);
        }
        else if(params().has_path("plane"))
        {
          const Node &plane= params()["plane"];
          double point[3], normal[3];;

          point[0] =  get_float64(plane["point/x"], data_object);
          point[1] =  get_float64(plane["point/y"], data_object);
          point[2] =  get_float64(plane["point/z"], data_object);
          normal[0] = get_float64(plane["normal/x"], data_object);
          normal[1] = get_float64(plane["normal/y"], data_object);
          normal[2] = get_float64(plane["normal/z"], data_object);
          slicer.SetPlaneSlice(point, normal);
        }

        slicer.Update();
        slice_output = slicer.GetOutput();
    }
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = new VTKHCollection();
    //= collection->copy_without_topology(topo_name);
    new_coll->add(*slice_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete slice_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHAutoSliceLevels::VTKHAutoSliceLevels()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHAutoSliceLevels::~VTKHAutoSliceLevels()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHAutoSliceLevels::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_autoslicelevels";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    vec3_schema(param_schema["properties/normal"], true);
    number_schema(param_schema["properties/levels"], true);

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "normal";
    param_schema["required"].append() = "levels";
    
    
}

//-----------------------------------------------------------------------------

viskores::Vec<viskores::Float32,3>
GetIntersectionPoint(viskores::Vec<viskores::Float32,3> normal)
{
  //point where normal intersects unit sphere
  viskores::Vec<viskores::Float32,3> point;

  //reverse normal
  //want camera point in the same dir as normal
  viskores::Vec<viskores::Float32,3> r_normal{((viskores::Float32)1.0)*normal[0],
		  			((viskores::Float32)1.0)*normal[1],
					((viskores::Float32)1.0)*normal[2]};

  //calc discriminant
  //a = dot(normal,normal)
  viskores::Float32 r_norm0 = r_normal[0]*r_normal[0];
  viskores::Float32 r_norm1 = r_normal[1]*r_normal[1];
  viskores::Float32 r_norm2 = r_normal[2]*r_normal[2];
  viskores::Float32 a = r_norm0 + r_norm1 + r_norm2;
  //b is 0
  //c is -1
  viskores::Float32 discriminant = 4.0*a;

  viskores::Float32 t =  sqrt(discriminant)/(2*a);
  viskores::Float32 t2 = -t;
  if(abs(t2) < abs(t)) 
    t = t2;

  point[0]= t * r_normal[0];
  point[1]= t * r_normal[1];
  point[2]= t * r_normal[2];

  return point;

}

void
SetCamera(viskores::rendering::Camera *camera, viskores::Vec<viskores::Float32,3> normal, viskores::Float32 radius)
{
  viskores::Vec<viskores::Float32,3> i_point = GetIntersectionPoint(normal);
  viskores::Vec<viskores::Float32,3> lookat = camera->GetLookAt();

  viskores::Vec<viskores::Float32,3> pos;
  viskores::Float32 zoom = 3;
  pos[0] = zoom*radius*i_point[0] + lookat[0];
  pos[1] = zoom*radius*i_point[1] + lookat[1];
  pos[2] = zoom*radius*i_point[2] + lookat[2];

  camera->SetPosition(pos);
}
//-----------------------------------------------------------------------------

void
VTKHAutoSliceLevels::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHAutoSliceLevels input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
        std::string field = params()["field"].as_string();
    if(!collection->has_field(field))
    {
      bool throw_error = false;
      detail::field_error(field, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field);
    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkh::AutoSliceLevels slicer;

    slicer.SetInput(&data);

    const Node &n_normal = params()["normal"];
    const int n_levels = params()["levels"].to_int32();

    using Vec3f = viskores::Vec<viskores::Float32,3>;
    viskores::Bounds bounds = data.GetGlobalBounds();

    Vec3f v_normal;
    v_normal[0] = get_float32(n_normal["x"], data_object);
    v_normal[1] = get_float32(n_normal["y"], data_object);
    v_normal[2] = get_float32(n_normal["z"], data_object);

    slicer.SetNormal(v_normal);
    slicer.SetLevels(n_levels);
    slicer.SetField(field);
    slicer.Update();

    vtkh::DataSet *slice_output = slicer.GetOutput();
    
    //TODO: implement auto camera based on input normal
    //
    //if(!graph().workspace().registry().has_entry("camera"))
    //{
    //  viskores::rendering::Camera *cam = new viskores::rendering::Camera;
    //  viskores::Bounds bounds = slicer.GetDataBounds();
    //  std::cerr << "In Ascent runtime filters" << std::endl;
    //  std::cerr << "X bounds: " << bounds.X.Min << " " << bounds.X.Max << " ";
    //  std::cerr << "Y bounds: " << bounds.Y.Min << " " << bounds.Y.Max << " ";
    //  std::cerr << "Z bounds: " << bounds.Z.Min << " " << bounds.Z.Max << " ";
    //  std::cerr<<std::endl;
    //  viskores::Vec<viskores::Float32,3> normal = slicer.GetNormal();
    //  std::cerr << "normal: " << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;
    //  viskores::Float32 radius = slicer.GetRadius();
    //  std::cerr << "radius: " << radius << std::endl;
    //  SetCamera(cam, normal, radius);
    //  std::cerr << "Cam before registry:" << std::endl;
    //  cam->Print();
    //  std::cerr << "Cam after registry:" << std::endl;
    //  graph().workspace().registry().add<viskores::rendering::Camera>("camera",cam,1);
    //}

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = new VTKHCollection();
    //= collection->copy_without_topology(topo_name);
    new_coll->add(*slice_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete slice_output;
    set_output<DataObject>(res);

}

//-----------------------------------------------------------------------------
VTKHGhostStripper::VTKHGhostStripper()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHGhostStripper::~VTKHGhostStripper()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHGhostStripper::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_ghost_stripper";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/min_value"], true);
    number_schema(param_schema["properties/max_value"], true);

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "min_value";
    param_schema["required"].append() = "max_value";
}

//-----------------------------------------------------------------------------
void
VTKHGhostStripper::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHGhostStripper input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    // ask what topology this field is associated with and
    // get the right data set
    std::string field_name = params()["field"].as_string();

    std::string topo_name = collection->field_topology(field_name);

    bool field_exists = topo_name != "";
    // Check to see of the ghost field even exists
    bool do_strip = field_exists;

    if(do_strip)
    {
      vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
      vtkh::GhostStripper stripper;

      stripper.SetInput(&data);
      stripper.SetField(field_name);

      const Node &n_min_val = params()["min_value"];
      const Node &n_max_val = params()["max_value"];

      int min_val = n_min_val.to_int32();
      int max_val = n_max_val.to_int32();

      stripper.SetMaxValue(max_val);
      stripper.SetMinValue(min_val);

      stripper.Update();

      vtkh::DataSet *stripper_output = stripper.GetOutput();

      // we need to pass through the rest of the topologies, untouched,
      // and add the result of this operation
      VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
      new_coll->add(*stripper_output, topo_name);
      // re wrap in data object
      DataObject *res =  new DataObject(new_coll);
      delete stripper_output;
      set_output<DataObject>(res);
    }
    else
    {
      set_output<DataObject>(data_object);
    }
}

//-----------------------------------------------------------------------------
VTKHAddRanks::VTKHAddRanks()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHAddRanks::~VTKHAddRanks()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHAddRanks::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_add_mpi_ranks";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    // optional
    string_schema(param_schema["properties/topology"]);
    string_schema(param_schema["properties/output"]);
}

//-----------------------------------------------------------------------------
void
VTKHAddRanks::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHAddRanks input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string output_field = "mpi_rank";
    if(params().has_child("output"))
    {
      output_field = params()["output"].as_string();
    }

    std::string topo_name = "";
    if(params().has_child("topology"))
    {
      topo_name = params()["topology"].as_string();
    }
    else
    {
      bool throw_error = false;
      topo_name = detail::resolve_topology(params(),
                                           this->name(),
                                           collection,
                                           throw_error);

      if(topo_name == "")
      {
        // this creates a data object with an invalid source
        set_output<DataObject>(new DataObject());
        return;
      }
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    data.AddConstantCellField(rank,output_field);
    new_coll->add(data, topo_name);
    
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHAddDomains::VTKHAddDomains()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHAddDomains::~VTKHAddDomains()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHAddDomains::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_add_domain_ids";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    // optional
    string_schema(param_schema["properties/topology"]);
    string_schema(param_schema["properties/output"]);
}

//-----------------------------------------------------------------------------
void
VTKHAddDomains::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHAddDomains input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string output_field = "domain_ids";
    if(params().has_child("output"))
    {
      output_field = params()["output"].as_string();
    }

    std::string topo_name = "";
    if(params().has_child("topology"))
    {
      topo_name = params()["topology"].as_string();
    }
    else
    {
      bool throw_error = false;
      topo_name = detail::resolve_topology(params(),
                                           this->name(),
                                           collection,
                                           throw_error);

      if(topo_name == "")
      {
        // this creates a data object with an invalid source
        set_output<DataObject>(new DataObject());
        return;
      }
    }

    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    data.AddDomainIdField(output_field);

    new_coll->add(data, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHThreshold::VTKHThreshold()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHThreshold::~VTKHThreshold()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHThreshold::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_threshold";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;
    param_schema["constraints/exclusiveChildren"].append() = "field";
    param_schema["constraints/exclusiveChildren"].append() = "sphere";
    param_schema["constraints/exclusiveChildren"].append() = "cylinder";
    param_schema["constraints/exclusiveChildren"].append() = "box";
    param_schema["constraints/exclusiveChildren"].append() = "plane";
    param_schema["constraints/exclusiveChildren"].append() = "multi_plane";
    param_schema["constraints/allowNoneInExclusiveGroup"] = false;

    // optional
    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/topology"]);
    number_schema(param_schema["properties/min_value"], true);
    number_schema(param_schema["properties/max_value"], true);
    string_schema(param_schema["properties/invert"]);
    string_schema(param_schema["properties/extract"]);

    param_schema["anyOf"].append() = "field";
    param_schema["anyOf"].append() = "topology";

    // --- sphere ---
    conduit::Node &sphere_schema = param_schema["properties/sphere"];
    sphere_schema["type"] = "object";
    sphere_schema["additionalProperties"] = false;
    vec3_schema(sphere_schema["properties/center"], true);
    number_schema(sphere_schema["properties/radius"], true);
    sphere_schema["required"].append() = "center";
    sphere_schema["required"].append() = "radius";

    // --- cylinder ---
    conduit::Node &cylinder_schema = param_schema["properties/cylinder"];
    cylinder_schema["type"] = "object";
    cylinder_schema["additionalProperties"] = false;
    vec3_schema(cylinder_schema["properties/center"], true);
    vec3_schema(cylinder_schema["properties/axis"], true);
    number_schema(cylinder_schema["properties/radius"], true);
    cylinder_schema["required"].append() = "center";
    cylinder_schema["required"].append() = "axis";
    cylinder_schema["required"].append() = "radius";

    // --- box ---
    conduit::Node &box_schema = param_schema["properties/box"];
    box_schema["type"] = "object";
    box_schema["additionalProperties"] = false;
    vec3_schema(box_schema["properties/min"], true);
    vec3_schema(box_schema["properties/max"], true);
    box_schema["required"].append() = "min";
    box_schema["required"].append() = "max";

    // --- plane ---
    conduit::Node &plane_schema = param_schema["properties/plane"];
    plane_schema["type"] = "object";
    plane_schema["additionalProperties"] = false;
    vec3_schema(plane_schema["properties/point"], true);
    vec3_schema(plane_schema["properties/normal"], true);
    plane_schema["required"].append() = "point";
    plane_schema["required"].append() = "normal";

    // --- multi plane ---
    conduit::Node &multi_plane_schema = param_schema["properties/multi_plane"];
    multi_plane_schema["type"] = "object";
    multi_plane_schema["additionalProperties"] = false;
    vec3_schema(multi_plane_schema["properties/point1"], true);
    vec3_schema(multi_plane_schema["properties/point2"], true);
    vec3_schema(multi_plane_schema["properties/normal1"], true);
    vec3_schema(multi_plane_schema["properties/normal2"], true);
    multi_plane_schema["required"].append() = "point1";
    multi_plane_schema["required"].append() = "point2";
    multi_plane_schema["required"].append() = "normal1";
    multi_plane_schema["required"].append() = "normal2";
}

//-----------------------------------------------------------------------------
void
VTKHThreshold::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_threshold input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    // find right topology, either via field name or topology param
    std::string topo_name = "";

    if(params().has_child("field"))
    {
        std::string field_name = params()["field"].as_string();
        if(!collection->has_field(field_name))
        {
          bool throw_error = false;
          detail::field_error(field_name, this->name(), collection, throw_error);
          // this creates a data object with an invalid source
          set_output<DataObject>(new DataObject());
          return;
        }

        topo_name = collection->field_topology(field_name);

    }
    else
    {
        bool throw_error = false;
        topo_name = detail::resolve_topology(params(),
                                           this->name(),
                                           collection,
                                           throw_error);
        if(topo_name == "")
        {
            // this creates a data object with an invalid source
            set_output<DataObject>(new DataObject());
            return;
        }
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Threshold thresher;
    thresher.SetInput(&data);

    if(params().has_child("invert"))
    {
      std::string invert = params()["invert"].as_string();
      if(invert == "true")
      {
        thresher.SetInvertThreshold(true);
      }
    }

    // field case
    if(params().has_child("field"))
    {
        std::string field_name = params()["field"].as_string();
        thresher.SetField(field_name);

        const Node &n_min_val = params()["min_value"];
        const Node &n_max_val = params()["max_value"];

        // convert to contig doubles
        double min_val = get_float64(n_min_val, data_object);
        double max_val = get_float64(n_max_val, data_object);
        thresher.SetFieldUpperThreshold(max_val);
        thresher.SetFieldLowerThreshold(min_val);
    }
    else // spatial select cases
    {
        if(params().has_path("sphere"))
        {
          const Node &sphere = params()["sphere"];
          double center[3];

          center[0] = get_float64(sphere["center/x"], data_object);
          center[1] = get_float64(sphere["center/y"], data_object);
          center[2] = get_float64(sphere["center/z"], data_object);
          double radius = get_float64(sphere["radius"], data_object);
          thresher.SetSphereThreshold(center, radius);
        }
        else if(params().has_path("cylinder"))
        {
          const Node &cylinder = params()["cylinder"];
          double center[3];
          double axis[3];

          center[0] = get_float64(cylinder["center/x"], data_object);
          center[1] = get_float64(cylinder["center/y"], data_object);
          center[2] = get_float64(cylinder["center/z"], data_object);

          axis[0] = get_float64(cylinder["axis/x"], data_object);
          axis[1] = get_float64(cylinder["axis/y"], data_object);
          axis[2] = get_float64(cylinder["axis/z"], data_object);

          double radius = get_float64(cylinder["radius"], data_object);
          thresher.SetCylinderThreshold(center, axis, radius);
        }
        else if(params().has_path("box"))
        {
          const Node &box = params()["box"];
          viskores::Bounds bounds;
          bounds.X.Min= get_float64(box["min/x"], data_object);
          bounds.Y.Min= get_float64(box["min/y"], data_object);
          bounds.Z.Min= get_float64(box["min/z"], data_object);
          bounds.X.Max = get_float64(box["max/x"], data_object);
          bounds.Y.Max = get_float64(box["max/y"], data_object);
          bounds.Z.Max = get_float64(box["max/z"], data_object);
          thresher.SetBoxThreshold(bounds);
        }
        else if(params().has_path("plane"))
        {
          const Node &plane= params()["plane"];
          double point[3], normal[3];;

          point[0] =  get_float64(plane["point/x"], data_object);
          point[1] =  get_float64(plane["point/y"], data_object);
          point[2] =  get_float64(plane["point/z"], data_object);
          normal[0] = get_float64(plane["normal/x"], data_object);
          normal[1] = get_float64(plane["normal/y"], data_object);
          normal[2] = get_float64(plane["normal/z"], data_object);
          thresher.SetPlaneThreshold(point, normal);
        }
        // else if(params().has_path("multi_plane"))
        // {
        //   const Node &plane= params()["multi_plane"];
        //   double point1[3], normal1[3], point2[3], normal2[3];
        //
        //   point1[0] = get_float64(plane["point1/x"], data_object);
        //   point1[1] = get_float64(plane["point1/y"], data_object);
        //   point1[2] = get_float64(plane["point1/z"], data_object);
        //   normal1[0] = get_float64(plane["normal1/x"], data_object);
        //   normal1[1] = get_float64(plane["normal1/y"], data_object);
        //   normal1[2] = get_float64(plane["normal1/z"], data_object);
        //   point2[0] = get_float64(plane["point2/x"], data_object);
        //   point2[1] = get_float64(plane["point2/y"], data_object);
        //   point2[2] = get_float64(plane["point2/z"], data_object);
        //   normal2[0] = get_float64(plane["normal2/x"], data_object);
        //   normal2[1] = get_float64(plane["normal2/y"], data_object);
        //   normal2[2] = get_float64(plane["normal2/z"], data_object);
        //   clipper.Set2PlaneClip(point1, normal1, point2, normal2);
        // }

    }
    
    thresher.Update();
    vtkh::DataSet *thresh_output = thresher.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*thresh_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete thresh_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHClip::VTKHClip()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHClip::~VTKHClip()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHClip::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_clip";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;
    param_schema["constraints/exclusiveChildren"].append() = "sphere";
    param_schema["constraints/exclusiveChildren"].append() = "cylinder";
    param_schema["constraints/exclusiveChildren"].append() = "box";
    param_schema["constraints/exclusiveChildren"].append() = "plane";
    param_schema["constraints/exclusiveChildren"].append() = "multi_plane";
    param_schema["constraints/allowNoneInExclusiveGroup"] = false;

    // optional
    string_schema(param_schema["properties/topology"]);
    string_schema(param_schema["properties/invert"]);

    // --- sphere ---
    conduit::Node &sphere_schema = param_schema["properties/sphere"];
    sphere_schema["type"] = "object";
    sphere_schema["additionalProperties"] = false;
    vec3_schema(sphere_schema["properties/center"], true);
    number_schema(sphere_schema["properties/radius"], true);
    sphere_schema["required"].append() = "center";
    sphere_schema["required"].append() = "radius";

    // --- cylinder ---
    conduit::Node &cylinder_schema = param_schema["properties/cylinder"];
    cylinder_schema["type"] = "object";
    cylinder_schema["additionalProperties"] = false;
    vec3_schema(cylinder_schema["properties/center"], true);
    vec3_schema(cylinder_schema["properties/axis"], true);
    number_schema(cylinder_schema["properties/radius"], true);
    cylinder_schema["required"].append() = "center";
    cylinder_schema["required"].append() = "axis";
    cylinder_schema["required"].append() = "radius";

    // --- box ---
    conduit::Node &box_schema = param_schema["properties/box"];
    box_schema["type"] = "object";
    box_schema["additionalProperties"] = false;
    vec3_schema(box_schema["properties/min"], true);
    vec3_schema(box_schema["properties/max"], true);
    box_schema["required"].append() = "min";
    box_schema["required"].append() = "max";

    // --- plane ---
    conduit::Node &plane_schema = param_schema["properties/plane"];
    plane_schema["type"] = "object";
    plane_schema["additionalProperties"] = false;
    vec3_schema(plane_schema["properties/point"], true);
    vec3_schema(plane_schema["properties/normal"], true);
    plane_schema["required"].append() = "point";
    plane_schema["required"].append() = "normal";

    // --- multi plane ---
    conduit::Node &multi_plane_schema = param_schema["properties/multi_plane"];
    multi_plane_schema["type"] = "object";
    multi_plane_schema["additionalProperties"] = false;
    vec3_schema(multi_plane_schema["properties/point1"], true);
    vec3_schema(multi_plane_schema["properties/point2"], true);
    vec3_schema(multi_plane_schema["properties/normal1"], true);
    vec3_schema(multi_plane_schema["properties/normal2"], true);
    multi_plane_schema["required"].append() = "point1";
    multi_plane_schema["required"].append() = "point2";
    multi_plane_schema["required"].append() = "normal1";
    multi_plane_schema["required"].append() = "normal2";
}

//-----------------------------------------------------------------------------
void
VTKHClip::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHClip input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = false;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
      // this creates a data object with an invalid source
      set_output<DataObject>(new DataObject());
      return;
    }


    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Clip clipper;

    clipper.SetInput(&data);

    if(params().has_path("sphere"))
    {
      const Node &sphere = params()["sphere"];
      double center[3];

      center[0] = get_float64(sphere["center/x"], data_object);
      center[1] = get_float64(sphere["center/y"], data_object);
      center[2] = get_float64(sphere["center/z"], data_object);
      double radius = get_float64(sphere["radius"], data_object);
      clipper.SetSphereClip(center, radius);
    }
    else if(params().has_path("cylinder"))
    {
      const Node &cylinder = params()["cylinder"];
      double center[3];
      double axis[3];

      center[0] = get_float64(cylinder["center/x"], data_object);
      center[1] = get_float64(cylinder["center/y"], data_object);
      center[2] = get_float64(cylinder["center/z"], data_object);

      axis[0] = get_float64(cylinder["axis/x"], data_object);
      axis[1] = get_float64(cylinder["axis/y"], data_object);
      axis[2] = get_float64(cylinder["axis/z"], data_object);

      double radius = get_float64(cylinder["radius"], data_object);
      clipper.SetCylinderClip(center, axis, radius);
    }
    else if(params().has_path("box"))
    {
      const Node &box = params()["box"];
      viskores::Bounds bounds;
      bounds.X.Min= get_float64(box["min/x"], data_object);
      bounds.Y.Min= get_float64(box["min/y"], data_object);
      bounds.Z.Min= get_float64(box["min/z"], data_object);
      bounds.X.Max = get_float64(box["max/x"], data_object);
      bounds.Y.Max = get_float64(box["max/y"], data_object);
      bounds.Z.Max = get_float64(box["max/z"], data_object);
      clipper.SetBoxClip(bounds);
    }
    else if(params().has_path("plane"))
    {
      const Node &plane= params()["plane"];
      double point[3], normal[3];;

      point[0] =  get_float64(plane["point/x"], data_object);
      point[1] =  get_float64(plane["point/y"], data_object);
      point[2] =  get_float64(plane["point/z"], data_object);
      normal[0] = get_float64(plane["normal/x"], data_object);
      normal[1] = get_float64(plane["normal/y"], data_object);
      normal[2] = get_float64(plane["normal/z"], data_object);
      clipper.SetPlaneClip(point, normal);
    }
    else if(params().has_path("multi_plane"))
    {
      const Node &plane= params()["multi_plane"];
      double point1[3], normal1[3], point2[3], normal2[3];

      point1[0] = get_float64(plane["point1/x"], data_object);
      point1[1] = get_float64(plane["point1/y"], data_object);
      point1[2] = get_float64(plane["point1/z"], data_object);
      normal1[0] = get_float64(plane["normal1/x"], data_object);
      normal1[1] = get_float64(plane["normal1/y"], data_object);
      normal1[2] = get_float64(plane["normal1/z"], data_object);
      point2[0] = get_float64(plane["point2/x"], data_object);
      point2[1] = get_float64(plane["point2/y"], data_object);
      point2[2] = get_float64(plane["point2/z"], data_object);
      normal2[0] = get_float64(plane["normal2/x"], data_object);
      normal2[1] = get_float64(plane["normal2/y"], data_object);
      normal2[2] = get_float64(plane["normal2/z"], data_object);
      clipper.Set2PlaneClip(point1, normal1, point2, normal2);
    }

    if(params().has_child("invert"))
    {
      std::string invert = params()["invert"].as_string();
      if(invert == "true")
      {
        clipper.SetInvertClip(true);
      }
    }

    clipper.Update();

    vtkh::DataSet *clip_output = clipper.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*clip_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete clip_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHClipWithField::VTKHClipWithField()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHClipWithField::~VTKHClipWithField()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHClipWithField::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_clip_with_field";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    number_schema(param_schema["properties/clip_value"], true);
    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/invert"]);

    param_schema["required"].append() = "clip_value";
    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHClipWithField::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHClipWithField input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::ClipField clipper;

    clipper.SetInput(&data);

    if(params().has_child("invert"))
    {
      std::string invert = params()["invert"].as_string();
      if(invert == "true")
      {
        clipper.SetInvertClip(true);
      }
    }

    viskores::Float64 clip_value = get_float64(params()["clip_value"], data_object);

    clipper.SetField(field_name);
    clipper.SetClipValue(clip_value);

    clipper.Update();

    vtkh::DataSet *clip_output = clipper.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*clip_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete clip_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHIsoVolume::VTKHIsoVolume()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHIsoVolume::~VTKHIsoVolume()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHIsoVolume::declare_interface(Node &i)
{
    i["type_name"] = "vtkh_iso_volume";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    number_schema(param_schema["properties/min_value"], true);
    number_schema(param_schema["properties/max_value"], true);
    string_schema(param_schema["properties/field"]);

    param_schema["required"].append() = "min_value";
    param_schema["required"].append() = "max_value";
    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHIsoVolume::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHIsoVolume input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::IsoVolume clipper;

    clipper.SetInput(&data);

    viskores::Range clip_range;
    clip_range.Min = get_float64(params()["min_value"], data_object);
    clip_range.Max = get_float64(params()["max_value"], data_object);

    clipper.SetField(field_name);
    clipper.SetRange(clip_range);

    clipper.Update();

    vtkh::DataSet *clip_output = clipper.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*clip_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete clip_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHLagrangian::VTKHLagrangian()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHLagrangian::~VTKHLagrangian()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHLagrangian::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_lagrangian";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/step_size"]);
    number_schema(param_schema["properties/write_frequency"]);
    number_schema(param_schema["properties/cust_res"]);
    number_schema(param_schema["properties/x_res"]);
    number_schema(param_schema["properties/y_res"]);
    number_schema(param_schema["properties/z_res"]);

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "step_size";
    param_schema["required"].append() = "write_frequency";
    param_schema["required"].append() = "cust_res";
    param_schema["required"].append() = "x_res";
    param_schema["required"].append() = "y_res";
    param_schema["required"].append() = "z_res";
}

//-----------------------------------------------------------------------------
void
VTKHLagrangian::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_lagrangian input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);


    double step_size = params()["step_size"].to_float64();
    int write_frequency = params()["write_frequency"].to_int32();
    int cust_res = params()["cust_res"].to_int32();
    int x_res = params()["x_res"].to_int32();
    int y_res = params()["y_res"].to_int32();
    int z_res = params()["z_res"].to_int32();

    vtkh::Lagrangian lagrangian;

    lagrangian.SetInput(&data);
    lagrangian.SetField(field_name);
    lagrangian.SetStepSize(step_size);
    lagrangian.SetWriteFrequency(write_frequency);
    lagrangian.SetCustomSeedResolution(cust_res);
    lagrangian.SetSeedResolutionInX(x_res);
    lagrangian.SetSeedResolutionInY(y_res);
    lagrangian.SetSeedResolutionInZ(z_res);
    lagrangian.Update();

    vtkh::DataSet *lagrangian_output = lagrangian.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*lagrangian_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete lagrangian_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHLog::VTKHLog()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHLog::~VTKHLog()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHLog::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_log";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    number_schema(param_schema["properties/clamp_min_value"], true);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHLog::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_log input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Log logger;
    logger.SetInput(&data);
    logger.SetField(field_name);
    if(params().has_path("output_name"))
    {
      logger.SetResultField(params()["output_name"].as_string());
    }

    if(params().has_path("clamp_min_value"))
    {
      double min_value = get_float64(params()["clamp_min_value"], data_object);
      logger.SetClampMin(min_value);
      logger.SetClampToMin(true);
    }

    logger.Update();

    vtkh::DataSet *log_output = logger.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*log_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete log_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHLog10::VTKHLog10()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHLog10::~VTKHLog10()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHLog10::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_log10";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    number_schema(param_schema["properties/clamp_min_value"], true);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHLog10::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_log10 input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Log10 logger;
    logger.SetInput(&data);
    logger.SetField(field_name);
    if(params().has_path("output_name"))
    {
      logger.SetResultField(params()["output_name"].as_string());
    }

    if(params().has_path("clamp_min_value"))
    {
      double min_value = get_float64(params()["clamp_min_value"], data_object);
      logger.SetClampMin(min_value);
      logger.SetClampToMin(true);
    }

    logger.Update();

    vtkh::DataSet *log_output = logger.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*log_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete log_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHLog2::VTKHLog2()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHLog2::~VTKHLog2()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHLog2::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_log2";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    number_schema(param_schema["properties/clamp_min_value"], true);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHLog2::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_log2 input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Log2 logger;
    logger.SetInput(&data);
    logger.SetField(field_name);
    if(params().has_path("output_name"))
    {
      logger.SetResultField(params()["output_name"].as_string());
    }

    if(params().has_path("clamp_min_value"))
    {
      double min_value = get_float64(params()["clamp_min_value"], data_object);
      logger.SetClampMin(min_value);
      logger.SetClampToMin(true);
    }

    logger.Update();

    vtkh::DataSet *log_output = logger.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*log_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete log_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHRecenter::VTKHRecenter()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHRecenter::~VTKHRecenter()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHRecenter::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_recenter";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/association"]);

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "association";
}

//-----------------------------------------------------------------------------
void
VTKHRecenter::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_recenter input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);


    std::string association = params()["association"].as_string();
    if(association != "vertex" && association != "element")
    {
      ASCENT_ERROR("Recenter: resulting field association '"<<association<<"'"
                   <<" must have a value of 'vertex' or 'element'");
    }

    vtkh::Recenter recenter;

    recenter.SetInput(&data);
    recenter.SetField(field_name);

    if(association == "vertex")
    {
      recenter.SetResultAssoc(viskores::cont::Field::Association::Points);
    }
    if(association == "element")
    {
      recenter.SetResultAssoc(viskores::cont::Field::Association::Cells);
    }

    recenter.Update();

    vtkh::DataSet *recenter_output = recenter.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*recenter_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete recenter_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHHistSampling::VTKHHistSampling()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHHistSampling::~VTKHHistSampling()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHHistSampling::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_hist_sampling";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/bins"], true);
    number_schema(param_schema["properties/sample_rate"], true);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHHistSampling::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_hist_sampling input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    float sample_rate = .1f;
    if(params().has_path("sample_rate"))
    {
      sample_rate = get_float32(params()["sample_rate"], data_object);
      if(sample_rate <= 0.f || sample_rate >= 1.f)
      {
        ASCENT_ERROR("vtkh_hist_sampling 'sample_rate' value '"<<sample_rate<<"'"
                     <<" not in the range (0,1)");
      }
    }

    int bins = 128;

    if(params().has_path("bins"))
    {
      bins = get_int32(params()["bins"], data_object);
      if(bins <= 0.f)
      {
        ASCENT_ERROR("vtkh_hist_sampling 'bins' value '"<<bins<<"'"
                     <<" must be positive");
      }
    }

    // TODO: write helper functions for this
    std::string ghost_field = "";
    Node meta = Metadata::n_metadata;

    if(meta.has_path("ghost_field"))
    {

      // there can be multiple ghost fields on different topologies
      // We should only find one(max) associated with this vtkh data set
      const conduit::Node ghost_list = meta["ghost_field"];
      const int num_ghosts = ghost_list.number_of_children();

      for(int i = 0; i < num_ghosts; ++i)
      {
        std::string ghost = ghost_list.child(i).as_string();
        if(data.GlobalFieldExists(ghost_field))
        {
          ghost_field = ghost;
          break;
        }
      }

    }

    vtkh::HistSampling hist;

    hist.SetInput(&data);
    hist.SetField(field_name);
    hist.SetNumBins(bins);
    hist.SetSamplingPercent(sample_rate);
    if(ghost_field != "")
    {
      hist.SetGhostField(ghost_field);
    }

    hist.Update();
    vtkh::DataSet *hist_output = hist.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*hist_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete hist_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHQCriterion::VTKHQCriterion()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHQCriterion::~VTKHQCriterion()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHQCriterion::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_qcriterion";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    string_schema(param_schema["properties/use_cell_gradient"]);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHQCriterion::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_qcriterion input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Gradient grad;
    grad.SetInput(&data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;
    grad_params.compute_qcriterion = true;

    // set the output name of the gradient result to
    // a unique temp name
    grad_params.output_name = "__tmp_gradient";

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }
    if(params().has_path("output_name"))
    {
      grad_params.qcriterion_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    // remove the gradient result (not the q-crit)
    // since downstream viskores filters may not be able to handle
    // the "vec of vec" gradient result
    grad_output->RemoveField("__tmp_gradient");

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*grad_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete grad_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHDivergence::VTKHDivergence()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHDivergence::~VTKHDivergence()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHDivergence::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_divergence";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    string_schema(param_schema["properties/use_cell_gradient"]);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHDivergence::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_divergence input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Gradient grad;
    grad.SetInput(&data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;
    grad_params.compute_divergence = true;
    // set the output name of the gradient result to
    // a unique temp name
    grad_params.output_name = "__tmp_gradient";

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }

    if(params().has_path("output_name"))
    {
      grad_params.divergence_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    // remove the gradient result
    // since downstream filters may not be able to handle
    grad_output->RemoveField("__tmp_gradient");

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*grad_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete grad_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHVorticity::VTKHVorticity()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVorticity::~VTKHVorticity()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHVorticity::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_curl";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    string_schema(param_schema["properties/use_cell_gradient"]);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHVorticity::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_vorticity input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Gradient grad;
    grad.SetInput(&data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;
    grad_params.compute_vorticity = true;
    // set the output name of the gradient result to
    // a unique temp name
    grad_params.output_name = "__tmp_gradient";

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }

    if(params().has_path("output_name"))
    {
      grad_params.vorticity_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    // remove the gradient result
    // since downstream filters may not be able to handle
    grad_output->RemoveField("__tmp_gradient");

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*grad_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete grad_output;
    set_output<DataObject>(res);
}
//-----------------------------------------------------------------------------

VTKHGradient::VTKHGradient()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHGradient::~VTKHGradient()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHGradient::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_gradient";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/output_name"]);
    string_schema(param_schema["properties/use_cell_gradient"]);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHGradient::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_gradient input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Gradient grad;
    grad.SetInput(&data);
    grad.SetField(field_name);
    vtkh::GradientParameters grad_params;

    if(params().has_path("use_cell_gradient"))
    {
      if(params()["use_cell_gradient"].as_string() == "true")
      {
        grad_params.use_point_gradient = false;
      }
    }

    if(params().has_path("output_name"))
    {
      grad_params.output_name = params()["output_name"].as_string();
    }

    grad.SetParameters(grad_params);
    grad.Update();

    vtkh::DataSet *grad_output = grad.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*grad_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete grad_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHUniformGrid::VTKHUniformGrid()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHUniformGrid::~VTKHUniformGrid()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHUniformGrid::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_uniform_grid";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    array_schema(param_schema["properties/fields"]);
    vec3_schema_anyOf(param_schema["properties/dims"], "i", "j", "k");
    vec3_schema_anyOf(param_schema["properties/origin"]);
    vec3_schema_anyOf(param_schema["properties/spacing"], "dx", "dy", "dz");
    number_schema(param_schema["properties/invalid_value"]);

    param_schema["anyOf"].append() = "field";
    param_schema["anyOf"].append() = "fields";
}

//-----------------------------------------------------------------------------
void
VTKHUniformGrid::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_uniform_grid input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::vector<std::string> field_selection;
    if(params().has_path("field"))
    {
      std::string field = params()["field"].as_string();

      field_selection.push_back(field);
    }
    else if(params().has_path("fields"))
    {
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("vtkh_uniform_grid fields selection list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("vtkh_uniform_grid fields selection list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
    }
    else
    {
      ASCENT_ERROR("vtkh_uniform_grid must specify a 'field' string or 'fields' list of strings");
    }

    int num_fields = field_selection.size(); 
    for(int i = 0; i < num_fields; ++i)
    {
      if(!collection->has_field(field_selection[i]))
      {
        bool throw_error = false;
        detail::field_error(field_selection[i], this->name(), collection, throw_error);
        // this creates a data object with an invalid soource
        set_output<DataObject>(new DataObject());
        return;
      }
    }

    std::string field = field_selection[0];
    std::string topo_name = collection->field_topology(field);
    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    viskores::Id global_cells = data.GetGlobalNumberOfCells();

    viskores::Bounds d_bounds = data.GetGlobalBounds();
    viskores::Float64 x_extents = d_bounds.X.Length() + 1; //add one b/c we are
    viskores::Float64 y_extents = d_bounds.Y.Length() + 1; //setting num points
    viskores::Float64 z_extents = d_bounds.Z.Length() + 1; //(not cells) in each dim

    viskores::Float64 invalid_value = 0.0;
    
    using Vec3f = viskores::Vec<viskores::Float64,3>;
    Vec3f v_dims    = {x_extents, y_extents, z_extents}; 
    Vec3f v_origin  = {d_bounds.X.Min,d_bounds.Y.Min,d_bounds.Z.Min};
    Vec3f v_spacing = {1.,1.,1.};

    if(params().has_path("dims"))
    {
      const Node &n_dims = params()["dims"];
      if(n_dims.has_path("i"))
        v_dims[0] = get_float64(n_dims["i"], data_object);
      if(n_dims.has_path("j"))
        v_dims[1] = get_float64(n_dims["j"], data_object);
      if(n_dims.has_path("k"))
        v_dims[2] = get_float64(n_dims["k"], data_object);

      v_dims[0] = (v_dims[0] > 0) ? (v_dims[0]) : 1;
      v_dims[1] = (v_dims[1] > 0) ? (v_dims[1]) : 1;
      v_dims[2] = (v_dims[2] > 0) ? (v_dims[2]) : 1;
      v_spacing[0] = x_extents/v_dims[0];
      v_spacing[1] = y_extents/v_dims[1];
      v_spacing[2] = z_extents/v_dims[2];
    }
    if(params().has_path("origin"))
    {
      const Node &n_origin = params()["origin"];
      if(n_origin.has_path("x"))
        v_origin[0] = get_float64(n_origin["x"], data_object);
      if(n_origin.has_path("y"))
        v_origin[1] = get_float64(n_origin["y"], data_object);
      if(n_origin.has_path("z"))
        v_origin[2] = get_float64(n_origin["z"], data_object);
    }
    if(params().has_path("spacing"))
    {
      const Node &n_spacing = params()["spacing"];
      if(n_spacing.has_path("dx"))
        v_spacing[0] = get_float64(n_spacing["dx"], data_object);
      if(n_spacing.has_path("dy"))
        v_spacing[1] = get_float64(n_spacing["dy"], data_object);
      if(n_spacing.has_path("dz"))
        v_spacing[2] = get_float64(n_spacing["dz"], data_object);

      v_spacing[0] = (v_spacing[0] > 0) ? (v_spacing[0]) : 1;
      v_spacing[1] = (v_spacing[1] > 0) ? (v_spacing[1]) : 1;
      v_spacing[2] = (v_spacing[2] > 0) ? (v_spacing[2]) : 1;
    }
    if(params().has_path("invalid_value"))
    {
      invalid_value = params()["invalid_value"].to_float64();
    }

    vtkh::Sample sampler;

    sampler.UniformGrid(v_dims, v_origin, v_spacing);
    sampler.InvalidValue(invalid_value);
    sampler.Fields(field_selection);
    sampler.SetInput(&data);

    sampler.Update();

    vtkh::DataSet *grid_output = sampler.GetOutput();

    VTKHCollection *new_coll = new VTKHCollection();
    new_coll->add(*grid_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete grid_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
VTKHSample::VTKHSample()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHSample::~VTKHSample()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHSample::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_sample";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    array_schema(param_schema["properties/fields"]);
    string_schema(param_schema["properties/topology"]);
    number_schema(param_schema["properties/invalid_value"]);

    // --- Line ---
    conduit::Node &line_schema = param_schema["properties/line"];
    line_schema["type"] = "object";
    line_schema["additionalProperties"] = false;
    number_schema(line_schema["properties/num_samples"]);
    vec3_schema_anyOf(line_schema["properties/start"]);
    vec3_schema_anyOf(line_schema["properties/end"]);

    // --- Points ---
    vec3_schema_anyOf(param_schema["properties/points"]);

    // --- Plane ---
    conduit::Node &plane_schema = param_schema["properties/plane"];
    plane_schema["type"] = "object";
    plane_schema["additionalProperties"] = false;
    vec3_schema(plane_schema["properties/point"], true);
    vec3_schema(plane_schema["properties/normal"], true);
    conduit::Node &plane_dims_schema = plane_schema["properties/dims"];
    plane_dims_schema["type"] = "object";
    plane_dims_schema["additionalProperties"] = false;
    number_schema(plane_dims_schema["properties/i"], true);
    number_schema(plane_dims_schema["properties/j"], true);
    number_schema(plane_dims_schema["properties/k"], true);
    conduit::Node &plane_spacing_schema = plane_schema["properties/spacing"];
    plane_spacing_schema["type"] = "object";
    plane_spacing_schema["additionalProperties"] = false;
    number_schema(plane_spacing_schema["properties/dx"], true);
    number_schema(plane_spacing_schema["properties/dy"], true);
    number_schema(plane_spacing_schema["properties/dz"], true);
    plane_schema["required"].append() = "point";
    plane_schema["required"].append() = "normal";
    plane_schema["required"].append() = "dims";
    plane_schema["required"].append() = "spacing";

    // --- Box ---
    conduit::Node &box_schema = param_schema["properties/box"];
    box_schema["type"] = "object";
    box_schema["additionalProperties"] = false;
    vec3_schema_anyOf(box_schema["properties/dims"]);
    vec3_schema_anyOf(box_schema["properties/min"]);
    vec3_schema_anyOf(box_schema["properties/max"]);

    // --- Uniform Grid ---
    conduit::Node &uniform_grid_schema = param_schema["properties/uniform_grid"];
    uniform_grid_schema["type"] = "object";
    uniform_grid_schema["additionalProperties"] = false;
    vec3_schema_anyOf(uniform_grid_schema["properties/dims"], "i", "j", "k");
    vec3_schema_anyOf(uniform_grid_schema["properties/origin"]);
    vec3_schema_anyOf(uniform_grid_schema["properties/spacing"], "dx", "dy", "dz");

    param_schema["anyOf"].append() = "field";
    param_schema["anyOf"].append() = "fields";
}

//-----------------------------------------------------------------------------
void
VTKHSample::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_sample input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::vector<std::string> field_selection;
    if(params().has_path("field"))
    {
      std::string field = params()["field"].as_string();

      field_selection.push_back(field);
    }
    else if(params().has_path("fields"))
    {
      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();
      if(num_fields == 0)
      {
        ASCENT_ERROR("vtkh_sample fields selection list must be non-empty");
      }
      for(int i = 0; i < num_fields; ++i)
      {
        const conduit::Node &f = flist.child(i);
        if(!f.dtype().is_string())
        {
           ASCENT_ERROR("vtkh_sample fields selection list values must be a string");
        }
        field_selection.push_back(f.as_string());
      }
    }
    else
    {
      ASCENT_ERROR("vtkh_sample must specify a 'field' string or 'fields' list of strings");
    }

    int num_fields = field_selection.size(); 
    for(int i = 0; i < num_fields; ++i)
    {
      if(!collection->has_field(field_selection[i]))
      {
        bool throw_error = false;
        detail::field_error(field_selection[i], this->name(), collection, throw_error);
        // this creates a data object with an invalid source
        set_output<DataObject>(new DataObject());
        return;
      }
    }

    std::string field = field_selection[0];
    std::string topo_name = collection->field_topology(field);
    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Sample sampler;
    std::string output_topo_name = topo_name;

    const bool has_topology = params().has_path("topology");
    if(has_topology)
    {
      std::string sample_topo_name = params()["topology"].as_string();
      if(!collection->has_topology(sample_topo_name))
      {
        ASCENT_ERROR("vtkh_sample topology '" << sample_topo_name
                     << "' does not exist. Known topologies: "
                     << detail::possible_topologies(collection));
      }

      std::shared_ptr<VTKHCollection> sample_collection;
#ifdef ASCENT_MPI_ENABLED
      MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
      int rank = 0;
      MPI_Comm_rank(mpi_comm, &rank);

      std::shared_ptr<conduit::Node> blueprint_data = data_object->as_low_order_bp();
      conduit::Node sample_mesh;
      if(rank == 0)
      {
        const conduit::Node &bp = *blueprint_data;
        const auto domains = blueprint::mesh::domains(bp);

        //make a conduit::Node sample mesh with input topo
        for(size_t i = 0; i < domains.size(); ++i)
        {
          const conduit::Node &src_dom = *domains[i];
          const std::string topo_path = "topologies/" + sample_topo_name;
          if(src_dom.has_path(topo_path))
          {
            conduit::Node &dst_dom = sample_mesh.append();
            if(src_dom.has_path("state"))
            {
              dst_dom["state"].set(src_dom["state"]);
            }

            dst_dom[topo_path].set(src_dom[topo_path]);
            const std::string coordset_name = src_dom[topo_path + "/coordset"].as_string();
            dst_dom["coordsets/" + coordset_name].set(src_dom["coordsets/" + coordset_name]);
          }
        }

        if(sample_mesh.number_of_children() == 0)
        {
          ASCENT_ERROR("vtkh_sample topology '" << sample_topo_name
                       << "' must be present on rank 0 for MPI topology sampling");
        }
      }
	  //broadcast created sample node and convert back to vtkh
      conduit::relay::mpi::broadcast_using_schema(sample_mesh, 0, mpi_comm);
      sample_collection.reset(VTKHDataAdapter::BlueprintToVTKHCollection(sample_mesh,
                                                                         false));
#else
      sample_collection = collection;
#endif

      vtkh::DataSet &sample_data = sample_collection->dataset_by_topology(sample_topo_name);
      std::vector<viskores::cont::DataSet> sample_domains;
      std::vector<viskores::Id> sample_domain_ids = sample_data.GetDomainIds();
      for(size_t i = 0; i < sample_domain_ids.size(); ++i)
      {
        sample_domains.push_back(sample_data.GetDomainById(sample_domain_ids[i]));
      }
      sampler.Topology(sample_domains, sample_domain_ids);
      output_topo_name = sample_topo_name;
    }
    else if(params().has_path("line"))
    {
      const Node &line_p = params()["line"];
      int num_samples = line_p["num_samples"].to_int();

      double sx = line_p["start/x"].to_double();
      double sy = line_p["start/y"].to_double();
      double sz = line_p["start"].has_child("z") ? line_p["start/z"].to_double() : 0.0;
      double ex = line_p["end/x"].to_double();
      double ey = line_p["end/y"].to_double();
      double ez = line_p["end"].has_child("z") ? line_p["end/z"].to_double() : 0.0;

      sampler.Line(num_samples,
                   sx, sy, sz, 
                   ex, ey, ez);
    }
    else if(params().has_path("points"))
    {
        viskores::cont::ArrayHandle<viskores::Float64> x_hnd, y_hnd, z_hnd;
        
        int spatial_dims = 2;
        float64_accessor x_vals, y_vals, z_vals;
        const Node &pts_p = params()["points"];
        x_vals = pts_p["x"].value();
        y_vals = pts_p["y"].value();

        int npts = x_vals.dtype().number_of_elements();
        x_hnd.Allocate(npts);
        y_hnd.Allocate(npts);

        if(pts_p.has_child("z"))
        {
          spatial_dims = 3;
          z_vals = pts_p["z"].value();
          z_hnd.Allocate(npts);
        }

        for(int i=0;i<npts;i++)
        {
            x_hnd.WritePortal().Set(i,x_vals[i]);
            y_hnd.WritePortal().Set(i,y_vals[i]);
            if(spatial_dims ==3)
            {
                z_hnd.WritePortal().Set(i,z_vals[i]);
            }
        }
        
        sampler.Points(x_hnd, y_hnd, z_hnd);

    }
    else if(params().has_path("plane"))
    {
      using Vec2_f64 = viskores::Vec<viskores::Float64,2>;
      using Vec3_f64 = viskores::Vec<viskores::Float64,3>;

      const Node &plane_p = params()["plane"];
      const Node &point_p = plane_p["point"];
      const Node &normal_p = plane_p["normal"];
      const Node &dims_p = plane_p["dims"];
      const Node &spacing_p = plane_p["spacing"];
      const std::string dim_names[3] = {"i", "j", "k"};
      const std::string spacing_names[3] = {"dx", "dy", "dz"};

      Vec3_f64 point = {
        get_float64(point_p["x"], data_object),
        get_float64(point_p["y"], data_object),
        get_float64(point_p["z"], data_object)
      };
      Vec3_f64 normal = {
        get_float64(normal_p["x"], data_object),
        get_float64(normal_p["y"], data_object),
        get_float64(normal_p["z"], data_object)
      };

      int axes[2];
      int num_axes = 0;
      for(int axis = 0; axis < 3; ++axis)
      {
        const bool has_dim = dims_p.has_path(dim_names[axis]);
        const bool has_spacing = spacing_p.has_path(spacing_names[axis]);
        if(has_dim != has_spacing)
        {
          ASCENT_ERROR("vtkh_sample plane dims and spacing must specify the same axes");
        }
        if(has_dim)
        {
          if(num_axes == 2)
          {
            ASCENT_ERROR("vtkh_sample plane must specify exactly two axes");
          }
          axes[num_axes++] = axis;
        }
      }
      if(num_axes != 2)
      {
        ASCENT_ERROR("vtkh_sample plane must specify exactly two axes");
      }

      Vec2_f64 dims = {
        get_float64(dims_p[dim_names[axes[0]]], data_object),
        get_float64(dims_p[dim_names[axes[1]]], data_object)
      };
      Vec2_f64 spacing = {
        get_float64(spacing_p[spacing_names[axes[0]]], data_object),
        get_float64(spacing_p[spacing_names[axes[1]]], data_object)
      };
      dims[0] = static_cast<viskores::Float64>(static_cast<int>(dims[0]));
      dims[1] = static_cast<viskores::Float64>(static_cast<int>(dims[1]));
      if(dims[0] <= 0.0 || dims[1] <= 0.0)
      {
        ASCENT_ERROR("vtkh_sample plane dims values must be greater than zero");
      }
      if(spacing[0] <= 0.0 || spacing[1] <= 0.0)
      {
        ASCENT_ERROR("vtkh_sample plane spacing values must be greater than zero");
      }

      sampler.Plane(point, normal, dims, spacing, axes);
    }
    else if(params().has_path("uniform_grid"))
    {
      viskores::Bounds d_bounds = data.GetGlobalBounds();
      viskores::Float64 x_extents = d_bounds.X.Length() + 1; //add one b/c we are
      viskores::Float64 y_extents = d_bounds.Y.Length() + 1; //setting num points
      viskores::Float64 z_extents = d_bounds.Z.Length() + 1; //(not cells) in each dim
  
      using Vec3_f64 = viskores::Vec<viskores::Float64,3>;
      Vec3_f64 v_dims    = {x_extents, y_extents, z_extents}; 
      Vec3_f64 v_origin  = {d_bounds.X.Min,d_bounds.Y.Min,d_bounds.Z.Min};
      Vec3_f64 v_spacing = {1.,1.,1.};
  
      if(params().has_path("uniform_grid/dims"))
      {
        const Node &n_dims = params()["uniform_grid/dims"];
        if(n_dims.has_path("i"))
          v_dims[0] = get_float64(n_dims["i"], data_object);
        if(n_dims.has_path("j"))
          v_dims[1] = get_float64(n_dims["j"], data_object);
        if(n_dims.has_path("k"))
          v_dims[2] = get_float64(n_dims["k"], data_object);
  
        v_dims[0] = (v_dims[0] > 0) ? (v_dims[0]) : 1;
        v_dims[1] = (v_dims[1] > 0) ? (v_dims[1]) : 1;
        v_dims[2] = (v_dims[2] > 0) ? (v_dims[2]) : 1;

        v_spacing[0] = x_extents/v_dims[0];
        v_spacing[1] = y_extents/v_dims[1];
        v_spacing[2] = z_extents/v_dims[2];
      }
      if(params().has_path("uniform_grid/origin"))
      {
        const Node &n_origin = params()["uniform_grid/origin"];
        if(n_origin.has_path("x"))
          v_origin[0] = get_float64(n_origin["x"], data_object);
        if(n_origin.has_path("y"))
          v_origin[1] = get_float64(n_origin["y"], data_object);
        if(n_origin.has_path("z"))
          v_origin[2] = get_float64(n_origin["z"], data_object);
      }
      if(params().has_path("uniform_grid/spacing"))
      {
        const Node &n_spacing = params()["uniform_grid/spacing"];
        if(n_spacing.has_path("dx"))
          v_spacing[0] = get_float64(n_spacing["dx"], data_object);
        if(n_spacing.has_path("dy"))
          v_spacing[1] = get_float64(n_spacing["dy"], data_object);
        if(n_spacing.has_path("dz"))
          v_spacing[2] = get_float64(n_spacing["dz"], data_object);

        v_spacing[0] = (v_spacing[0] > 0) ? (v_spacing[0]) : 1;
        v_spacing[1] = (v_spacing[1] > 0) ? (v_spacing[1]) : 1;
        v_spacing[2] = (v_spacing[2] > 0) ? (v_spacing[2]) : 1;
      }
      sampler.UniformGrid(v_dims, v_origin, v_spacing);
    }
    else if(params().has_path("box"))
    {
      int dims[3];
      viskores::Float64 x_min, x_max, y_min, y_max, z_min, z_max;
      viskores::Bounds g_bounds = data.GetGlobalBounds();
      const Node &dims_b = params()["box/dims"];
      const Node &min_b = params()["box/min"];
      const Node &max_b = params()["box/max"];

      //Grab Dims
      if(dims_b.has_child("i"))
      {
        dims[0] = dims_b["i"].to_int();
      }
      else
        dims[0] = 1;

      if(dims_b.has_child("j"))
      {
        dims[1] = dims_b["j"].to_int();
      }
      else
        dims[1] = 1;

      if(dims_b.has_child("k"))
      {
        dims[2] = dims_b["k"].to_int();
      }
      else
        dims[2] = 1;
      
      //Grab Mins
      if(min_b.has_child("x"))
      {
        if(min_b["x"].dtype().is_string())
        {
          if(min_b["x"].as_string() != "min")
            ASCENT_ERROR("minimum value for x must be the string `min` or a scalar double");
          x_min = g_bounds.X.Min;
        }
        else
        {
          x_min = min_b["x"].to_float64();
        }
      }
      else //not set; default min
      {
        x_min = g_bounds.X.Min;
      }

      if(min_b.has_child("y"))
      {
        if(min_b["y"].dtype().is_string())
        {
          if(min_b["y"].as_string() != "min")
            ASCENT_ERROR("minimum value for y must be the string `min` or a scalar double");
          y_min = g_bounds.Y.Min;
        }
        else
        {
          y_min = min_b["y"].to_float64();
        }
      }
      else //not set; default min
      {
        y_min = g_bounds.Y.Min;
      }

      if(min_b.has_child("z"))
      {
        if(min_b["z"].dtype().is_string())
        {
          if(min_b["z"].as_string() != "min")
            ASCENT_ERROR("minimum value for z must be the string `min` or a scalar double");
          z_min = g_bounds.Z.Min;
        }
        else
        {
          z_min = min_b["z"].to_float64();
        }
      }
      else //not set; default min
      {
        z_min = g_bounds.Z.Min;
      }

      //Grab Maxes
      if(max_b.has_child("x"))
      {
        if(max_b["x"].dtype().is_string())
        {
          if(max_b["x"].as_string() != "max")
            ASCENT_ERROR("maximum value for x must be the string `max` or a scalar double");
          x_max = g_bounds.X.Max;
        }
        else
        {
          x_max = max_b["x"].to_float64();
        }
      }
      else //not set; default max
      {
        x_max = g_bounds.X.Max;
      }

      if(max_b.has_child("y"))
      {
        if(max_b["y"].dtype().is_string())
        {
          if(max_b["y"].as_string() != "max")
            ASCENT_ERROR("maximum value for y must be the string `max` or a scalar double");
          y_max = g_bounds.Y.Max;
        }
        else
        {
          y_max = max_b["y"].to_float64();
        }
      }
      else //not set; default max
      {
        y_max = g_bounds.Y.Max;
      }

      if(max_b.has_child("z"))
      {
        if(max_b["z"].dtype().is_string())
        {
          if(max_b["z"].as_string() != "max")
            ASCENT_ERROR("maximum value for z must be the string `max` or a scalar double");
          z_max = g_bounds.Z.Max;
        }
        else
        {
          z_max = max_b["z"].to_float64();
        }
      }
      else //not set; default max
      {
        z_max = g_bounds.Z.Max;
      }

      sampler.Box(dims, x_min, y_min, z_min, x_max, y_max, z_max);
    }

    double invalid_value = 0.0;
    if(params().has_path("invalid_value"))
    {
      invalid_value = params()["invalid_value"].to_float64();
    }

    sampler.InvalidValue(invalid_value);
    sampler.Fields(field_selection);
    sampler.SetInput(&data);

    sampler.Update();

    vtkh::DataSet *grid_output = sampler.GetOutput();

    VTKHCollection *new_coll = new VTKHCollection();
    new_coll->add(*grid_output, output_topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete grid_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

VTKHStats::VTKHStats()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHStats::~VTKHStats()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHStats::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_stats";
    i["port_names"].append() = "in";
    i["output_port"] = "false";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHStats::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_stats input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::Statistics stats;
    stats.SetField(field_name);
    stats.SetInput(&data);
    stats.Update();

    vtkh::DataSet* res = stats.GetOutput();
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    // if(rank == 0)
    // {
    //   res->PrintSummary(std::cout);
    // }
}
//-----------------------------------------------------------------------------

VTKHHistogram::VTKHHistogram()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHHistogram::~VTKHHistogram()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHHistogram::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_histogram";
    i["port_names"].append() = "in";
    i["output_port"] = "false";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/bins"], true);

    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHHistogram::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_histogram input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    int bins = 128;
    if(params().has_path("bins"))
    {
      bins = get_int32(params()["bins"], data_object);
    }

    vtkh::Histogram hist;

    hist.SetNumBins(bins);
    vtkh::Histogram::HistogramResult res = hist.Run(data, field_name);
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    // if(rank == 0)
    // {
    //   res.Print(std::cout);
    // }
}
//-----------------------------------------------------------------------------

VTKHProject2d::VTKHProject2d()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHProject2d::~VTKHProject2d()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHProject2d::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_project_2d";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/topology"]);
    integer_schema(param_schema["properties/image_width"]);
    integer_schema(param_schema["properties/image_height"]);
    ignore_schema(param_schema["properties/dataset_bounds"]);
    ignore_schema(param_schema["properties/camera"]);
    ignore_schema(param_schema["properties/rays"]);
    ignore_schema(param_schema["properties/result"]);
    array_schema(param_schema["properties/fields"]);
}

//-----------------------------------------------------------------------------
void
VTKHProject2d::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_project2d input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool throw_error = false;
    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     throw_error);
    if(topo_name == "")
    {
      // this creates a data object with an invalid source
      set_output<DataObject>(new DataObject());
      return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    viskores::Bounds bounds = data.GetGlobalBounds();

    if(params().has_path("dataset_bounds"))
    {
        float64_accessor d_bounds = params()["dataset_bounds"].value();
        int num_bounds = d_bounds.number_of_elements();

        if(num_bounds != 6)
        {
            std::string fpath = filter_to_path(this->name());
            ASCENT_ERROR("project_2d (" << fpath << ")" <<
                         " only provided " << num_bounds <<
                         " dataset_bounds when 6 are required:" <<
                         " [xMin,xMax,yMin,yMax,zMin,zMax]");
        }

        bounds.X.Min = d_bounds[0];
        bounds.X.Max = d_bounds[1];
        bounds.Y.Min = d_bounds[2];
        bounds.Y.Max = d_bounds[3];
        bounds.Z.Min = d_bounds[4];
        bounds.Z.Max = d_bounds[5];
    }

    viskores::rendering::Camera camera;
    camera.ResetToBounds(bounds);

    std::string source = "camera"; // or rays
    std::string result = "image";  // or rays

    std::vector<std::string> field_names;

    if(params().has_path("result"))
    {
      result = params()["result"].as_string();
    }

    if(params().has_path("camera"))
    {
      parse_camera(params()["camera"], camera);
    }

    if(params().has_path("fields"))
    {

      const conduit::Node &flist = params()["fields"];
      const int num_fields = flist.number_of_children();

      if(num_fields == 0)
      {
        ASCENT_ERROR("'fields' list must be non-empty");
      }

      for(int i = 0; i < num_fields; i++)
      {
        const conduit::Node &f = flist.child(i); 
        if(!f.dtype().is_string())
        {
            ASCENT_ERROR("'fields' list values must be a string");
        }
        field_names.push_back(f.as_string());
      }
    }

    int width  = 512;
    int height = 512;
    if(params().has_path("image_width"))
    {
      width = params()["image_width"].to_int32();
    }
    if(params().has_path("image_height"))
    {
      height = params()["image_height"].to_int32();
    }
    
    // handle rays
    if(params().has_path("rays"))
    {
        // points/x,y,z
        // normals/x,y,z
        source = "rays";
    }


    vtkh::ScalarRenderer tracer;
    tracer.SetInput(&data);
    tracer.SetFields(field_names);
    
    if(source == "camera")
    {
        tracer.SetCamera(camera);
        tracer.SetWidth(width);
        tracer.SetHeight(height);
    }
    else if( source == "rays")
    {
        float64_accessor pts  = params()["rays/points"].value();
        float64_accessor nmls = params()["rays/normals"].value();
        float64 max_dist      = params()["rays/max_distance"].to_value();
        index_t number_of_rays = pts.number_of_elements() / 3;

        viskores::cont::ArrayHandle<viskores::Float64> pts_x;
        viskores::cont::ArrayHandle<viskores::Float64> pts_y;
        viskores::cont::ArrayHandle<viskores::Float64> pts_z;

        viskores::cont::ArrayHandle<viskores::Float64> dirs_x;
        viskores::cont::ArrayHandle<viskores::Float64> dirs_y;
        viskores::cont::ArrayHandle<viskores::Float64> dirs_z;

        pts_x.Allocate(number_of_rays);
        pts_y.Allocate(number_of_rays);
        pts_z.Allocate(number_of_rays);

        dirs_x.Allocate(number_of_rays);
        dirs_y.Allocate(number_of_rays);
        dirs_z.Allocate(number_of_rays);
  
        auto pts_x_portal = pts_x.WritePortal();
        auto pts_y_portal = pts_y.WritePortal();
        auto pts_z_portal = pts_z.WritePortal();

        auto dirs_x_portal = dirs_x.WritePortal();
        auto dirs_y_portal = dirs_y.WritePortal();
        auto dirs_z_portal = dirs_z.WritePortal();

        index_t idx = 0;
        for(index_t i=0; i < number_of_rays; i++)
        {
            pts_x_portal.Set(i,pts[idx]);
            pts_y_portal.Set(i,pts[idx+1]);
            pts_z_portal.Set(i,pts[idx+2]);
            dirs_x_portal.Set(i,nmls[idx]);
            dirs_y_portal.Set(i,nmls[idx+1]);
            dirs_z_portal.Set(i,nmls[idx+2]);

            idx+=3;
        }

        tracer.SetRays(pts_x, pts_y, pts_z,
                       dirs_x, dirs_y, dirs_z,
                       max_dist);
    }

    tracer.Update();

    vtkh::DataSet *output = tracer.GetOutput();
    VTKHCollection *new_coll = new VTKHCollection();
    new_coll->add(*output, topo_name);
    
    // TODO: Add rays mesh as well?
    //tracer.GenerateRaysMesh()
    
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHNoOp::VTKHNoOp()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHNoOp::~VTKHNoOp()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHNoOp::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_no_op";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    param_schema["required"].append() = "field";
}

//-----------------------------------------------------------------------------
void
VTKHNoOp::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_no_op input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::NoOp noop;

    noop.SetInput(&data);
    noop.SetField(field_name);

    noop.Update();

    vtkh::DataSet *noop_output = noop.GetOutput();
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*noop_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete noop_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHVectorComponent::VTKHVectorComponent()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVectorComponent::~VTKHVectorComponent()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHVectorComponent::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_vector_component";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/component"]);
    string_schema(param_schema["properties/output_name"]);

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "component";
    param_schema["required"].append() = "output_name";
}

//-----------------------------------------------------------------------------
void
VTKHVectorComponent::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_vector_component input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }
    int component = params()["component"].to_int32();
    std::string res_name = params()["output_name"].as_string();

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::VectorComponent comp;

    comp.SetInput(&data);
    comp.SetField(field_name);
    comp.SetComponent(component);
    comp.SetResultField(res_name);

    comp.Update();

    vtkh::DataSet *comp_output = comp.GetOutput();
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*comp_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete comp_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHCompositeVector::VTKHCompositeVector()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHCompositeVector::~VTKHCompositeVector()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHCompositeVector::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_composite_vector";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field1"]);
    string_schema(param_schema["properties/field2"]);
    string_schema(param_schema["properties/field3"]);
    string_schema(param_schema["properties/output_name"]);

    param_schema["required"].append() = "field1";
    param_schema["required"].append() = "field2";
    param_schema["required"].append() = "output_name";
}

//-----------------------------------------------------------------------------
void
VTKHCompositeVector::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_composite_vector input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name1 = params()["field1"].as_string();
    if(!collection->has_field(field_name1))
    {
      bool throw_error = false;
      detail::field_error(field_name1, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string field_name2 = params()["field2"].as_string();
    if(!collection->has_field(field_name2))
    {
      bool throw_error = false;
      detail::field_error(field_name2, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string field_name3;

    if(params().has_path("field3"))
    {
      field_name3 = params()["field3"].as_string();
      if(!collection->has_field(field_name3))
      {
        bool throw_error = false;
        detail::field_error(field_name3, this->name(), collection, throw_error);
        // this creates a data object with an invalid soource
        set_output<DataObject>(new DataObject());
        return;
      }
    }

    std::string topo_name = collection->field_topology(field_name1);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);


    vtkh::CompositeVector comp;

    comp.SetInput(&data);
    if(field_name3 == "")
    {
      comp.SetFields(field_name1, field_name2);
    }
    else
    {
      comp.SetFields(field_name1, field_name2, field_name3);
    }
    std::string res_name = params()["output_name"].as_string();
    comp.SetResultField(res_name);
    comp.Update();

    vtkh::DataSet *comp_output = comp.GetOutput();
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*comp_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete comp_output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHScale::VTKHScale()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHScale::~VTKHScale()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHScale::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_scale_transform";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    vec3_schema(i["param_schema"], "x_scale", "y_scale", "z_scale", true);
}

//-----------------------------------------------------------------------------
void
VTKHScale::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_point_transform input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    float x_scale = get_float32(params()["x_scale"], data_object);
    float y_scale = get_float32(params()["y_scale"], data_object);
    float z_scale = get_float32(params()["z_scale"], data_object);

    std::vector<std::string> topo_names = collection->topology_names();
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif

    VTKHCollection *new_coll = new VTKHCollection();
    for(auto &topo : topo_names)
    {
      vtkh::DataSet &data = collection->dataset_by_topology(topo);
      vtkh::PointTransform transform;
      transform.SetScale(x_scale, y_scale, z_scale);
      transform.SetInput(&data);
      transform.Update();
      vtkh::DataSet *trans_output = transform.GetOutput();
      new_coll->add(*trans_output, topo);
      delete trans_output;
    }

    //// re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    set_output<DataObject>(res);
}


//-----------------------------------------------------------------------------

VTKHTransform::VTKHTransform()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHTransform::~VTKHTransform()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHTransform::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_transform";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;
    param_schema["constraints/exclusiveChildren"].append() = "scale";
    param_schema["constraints/exclusiveChildren"].append() = "translate";
    param_schema["constraints/exclusiveChildren"].append() = "reflect";
    param_schema["constraints/exclusiveChildren"].append() = "rotate";
    param_schema["constraints/exclusiveChildren"].append() = "matrix";
    param_schema["constraints/allowNoneInExclusiveGroup"] = false;

    vec3_schema_anyOf(param_schema["properties/scale"], true);
    vec3_schema_anyOf(param_schema["properties/translate"], true);

    // --- rotate ---
    conduit::Node &rotate_schema = param_schema["properties/rotate"];
    rotate_schema["type"] = "object";
    rotate_schema["additionalProperties"] = false;
    number_schema(rotate_schema["properties/angle"], true);
    vec3_schema_anyOf(rotate_schema["properties/axis"], true);
    rotate_schema["required"].append() = "angle";
    rotate_schema["required"].append() = "axis";

    // --- reflect ---
    conduit::Node &reflect_schema = param_schema["properties/reflect"];
    reflect_schema["type"] = "object";
    reflect_schema["additionalProperties"] = false;
    vec3_schema_anyOf(reflect_schema["properties/normal"], true);
    vec3_schema_anyOf(reflect_schema["properties/point"], true);

    // --- matrix ---
    conduit::Node matrix_item_schema;
    conduit::Node &matrix_schema = array_schema(param_schema["properties/matrix"], number_schema(matrix_item_schema, true), 16, 16);
}

//-----------------------------------------------------------------------------
void
VTKHTransform::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_transform input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    bool use_scale     = false;
    bool use_translate = false;
    bool use_rotate    = false;
    bool use_reflect   = false;
    bool use_matrix    = false;

    double t_scale[3]          = {1.0, 1.0, 1.0};
    double t_translate[3]      = {0.0, 0.0, 0.0};
    double t_rotate_angle      =  0.0;
    double t_rotate_axis[3]    = {0.0, 0.0, 0.0};
    double t_reflect_normal[3] = {0.0, 0.0, 0.0}; 
    double t_reflect_point[3]  = {0.0, 0.0, 0.0};
    double t_matrix[16]        = {0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0};
    ParamSpec reflect_param[3];

    if(params().has_child("scale"))
    {
        use_scale = true;
        const Node &p_vals = params()["scale"];
        if(p_vals.has_child("x"))
        {
            t_scale[0] = get_float64(p_vals["x"],data_object);
        }

        if(p_vals.has_child("y"))
        {
            t_scale[1] = get_float64(p_vals["y"],data_object);
        }

        if(p_vals.has_child("z"))
        {
            t_scale[2] = get_float64(p_vals["z"],data_object);
        }
    }

    if(params().has_child("translate"))
    {
        use_translate = true;
        const Node &p_vals = params()["translate"];
        if(p_vals.has_child("x"))
        {
            t_translate[0] = get_float64(p_vals["x"],data_object);
        }

        if(p_vals.has_child("y"))
        {
            t_translate[1] = get_float64(p_vals["y"],data_object);
        }

        if(p_vals.has_child("z"))
        {
            t_translate[2] = get_float64(p_vals["z"],data_object);
        }
    }

    if(params().has_child("rotate"))
    {
        use_rotate = true;
        const Node &p_vals = params()["rotate"];

        t_rotate_angle = get_float64(p_vals["angle"],data_object);

        const Node &p_axis = p_vals["axis"];
        if(p_axis.has_child("x"))
        {
            t_rotate_axis[0] = get_float64(p_axis["x"],data_object);
        }

        if(p_axis.has_child("y"))
        {
            t_rotate_axis[1] = get_float64(p_axis["y"],data_object);
        }

        if(p_axis.has_child("z"))
        {
            t_rotate_axis[2] = get_float64(p_axis["z"],data_object);
        }
    }

    if(params().has_child("reflect"))
    {
      use_reflect = true;
      const Node &n_reflect = params()["reflect"];
      const Node &n_vals = n_reflect["normal"];
      if(n_vals.has_child("x"))
      {
        t_reflect_normal[0] = get_float64(n_vals["x"], data_object);
      }

      if(n_vals.has_child("y"))
      {
        t_reflect_normal[1] = get_float64(n_vals["y"], data_object);
      }

      if(n_vals.has_child("z"))
      {
        t_reflect_normal[2] = get_float64(n_vals["z"], data_object);
      }

      if((t_reflect_normal[0] == 0.0) &&
         (t_reflect_normal[1] == 0.0) && 
         (t_reflect_normal[2] == 0.0))
        ASCENT_ERROR("A non-zero normal must be specified.");
      
      if(n_reflect.has_child("point"))
      {
        const Node &p_vals = n_reflect["point"];
        if(p_vals.has_child("x"))
        {
          reflect_param[0] = assign_param_spec(p_vals["x"],data_object);
        }

        if(p_vals.has_child("y"))
        {
          reflect_param[1] = assign_param_spec(p_vals["y"],data_object);
        }

        if(p_vals.has_child("z"))
        {
          reflect_param[2] = assign_param_spec(p_vals["z"],data_object);
        }
      }
    }

    if(params().has_child("matrix"))
    {
        use_matrix = true;
        // matrix
        float64_accessor matrix_vals = params()["matrix"].value();
        for(index_t i=0;i<16;i++)
        {
            t_matrix[i] = matrix_vals[i];
        }
    }

    std::vector<std::string> topo_names = collection->topology_names();
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif

    VTKHCollection *new_coll = new VTKHCollection();
    for(auto &topo : topo_names)
    {
      vtkh::DataSet &data = collection->dataset_by_topology(topo);
      vtkh::PointTransform transform;

      if(use_scale)
      {
          transform.SetScale(t_scale[0],
                             t_scale[1],
                             t_scale[2]);
      }

      if(use_translate)
      {
          transform.SetTranslation(t_translate[0],
                                   t_translate[1],
                                   t_translate[2]);
      }

      if(use_rotate)
      {
          transform.SetRotation(t_rotate_angle,
                                t_rotate_axis[0],
                                t_rotate_axis[1],
                                t_rotate_axis[2]);
      }

      if(use_reflect)
      {
          viskores::Bounds bounds = data.GetGlobalBounds();
          //resolve the ParamSpec types now that we have bounds 
          auto resolve = [&](int axis) -> double
          {
            switch(reflect_param[axis].mode)
            {
              case ParamVal::Unset:     
                return 0.0;//default

              case ParamVal::Value:     
                return reflect_param[axis].value;

              case ParamVal::BoundsMin:       
                switch(axis)
                {
                  case 0: return bounds.X.Min;
                  case 1: return bounds.Y.Min;
                  case 2: return bounds.Z.Min;
                }
                break;

              case ParamVal::BoundsMax:
                switch(axis)
                {
                  case 0: return bounds.X.Max;
                  case 1: return bounds.Y.Max;
                  case 2: return bounds.Z.Max;
                }
                break;
            }
            return 0.0;
          };

          t_reflect_point[0] = resolve(0);
          t_reflect_point[1] = resolve(1);
          t_reflect_point[2] = resolve(2);
          
          transform.SetReflect(t_reflect_point[0],
                               t_reflect_point[1],
                               t_reflect_point[2],
                               t_reflect_normal[0],
                               t_reflect_normal[1],
                               t_reflect_normal[2]);
      }

      if(use_matrix)
      {
          transform.SetTransform(t_matrix);
      }

      transform.SetInput(&data);
      transform.Update();
      vtkh::DataSet *trans_output = transform.GetOutput();
      new_coll->add(*trans_output, topo);
      delete trans_output;
    }

    //// re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHParticleAdvection::VTKHParticleAdvection()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHParticleAdvection::~VTKHParticleAdvection()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHParticleAdvection::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_particle_advection";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    number_schema(param_schema["properties/num_steps"], true);
    number_schema(param_schema["properties/step_size"], true);

    // --- seed ---
    conduit::Node &seeds_schema = param_schema["properties/seeds"];
    seeds_schema["type"] = "object";
    seeds_schema["additionalProperties"] = false;
    string_schema(seeds_schema["properties/type"]);
    number_schema(seeds_schema["properties/location"]);
    number_schema(seeds_schema["properties/start"]);
    number_schema(seeds_schema["properties/end"]);
    number_schema(seeds_schema["properties/num_seeds"]);
    number_schema(seeds_schema["properties/num_seeds_x"]);
    number_schema(seeds_schema["properties/num_seeds_y"]);
    number_schema(seeds_schema["properties/num_seeds_z"]);
    number_schema(seeds_schema["properties/extents_x"]);
    number_schema(seeds_schema["properties/extents_y"]);
    number_schema(seeds_schema["properties/extents_z"]);
    string_schema(seeds_schema["properties/sampling_type"]);
    string_schema(seeds_schema["properties/sampling_space"]);
    seeds_schema["required"].append() = "type";

    seeds_schema["constraints/dependencies/extents_x"].append() = "extents_y";
    seeds_schema["constraints/dependencies/extents_x"].append() = "extents_z";
    seeds_schema["constraints/dependencies/extents_y"].append() = "extents_x";
    seeds_schema["constraints/dependencies/extents_y"].append() = "extents_z";
    seeds_schema["constraints/dependencies/extents_z"].append() = "extents_x";
    seeds_schema["constraints/dependencies/extents_z"].append() = "extents_y";

    // type == point
    conduit::Node &point_option = seeds_schema["oneOf"].append();
    point_option["type"] = "object";
    point_option["properties/type/type"] = "string";
    point_option["properties/type/constraints/const"] = "point";
    point_option["required"].append() = "type";
    point_option["required"].append() = "location";

    // type == point_list
    conduit::Node &point_list_option = seeds_schema["oneOf"].append();
    point_list_option["type"] = "object";
    point_list_option["properties/type/type"] = "string";
    point_list_option["properties/type/constraints/const"] = "point_list";
    point_list_option["required"].append() = "type";
    point_list_option["required"].append() = "location";

    // type == line
    conduit::Node &line_option = seeds_schema["oneOf"].append();
    line_option["type"] = "object";
    line_option["properties/type/type"] = "string";
    line_option["properties/type/constraints/const"] = "line";
    line_option["required"].append() = "type";
    line_option["required"].append() = "start";
    line_option["required"].append() = "end";
    line_option["required"].append() = "num_seeds";
    line_option["required"].append() = "sampling_type";

    // type == box
    conduit::Node &box_option = seeds_schema["oneOf"].append();
    box_option["type"] = "object";
    box_option["properties/type/type"] = "string";
    box_option["properties/type/constraints/const"] = "box";
    box_option["required"].append() = "type";
    box_option["required"].append() = "sampling_space";
    box_option["required"].append() = "sampling_type";
    {
        conduit::Node &box_option_uniform = box_option["oneOf"].append();
        box_option_uniform["type"] = "object";
        box_option_uniform["properties/sampling_type/type"] = "string";
        box_option_uniform["properties/sampling_type/constraints/const"] = "uniform";
        box_option_uniform["required"].append() = "sampling_type";
        box_option_uniform["required"].append() = "num_seeds_x";
        box_option_uniform["required"].append() = "num_seeds_y";
        box_option_uniform["required"].append() = "num_seeds_z";
    }
    {
        conduit::Node &box_option_non_uniform = box_option["oneOf"].append();
        box_option_non_uniform["type"] = "object";
        box_option_non_uniform["required"].append() = "sampling_type";
        box_option_non_uniform["required"].append() = "num_seeds";
        box_option_non_uniform["constraints/not_const/sampling_type"] = "uniform";
    }

    // --- rendering ---
    conduit::Node &rendering_schema = param_schema["properties/rendering"];
    rendering_schema["type"] = "object";
    rendering_schema["additionalProperties"] = false;
    string_schema(rendering_schema["properties/enable_tubes"]);
    string_schema(rendering_schema["properties/tube_capping"]);
    number_schema(rendering_schema["properties/tube_size"]);
    number_schema(rendering_schema["properties/tube_sides"]);
    number_schema(rendering_schema["properties/tube_value"]);
    string_schema(rendering_schema["properties/output_field"]);

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "num_steps";
    param_schema["required"].append() = "step_size";
    param_schema["required"].append() = "seeds";
}

//-----------------------------------------------------------------------------
void
VTKHParticleAdvection::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_particle_advection input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
        set_output<DataObject>(data_object);
        return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
        bool throw_error = false;
        detail::field_error(field_name, this->name(), collection, throw_error);
        // this creates a data object with an invalid soource
        set_output<DataObject>(new DataObject());
        return;
    }

    std::string topo_name = collection->field_topology(field_name);
    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    int numSteps = get_int32(params()["num_steps"], data_object);
    float stepSize = get_float32(params()["step_size"], data_object);
    std::random_device device;
    std::default_random_engine generator(0);
    float  zero(0), one(1);
    std::uniform_real_distribution<viskores::FloatDefault> distribution(zero, one);

    conduit::Node n_seeds = params()["seeds"];
    std::string seed_type = n_seeds["type"].as_string();
    std::vector<viskores::Particle> seeds;
    if(seed_type == "point")
    {
        const Node &n_loc_vals = n_seeds["location"];

        //convert to contig doubles
        Node n_loc_vals_dbls;
        n_loc_vals.to_float64_array(n_loc_vals_dbls);

        double* location = n_loc_vals_dbls.as_double_ptr();
        double x = location[0];
        double y = location[1];
        double z = location[2];
        //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
        seeds.push_back(viskores::Particle({x,y,z}, 0));
    }
    else if(seed_type == "point_list")
    {
        const Node &n_loc_vals = n_seeds["location"];

        //convert to contig doubles
        Node n_loc_vals_dbls;
        n_loc_vals.to_float64_array(n_loc_vals_dbls);

        double* location = n_loc_vals_dbls.as_double_ptr();
      
        int num_points = (n_loc_vals_dbls.dtype().number_of_elements());
        //std::cerr << "num_points: " << num_points << std::endl;
        for(int i = 0; i < num_points; i+=3)
        {
            double x = location[i];
            double y = location[i+1];
            double z = location[i+2];
            //std::cerr << "seed point " << i/3 <<  ": " << x << " " << y << " " << z << std::endl;
            seeds.push_back(viskores::Particle({x,y,z}, i/3));
        }
    }
    else if(seed_type == "line")
    {
        const Node &n_start_vals = n_seeds["start"];
        const Node &n_end_vals = n_seeds["end"];
        std::string sampling = n_seeds["sampling_type"].as_string();
        int num_seeds = n_seeds["num_seeds"].to_int();


        //convert to contig doubles
        Node n_start_vals_dbls;
        n_start_vals.to_float64_array(n_start_vals_dbls);
        Node n_end_vals_dbls;
        n_end_vals.to_float64_array(n_end_vals_dbls);

        double* start = n_start_vals_dbls.as_double_ptr();
        double* end = n_end_vals_dbls.as_double_ptr();

        double dist_x = end[0] - start[0];
        double dist_y = end[1] - start[1];
        double dist_z = end[2] - start[2];

        if(sampling == "uniform")
        {
            double dx = (dist_x)/(num_seeds-1);
            double dy = (dist_y)/(num_seeds-1);
            double dz = (dist_z)/(num_seeds-1);
            for(int i = 0; i < num_seeds; ++i)
            {
                double x = start[0] + dx*i;
                double y = start[1] + dy*i;
                double z = start[2] + dz*i;
                //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
                seeds.push_back(viskores::Particle({x,y,z}, i));
            }
        }
        else
        {
            std::random_device device;
            std::default_random_engine generator(0);
            float  zero(0), one(1);
            std::uniform_real_distribution<viskores::FloatDefault> distribution(zero, one);
            for(int i = 0; i < num_seeds; ++i)
            {
                double rand = distribution(generator);
                double x = start[0] + dist_x*rand;
                double y = start[1] + dist_y*rand;
                double z = start[2] + dist_z*rand;
                //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
                seeds.push_back(viskores::Particle({x,y,z}, i));
	          }
        }
    }
    else if(seed_type == "box")
    {
        double dist_x, dist_y, dist_z;
        double x_min, y_min, z_min;
        double x_max, y_max, z_max;
        if(n_seeds.has_child("extents_x"))
        {
            const Node &n_extents_x_vals = n_seeds["extents_x"];
            const Node &n_extents_y_vals = n_seeds["extents_y"];
            const Node &n_extents_z_vals = n_seeds["extents_z"];
            Node n_extents_x_vals_dbls;
            Node n_extents_y_vals_dbls;
            Node n_extents_z_vals_dbls;
            n_extents_x_vals.to_float64_array(n_extents_x_vals_dbls);
            n_extents_y_vals.to_float64_array(n_extents_y_vals_dbls);
            n_extents_z_vals.to_float64_array(n_extents_z_vals_dbls);
            double* extents_x = n_extents_x_vals_dbls.as_double_ptr();
            double* extents_y = n_extents_y_vals_dbls.as_double_ptr();
            double* extents_z = n_extents_z_vals_dbls.as_double_ptr();
            dist_x = extents_x[1] - extents_x[0];
            dist_y = extents_y[1] - extents_y[0];
            dist_z = extents_z[1] - extents_z[0];
            x_min = extents_x[0];
            y_min = extents_y[0];
            z_min = extents_z[0];
            x_max = extents_x[1];
            y_max = extents_y[1];
            z_max = extents_z[1];
        }
        else// whole dataset
        {
            viskores::Bounds global_bounds = data.GetGlobalBounds();
            dist_x = global_bounds.X.Length();
            dist_y = global_bounds.Y.Length();
            dist_z = global_bounds.Z.Length();
            x_min = global_bounds.X.Min;
            y_min = global_bounds.Y.Min;
            z_min = global_bounds.Z.Min;
            x_max = global_bounds.X.Max;
            y_max = global_bounds.Y.Max;
            z_max = global_bounds.Z.Max;
        }
        std::string sampling_type = n_seeds["sampling_type"].as_string();
        std::string sampling_space = n_seeds["sampling_space"].as_string();
        if(sampling_type != "uniform" && sampling_type != "random")
        {
            ASCENT_ERROR("Particle Advection box seeds accepts either 'uniform' or 'random' as the 'sampling_type'");
        }

        if(sampling_space == "interior")
        {
            if(sampling_type == "uniform")
            {
                int num_seeds_x = n_seeds["num_seeds_x"].to_int();
                int num_seeds_y = n_seeds["num_seeds_y"].to_int();
                int num_seeds_z = n_seeds["num_seeds_z"].to_int();
                
                double dx = 1, dy = 1, dz = 1;
                if(num_seeds_x != 0)
                {
                    if(num_seeds_x != 1)
                    {
                        dx = dist_x/(num_seeds_x-1);
                    }
                    else
                    {
                        dx = dist_x/num_seeds_x;
                    }
                }
                if(num_seeds_y != 0)
                {
                    if(num_seeds_y != 1)
                    {
                         dy = dist_y/(num_seeds_y-1);
                    }
                    else
                    {
                         dy = dist_y/num_seeds_y;
                    }
                }
                if(num_seeds_z != 0)
                {
                    if(num_seeds_z != 1)
                    {
                         dz = dist_z/(num_seeds_z-1);
                    }
                    else
                    {
                         dz = dist_z/num_seeds_z;
                    }
                }
                for(int i = 0; i < num_seeds_x; ++i)
                {
                    double x = x_min + dx*i;
                    for(int j = 0; j < num_seeds_y; ++j)
                    {
                        double y = y_min + dy*j;
                        for(int k = 0; k < num_seeds_z; ++k)
                        {
                            double z = z_min + dz*k;
                            //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
                            seeds.push_back(viskores::Particle({x,y,z}, i));
                        }
                    }
                }
            }
            else //random
            {
                std::random_device device;
                std::default_random_engine generator(0);
                float  zero(0), one(1);
                std::uniform_real_distribution<viskores::FloatDefault> distribution(zero, one);
                int num_seeds = n_seeds["num_seeds"].to_int();
                for(int i = 0; i < num_seeds; ++i)
                {
                    double rand = distribution(generator);
                    double x = x_min + dist_x*distribution(generator);
                    double y = y_min + dist_y*distribution(generator);
                    double z = z_min + dist_z*distribution(generator);
                    //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
                    seeds.push_back(viskores::Particle({x,y,z}, i));
                }
            }
        }
        else if (sampling_space == "boundary") 
        {
            if(sampling_type == "uniform")
            {
                int num_seeds_x = n_seeds["num_seeds_x"].to_int();
                int num_seeds_y = n_seeds["num_seeds_y"].to_int();
                int num_seeds_z = n_seeds["num_seeds_z"].to_int();

                double dx = 1, dy = 1, dz = 1;
                if(num_seeds_x != 0)
                {
                    if(num_seeds_x != 1)
                    {
                        dx = dist_x/(num_seeds_x-1);
                    }
                    else
                    {
                        dx = dist_x/num_seeds_x;
                    }
                }
                if(num_seeds_y != 0)
                {
                    if(num_seeds_y != 1)
                    {
                        dy = dist_y/(num_seeds_y-1);
                    }
                    else
                    {
                        dy = dist_y/num_seeds_y;
                    }
                }
                if(num_seeds_z != 0)
                {
                    if(num_seeds_z != 1)
                    {
                        dz = dist_z/(num_seeds_z-1);
                    }
                    else
                    {
                        dz = dist_z/num_seeds_z;
                    }
                }
                int seed_count = 0;
                for(int i = 0; i < num_seeds_x; ++i)
                {
                    double x = x_min + dx*i;
                    for(int j = 0; j < num_seeds_z; ++j)
                    {
                         double z = z_min + dz*j;
                         //std::cerr << "seed point" << ": " << x << " " << y_min << " " << z << std::endl;
                         //std::cerr << "seed point" << ": " << x << " " << y_max << " " << z << std::endl;
                        //std::cerr << "seed_count: " << seed_count << std::endl;
                         seeds.push_back(viskores::Particle({x,y_min,z}, seed_count++));
                         seeds.push_back(viskores::Particle({x,y_max,z}, seed_count++));
                    }
                }
                for(int j = 0; j < num_seeds_y; ++j)
                {
                     double y = y_min + dy*j;
                     for(int k = 0; k < num_seeds_z; ++k)
                    {
                         double z = z_min + dz*k;
                         //std::cerr << "seed point" << ": " << x_min << " " << y << " " << z << std::endl;
                         //std::cerr << "seed point" << ": " << x_max << " " << y << " " << z << std::endl;
                         //std::cerr << "seed_count: " << seed_count << std::endl;
                         seeds.push_back(viskores::Particle({x_min,y,z}, seed_count++));
                         seeds.push_back(viskores::Particle({x_max,y,z}, seed_count++));
                    }
                }
            }
            else //random
            {
                std::random_device device;
                std::default_random_engine generator(0);
                float  zero(0), one(1);
                std::uniform_real_distribution<viskores::FloatDefault> distribution(zero, one);
                int num_seeds = n_seeds["num_seeds"].to_int();
                for(int i = 0; i < num_seeds; ++i)
                {
                    int side = std::rand()%4;
                    //std::cerr << "side: " << side << std::endl;
                    if(side == 0) //x_max
                    {
                        double y = y_min + dist_y*distribution(generator);
                        double z = z_min + dist_z*distribution(generator);
                        seeds.push_back(viskores::Particle({x_max,y,z}, i));
                        //std::cerr << "seed point" << ": " << x_max << " " << y << " " << z << std::endl;
                    }
                    else if(side == 1) //x_min
                    {
                        double y = y_min + dist_y*distribution(generator);
                        double z = z_min + dist_z*distribution(generator);
                        seeds.push_back(viskores::Particle({x_min,y,z}, i));
                        //std::cerr << "seed point" << ": " << x_min << " " << y << " " << z << std::endl;
                    }
                    else if(side == 2) //y_max
                    {
                        double x = x_min + dist_x*distribution(generator);
                        double z = z_min + dist_z*distribution(generator);
                        seeds.push_back(viskores::Particle({x,y_max,z}, i));
                        //std::cerr << "seed point" << ": " << x << " " << y_max << " " << z << std::endl;
                    }
                    else //y_min
                    {
                        double x = x_min + dist_x*distribution(generator);
                        double z = z_min + dist_z*distribution(generator);
                        seeds.push_back(viskores::Particle({x,y_min,z}, i));
                        //std::cerr << "seed point" << ": " << x << " " << y_min << " " << z << std::endl;
                    }
                }
            }
        }
        else //error
        {
            ASCENT_ERROR("Particle Advection box seeds accepts either 'interior' or 'boundary' as the 'sampling_space'");
        }
    }

    auto seedArray = viskores::cont::make_ArrayHandle(seeds, viskores::CopyFlag::On);
    //int numSeeds = get_int32(params()["num_seeds"], data_object);
    
    //tube params
    std::string output_field = field_name + "_streamlines";

    bool draw_tubes = true;
    if(params().has_path("rendering/enable_tubes"))
    {
        if(params()["rendering/enable_tubes"].as_string() == "false")
        {
            draw_tubes = false;
        }
    }

    //float seedBBox[6];
    //seedBBox[0] = get_float32(params()["seed_bounding_box_xmin"], data_object);
    //seedBBox[1] = get_float32(params()["seed_bounding_box_xmax"], data_object);
    //seedBBox[2] = get_float32(params()["seed_bounding_box_ymin"], data_object);
    //seedBBox[3] = get_float32(params()["seed_bounding_box_ymax"], data_object);
    //seedBBox[4] = get_float32(params()["seed_bounding_box_zmin"], data_object);
    //seedBBox[5] = get_float32(params()["seed_bounding_box_zmax"], data_object);

    //float dx = seedBBox[1] - seedBBox[0];
    //float dy = seedBBox[3] - seedBBox[2];
    //float dz = seedBBox[5] - seedBBox[4];


    //Generate seeds

    //std::vector<viskores::Particle> seeds;
    //for (int i = 0; i < numSeeds; i++)
    //{
    //  float x = seedBBox[0] + dx * distribution(generator);
    //  float y = seedBBox[2] + dy * distribution(generator);
    //  float z = seedBBox[4] + dz * distribution(generator);
    //  std::cerr << "seed " << i << ": " << x << " " << y << " " << z << std::endl;
    //  seeds.push_back(viskores::Particle({x,y,z}, i));
    //}
    //auto seedArray = viskores::cont::make_ArrayHandle(seeds, viskores::CopyFlag::On);


    vtkh::DataSet *output = nullptr;
    if (record_trajectories)
    {
        vtkh::Streamline sl;
        sl.SetStepSize(stepSize);
        sl.SetNumberOfSteps(numSteps);
        sl.SetSeeds(seeds);
        sl.SetField(field_name);
        if(draw_tubes)
        {
            sl.SetTubes(true);
            if(params().has_path("rendering/output_field")) 
            {
                std::string output_field = params()["rendering/output_field"].as_string();
                sl.SetOutputField(output_field);
            }
            else
            {
               std::string output_field = field_name + "_streamlines";
                sl.SetOutputField(output_field);
            }
            if(params().has_path("rendering/tube_value")) 
            {
                double tube_value = params()["rendering/tube_value"].to_float64();
                sl.SetTubeValue(tube_value);
            }
            if(params().has_path("rendering/tube_size")) 
            {
                double tube_size = params()["rendering/tube_size"].to_float64();
                sl.SetTubeSize(tube_size);
            }
            if(params().has_path("rendering/tube_sides")) 
            {
                int tube_sides = params()["rendering/tube_sides"].to_int32();
                sl.SetTubeSides(tube_sides);
            }
            if(params().has_path("rendering/tube_capping"))
            {
                bool tube_capping = true;
                if(params()["rendering/tube_capping"].as_string() == "false")
                {
                    tube_capping = false;
                }
                sl.SetTubeCapping(tube_capping);
            }
        }
    
        sl.SetInput(&data);
        sl.Update();
        output = sl.GetOutput();
    }
    else
    {
        vtkh::ParticleAdvection pa;
        pa.SetStepSize(stepSize);
        pa.SetNumberOfSteps(numSteps);
        pa.SetSeeds(seeds);
        pa.SetField(field_name);
        pa.SetInput(&data);
        pa.Update();
        output = pa.GetOutput();
    }

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------

VTKHStreamline::VTKHStreamline()
:VTKHParticleAdvection()
{
  record_trajectories = true;
}

void
VTKHStreamline::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_streamline";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
VTKHStreamline::~VTKHStreamline()
{
// empty
}

//-----------------------------------------------------------------------------

VTKHWarpXStreamline::VTKHWarpXStreamline()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHWarpXStreamline::~VTKHWarpXStreamline()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHWarpXStreamline::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_warpx_streamline";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/b_field"]);
    string_schema(param_schema["properties/e_field"]);
    number_schema(param_schema["properties/num_steps"], true);
    number_schema(param_schema["properties/step_size"], true);

    // --- rendering ---
    conduit::Node &rendering_schema = param_schema["properties/rendering"];
    rendering_schema["type"] = "object";
    rendering_schema["additionalProperties"] = false;
    string_schema(rendering_schema["properties/enable_tubes"]);
    string_schema(rendering_schema["properties/tube_capping"]);
    number_schema(rendering_schema["properties/tube_size"]);
    number_schema(rendering_schema["properties/tube_sides"]);
    number_schema(rendering_schema["properties/tube_value"]);
    string_schema(rendering_schema["properties/output_field"]);

    param_schema["required"].append() = "num_steps";
    param_schema["required"].append() = "step_size";
}

//-----------------------------------------------------------------------------
void
VTKHWarpXStreamline::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_warpx_streamline input must be a data object");
    }

    // grab the data collection and ask for a vtkh collection
    // which is one vtkh data set per topology
    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string b_field = "B";
    std::string e_field = "E";
    std::string charge_field = "Charge";
    std::string mass_field = "Mass";
    std::string momentum_field = "Momentum";
    std::string weighting_field = "Weighting";
    if(params().has_path("b_field"))
      b_field = params()["b_field"].as_string();
    if(params().has_path("e_field"))
      e_field = params()["e_field"].as_string();
    if(params().has_path("charge_field"))
      charge_field = params()["charge_field"].as_string();
    if(params().has_path("mass_field"))
      mass_field = params()["mass_field"].as_string();
    if(params().has_path("momentum_field"))
      momentum_field = params()["momentum_field"].as_string();
    if(params().has_path("weighting_field"))
      weighting_field = params()["weighting_field"].as_string();

    if(!collection->has_field(b_field))
    {
        bool throw_error = false;
        detail::field_error(b_field, this->name(), collection, throw_error);
        // this creates a data object with an invalid soource
        set_output<DataObject>(new DataObject());
        return;
    }
    if(!collection->has_field(e_field))
    {
        bool throw_error = false;
        detail::field_error(e_field, this->name(), collection, throw_error);
        // this creates a data object with an invalid soource
        set_output<DataObject>(new DataObject());
        return;
    }
    
    std::string topo_name = collection->field_topology(b_field);
    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);


    int numSteps = get_int32(params()["num_steps"], data_object);
    float stepSize = get_float32(params()["step_size"], data_object);

    //tube params
    bool draw_tubes = false;
    if(params().has_path("enable_tubes"))
    {
        if(params()["rendering/enable_tubes"].as_string() == "true")
        {
            draw_tubes = true;
        }
    }

    vtkh::DataSet *output = nullptr;
    vtkh::WarpXStreamline sl;
    sl.SetStepSize(stepSize);
    sl.SetNumberOfSteps(numSteps);
    sl.SetBField(b_field);
    sl.SetEField(e_field);
    sl.SetChargeField(charge_field);
    sl.SetMassField(mass_field);
    sl.SetMomentumField(momentum_field);
    sl.SetWeightingField(weighting_field);

    if(draw_tubes)
    {
        sl.SetTubes(true);
        if(params().has_path("output_field")) 
        {
            std::string output_field = params()["rendering/output_field"].as_string();
            sl.SetOutputField(output_field);
        }
        else
        {
            std::string output_field = b_field+ "_" + e_field + "_streamlines";
            sl.SetOutputField(output_field);
        }
        if(params().has_path("tube_value")) 
        {
            double tube_value = params()["rendering/tube_value"].to_float64();
            sl.SetTubeValue(tube_value);
        }
        if(params().has_path("tube_size")) 
        {
            double tube_size = params()["rendering/tube_size"].to_float64();
            sl.SetTubeSize(tube_size);
        }
        if(params().has_path("tube_sides")) 
        {
            int tube_sides = params()["rendering/tube_sides"].to_int32();
            sl.SetTubeSides(tube_sides);
        }
        if(params().has_path("tube_capping"))
        {
            bool tube_capping = true;
            if(params()["rendering/tube_capping"].as_string() == "false")
            {
                tube_capping = false;
            }
            sl.SetTubeCapping(tube_capping);
        }
    }

    sl.SetInput(&data);
    sl.Update();
    output = sl.GetOutput();

    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete output;
    set_output<DataObject>(res);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// VTKHVTKFileExtract
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
VTKHVTKFileExtract::VTKHVTKFileExtract()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHVTKFileExtract::~VTKHVTKFileExtract()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHVTKFileExtract::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_vtk_file_extract";
    i["port_names"].append() = "in";
    i["output_port"] = "false";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/path"]);
    string_schema(param_schema["properties/topology"]);

    param_schema["required"].append() = "path";
}

//-----------------------------------------------------------------------------
void
VTKHVTKFileExtract::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("VTKHVTKFileExtract input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      return;
    }

    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string topo_name = detail::resolve_topology(params(),
                                                     this->name(),
                                                     collection,
                                                     true // throw error
                                                     );
    // we create
    // file: basename.visit
    // directory: basename + "_vtk_files"
    // files: basename + "_vtk_files/basename_%08d.vtk"

    int par_rank = 0;
    int mpi_comm_id = -1; 
#ifdef ASCENT_MPI_ENABLED
    mpi_comm_id = Workspace::default_mpi_comm();
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
    MPI_Comm_rank(mpi_comm, &par_rank);
#endif

    std::string output_base = expand_path_special_variables(params()["path"].as_string(),
                                                            ".visit",
                                                            mpi_comm_id);
    
    std::string output_files_dir  = output_base + "_vtk_files";
    std::string output_visit_file = output_base + ".visit";

    std::string output_file_pattern = conduit::utils::join_path(output_files_dir,
                                                                "domain_{:08d}.vtk");

    if(par_rank == 0 && !conduit::utils::is_directory(output_files_dir))
    {
        // mkdir output dir
        conduit::utils::create_directory(output_files_dir);
    }

    int error_occured = 0;
    std::string error_message;

    vtkh::DataSet &vtkh_dset = collection->dataset_by_topology(topo_name);

    // loop over all local domains and save each to a legacy vtk file.

    viskores::cont::DataSet viskores_dset;
    viskores::Id            domain_id;

    viskores::Id num_local_domains  = vtkh_dset.GetNumberOfDomains();
    viskores::Id num_global_domains = vtkh_dset.GetGlobalNumberOfDomains();

    // keep list of domain ids
    Node n_local_domain_ids(DataType::index_t(num_local_domains));
    index_t_array local_domain_ids = n_local_domain_ids.value();

    for(viskores::Id idx = 0; idx < num_local_domains; idx++ )
    {
        vtkh_dset.GetDomain(idx,
                            viskores_dset,
                            domain_id);
        local_domain_ids[idx] = domain_id;
        viskores::io::VTKDataSetWriter writer(conduit_fmt::format(output_file_pattern,
                                                              domain_id));
        writer.WriteDataSet(viskores_dset);
    }

    // create .visit file on rank 0

    // broadcast list of domain ids for mpi case
    Node n_global_domain_ids;
#ifdef ASCENT_MPI_ENABLED
        Node n_recv;
        conduit::relay::mpi::all_gather_using_schema(n_local_domain_ids,
                                                     n_recv,
                                                     mpi_comm);
        n_global_domain_ids.set(DataType::index_t(num_global_domains));
        //n_global_domain_ids.print();
        index_t_array global_vals = n_global_domain_ids.value();
        // each child will an array with its domain ids
        index_t idx = 0;
        for(index_t chld_idx = 0; chld_idx < n_recv.number_of_children();chld_idx++)
        {
            const Node &cld = n_recv.child(chld_idx);
            index_t_array cld_vals = cld.value();
            for(index_t local_idx = 0; local_idx < cld_vals.number_of_elements();local_idx++)
            {
              global_vals[idx] = cld_vals[local_idx];
              idx++;
            } 
        }
#else
        n_global_domain_ids.set_external(n_local_domain_ids);
#endif

    if(par_rank == 0)
    {
        std::ofstream ofs;
        ofs.open(output_visit_file.c_str());

        if(!ofs.is_open())
        {
          error_occured = 1;
        }
        else
        {
      
          // make sure this is relative to output dir
          std::string output_files_dir_rel;
          std::string tmp;
          utils::rsplit_path(output_files_dir,
                             output_files_dir_rel,
                             tmp);
        
          std::string output_file_pattern_rel = conduit::utils::join_path(output_files_dir_rel,
                                                                  "domain_{:08d}.vtk");

          index_t_array global_domain_ids = n_global_domain_ids.value();
          ofs << "!NBLOCKS " << num_global_domains << std::endl;
          for(size_t i=0;i< global_domain_ids.number_of_elements();i++)
          {
              ofs << conduit_fmt::format(output_file_pattern_rel,
                                         global_domain_ids[i]) << std::endl;
          }
        }
    }

#ifdef ASCENT_MPI_ENABLED
    Node n_local_err, n_global_err;
    n_local_err = error_occured;
    conduit::relay::mpi::max_all_reduce(n_local_err,
                                        n_global_err,
                                        mpi_comm);
    n_local_err = n_global_err.to_index_t();
#endif

    if(error_occured == 1)
    {
        ASCENT_ERROR("failed to save vtk files to path:" << output_base);
    }

}


//-----------------------------------------------------------------------------

VTKHMIR::VTKHMIR()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHMIR::~VTKHMIR()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHMIR::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_mir";
    i["port_names"].append() = "in";
    i["output_port"] = "true";

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/matset"]);
    string_schema(param_schema["properties/output_name"]);
    number_schema(param_schema["properties/error_scaling"]);
    number_schema(param_schema["properties/scaling_decay"]);
    number_schema(param_schema["properties/iterations"]);
    number_schema(param_schema["properties/max_error"]);

    param_schema["required"].append() = "matset";
}

//-----------------------------------------------------------------------------
void
VTKHMIR::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_MIR input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      set_output<DataObject>(data_object);
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string matset_name = params()["matset"].as_string();
    std::string ids_name = "material_ids"; 
    if(!collection->has_field(ids_name))
    {
      bool throw_error = false;
      detail::field_error(ids_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::string topo_name = collection->field_topology(ids_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    double error_scaling = 0.0; 
    double scaling_decay = 0.0; 
    double max_error = 0.00001;
    int iterations = 0;
    std::string output_name = matset_name;
    if(params().has_path("error_scaling"))
      error_scaling = params()["error_scaling"].to_float64();
    if(params().has_path("scaling_decay"))
      scaling_decay = params()["scaling_decay"].to_float64();
    if(params().has_path("iterations"))
      iterations = params()["iterations"].to_int64();
    if(params().has_path("max_error"))
      max_error = params()["max_error"].to_float64();
    if(params().has_path("output_name"))
      output_name = params()["output_name"].as_string();

    vtkh::MIR mir;
    mir.SetErrorScaling(error_scaling);
    mir.SetScalingDecay(scaling_decay);
    mir.SetIterations(iterations);
    mir.SetMaxError(max_error);
    mir.SetOutputName(output_name);
    mir.SetMatSet(matset_name);
    mir.SetInput(&data);
    mir.Update();
    vtkh::DataSet *mir_output = mir.GetOutput();

    //// we need to pass through the rest of the topologies, untouched,
    //// and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*mir_output, topo_name);
    // re wrap in data object
    DataObject *res =  new DataObject(new_coll);
    delete mir_output;
    set_output<DataObject>(res);
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
