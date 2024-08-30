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

#if defined(ASCENT_VTKM_ENABLED)
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
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/filters/Log.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/filters/UniformGrid.hpp>
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
#include <vtkm/cont/DataSet.h>
#include <vtkm/io/VTKDataSetWriter.h>
#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
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
}

//-----------------------------------------------------------------------------
bool
VTKHMarchingCubes::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    bool has_values = check_numeric("iso_values",params, info, false);
    bool has_levels = check_numeric("levels",params, info, false);

    if(!has_values && !has_levels)
    {
        info["errors"].append() = "Missing required numeric parameter. Contour must"
                                  " specify 'iso_values' or 'levels'.";
        res = false;
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("levels");
    valid_paths.push_back("iso_values");
    valid_paths.push_back("use_contour_tree");
    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
VTKHMarchingCubes::execute()
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
}

//-----------------------------------------------------------------------------
bool
VTKHVectorMagnitude::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res = check_string("output_name",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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
}

//-----------------------------------------------------------------------------
bool
VTKH3Slice::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();
    bool res = true;
    std::vector<std::string> valid_paths;
    res &= check_string("topology",params, info, false);
    valid_paths.push_back("topology");

    res &= check_numeric("x_offset",params, info, false, true);
    res &= check_numeric("y_offset",params, info, false, true);
    res &= check_numeric("z_offset",params, info, false, true);
    res = check_string("topology",params, info, false) && res;

    valid_paths.push_back("x_offset");
    valid_paths.push_back("y_offset");
    valid_paths.push_back("z_offset");

    std::string surprises = surprise_check(valid_paths, params);
    if(surprises != "")
    {
       res = false;
       info["errors"].append() = surprises;
    }

    return res;
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

    using Vec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkm::Bounds bounds = data.GetGlobalBounds();
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
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      x_point[0] = bounds.X.Min + t * (bounds.X.Max - bounds.X.Min);
    }

    if(params().has_path("y_offset"))
    {
      float offset = get_float32(params()["y_offset"], data_object);
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      y_point[1] = bounds.Y.Min + t * (bounds.Y.Max - bounds.Y.Min);
    }

    if(params().has_path("z_offset"))
    {
      float offset = get_float32(params()["z_offset"], data_object);
      std::max(-1.f, std::min(1.f, offset));
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
}

//-----------------------------------------------------------------------------
bool
VTKHTriangulate::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
    info.reset();

    bool res = true;

    res = check_string("topology",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("topology");


    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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
}

//-----------------------------------------------------------------------------
bool
VTKHCleanGrid::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = true;

    res = check_string("topology",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("topology");


    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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
}

//-----------------------------------------------------------------------------
bool
VTKHSlice::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = true;


    if(params.has_path("point/x_offset") && params.has_path("point/x"))
    {
      info["errors"]
        .append() = "Cannot specify the plane point as both an offset and explicit point";
      res = false;
    }

    if(params.has_path("point/x"))
    {
      res &= check_numeric("point/x",params, info, true, true);
      res = check_numeric("point/y",params, info, true, true) && res;
      res = check_numeric("point/z",params, info, true, true) && res;
    }
    else if(params.has_path("point/x_offset"))
    {
      res &= check_numeric("point/x_offset",params, info, true, true);
      res = check_numeric("point/y_offset",params, info, true, true) && res;
      res = check_numeric("point/z_offset",params, info, true, true) && res;
    }
    else
    {
      info["errors"]
        .append() = "Slice must specify a point for the plane.";
      res = false;
    }

    res = check_string("topology",params, info, false) && res;

    res = check_numeric("normal/x",params, info, true, true) && res;
    res = check_numeric("normal/y",params, info, true, true) && res;
    res = check_numeric("normal/z",params, info, true, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("point/x");
    valid_paths.push_back("point/y");
    valid_paths.push_back("point/z");
    valid_paths.push_back("point/x_offset");
    valid_paths.push_back("point/y_offset");
    valid_paths.push_back("point/z_offset");
    valid_paths.push_back("normal/x");
    valid_paths.push_back("normal/y");
    valid_paths.push_back("normal/z");
    valid_paths.push_back("topology");


    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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
    vtkh::Slice slicer;

    slicer.SetInput(&data);

    const Node &n_point = params()["point"];
    const Node &n_normal = params()["normal"];

    using Vec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkm::Bounds bounds = data.GetGlobalBounds();
    Vec3f point;

    const float eps = 1e-5; // ensure that the slice is always inside the data set

    if(n_point.has_path("x_offset"))
    {
      float offset = get_float32(n_point["x_offset"], data_object);
      std::max(-1.f, std::min(1.f, offset));
      float t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      point[0] = bounds.X.Min + t * (bounds.X.Max - bounds.X.Min);

      offset = get_float32(n_point["y_offset"], data_object);
      std::max(-1.f, std::min(1.f, offset));
      t = (offset + 1.f) / 2.f;
      t = std::max(0.f + eps, std::min(1.f - eps, t));
      point[1] = bounds.Y.Min + t * (bounds.Y.Max - bounds.Y.Min);

      offset = get_float32(n_point["z_offset"], data_object);
      std::max(-1.f, std::min(1.f, offset));
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

    vtkh::DataSet *slice_output = slicer.GetOutput();

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
}

//-----------------------------------------------------------------------------
bool
VTKHAutoSliceLevels::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    if(!params.has_path("levels"))
    {
      info["errors"]
        .append() = "AutoSliceLevels must specify number of slices to consider via 'levels'.";
      res = false;
    }

    res = check_string("topology",params, info, false) && res;

    res = check_numeric("normal/x",params, info, true, true) && res;
    res = check_numeric("normal/y",params, info, true, true) && res;
    res = check_numeric("normal/z",params, info, true, true) && res;

    res = check_numeric("levels",params, info, true, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("levels");
    valid_paths.push_back("field");
    valid_paths.push_back("normal/x");
    valid_paths.push_back("normal/y");
    valid_paths.push_back("normal/z");
    valid_paths.push_back("topology");


    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------

vtkm::Vec<vtkm::Float32,3>
GetIntersectionPoint(vtkm::Vec<vtkm::Float32,3> normal)
{
  //point where normal intersects unit sphere
  vtkm::Vec<vtkm::Float32,3> point;

  //reverse normal
  //want camera point in the same dir as normal
  vtkm::Vec<vtkm::Float32,3> r_normal{((vtkm::Float32)1.0)*normal[0],
		  			((vtkm::Float32)1.0)*normal[1],
					((vtkm::Float32)1.0)*normal[2]};

  //calc discriminant
  //a = dot(normal,normal)
  vtkm::Float32 r_norm0 = r_normal[0]*r_normal[0];
  vtkm::Float32 r_norm1 = r_normal[1]*r_normal[1];
  vtkm::Float32 r_norm2 = r_normal[2]*r_normal[2];
  vtkm::Float32 a = r_norm0 + r_norm1 + r_norm2;
  //b is 0
  //c is -1
  vtkm::Float32 discriminant = 4.0*a;

  vtkm::Float32 t =  sqrt(discriminant)/(2*a);
  vtkm::Float32 t2 = -t;
  if(abs(t2) < abs(t)) 
    t = t2;

  point[0]= t * r_normal[0];
  point[1]= t * r_normal[1];
  point[2]= t * r_normal[2];

  return point;

}

void
SetCamera(vtkm::rendering::Camera *camera, vtkm::Vec<vtkm::Float32,3> normal, vtkm::Float32 radius)
{
  vtkm::Vec<vtkm::Float32,3> i_point = GetIntersectionPoint(normal);
  vtkm::Vec<vtkm::Float32,3> lookat = camera->GetLookAt();

  vtkm::Vec<vtkm::Float32,3> pos;
  vtkm::Float32 zoom = 3;
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
    vtkh::AutoSliceLevels slicer;

    slicer.SetInput(&data);

    const Node &n_normal = params()["normal"];
    const int n_levels = params()["levels"].to_int32();
    std::string field = params()["field"].as_string();

    using Vec3f = vtkm::Vec<vtkm::Float32,3>;
    vtkm::Bounds bounds = data.GetGlobalBounds();

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
    //  vtkm::rendering::Camera *cam = new vtkm::rendering::Camera;
    //  vtkm::Bounds bounds = slicer.GetDataBounds();
    //  std::cerr << "In Ascent runtime filters" << std::endl;
    //  std::cerr << "X bounds: " << bounds.X.Min << " " << bounds.X.Max << " ";
    //  std::cerr << "Y bounds: " << bounds.Y.Min << " " << bounds.Y.Max << " ";
    //  std::cerr << "Z bounds: " << bounds.Z.Min << " " << bounds.Z.Max << " ";
    //  std::cerr<<std::endl;
    //  vtkm::Vec<vtkm::Float32,3> normal = slicer.GetNormal();
    //  std::cerr << "normal: " << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;
    //  vtkm::Float32 radius = slicer.GetRadius();
    //  std::cerr << "radius: " << radius << std::endl;
    //  SetCamera(cam, normal, radius);
    //  std::cerr << "Cam before registry:" << std::endl;
    //  cam->Print();
    //  std::cerr << "Cam after registry:" << std::endl;
    //  graph().workspace().registry().add<vtkm::rendering::Camera>("camera",cam,1);
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
}

//-----------------------------------------------------------------------------
bool
VTKHGhostStripper::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    res = check_numeric("min_value",params, info, true, true) && res;
    res = check_numeric("max_value",params, info, true, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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

      seeds.push_back(vtkm::Particle({x,y,z}, 0));
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
        seeds.push_back(vtkm::Particle({x,y,z}, i/3));
      }
    }
    else if(seed_type == "line")
    {
      const Node &n_start_vals = n_seeds["start"];
      const Node &n_end_vals = n_seeds["end"];
      std::string sampling = n_seeds["sampling_type"].as_string();
      int num_seeds = n_seeds["num_seeds"].as_int();


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
          seeds.push_back(vtkm::Particle({x,y,z}, i));
	}
      }
      else
      {
        std::random_device device;
        std::default_random_engine generator(0);
        float  zero(0), one(1);
        std::uniform_real_distribution<vtkm::FloatDefault> distribution(zero, one);
        for(int i = 0; i < num_seeds; ++i)
	{
	  double rand = distribution(generator);
          double x = start[0] + dist_x*rand;
          double y = start[1] + dist_y*rand;
          double z = start[2] + dist_z*rand;
          //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
          seeds.push_back(vtkm::Particle({x,y,z}, i));
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
        vtkm::Bounds global_bounds = data.GetGlobalBounds();
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
	  
	  int num_seeds_x = n_seeds["num_seeds_x"].as_int();
	  int num_seeds_y = n_seeds["num_seeds_y"].as_int();
	  int num_seeds_z = n_seeds["num_seeds_z"].as_int();

	  double dx = 1, dy = 1, dz = 1;
	  if(num_seeds_x != 0)
	    if(num_seeds_x != 1)
              dx = dist_x/(num_seeds_x-1);
	    else
              dx = dist_x/num_seeds_x;
	  if(num_seeds_y != 0)
	    if(num_seeds_y != 1)
              dy = dist_y/(num_seeds_y-1);
	    else
              dy = dist_y/num_seeds_y;
	  if(num_seeds_z != 0)
	    if(num_seeds_z != 1)
              dz = dist_z/(num_seeds_z-1);
	    else
              dz = dist_z/num_seeds_z;
 
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
                seeds.push_back(vtkm::Particle({x,y,z}, i));
	      }
	    }
	  }
        }
        else //random
        {
          std::random_device device;
          std::default_random_engine generator(0);
          float  zero(0), one(1);
          std::uniform_real_distribution<vtkm::FloatDefault> distribution(zero, one);
	  int num_seeds = n_seeds["num_seeds"].as_int();
          for(int i = 0; i < num_seeds; ++i)
	  {
	    double rand = distribution(generator);
            double x = x_min + dist_x*distribution(generator);
            double y = y_min + dist_y*distribution(generator);
            double z = z_min + dist_z*distribution(generator);
            //std::cerr << "seed point" << ": " << x << " " << y << " " << z << std::endl;
            seeds.push_back(vtkm::Particle({x,y,z}, i));
	  }
        }

      }
      else if (sampling_space == "boundary") 
      {
        if(sampling_type == "uniform")
        {
	  int num_seeds_x = n_seeds["num_seeds_x"].as_int();
	  int num_seeds_y = n_seeds["num_seeds_y"].as_int();
	  int num_seeds_z = n_seeds["num_seeds_z"].as_int();

	  double dx = 1, dy = 1, dz = 1;
	  if(num_seeds_x != 0)
	    if(num_seeds_x != 1)
              dx = dist_x/(num_seeds_x-1);
	    else
              dx = dist_x/num_seeds_x;
	  if(num_seeds_y != 0)
	    if(num_seeds_y != 1)
              dy = dist_y/(num_seeds_y-1);
	    else
              dy = dist_y/num_seeds_y;
	  if(num_seeds_z != 0)
	    if(num_seeds_z != 1)
              dz = dist_z/(num_seeds_z-1);
	    else
              dz = dist_z/num_seeds_z;
 
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
              seeds.push_back(vtkm::Particle({x,y_min,z}, seed_count++));
              seeds.push_back(vtkm::Particle({x,y_max,z}, seed_count++));
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
              seeds.push_back(vtkm::Particle({x_min,y,z}, seed_count++));
              seeds.push_back(vtkm::Particle({x_max,y,z}, seed_count++));
	    }
	  }
        }
        else //random
        {
          std::random_device device;
          std::default_random_engine generator(0);
          float  zero(0), one(1);
          std::uniform_real_distribution<vtkm::FloatDefault> distribution(zero, one);
	  int num_seeds = n_seeds["num_seeds"].as_int();
	  for(int i = 0; i < num_seeds; ++i)
	  {
	    int side = std::rand()%4;
	    //std::cerr << "side: " << side << std::endl;
	    if(side == 0) //x_max
	    {
              double y = y_min + dist_y*distribution(generator);
              double z = z_min + dist_z*distribution(generator);
              seeds.push_back(vtkm::Particle({x_max,y,z}, i));
              //std::cerr << "seed point" << ": " << x_max << " " << y << " " << z << std::endl;
	    }
	    else if(side == 1) //x_min
	    {
              double y = y_min + dist_y*distribution(generator);
              double z = z_min + dist_z*distribution(generator);
              seeds.push_back(vtkm::Particle({x_min,y,z}, i));
              //std::cerr << "seed point" << ": " << x_min << " " << y << " " << z << std::endl;
	    }
	    else if(side == 2) //y_max
	    {
              double x = x_min + dist_x*distribution(generator);
              double z = z_min + dist_z*distribution(generator);
              seeds.push_back(vtkm::Particle({x,y_max,z}, i));
              //std::cerr << "seed point" << ": " << x << " " << y_max << " " << z << std::endl;
	    }
	    else //y_min
	    {
              double x = x_min + dist_x*distribution(generator);
              double z = z_min + dist_z*distribution(generator);
              seeds.push_back(vtkm::Particle({x,y_min,z}, i));
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

    auto seedArray = vtkm::cont::make_ArrayHandle(seeds, vtkm::CopyFlag::On);
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

    //std::vector<vtkm::Particle> seeds;
    //for (int i = 0; i < numSeeds; i++)
    //{
    //  float x = seedBBox[0] + dx * distribution(generator);
    //  float y = seedBBox[2] + dy * distribution(generator);
    //  float z = seedBBox[4] + dz * distribution(generator);
    //  std::cerr << "seed " << i << ": " << x << " " << y << " " << z << std::endl;
    //  seeds.push_back(vtkm::Particle({x,y,z}, i));
    //}
    //auto seedArray = vtkm::cont::make_ArrayHandle(seeds, vtkm::CopyFlag::On);


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
          double tube_value = params()["rendering/tube_value"].as_float64();
          sl.SetTubeValue(tube_value);
	}
        if(params().has_path("rendering/tube_size")) 
	{
          double tube_size = params()["rendering/tube_size"].as_float64();
          sl.SetTubeSize(tube_size);
	}
        if(params().has_path("rendering/tube_sides")) 
	{
          int tube_sides = params()["rendering/tube_sides"].as_int32();
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
}

//-----------------------------------------------------------------------------
bool
VTKHWarpXStreamline::verify_params(const conduit::Node &params,
                                     conduit::Node &info)
{
    info.reset();
    bool res = check_string("b_field", params, info, false);
    res &= check_string("e_field", params, info, false);
    res &= check_numeric("num_steps", params, info, true, true);
    res &= check_numeric("step_size", params, info, true, true);

    if(params.has_child("rendering"))
    {
      res &= check_string("rendering/enable_tubes", params, info, false);
      res &= check_string("rendering/tube_capping", params, info, false);
      res &= check_numeric("rendering/tube_size", params, info, false);
      res &= check_numeric("rendering/tube_sides", params, info, false);
      res &= check_numeric("rendering/tube_value", params, info, false);
      res &= check_string("rendering/output_field", params, info, false);
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("b_field");
    valid_paths.push_back("e_field");
    valid_paths.push_back("charge_field");
    valid_paths.push_back("mass_field");
    valid_paths.push_back("momentum_field");
    valid_paths.push_back("weighting_field");
    valid_paths.push_back("num_steps");
    valid_paths.push_back("step_size");
    valid_paths.push_back("rendering/enable_tubes");
    valid_paths.push_back("rendering/tube_capping");
    valid_paths.push_back("rendering/tube_size");
    valid_paths.push_back("rendering/tube_sides");
    valid_paths.push_back("rendering/tube_value");
    valid_paths.push_back("rendering/output_field");

    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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
        double tube_value = params()["rendering/tube_value"].as_float64();
        sl.SetTubeValue(tube_value);
      }
      if(params().has_path("tube_size")) 
      {
        double tube_size = params()["rendering/tube_size"].as_float64();
        sl.SetTubeSize(tube_size);
      }
      if(params().has_path("tube_sides")) 
      {
        int tube_sides = params()["rendering/tube_sides"].as_int32();
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
}

//-----------------------------------------------------------------------------
bool
VTKHVTKFileExtract::verify_params(const conduit::Node &params,
                                  conduit::Node &info)
{
    info.reset();

    bool res = true;

    if( !params.has_child("path") )
    {
        info["errors"].append() = "missing required entry 'path'";
        res = false;
    }

    res = check_string("topology",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("path");
    valid_paths.push_back("topology");

    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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

    std::string output_base = params()["path"].as_string();
    
    std::string output_files_dir  = output_base + "_vtk_files";
    std::string output_visit_file = output_base + ".visit";

    std::string output_file_pattern = conduit::utils::join_path(output_files_dir,
                                                                "domain_{:08d}.vtk");

    int par_rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &par_rank);
#endif

    if(par_rank == 0 && !conduit::utils::is_directory(output_files_dir))
    {
        // mkdir output dir
        conduit::utils::create_directory(output_files_dir);
    }

    int error_occured = 0;
    std::string error_message;

    vtkh::DataSet &vtkh_dset = collection->dataset_by_topology(topo_name);

    // loop over all local domains and save each to a legacy vtk file.

    vtkm::cont::DataSet vtkm_dset;
    vtkm::Id            domain_id;

    vtkm::Id num_local_domains  = vtkh_dset.GetNumberOfDomains();
    vtkm::Id num_global_domains = vtkh_dset.GetGlobalNumberOfDomains();

    // keep list of domain ids
    Node n_local_domain_ids(DataType::index_t(num_local_domains));
    index_t_array local_domain_ids = n_local_domain_ids.value();

    for(vtkm::Id idx = 0; idx < num_local_domains; idx++ )
    {
        vtkh_dset.GetDomain(idx,
                            vtkm_dset,
                            domain_id);
        local_domain_ids[idx] = domain_id;
        vtkm::io::VTKDataSetWriter writer(conduit_fmt::format(output_file_pattern,
                                                              domain_id));
        writer.WriteDataSet(vtkm_dset);
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
        n_global_domain_ids.print();
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
