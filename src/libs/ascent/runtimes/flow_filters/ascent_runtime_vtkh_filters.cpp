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
#include <vtkh/filters/SampleGrid.hpp>
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
      // this creates a data object with an invalid soource
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
}

//-----------------------------------------------------------------------------
bool
VTKHThreshold::verify_params(const conduit::Node &params,
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

    vtkh::Threshold thresher;

    thresher.SetInput(&data);
    thresher.SetField(field_name);

    const Node &n_min_val = params()["min_value"];
    const Node &n_max_val = params()["max_value"];

    // convert to contig doubles
    double min_val = get_float64(n_min_val, data_object);
    double max_val = get_float64(n_max_val, data_object);
    thresher.SetUpperThreshold(max_val);
    thresher.SetLowerThreshold(min_val);

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
}

//-----------------------------------------------------------------------------
bool
VTKHClip::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = true;

    bool type_present = false;

    if(params.has_child("sphere"))
    {
      type_present = true;
    }
    else if(params.has_child("box"))
    {
      type_present = true;
    }
    else if(params.has_child("plane"))
    {
      type_present = true;
    }
    else if(params.has_child("multi_plane"))
    {
      type_present = true;
    }

    if(!type_present)
    {
        info["errors"].append() = "Missing required parameter. Clip must specify a 'sphere', 'box', 'plane', or 'mulit_plane'";
        res = false;
    }
    else
    {

      res &= check_string("topology",params, info, false);
      if(params.has_child("sphere"))
      {
         res = check_numeric("sphere/center/x",params, info, true, true) && res;
         res = check_numeric("sphere/center/y",params, info, true, true) && res;
         res = check_numeric("sphere/center/z",params, info, true, true) && res;
         res = check_numeric("sphere/radius",params, info, true, true) && res;

      }
      else if(params.has_child("box"))
      {
         res = check_numeric("box/min/x",params, info, true, true) && res;
         res = check_numeric("box/min/y",params, info, true, true) && res;
         res = check_numeric("box/min/z",params, info, true, true) && res;
         res = check_numeric("box/max/x",params, info, true, true) && res;
         res = check_numeric("box/max/y",params, info, true, true) && res;
         res = check_numeric("box/max/z",params, info, true, true) && res;
      }
      else if(params.has_child("plane"))
      {
         res = check_numeric("plane/point/x",params, info, true, true) && res;
         res = check_numeric("plane/point/y",params, info, true, true) && res;
         res = check_numeric("plane/point/z",params, info, true, true) && res;
         res = check_numeric("plane/normal/x",params, info, true, true) && res;
         res = check_numeric("plane/normal/y",params, info, true, true) && res;
         res = check_numeric("plane/normal/z",params, info, true, true) && res;
      }
      else if(params.has_child("multi_plane"))
      {
         res = check_numeric("multi_plane/point1/x",params, info, true, true) && res;
         res = check_numeric("multi_plane/point1/y",params, info, true, true) && res;
         res = check_numeric("multi_plane/point1/z",params, info, true, true) && res;
         res = check_numeric("multi_plane/normal1/x",params, info, true, true) && res;
         res = check_numeric("multi_plane/normal1/y",params, info, true, true) && res;
         res = check_numeric("multi_plane/normal1/z",params, info, true, true) && res;

         res = check_numeric("multi_plane/point2/x",params, info, true, true) && res;
         res = check_numeric("multi_plane/point2/y",params, info, true, true) && res;
         res = check_numeric("multi_plane/point2/z",params, info, true, true) && res;
         res = check_numeric("multi_plane/normal2/x",params, info, true, true) && res;
         res = check_numeric("multi_plane/normal2/y",params, info, true, true) && res;
         res = check_numeric("multi_plane/normal2/z",params, info, true, true) && res;
      }
    }

    res = check_string("invert",params, info, false) && res;
    res = check_string("topology",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("topology");
    valid_paths.push_back("invert");
    valid_paths.push_back("sphere/center/x");
    valid_paths.push_back("sphere/center/y");
    valid_paths.push_back("sphere/center/z");
    valid_paths.push_back("sphere/radius");
    valid_paths.push_back("box/min/x");
    valid_paths.push_back("box/min/y");
    valid_paths.push_back("box/min/z");
    valid_paths.push_back("box/max/x");
    valid_paths.push_back("box/max/y");
    valid_paths.push_back("box/max/z");
    valid_paths.push_back("plane/point/x");
    valid_paths.push_back("plane/point/y");
    valid_paths.push_back("plane/point/z");
    valid_paths.push_back("plane/normal/x");
    valid_paths.push_back("plane/normal/y");
    valid_paths.push_back("plane/normal/z");

    valid_paths.push_back("multi_plane/point1/x");
    valid_paths.push_back("multi_plane/point1/y");
    valid_paths.push_back("multi_plane/point1/z");
    valid_paths.push_back("multi_plane/normal1/x");
    valid_paths.push_back("multi_plane/normal1/y");
    valid_paths.push_back("multi_plane/normal1/z");

    valid_paths.push_back("multi_plane/point2/x");
    valid_paths.push_back("multi_plane/point2/y");
    valid_paths.push_back("multi_plane/point2/z");
    valid_paths.push_back("multi_plane/normal2/x");
    valid_paths.push_back("multi_plane/normal2/y");
    valid_paths.push_back("multi_plane/normal2/z");
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
      // this creates a data object with an invalid soource
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
    else if(params().has_path("box"))
    {
      const Node &box = params()["box"];
      vtkm::Bounds bounds;
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
}

//-----------------------------------------------------------------------------
bool
VTKHClipWithField::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_numeric("clip_value",params, info, true, true);
    res = check_string("field",params, info, true) && res;
    res = check_string("invert",params, info, false) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("clip_value");
    valid_paths.push_back("invert");
    valid_paths.push_back("field");
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

    vtkm::Float64 clip_value = get_float64(params()["clip_value"], data_object);

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
}

//-----------------------------------------------------------------------------
bool
VTKHIsoVolume::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_numeric("min_value",params, info, true, true);
    res = check_numeric("max_value",params, info, true, true) && res;
    res = check_string("field",params, info, true) && res;

    std::vector<std::string> valid_paths;
    valid_paths.push_back("min_value");
    valid_paths.push_back("max_value");
    valid_paths.push_back("field");
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

    vtkm::Range clip_range;
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
}

//-----------------------------------------------------------------------------
bool
VTKHLagrangian::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("step_size", params, info, true);
    res &= check_numeric("write_frequency", params, info, true);
    res &= check_numeric("cust_res", params, info, true);
    res &= check_numeric("x_res", params, info, true);
    res &= check_numeric("y_res", params, info, true);
    res &= check_numeric("z_res", params, info, true);


    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("step_size");
    valid_paths.push_back("write_frequency");
    valid_paths.push_back("cust_res");
    valid_paths.push_back("x_res");
    valid_paths.push_back("y_res");
    valid_paths.push_back("z_res");

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
}

//-----------------------------------------------------------------------------
bool
VTKHLog::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_numeric("clamp_min_value",params, info, false, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("clamp_min_value");

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
}

//-----------------------------------------------------------------------------
bool
VTKHLog10::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_numeric("clamp_min_value",params, info, false, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("clamp_min_value");

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
}

//-----------------------------------------------------------------------------
bool
VTKHLog2::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_numeric("clamp_min_value",params, info, false, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("clamp_min_value");

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
}

//-----------------------------------------------------------------------------
bool
VTKHRecenter::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("association",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("association");

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
      recenter.SetResultAssoc(vtkm::cont::Field::Association::Points);
    }
    if(association == "element")
    {
      recenter.SetResultAssoc(vtkm::cont::Field::Association::Cells);
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
}

//-----------------------------------------------------------------------------
bool
VTKHHistSampling::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("bins",params, info, false, true);
    res &= check_numeric("sample_rate",params, info, false, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("bins");
    valid_paths.push_back("sample_rate");

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
}

//-----------------------------------------------------------------------------
bool
VTKHQCriterion::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
    // since downstream vtk-m filters may not be able to handle
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
}

//-----------------------------------------------------------------------------
bool
VTKHDivergence::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
}

//-----------------------------------------------------------------------------
bool
VTKHVorticity::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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
}

//-----------------------------------------------------------------------------
bool
VTKHGradient::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_string("output_name",params, info, false);
    res &= check_string("use_cell_gradient",params, info, false);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("output_name");
    valid_paths.push_back("use_cell_gradient");

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

VTKHSampleGrid::VTKHSampleGrid()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
VTKHSampleGrid::~VTKHSampleGrid()
{
// empty
}

//-----------------------------------------------------------------------------
void
VTKHSampleGrid::declare_interface(Node &i)
{
    i["type_name"]   = "vtkh_sample_grid";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
VTKHSampleGrid::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("dims/i",params, info, true);
    res &= check_numeric("dims/j",params, info, true);
    res &= check_numeric("dims/k",params, info, true);
    res &= check_numeric("origin/x",params, info, true);
    res &= check_numeric("origin/y",params, info, true);
    res &= check_numeric("origin/z",params, info, true);
    res &= check_numeric("spacing/dx",params, info, true);
    res &= check_numeric("spacing/dx",params, info, true);
    res &= check_numeric("spacing/dy",params, info, true);
    res &= check_numeric("spacing/dz",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("dims/i");
    valid_paths.push_back("dims/j");
    valid_paths.push_back("dims/k");
    valid_paths.push_back("origin/x");
    valid_paths.push_back("origin/y");
    valid_paths.push_back("origin/z");
    valid_paths.push_back("spacing/dx");
    valid_paths.push_back("spacing/dy");
    valid_paths.push_back("spacing/dz");
    valid_paths.push_back("invalid_value");

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
VTKHSampleGrid::execute()
{

    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("vtkh_sample_grid input must be a data object");
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
      // this creates a data object with an invalid source
      set_output<DataObject>(new DataObject());
      return;
    }

    const Node &n_origin = params()["origin"];
    const Node &n_spacing = params()["spacing"];
    const Node &n_dims = params()["dims"];

    using Vec3f = vtkm::Vec<vtkm::Float64,3>;

    Vec3f v_dims;
    v_dims[0] = get_float32(n_dims["i"], data_object);
    v_dims[1] = get_float32(n_dims["j"], data_object);
    v_dims[2] = get_float32(n_dims["k"], data_object);
    Vec3f v_origin;
    v_origin[0] = get_float32(n_origin["x"], data_object);
    v_origin[1] = get_float32(n_origin["y"], data_object);
    v_origin[2] = get_float32(n_origin["z"], data_object);
    Vec3f v_spacing;
    v_spacing[0] = get_float32(n_spacing["dx"], data_object);
    v_spacing[1] = get_float32(n_spacing["dy"], data_object);
    v_spacing[2] = get_float32(n_spacing["dz"], data_object);

    


    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

    vtkh::SampleGrid grid_probe;

    if(params().has_path("invalid_value"))
    {
      vtkm::Float64 invalid_value = params()["invalid_value"].as_float64();
      grid_probe.InvalidValue(invalid_value);
    }
    grid_probe.Dims(v_dims);
    grid_probe.Origin(v_origin);
    grid_probe.Spacing(v_spacing);
    grid_probe.SetInput(&data);

    grid_probe.Update();

    vtkh::DataSet *grid_output = grid_probe.GetOutput();
    // we need to pass through the rest of the topologies, untouched,
    // and add the result of this operation
    VTKHCollection *new_coll = collection->copy_without_topology(topo_name);
    new_coll->add(*grid_output, topo_name);
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
}

//-----------------------------------------------------------------------------
bool
VTKHStats::verify_params(const conduit::Node &params,
                         conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");

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

    vtkh::Statistics::Result res = stats.Run(data, field_name);
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(Workspace::default_mpi_comm());
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    if(rank == 0)
    {
      res.Print(std::cout);
    }
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
}

//-----------------------------------------------------------------------------
bool
VTKHHistogram::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("bins",params, info, false, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("bins");

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
    if(rank == 0)
    {
      res.Print(std::cout);
    }
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
}

//-----------------------------------------------------------------------------
bool
VTKHProject2d::verify_params(const conduit::Node &params,
                             conduit::Node &info)
{
    info.reset();
    bool res = check_string("topology",params, info, false);
    res &= check_numeric("image_width",params, info, false);
    res &= check_numeric("image_height",params, info, false);

    std::vector<std::string> valid_paths;
    std::vector<std::string> ignore_paths;
    valid_paths.push_back("topology");
    valid_paths.push_back("image_width");
    valid_paths.push_back("image_height");
    valid_paths.push_back("camera");
    ignore_paths.push_back("camera");

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
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
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    vtkh::DataSet &data = collection->dataset_by_topology(topo_name);
    vtkm::Bounds bounds = data.GetGlobalBounds();
    vtkm::rendering::Camera camera;
    camera.ResetToBounds(bounds);

    if(params().has_path("camera"))
    {
      parse_camera(params()["camera"], camera);
    }

    int width = 512;
    int height = 512;
    if(params().has_path("image_width"))
    {
      width = params()["image_width"].to_int32();
    }
    if(params().has_path("image_height"))
    {
      height = params()["image_height"].to_int32();
    }

    vtkh::ScalarRenderer tracer;

    tracer.SetWidth(width);
    tracer.SetHeight(height);
    tracer.SetInput(&data);
    tracer.SetCamera(camera);

    tracer.Update();

    vtkh::DataSet *output = tracer.GetOutput();
    VTKHCollection *new_coll = new VTKHCollection();
    new_coll->add(*output, topo_name);
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
}

//-----------------------------------------------------------------------------
bool
VTKHNoOp::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");

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
}

//-----------------------------------------------------------------------------
bool
VTKHVectorComponent::verify_params(const conduit::Node &params,
                                   conduit::Node &info)
{
    info.reset();

    bool res = check_string("field",params, info, true);
    res &= check_numeric("component",params, info, true);
    res &= check_string("output_name",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("component");
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
}

//-----------------------------------------------------------------------------
bool
VTKHCompositeVector::verify_params(const conduit::Node &params,
                        conduit::Node &info)
{
    info.reset();

    bool res = check_string("field1",params, info, true);
    res &= check_string("field2",params, info, true);
    res &= check_string("field3",params, info, false);
    res &= check_string("output_name",params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field1");
    valid_paths.push_back("field2");
    valid_paths.push_back("field3");
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
}

//-----------------------------------------------------------------------------
bool
VTKHScale::verify_params(const conduit::Node &params,
                                  conduit::Node &info)
{
    info.reset();

    bool res = check_numeric("x_scale",params, info, true, true);
    res &= check_numeric("y_scale",params, info, true, true);
    res &= check_numeric("z_scale",params, info, true, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("x_scale");
    valid_paths.push_back("y_scale");
    valid_paths.push_back("z_scale");

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
}

//-----------------------------------------------------------------------------
bool
VTKHParticleAdvection::verify_params(const conduit::Node &params,
                                     conduit::Node &info)
{
    info.reset();
    bool res = check_string("field", params, info, true);
    res &= check_numeric("num_seeds", params, info, true, true);
    res &= check_numeric("num_steps", params, info, true, true);
    res &= check_numeric("step_size", params, info, true, true);
    res &= check_numeric("seed_bounding_box_xmin", params, info, true, true);
    res &= check_numeric("seed_bounding_box_xmax", params, info, true, true);
    res &= check_numeric("seed_bounding_box_ymin", params, info, true, true);
    res &= check_numeric("seed_bounding_box_ymax", params, info, true, true);
    res &= check_numeric("seed_bounding_box_zmin", params, info, true, true);
    res &= check_numeric("seed_bounding_box_zmax", params, info, true, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("num_seeds");
    valid_paths.push_back("num_steps");
    valid_paths.push_back("step_size");
    valid_paths.push_back("seed_bounding_box_xmin");
    valid_paths.push_back("seed_bounding_box_xmax");
    valid_paths.push_back("seed_bounding_box_ymin");
    valid_paths.push_back("seed_bounding_box_ymax");
    valid_paths.push_back("seed_bounding_box_zmin");
    valid_paths.push_back("seed_bounding_box_zmax");

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


    int numSeeds = get_int32(params()["num_seeds"], data_object);
    int numSteps = get_int32(params()["num_steps"], data_object);
    float stepSize = get_float32(params()["step_size"], data_object);

    float seedBBox[6];
    seedBBox[0] = get_float32(params()["seed_bounding_box_xmin"], data_object);
    seedBBox[1] = get_float32(params()["seed_bounding_box_xmax"], data_object);
    seedBBox[2] = get_float32(params()["seed_bounding_box_ymin"], data_object);
    seedBBox[3] = get_float32(params()["seed_bounding_box_ymax"], data_object);
    seedBBox[4] = get_float32(params()["seed_bounding_box_zmin"], data_object);
    seedBBox[5] = get_float32(params()["seed_bounding_box_zmax"], data_object);

    float dx = seedBBox[1] - seedBBox[0];
    float dy = seedBBox[3] - seedBBox[2];
    float dz = seedBBox[5] - seedBBox[4];

    if (dx < 0 || dy < 0 || dz < 0)
    {
      bool throw_error = false;
      detail::field_error(field_name, this->name(), collection, throw_error);
      // this creates a data object with an invalid soource
      set_output<DataObject>(new DataObject());
      return;
    }

    std::random_device device;
    std::default_random_engine generator(0);
    float  zero(0), one(1);
    std::uniform_real_distribution<vtkm::FloatDefault> distribution(zero, one);
    //Generate seeds

    std::vector<vtkm::Particle> seeds;
    for (int i = 0; i < numSeeds; i++)
    {
      float x = seedBBox[0] + dx * distribution(generator);
      float y = seedBBox[2] + dy * distribution(generator);
      float z = seedBBox[4] + dz * distribution(generator);
      seeds.push_back(vtkm::Particle({x,y,z}, i));
    }
    auto seedArray = vtkm::cont::make_ArrayHandle(seeds, vtkm::CopyFlag::On);


    vtkh::DataSet *output = nullptr;
    if (record_trajectories)
    {
      vtkh::Streamline sl;
      sl.SetStepSize(stepSize);
      sl.SetNumberOfSteps(numSteps);
      sl.SetSeeds(seeds);
      sl.SetField(field_name);
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

    std::vector<std::string> valid_paths;
    valid_paths.push_back("b_field");
    valid_paths.push_back("e_field");
    valid_paths.push_back("charge_field");
    valid_paths.push_back("mass_field");
    valid_paths.push_back("momentum_field");
    valid_paths.push_back("weighting_field");
    valid_paths.push_back("num_steps");
    valid_paths.push_back("step_size");

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
                                                     MPI_COMM_WORLD);
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
