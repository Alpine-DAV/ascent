#include <expressions/ascent_array_utils.hpp>
#include <ascent_config.h>
#include <expressions/ascent_dispatch.hpp>
#include <expressions/ascent_blueprint_architect.hpp>

#include <RAJA/RAJA.hpp>

#include <flow_workspace.hpp>
#include <map>

#ifdef ASCENT_MPI_ENABLED
#include <conduit_relay_mpi.hpp>
#include <mpi.h>
#endif
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
// -- begin ascent::runtime::expressions--
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{

int component_str_to_id(const conduit::Node &dataset,
                        const std::string field_name,
                        const std::string component)
{
  int comp_idx = -1;
  conduit::Node n_error;
  const int num_domains = dataset.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    if(dom.has_path("fields/"+field_name))
    {
      const conduit::Node &field = dom["fields/"+field_name];
      int num_components = field["values"].number_of_children();
      if((num_components == 0 || num_components == 1) && component == "")
      {
        comp_idx = 0;
      }
      else if(num_components == 0 && component != "")
      {
        n_error.append() = "specified a component name but no name exists";
      }
      else
      {
        std::vector<std::string> comp_names = field["values"].child_names();
        for(int name_idx = 0; name_idx < comp_names.size(); ++name_idx)
        {
          if(comp_names[name_idx] == component)
          {
            comp_idx = name_idx;
            break;
          }
        }
      }
      break;
    }
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  int global_idx;
  MPI_Allreduce(&comp_idx, &global_idx, 1, MPI_INT, MPI_MAX, mpi_comm);
  comp_idx = global_idx;
#endif

  return comp_idx;
}

// TODO If someone names their fields x,y,z things will go wrong
bool
is_xyz(const std::string &axis_name)
{
  return axis_name == "x" || axis_name == "y" || axis_name == "z";
}

int spatial_component(const std::string axis)
{
  int res = -1;
  if(axis == "x")
  {
    res = 0;
  }
  else if(axis == "y")
  {
    res = 1;
  }
  else if(axis == "z")
  {
    res = 2;
  }
  return res;
}

Array<double> allocate_bins(const std::string reduction_op, const conduit::Node &axes)
{
  const int num_axes = axes.number_of_children();

  // allocate memory for the total number of bins
  size_t num_bins = 1;
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    num_bins *= axes.child(axis_index)["bins"].dtype().number_of_elements() - 1;
  }
  std::cout<<"Total bins "<<num_bins<<"\n";

  // we might need additional space to keep track of statistics,
  // i.e., we might need to keep track of the bin sum and counts for
  // average
  int num_bin_vars = 2;
  if(reduction_op == "var" || reduction_op == "std")
  {
    num_bin_vars = 3;
  }
  else if(reduction_op == "min" || reduction_op == "max")
  {
    num_bin_vars = 1;
  }

  const int bins_size = num_bins * num_bin_vars;
  Array<double> bins;
  bins.resize(bins_size);
  return bins;
}

// get the association and topology and ensure they are the same
conduit::Node
verify_topo_and_assoc(const conduit::Node &dataset,
                      const std::vector<std::string> var_names)
{
  std::string assoc_str;
  std::string topo_name;
  bool error = false;
  conduit::Node error_msg;

  for(int dom_index = 0; dom_index < dataset.number_of_children(); ++dom_index)
  {
    const conduit::Node &dom = dataset.child(dom_index);
    for(const std::string &var_name : var_names)
    {
      if(dom.has_path("fields/" + var_name))
      {
        const std::string cur_assoc_str =
            dom["fields/" + var_name + "/association"].as_string();
        if(assoc_str.empty())
        {
          assoc_str = cur_assoc_str;
        }
        else if(assoc_str != cur_assoc_str)
        {
          error_msg.append() = "All Binning fields must have the same association.";
          error = true;
        }

        const std::string cur_topo_name =
            dom["fields/" + var_name + "/topology"].as_string();
        if(topo_name.empty())
        {
          topo_name = cur_topo_name;
        }
        else if(topo_name != cur_topo_name)
        {
          error_msg.append() = "All Binning fields must have the same topology.";
          error = true;
        }
      }
    }
  }
#ifdef ASCENT_MPI_ENABLED
  int rank;
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);

  struct MaxLoc
  {
    double size;
    int rank;
  };

  // there is no MPI_INT_INT so shove the "small" size into double
  MaxLoc maxloc = {(double)topo_name.length(), rank};
  MaxLoc maxloc_res;
  MPI_Allreduce(&maxloc, &maxloc_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi_comm);

  conduit::Node msg;
  msg["assoc_str"] = assoc_str;
  msg["topo_name"] = topo_name;
  conduit::relay::mpi::broadcast_using_schema(msg, maxloc_res.rank, mpi_comm);

  if(assoc_str != "" && assoc_str != msg["assoc_str"].as_string())
  {
    error_msg.append() = "All Binning fields must have the same association.";
    error = true;
  }
  if(topo_name != "" && topo_name != msg["topo_name"].as_string())
  {
    error_msg.append() = "All Binning fields must have the same topology.";
    error = true;
  }
  assoc_str = msg["assoc_str"].as_string();
  topo_name = msg["topo_name"].as_string();
#endif
  if(assoc_str.empty())
  {

    error_msg.append() = "Could not determine the associate from the given "
                          "reduction_var and axes. Try supplying a field.";
    error = true;
  }
  else if(assoc_str != "vertex" && assoc_str != "element")
  {
    error_msg.append() = "Unknown association: '"
                       + assoc_str
                       + "'. Binning only supports vertex and element association.";
  }

#ifdef ASCENT_MPI_ENABLED
  int error_int = error ? 1 : 0;
  int global_error;
  MPI_Allreduce(&error_int, &global_error, 1, MPI_INT, MPI_MAX, mpi_comm);
  conduit::Node global_msg;
  error = global_error == 1;
  if(global_error == 1)
  {
    conduit::relay::mpi::all_gather_using_schema(error_msg, global_msg, mpi_comm);
    error_msg = global_msg;
  }
#endif

  if(error)
  {
    ASCENT_ERROR(error_msg.to_yaml());
  }

  conduit::Node res;
  res["topo_name"] = topo_name;
  res["assoc_str"] = assoc_str;
  return res;
}

int coord_axis_to_int(const std::string coord)
{
  int axis = -1;
  if(coord == "x")
  {
    axis = 0;
  }
  else if(coord == "y")
  {
    axis = 1;
  }
  else if(coord == "z")
  {
    axis = 2;
  }
  if(axis == -1)
  {
    ASCENT_ERROR("Unknown coord axis '"<<coord
                 <<"'. This should not happen");
  }
  return axis;
}

// Create a set of explicit bins for all axes
conduit::Node
create_bins_axes(conduit::Node &bin_axes,
                 const conduit::Node &dataset,
                 std::string topo_name)
{

  // we are most likely to have some spatial bins so just calculate
  // the global bounds
  const conduit::Node &bounds = global_bounds(dataset, topo_name);
  const double *min_coords = bounds["min_coords"].value();
  const double *max_coords = bounds["max_coords"].value();

  bin_axes.print();
  conduit::Node res;

  const int num_axes = bin_axes.number_of_children();
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = bin_axes.child(i);
    const std::string axis_name = axis.name();
    std::cout<<"Axis name "<<axis_name<<"\n";
    res[axis_name + "/clamp"] = axis["clamp"];
    if(axis.has_path("bins"))
    {
      // we have explicit bins, so just use them
      // TODO: we could check and sort the bins
      // just in case there was a typo
      res[axis_name] = axis;
      continue;
    }

    // ok, we need to create a uniform binning for this
    // axis. So figure out the mins and maxs
    if(!axis.has_path("num_bins"))
    {
      ASCENT_ERROR("Axis missing num_bins and explicit bins");
    }

    const int num_bins = axis["num_bins"].to_int32();
    if(num_bins < 1)
    {
      ASCENT_ERROR("Binning: There must be at least one bin for axis '"<<
                   axis_name<<"'");
    }

    double min_val, max_val;
    bool has_min = axis.has_path("min_val");
    bool has_max = axis.has_path("max_val");
    if(has_min)
    {
      min_val = axis["min_val"].as_float64();
    }
    if(has_min)
    {
      max_val = axis["max_val"].as_float64();
    }

    if(is_xyz(axis_name))
    {
      int axis_id = coord_axis_to_int(axis_name);
      if(!has_min)
      {
        min_val = min_coords[axis_id];
      }
      if(!has_min)
      {
        max_val = max_coords[axis_id];
      }
      std::cout<<"spatial axis "<<axis_id<<"\n";
    }
    else
    {
      // this is a field, so
      std::cout<<"this is a field\n";
      if(!has_min)
      {
        min_val = field_min(dataset, axis_name)["value"].as_float64();
      }
      if(!has_max)
      {
        max_val = field_max(dataset, axis_name)["value"].as_float64();
      }
    }

    if(min_val > max_val)
    {
      ASCENT_ERROR("Binning: axis '"<<axis_name<<"' has invalid bounds "
                   <<"min "<<min_val<<" max "<<max_val);
    }

    const int divs = num_bins + 1;
    res[axis_name +"/bins"].set(conduit::DataType::float64(divs));
    double *bins = res[axis_name +"/bins"].value();
    const double delta = (max_val - min_val) / double(num_bins);
    bins[0] = min_val;

    for(int j = 1; j < divs - 1; ++j)
    {
      bins[j] = min_val + j * delta;
    }
    bins[divs - 1] = max_val;


  }
  res.print();
  return res;
}


template<typename Exec>
void calc_bindex(const Array<double> &values,
                 const int num_components,
                 const int component_id,
                 const int bin_stride,
                 const conduit::Node &axis,
                 Array<int> &bindexes,
                 Exec)
{
  const std::string mem_space = Exec::memory_space;
  using fp = typename Exec::for_policy;

  const int size = values.size();
  const double *values_ptr = values.ptr_const(mem_space);
  int *bindex_ptr = bindexes.ptr(mem_space);
  double *bins_node_ptr = const_cast<double*>(axis["bins"].as_float64_ptr());
  Array<double> bins( bins_node_ptr, axis["bins"].dtype().number_of_elements());
  const double *bins_ptr = bins.ptr_const(mem_space);
  const int bins_size = bins.size();
  bool clamp = axis["clamp"].to_int32() == 1;
  std::cout<<"**** bindex size "<<size<<"\n";

  RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
  {
    const int v_offset = i * num_components + component_id;
    const double value = values_ptr[v_offset];
    // just scan throught the bins, be facier later
    // we should get the index of the first bin that
    //
    int bindex = 0;
    while(value > bins_ptr[bindex])
    {
      ++bindex;
      if(bindex >= bins_size)
      {
        break;
      }
    }
    // make sure the last bin is inclusive
    if(value == bins_ptr[bins_size-1])
    {
      bindex = bins_size - 1;
    }
    if(!clamp && bindex == 0 && (value < bins_ptr[0]))
    {
      bindex = -1;
    }
    else if(!clamp && bindex == bins_size)
    {
      bindex = -1;
    }
    else
    {
      bindex = max(0,min(bindex,bins_size - 1));
    }
    int current_bindex = bins_ptr[i];
    bool valid = true;
    // this is missed some other bin, so just keep it that way
    if(current_bindex == -1 || bindex == -1)
    {
      valid = false;
    }

    int bin_value = bindex * bin_stride + current_bindex;
    if(!valid)
    {
      // keep it invalid
      bin_value = -1;
    }
    bindex_ptr[i] = bin_value;
  });
  ASCENT_ERROR_CHECK();
}

template<typename T, typename Exec>
Array<double> cast_to_float64(conduit::Node &field, const std::string component)
{
  const std::string mem_space = Exec::memory_space;
  using fp = typename Exec::for_policy;

  MemoryInterface<T> farray(field);
  MemoryAccessor<T> accessor = farray.accessor(mem_space,component);
  Array<double> res;
  const int size = accessor.m_size;
  res.resize(size);
  double *res_ptr = res.ptr(mem_space);

  RAJA::forall<fp> (RAJA::RangeSegment (0, size), [=] ASCENT_LAMBDA (RAJA::Index_type i)
  {
    res_ptr[i] = static_cast<double>(accessor[i]);
  });
  ASCENT_ERROR_CHECK();
  return res;
}

template<typename Exec>
Array<double> cast_field_values(conduit::Node &field, const std::string component, Exec )
{
  Array<double> res;
  if(field_is_float32(field))
  {
    res = cast_to_float64<conduit::float32,Exec>(field, component);
  }
  else if(field_is_float64(field))
  {
    res = cast_to_float64<conduit::float64,Exec>(field, component);
  }
  else if(field_is_int32(field))
  {
    res = cast_to_float64<conduit::int32,Exec>(field, component);
  }
  else if(field_is_int64(field))
  {
    res = cast_to_float64<conduit::int64,Exec>(field, component);
  }
  else
  {
#warning "this is a bad idea, one rank might have this field and others not"
// TODO: have a way to propgate errors that could hang things
    ASCENT_ERROR("Type dispatch: unsupported array type "<<
                  field.schema().to_string());
  }
  return res;
}

struct BindexingFunctor
{

  // map of domain_id to bin indexes
  std::map<int,Array<int>> m_bindexes;
  const conduit::Node &m_axes;
  conduit::Node &m_dataset;
  const std::string m_topo_name;
  const std::string m_assoc;
  const std::string m_component;
  BindexingFunctor(conduit::Node &dataset,
                   const conduit::Node &axes,
                   const std::string topo_name,
                   const std::string assoc,
                   const std::string component)
    : m_axes(axes),
      m_dataset(dataset),
      m_topo_name(topo_name),
      m_assoc(assoc),
      m_component(component)
  {}

  template<typename Exec>
  void operator()(const Exec &)
  {
    const int num_axes = m_axes.number_of_children();
    const int num_domains = m_dataset.number_of_children();
    for(int dom_index = 0; dom_index < num_domains; ++dom_index)
    {
      // ensure this domain has the necessary fields
      conduit::Node &dom = m_dataset.child(dom_index);
      for(int axis_index = 0; axis_index < num_axes; ++axis_index)
      {
        const conduit::Node &axis = m_axes.child(axis_index);
        const std::string axis_name = axis.name();
        if(!dom.has_path("fields/" + axis_name) && !is_xyz(axis_name))
        {
          continue;
        }
      }

      // Calculate the size of homes
      conduit::index_t homes_size = 0;
      if(m_assoc== "vertex")
      {
        homes_size = num_points(dom, m_topo_name);
      }
      else if(m_assoc == "element")
      {
        homes_size = num_cells(dom, m_topo_name);
      }
      // we have the assumption that ascent ensured that there
      // are in fact domain ids
      const int domain_id = dom["state/domain_id"].to_int32();
      Array<int> &bindexes = m_bindexes[domain_id];
      std::cout<<"*** Homes size "<<homes_size<<"\n";
      bindexes.resize(homes_size);
      array_memset(bindexes, 0);

      int *bins_ptr = bindexes.ptr(Exec::memory_space);

      bool do_once = true;
      // either vertex locations or centroids based on the
      // centering of the reduction variable
      Array<double> spatial_values;

      int bin_stride = 1;
      for(int axis_index = 0; axis_index < num_axes; ++axis_index)
      {
        const conduit::Node &axis = m_axes.child(axis_index);
        const std::string axis_name = axis.name();
        if(dom.has_path("fields/" + axis_name))
        {
          Array<double> values;
          std::cout<<"**** Casting field to double\n";
          conduit::Node &field = dom["fields/"+axis_name];
          values = cast_field_values(field, m_component, Exec());
          detail::calc_bindex(values,
                              1, // number of components
                              0, // which component
                              bin_stride,
                              axis,
                              bindexes,
                              Exec());

        }
        else // this is a spatatial axis
        {
          // this is a coordinate axis so we need the spatial information
          if(do_once)
          {
            do_once = false;
            if(m_assoc == "vertex")
            {
              std::cout<<"**** Getting vertices\n";
              spatial_values = vertices(dom, m_topo_name);
            }
            else
            {
              std::cout<<"**** Getting centroids\n";
              spatial_values = centroids(dom, m_topo_name);
            }
            std::cout<<"*** spatial values "<<spatial_values.size()<<"\n";
          }
          std::cout<<"Spatial axis "<<axis_name<<"\n";
          int comp = detail::spatial_component(axis_name);
          detail::calc_bindex(spatial_values,
                              3, // number of components
                              comp, // which component
                              bin_stride,
                              axis,
                              bindexes,
                              Exec());
        }

        bin_stride *= axis["bins"].dtype().number_of_elements() - 1;
      }

      // debug
      int bsize = bindexes.size();
      int *b_ptr = bindexes.host_ptr();
      for(int i = 0; i < bsize; ++i)
      {
        std::cout<<"Index "<<i<<" bin "<<b_ptr[i]<<"\n";
      }

    }
  }
};

} // namespace detail

conduit::Node data_binning(conduit::Node &dataset,
                           conduit::Node &bin_axes,
                           const std::string &reduction_var,
                           const std::string &reduction_op,
                           const double empty_bin_val,
                           const std::string &component)
{
  conduit::Node res;

  bin_axes.print();

  // first verify that all variables have matching associations
  // and are part of the same topology
  std::vector<std::string> var_names = bin_axes.child_names();
  if(!reduction_var.empty())
  {
    var_names.push_back(reduction_var);
  }
  const conduit::Node &topo_and_assoc =
      detail::verify_topo_and_assoc(dataset, var_names);
  const std::string topo_name = topo_and_assoc["topo_name"].as_string();
  const std::string assoc_str = topo_and_assoc["assoc_str"].as_string();

  // expand optional / automatic axes into explicit bins
  conduit::Node axes = detail::create_bins_axes(bin_axes, dataset, topo_name);
  std::cout<<"DONE BINS\n";

  Array<double> bins = detail::allocate_bins(reduction_op, axes);

  //int component_idx = detail::component_str_to_id(dataset,
  //                                                reduction_var,
  //                                                component);
  //if(component_idx == -1)
  //{
  //  ASCENT_ERROR("Binning: unable to resolve component");
  //}
  //std::cout<<"Component index "<<component_idx<<"\n";
  detail::BindexingFunctor bindexer(dataset,
                                    axes,
                                    topo_name,
                                    assoc_str,
                                    component);

  exec_dispatch(bindexer);

  ASCENT_ERROR("not done implementing");
  return res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::expressions--
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
