#include <expressions/ascent_array_utils.hpp>
#include <ascent_config.h>
#include <expressions/ascent_math.hpp>
#include <expressions/ascent_blueprint_device_dispatch.hpp>
#include <expressions/ascent_blueprint_architect.hpp>

#if defined(ASCENT_RAJA_ENABLED)
#include <RAJA/RAJA.hpp>
#endif

#include <flow_workspace.hpp>
#include <map>

#ifdef ASCENT_MPI_ENABLED
#include <conduit_relay_mpi.hpp>
#include <mpi.h>
#endif

// ============================================================================
// NOTE THIS IS AN INCOMPLETE RAJA VERSION OF BINNING , IT IS NOT IN USE YET
// ============================================================================


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

Array<double> allocate_bins(const std::string reduction_op,
                            const conduit::Node &axes,
                            int &total_bins)
{
  const int num_axes = axes.number_of_children();

  // allocate memory for the total number of bins
  size_t num_bins = 1;
  for(int axis_index = 0; axis_index < num_axes; ++axis_index)
  {
    num_bins *= axes.child(axis_index)["bins"].dtype().number_of_elements() - 1;
  }
  //std::cout<<"Total bins "<<num_bins<<"\n";

  // we might need additional space to keep track of statistics,
  // i.e., we might need to keep track of the bin sum and counts for
  // average
  int num_bin_vars = 2;
  if(reduction_op == "var" ||
     reduction_op == "std")
  {
    num_bin_vars = 3;
  }
  else if(reduction_op == "min"   ||
          reduction_op == "max"   ||
          reduction_op == "pdf"   ||
          reduction_op == "count" )
  {
    num_bin_vars = 1;
  }

  const int bins_size = num_bins * num_bin_vars;
  Array<double> bins;
  bins.resize(bins_size);

  double init_val = 0.;

  // init the memory
  if(reduction_op == "max")
  {
    init_val = std::numeric_limits<double>::lowest();
  }
  else if (reduction_op == "min")
  {
    init_val = std::numeric_limits<double>::max();
  }

  array_memset(bins, init_val);

  total_bins = static_cast<int>(num_bins);
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

    error_msg.append() = "Could not determine the association from the given "
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

  // bin_axes.print();
  conduit::Node res;

  const int num_axes = bin_axes.number_of_children();
  for(int i = 0; i < num_axes; ++i)
  {
    const conduit::Node &axis = bin_axes.child(i);
    const std::string axis_name = axis.name();
    //std::cout<<"Axis name "<<axis_name<<"\n";
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
      //std::cout<<"spatial axis "<<axis_id<<"\n";
    }
    else
    {
      // this is a field, so
      //std::cout<<"this is a field\n";
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

    // if we had to calc them, propgate our min and max vals
    if(!has_min)
    {
      bin_axes.child(i)["min_val"] = min_val;
    }
    if(!has_max)
    {
      bin_axes.child(i)["max_val"] = max_val;
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
  // res.print();
  return res;
}


///
/// `values` can be 2D multi-component array
/// bindexes is 1D, either number of verts or number of eles long
///

template<typename Exec>
void calc_bindex(const Array<double> &values,
                 const int num_components,
                 const int component_id,
                 const conduit::Node &axis,
                 const int bin_stride,
                 Array<int> &bindexes,
                 Exec)
{
  const std::string mem_space = Exec::memory_space;
  axis.print();
  // number of values to bin
  const int size = bindexes.size();
  // values we want to bin
  const double *values_ptr = values.get_ptr_const(mem_space);
  // bindexs (binning index result for each value)
  int *bindex_ptr = bindexes.get_ptr(mem_space);

  //  the bin extents for given axis
  double *bins_node_ptr = const_cast<double*>(axis["bins"].as_float64_ptr());
  Array<double> bins( bins_node_ptr, axis["bins"].dtype().number_of_elements());
  const int bins_size = bins.size();
  const double *bins_ptr = bins.get_ptr_const(mem_space);

  bool clamp = axis["clamp"].to_int32() == 1;
  // std::cout<<"**** bindex size "<<size<<"\n";

  using for_policy = typename Exec::for_policy;

  ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
  {
    // std::cout<<"**** bindex idx "<< i <<"\n";
    // calc location of bin result
    const int v_offset = i * num_components + component_id;
    const double value = values_ptr[v_offset];

    // just scan through the bins, be fancier later
    // we should get the index of the first bin that
    // is greater than the value

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
      bindex = bins_size - 2;
    }
    // if we aren't clamping and we are less
    // than the first bin, invalidate the index
    else if(!clamp && bindex == 0 && (value < bins_ptr[0]))
    {
      bindex = -1;
    }
    // if we aren't clamping and we above
    // than the last bin, invalidate the index
    else if(!clamp && bindex == bins_size)
    {
      bindex = -1;
    }
    else // otherwise we min/max to clamp
    {
      // adj back to zero-based, we have one less bin
      // than we have bin bounds
      bindex--;
      bindex = max(0,min(bindex,bins_size - 2));
    }

    // check bindex from prior passes
    // if any were -1, that means we are out of bin range
    // keep -1
    int prev_bindex = bindex_ptr[i];
    if (prev_bindex == -1 || bindex == -1)
    {
      bindex = -1;
    }
    // we may have prev pass, apply striding to new value
    // to eventually land
    else 
    {
      bindex =  bindex * bin_stride + prev_bindex;
    }
    bindex_ptr[i] = bindex;
    // std::cout << "final bindex for " << i << " " << bindex_ptr[i] << std::endl;
  });
  ASCENT_DEVICE_ERROR_CHECK();
}

template<typename T, typename Exec>
Array<double> cast_to_float64(conduit::Node &field, const std::string component)
{
  const std::string mem_space = Exec::memory_space;


  MCArray<T> farray(field["values"]);
  DeviceAccessor<T> accessor = farray.accessor(mem_space,component);
  Array<double> res;
  const int size = accessor.m_size;
  res.resize(size);
  double *res_ptr = res.get_ptr(mem_space);
  using for_policy = typename Exec::for_policy;
  ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
  {
    res_ptr[i] = static_cast<double>(accessor[i]);
  });
  //std::cout<<"Cast to float64 "<<mem_space<<"\n";
  // res.status();
  ASCENT_DEVICE_ERROR_CHECK();
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
    // This error is a bad idea, one rank might have this field and others not
    // TODO: have a way to propgate errors that could hang things
    ASCENT_ERROR("Type dispatch: unsupported array type "<<
                 field.schema().to_string());
  }
  return res;
}

struct BinningFunctor
{
  // per domain bindexes
  // per domain values
  std::map<int,Array<int>> &m_bindexes;
  std::map<int,Array<double>> &m_values;
  Array<double> &m_bins;
  const std::string m_op;
  std::vector<int> m_domain_ids;


  BinningFunctor() = delete;
  BinningFunctor(std::map<int,Array<int>> &bindexes,
                 std::map<int,Array<double>> &values,
                 Array<double> &bins,
                 const std::string op)
    : m_bindexes(bindexes),
      m_values(values),
      m_bins(bins),
      m_op(op)
  {
    for(auto &pair : m_values)
    {
      m_domain_ids.push_back(pair.first);
    }
  }

  template<typename Exec>
  void operator()(const Exec &)
  {

    using for_policy = typename Exec::for_policy;
    using atomic_policy = typename Exec::atomic_policy;

    for(auto dom_id : m_domain_ids)
    {
      const int size = m_values[dom_id].size();
      const int *bindex_ptr = m_bindexes[dom_id].get_ptr_const(Exec::memory_space);
      const double *values_ptr = m_values[dom_id].get_ptr_const(Exec::memory_space);
      double *bins_ptr = m_bins.get_ptr(Exec::memory_space);

      // m_values[dom_id].status();
      // m_values[dom_id].summary();
      // m_bindexes[dom_id].summary();

      if(m_op == "min")
      {
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            const double value = values_ptr[i];
            ascent::atomic_min<atomic_policy>(bins_ptr + bindex, value);
          }
        });
      }
      else if(m_op == "max")
      {
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            const double value = values_ptr[i];
            ascent::atomic_max<atomic_policy>(bins_ptr + bindex, value);
          }
        });
      }
      else if(m_op == "count")
      {
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {

          const int bindex = bindex_ptr[i];
          // std::cout << " size "<< size << " i" << i << " bindex" << bindex << std::endl;
          if(bindex>= 0)
          {
            ascent::atomic_add<atomic_policy>(bins_ptr + bindex, 1.);
          }
        });
      }
      else if(m_op == "pdf")
      {
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            ascent::atomic_add<atomic_policy>(bins_ptr + bindex, 1.);
          }
        });
      }
      else if(m_op == "avg" || m_op == "sum")
      {
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            const double value = values_ptr[i];
            const int offset = bindex * 2;
            ascent::atomic_add<atomic_policy>(bins_ptr + offset, value);
            ascent::atomic_add<atomic_policy>(bins_ptr + offset + 1, 1.);
          }
        });
      }
      else if(m_op == "rms")
      {
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            const double value = values_ptr[i];
            const int offset = bindex * 2;
            ascent::atomic_add<atomic_policy>(bins_ptr + offset, value * value);
            ascent::atomic_add<atomic_policy>(bins_ptr + offset + 1, 1.);
          }
        });
      }
      else if(m_op == "var" || m_op == "std")
      {
        // use basic 2-pass, since single pass sum of squares method is not
        // numerically stable
        // http://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            const double value = values_ptr[i];
            const int offset = bindex * 3;
            // accum value
            ascent::atomic_add<atomic_policy>(bins_ptr + offset, value);
            // count
            ascent::atomic_add<atomic_policy>(bins_ptr + offset + 1, 1.);
          }
        });
        // NOTE: in second pass we only read [0] and [1], and write [2]
        ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA (index_t i)
        {
          const int bindex = bindex_ptr[i];
          if(bindex >= 0)
          {
            const double value = values_ptr[i];
            const int offset = bindex * 3;
            const double val_avg = bins_ptr[offset] / bins_ptr[offset +1];
            // (v - mean) ^ 2
            double val_cal = (value - val_avg); 
            val_cal = val_cal * val_cal;
            ascent::atomic_add<atomic_policy>(bins_ptr + offset + 2, val_cal);
          }
        });
      }

      // DEBUGGING:
      // double *host_ptr = m_bins.get_host_ptr();
      // for(int i = 0; i < m_bins.size(); ++i)
      // {
      //   std::cout<<"bin accumulate results index "<<i<<" "<<host_ptr[i]<<"\n";
      // }
    }
  }
};

struct BindexingFunctor
{

  // map of domain_id to bin indexes
  std::map<int,Array<int>> m_bindexes;
  std::map<int,Array<double>> m_values;
  const conduit::Node &m_axes;
  conduit::Node &m_dataset;
  const std::string m_topo_name;
  const std::string m_assoc;
  const std::string m_component;
  const std::string m_reduction_var;
  BindexingFunctor(conduit::Node &dataset,
                   const conduit::Node &axes,
                   const std::string topo_name,
                   const std::string assoc,
                   const std::string component,
                   const std::string reduction_var)
    : m_axes(axes),
      m_dataset(dataset),
      m_topo_name(topo_name),
      m_assoc(assoc),
      m_component(component),
      m_reduction_var(reduction_var)
  {}

  template<typename Exec>
  void operator()(const Exec &)
  {
    const int num_domains = m_dataset.number_of_children();
    const int num_axes = m_axes.number_of_children();

    // loop over domains
    for(int dom_index = 0; dom_index < num_domains; ++dom_index)
    {
      // ensure this domain has the necessary topo and fields
      conduit::Node &dom = m_dataset.child(dom_index);

      if(!dom["topologies"].has_child(m_topo_name))
      {
        continue;
      }

      for(int axis_index = 0; axis_index < num_axes; ++axis_index)
      {
        const conduit::Node &axis   = m_axes.child(axis_index);
        const std::string axis_name = axis.name();
        // skip the axis name is not one of the spatial axes
        // or if the domain is missing the named field
        if(!dom.has_path("fields/" + axis_name) && !is_xyz(axis_name))
        {
          continue;
        }
      }
      // find the coordset name for the topo
      std::string coords_name = dom["topologies"][m_topo_name]["coordset"].as_string();

      // find the mesh dims
      int coords_dims = conduit::blueprint::mesh::coordset::dims(dom["coordsets"][coords_name]);

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

      //std::cout<<"*** Homes size "<<homes_size<<"\n";
      bindexes.resize(homes_size);
      array_memset(bindexes, 0);

      int *bins_ptr = bindexes.get_ptr(Exec::memory_space);

      // we need to track if values were pulled out
      // things downstream expect the bindexer to do this

      bool reduction_op_values_found = false;
      bool do_once = true;
      // either vertex locations or centroids based on the
      // centering of the reduction variable
      Array<double> spatial_values;

      int bin_stride = 1;

      for(int axis_index = 0; axis_index < num_axes; ++axis_index)
      {
        const conduit::Node &axis   = m_axes.child(axis_index);
        const std::string axis_name = axis.name();
        // std::cout << "axis index = " <<  axis_index << " " << axis_name << std::endl;
        // case where bin axis is a field
        if(dom.has_path("fields/" + axis_name))
        {
          Array<double> values;
          //std::cout<<"**** Casting field to double\n";
          conduit::Node &field = dom["fields/"+axis_name];
          values = cast_field_values(field, m_component, Exec());
          detail::calc_bindex(values,
                              1, // number of components
                              0, // which component
                              axis,
                              bin_stride,
                              bindexes,
                              Exec());
          if(axis_name == m_reduction_var)
          {
            reduction_op_values_found = true;
            m_values[domain_id] = values;
            //std::cout<<"**** VALUES **** \n";
            // values.status();
            // values.summary();
          }
        }
        // case where bin axis is one of the spatial axes
        else // this is a spatial axis
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
              std::cout << spatial_values.size() << std::endl;
              double *sval_ptr = spatial_values.get_host_ptr();
              index_t sidx = 0;
              for(index_t e_idx = 0; e_idx < homes_size; e_idx++)
              {
                std::cout << sval_ptr[sidx]   << " "
                          << sval_ptr[sidx+1] << " ";

                if(coords_dims == 3)
                { 
                    std::cout << sval_ptr[sidx+2] << "   ";
                }
                else 
                {
                    std::cout << "  ";
                }
                sidx+=coords_dims;
              }
              std::cout << std::endl;
            }
            //std::cout<<"*** spatial values "<<spatial_values.size()<<"\n";
          }
          //std::cout<<"Spatial axis "<<axis_name<<"\n";

          int comp = detail::spatial_component(axis_name);
          detail::calc_bindex(spatial_values,
                              coords_dims, // number of components
                              comp, // which component
                              axis,
                              bin_stride,
                              bindexes,
                              Exec());
        }

        bin_stride *= axis["bins"].dtype().number_of_elements() - 1;
      }

      // we need to extract the values if we haven't already
      // pulled them out
      if(!reduction_op_values_found)
      { 
         // empty reduction var is supported for count and pdf
         if(m_reduction_var != "")
         {
           conduit::Node &field = dom["fields/"+m_reduction_var];
           m_values[domain_id] = cast_field_values(field, m_component, Exec()); 
         }
      }

      // debug
      // int bsize = bindexes.size();
      // int *b_ptr = bindexes.get_host_ptr();
      // for(int i = 0; i < bsize; ++i)
      // {
      //   std::cout<<"Index "<<i<<" bin "<<b_ptr[i]<<"\n";
      // }

    }
  }
};

void exchange_bins(Array<double> &bins, const std::string op)
{
#ifdef ASCENT_MPI_ENABLED
  double *bins_ptr = bins.get_host_ptr();
  const int bins_size = bins.size();
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  Array<double> global_bins;
  global_bins.resize(bins_size);
  double *global_ptr = global_bins.get_host_ptr();

  if(op == "sum" || op == "pdf" || op == "avg" || op == "count" ||
     op == "std" || op == "var" || op == "rms")
  {
    MPI_Allreduce(bins_ptr, global_ptr, bins_size, MPI_DOUBLE, MPI_SUM, mpi_comm);
  }
  else if(op == "min")
  {
    MPI_Allreduce(bins_ptr, global_ptr, bins_size, MPI_DOUBLE, MPI_MIN, mpi_comm);
  }
  else if(op == "max")
  {
    MPI_Allreduce(bins_ptr, global_ptr, bins_size, MPI_DOUBLE, MPI_MAX, mpi_comm);
  }
  bins = global_bins;
#endif
}

struct BinningReductionFunctor
{
  conduit::Node     &m_res;
  Array<double>     &m_bins;
  const int          m_num_bins;
  const std::string  m_op;
  const double       m_empty_val;

  BinningReductionFunctor() = delete;
  BinningReductionFunctor(conduit::Node &res,
                          Array<double> &bins,
                          const int num_bins,
                          const std::string op,
                          const double empty_val)
    : m_res(res),
      m_bins(bins),
      m_num_bins(num_bins),
      m_op(op),
      m_empty_val(empty_val)
  {
    // empty
  }

  template<typename Exec>
  void operator()(const Exec &)
  {
    double *bins_ptr = m_bins.get_ptr(Exec::memory_space);
    const int size = m_num_bins;

    const double min_default = std::numeric_limits<double>::max();
    const double max_default = std::numeric_limits<double>::lowest();
    const double empty_val = m_empty_val;
    double pdf_total = 0;

    using for_policy = typename Exec::for_policy;
    using reduce_policy = typename Exec::reduce_policy;

    Array<double> results;
    results.resize(size);
    //std::cout<<"Num bins "<<size<<"\n";
    array_memset(results, m_empty_val);
    double *res_ptr = results.get_ptr(Exec::memory_space);

    // break in to per op foralls

    if(m_op == "min")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // min
        double val = bins_ptr[i];
        if(val == min_default)
        {
          val = empty_val;
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "max")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // max
        double val = bins_ptr[i];
        if(val == max_default)
        {
          val = empty_val;
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "sum")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // sum
        double val = bins_ptr[i*2];
        double count = bins_ptr[i*2+1];
        if(count == 0.)
        {
          val = empty_val;
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "avg")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // avg
        const double sum   = bins_ptr[2 * i];
        const double count = bins_ptr[2 * i + 1];
        double val;
        if(count == 0)
        {
          val = empty_val;
        }
        else
        {
          val = sum / count;
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "rms")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // rms
        const double sum_x = bins_ptr[2 * i];
        const double n = bins_ptr[2 * i + 1];
        double val;
        if(n == 0)
        {
          val = empty_val;
        }
        else
        {
          val = sqrt(sum_x / n);
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "var")
    { 
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // std =  sum( v[i] - avg(vs)^2) / count
        const double count = bins_ptr[3 * i + 1];
        const double var   = bins_ptr[3 * i + 2];
        double val;
        if(count == 0)
        {
          val = empty_val;
        }
        else
        {
          val = var/count;
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "std")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // std =  sqrt( sum( v[i] - avg(vs)^2) / count )
        const double count = bins_ptr[3 * i + 1];
        const double var   = bins_ptr[3 * i + 2];

        double val;
        if(count == 0)
        {
          val = empty_val;
        }
        else
        {
          val = sqrt(var/count);
        }
        res_ptr[i] = val;
      });
    }
    
    else if(m_op == "pdf")
    {
      // two pass
      double pdf_total = 0.0;
      // we need the total count to generate a pdf
      ascent::ReduceSum<reduce_policy, double> sum(0.0);
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        sum += bins_ptr[i];
      });
      pdf_total = sum.get();

      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // pdf
        double val = bins_ptr[i];
        if(val == 0)
        {
          val = empty_val;
        }
        else
        {
          val /= pdf_total;
        }
        res_ptr[i] = val;
      });
    }
    else if(m_op == "count")
    {
      ascent::forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
        // count
        double val = bins_ptr[i];
        if(val == 0)
        {
          val = empty_val;
        }
        res_ptr[i] = val;
      });
    }
    // // debugging: 
    // double *host_ptr = results.get_host_ptr();
    // for(int i = 0; i < m_num_bins; ++i)
    // {
    //   //std::cout<<"res bin "<<i<<" "<<host_ptr[i]<<"\n";
    // }
    m_res["value"].set(results.get_host_ptr(), m_num_bins);
  }
};

} // namespace detail

conduit::Node data_binning(conduit::Node &dataset,
                           conduit::Node &bin_axes,
                           const std::string &reduction_var,
                           const std::string &reduction_op,
                           const double empty_bin_val,
                           const std::string &component,
                           std::map<int,Array<int>> &bindexes)
{
  // bin_axes.print();

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
                                    component,
                                    reduction_var);

  exec_dispatch_function(bindexer);

  // return the bindexes so we can paint later
  bindexes = bindexer.m_bindexes;

  // we now have the all of the bin setup, all we have to
  // do now is the reduction
  int num_bins;
  Array<double> bins = detail::allocate_bins(reduction_op, axes, num_bins);

  detail::BinningFunctor binner(bindexer.m_bindexes,
                                bindexer.m_values,
                                bins,
                                reduction_op);

  exec_dispatch_function(binner);

  //std::cout<<"DONE BINinng\n";
  // mpi exchange
  detail::exchange_bins(bins, reduction_op);

  // use the intermediate results to calc the final bin values
  conduit::Node res;
  detail::BinningReductionFunctor reducer(res,
                                          bins,
                                          num_bins,
                                          reduction_op,
                                          empty_bin_val);

  exec_dispatch_function(reducer);

  res["association"] = assoc_str;
  //std::cout<<"res "<<res.to_summary_string()<<"\n";
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
