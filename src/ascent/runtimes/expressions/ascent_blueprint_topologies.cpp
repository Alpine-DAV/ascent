#include "ascent_blueprint_topologies.hpp"

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
// -- begin ascent::runtime::expressions --
//-----------------------------------------------------------------------------
namespace expressions
{

namespace detail
{
// double or float for a topology in a given domain
std::string
coord_dtype(const std::string &topo_name, const conduit::Node &domain)
{
  // ok, so we can have a mix of uniform and non-uniform
  // coords, where non-uniform coords have arrays
  // if we only have unirform, the double,
  // if some have arrays, then go with whatever
  // that is.
  bool is_float = false;
  bool has_array = false;
  bool error = false;

  const std::string topo_path = "topologies/" + topo_name;
  std::string type_name;

  if(domain.has_path(topo_path))
  {
    std::string coord_name = domain[topo_path + "/coordset"].as_string();
    const conduit::Node &n_coords = domain["coordsets/" + coord_name];
    const std::string coords_type = n_coords["type"].as_string();
    if(coords_type != "uniform")
    {
      has_array = true;

      if(n_coords["values/x"].dtype().is_float32())
      {
        is_float = true;
      }
      else if(!n_coords["values/x"].dtype().is_float64())
      {
        is_float = false;
        type_name = n_coords["/values/x"].dtype().name();
        error = true;
      }
    }
  }
  else
  {
    ASCENT_ERROR("Could not determine the data type of topology '"
                 << topo_name << "' in domain '" << domain.name()
                 << "' because it was not found there.");
  }

  if(error)
  {

    ASCENT_ERROR("Coords array from topo '" << topo_name
                                            << "' is neither float or double."
                                            << " type is '" << type_name << "'."
                                            << " Contact someone.");
  }

  bool my_vote = has_array && is_float;

  return my_vote ? "float" : "double";
}



template <size_t num_dims>
std::array<size_t, num_dims>
logical_index(const size_t index, const std::array<size_t, num_dims> &dims)
{
  ASCENT_ERROR("Unsupported number of dimensions: " << num_dims);
}

template <>
std::array<size_t, 1>
logical_index(const size_t index, const std::array<size_t, 1> &dims)
{
  return {index};
}

template <>
std::array<size_t, 2>
logical_index(const size_t index, const std::array<size_t, 2> &dims)
{
  return {index % dims[0], index / dims[0]};
}

template <>
std::array<size_t, 3>
logical_index(const size_t index, const std::array<size_t, 3> &dims)
{
  return {index % dims[0],
          (index / dims[0]) % dims[0],
          index / (dims[0] * dims[1])};
}

}  // namespace detail

//-----------------------------------------------------------------------------
// -- Topology
//-----------------------------------------------------------------------------
Topology::Topology(const std::string &topo_name,
                   const conduit::Node &domain,
                   const size_t num_dims)
    : domain(domain), topo_name(topo_name),
      topo_type(domain["topologies/" + topo_name + "/type"].as_string()),
      coords_name(domain["topologies/" + topo_name + "/coordset"].as_string()),
      coords_type(domain["coordsets/" + coords_name + "/type"].as_string()),
      num_dims(num_dims)
{
}

size_t
Topology::get_num_points() const
{
  return num_points;
}

size_t
Topology::get_num_cells() const
{
  return num_cells;
}

//-----------------------------------------------------------------------------
// -- PointTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
PointTopology<T, N>::PointTopology(const std::string &topo_name,
                                   const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "point")
  {
    ASCENT_ERROR("Cannot initialize a PointTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }

  if(this->coord_type == "uniform")
  {
    const conduit::Node &n_coords = domain["coordsets/" + this->coords_name];
    const conduit::Node &n_dims = n_coords["dims"];
    const conduit::Node &n_origin = n_coords["origin"];
    const conduit::Node &n_spacing = n_coords["spacing"];
    for(size_t i = 0; i < N; ++i)
    {
      const std::string dim = std::string(1, 'i' + i);
      const std::string coord = std::string(1, 'x' + i);
      dims[i] = n_dims[dim].to_int();
      origin[i] = n_origin[dim].to_float64();
      spacing[i] = n_spacing["d" + coord].to_float64();
      num_points *= dims[i];
      num_cells *= dims[i] - 1;
    }
  }
  else if(this->coord_type == "rectilinear")
  {
    const conduit::Node &values =
        domain["coordsets/" + this->coords_name + "/values"];
    num_points = 1;
    for(size_t i = 0; i < N; ++i)
    {
      const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
      coords[i] = coord_values.value();
      num_points *= coords[i].dtype().number_of_elements();
    }
  }
  else if(this->coord_type == "explicit")
  {
    const conduit::Node &values =
        domain["coordsets/" + this->coords_name + "/values"];
    for(size_t i = 0; i < N; ++i)
    {
      const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
      coords[i] = coord_values.value();
    }
    num_points = coords[0].dtype().number_of_elements();
  }
  else
  {
    ASCENT_ERROR("Unknown coordinate type '"
                 << this->coord_type << "' for point topology '" << topo_name
                 << "' in domain " << domain.name() << ".");
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
PointTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  if(this->coord_type == "uniform")
  {
    auto l_index = detail::logical_index(index, dims);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] = origin[i] + l_index[i] * spacing[i];
    }
  }
  else if(this->coord_type == "rectilinear")
  {
    std::array<size_t, N> dims;
    for(size_t i = 0; i < N; ++i)
    {
      dims[i] = coords[i].number_of_elements();
    }
    const auto l_index = detail::logical_index(index, dims);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] = coords[i][l_index[i]];
    }
  }
  else if(this->coord_type == "explicit")
  {
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] = coords[i][index];
    }
  }
  else
  {
    ASCENT_ERROR("Unknown coordinate type '"
                 << this->coord_type << "' for point topology '" << topo_name
                 << "' in domain " << domain.name() << ".");
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
PointTopology<T, N>::element_location(const size_t index) const
{
  ASCENT_ERROR("Cannot get the element location of a point topology '"
               << topo_name << "'.");
  // return something so we don't get compiler warnings
  std::array<conduit::float64, 3> loc{};
  return loc;
}

template <typename T, size_t N>
size_t
PointTopology<T, N>::get_num_cells() const
{
  ASCENT_ERROR("Cannot get the number of cells in a point topology '"
               << topo_name << "'.");
  // return something so we don't get compiler warnings
  return 0;
}

//-----------------------------------------------------------------------------
// -- UniformTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
UniformTopology<T, N>::UniformTopology(const std::string &topo_name,
                                       const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "uniform")
  {
    ASCENT_ERROR("Cannot initialize a UniformTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }

  const conduit::Node &n_coords = domain["coordsets/" + this->coords_name];
  const conduit::Node &n_dims = n_coords["dims"];
  const conduit::Node &n_origin = n_coords["origin"];
  const conduit::Node &n_spacing = n_coords["spacing"];
  num_points = 1;
  num_cells = 1;
  for(size_t i = 0; i < N; ++i)
  {
    const std::string dim = std::string(1, 'i' + i);
    const std::string coord = std::string(1, 'x' + i);
    dims[i] = n_dims[dim].to_int32();
    origin[i] = n_origin[coord].to_float64();
    spacing[i] = n_spacing["d" + coord].to_float64();
    num_points *= dims[i];
    num_cells *= dims[i] - 1;
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UniformTopology<T, N>::vertex_location(const size_t index) const
{
  auto l_index = detail::logical_index(index, dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = origin[i] + l_index[i] * spacing[i];
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UniformTopology<T, N>::element_location(const size_t index) const
{
  std::array<size_t, N> element_dims;
  for(size_t i = 0; i < N; ++i)
  {
    element_dims[i] = dims[i] - 1;
  }
  const auto l_index = detail::logical_index(index, element_dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = origin[i] + (l_index[i] + 0.5) * spacing[i];
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- RectilinearTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
RectilinearTopology<T, N>::RectilinearTopology(const std::string &topo_name,
                                               const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "rectilinear")
  {
    ASCENT_ERROR("Cannot initialize a RectilinearTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }

  const conduit::Node &values =
      domain["coordsets/" + this->coords_name + "/values"];
  num_points = 1;
  num_cells = 1;
  for(size_t i = 0; i < N; ++i)
  {
    const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
    coords[i] = coord_values.value();
    num_points *= coords[i].dtype().number_of_elements();
    num_cells *= coords[i].dtype().number_of_elements() - 1;
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
RectilinearTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<size_t, N> dims;
  for(size_t i = 0; i < N; ++i)
  {
    dims[i] = coords[i].number_of_elements();
  }
  const auto l_index = detail::logical_index(index, dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = coords[i][l_index[i]];
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
RectilinearTopology<T, N>::element_location(const size_t index) const
{
  std::array<size_t, N> dims;
  for(size_t i = 0; i < N; ++i)
  {
    dims[i] = coords[i].number_of_elements() - 1;
  }
  const auto l_index = detail::logical_index(index, dims);
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = (coords[i][l_index[i]] + coords[i][l_index[i] + 1]) / 2;
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- StructuredTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
StructuredTopology<T, N>::StructuredTopology(const std::string &topo_name,
                                             const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "structured")
  {
    ASCENT_ERROR("Cannot initialize a StructuredTopology class from topology '"
                 << topo_name << "' in domain " << domain.name()
                 << " which has type '" << this->topo_type << "'.");
  }
  const conduit::Node &values =
      domain["coordsets/" + this->coords_name + "/values"];
  const conduit::Node &n_dims =
      domain["topologies/" + topo_name + "/elements/dims"];
  num_points = 1;
  num_cells = 1;
  for(size_t i = 0; i < N; ++i)
  {
    const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
    const std::string &dim = std::string(1, 'i' + i);
    coords[i] = coord_values.value();
    // the blueprint gives structured dims in terms of elements not vertices so
    // we change it to vertices so that it's consistent with uniform
    dims[i] = n_dims[dim].to_int32() + 1;
    num_points *= dims[i];
    num_cells *= dims[i] - 1;
  }
  // check that number of vertices in coordset matches dims
  if((size_t)coords[0].dtype().number_of_elements() != num_points)
  {
    ASCENT_ERROR(
        "StructuredTopology ("
        << topo_name << "): The number of points calculated (" << num_points
        << ") differs from the number of vertices in corresponding coordset ("
        << coords[0].dtype().number_of_elements() << ").");
  }
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
StructuredTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = coords[i][index];
  }
  return loc;
}

constexpr size_t
constexpr_pow(size_t x, size_t y)
{
  return y == 0 ? 1 : x * constexpr_pow(x, y - 1);
}

// vertices are ordered in the VTK format
// https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
template <typename T, size_t N>
std::array<conduit::float64, 3>
StructuredTopology<T, N>::element_location(const size_t index) const
{

  std::array<size_t, N> element_dims;
  for(size_t i = 0; i < N; ++i)
  {
    element_dims[i] = dims[i] - 1;
  }
  const auto element_index = detail::logical_index(index, element_dims);

  constexpr size_t num_vertices = constexpr_pow(2, N);
  std::array<size_t, num_vertices> vertices;
  if(num_vertices == 2)
  {
    vertices[0] = element_index[0];
    vertices[1] = vertices[0] + 1;
  }
  else if(num_vertices == 4)
  {
    vertices[0] = element_index[1] * dims[0] + element_index[0];
    vertices[1] = vertices[0] + 1;
    vertices[2] = vertices[1] + dims[0];
    vertices[3] = vertices[2] - 1;
  }
  else if(num_vertices == 8)
  {
    vertices[0] = (element_index[2] * dims[1] + element_index[1]) * dims[0] +
                  element_index[0];
    vertices[1] = vertices[0] + 1;
    vertices[2] = vertices[1] + dims[0];
    vertices[3] = vertices[2] - 1;
    vertices[4] = vertices[0] + dims[0] * dims[1];
    vertices[5] = vertices[4] + 1;
    vertices[6] = vertices[5] + dims[0];
    vertices[7] = vertices[6] - 1;
  }

  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < num_vertices; ++i)
  {
    const auto vert_loc = vertex_location(vertices[i]);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] += vert_loc[i];
    }
  }
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] /= num_vertices;
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- UnstructuredTopology
//-----------------------------------------------------------------------------
template <typename T, size_t N>
UnstructuredTopology<T, N>::UnstructuredTopology(const std::string &topo_name,
                                                 const conduit::Node &domain)
    : Topology(topo_name, domain, N)
{
  if(this->topo_type != "unstructured")
  {
    ASCENT_ERROR(
        "Cannot initialize a UnstructuredTopology class from topology '"
        << topo_name << "' in domain " << domain.name() << " which has type '"
        << this->topo_type << "'.");
  }
  const conduit::Node &values =
      domain["coordsets/" + this->coords_name + "/values"];
  for(size_t i = 0; i < N; ++i)
  {
    const conduit::Node &coord_values = values.fetch(std::string(1, 'x' + i));
    coords[i] = coord_values.value();
  }
  const conduit::Node &elements =
      domain["topologies/" + topo_name + "/elements"];
  shape = elements["shape"].as_string();
  if(shape == "polyhedral")
  {
    polyhedral_connectivity = elements["connectivity"].value();
    polyhedral_sizes = elements["sizes"].value();
    polyhedral_offsets = elements["offsets"].value();
    num_cells = polyhedral_sizes.dtype().number_of_elements();

    const conduit::Node &subelements =
        domain["topologies/" + topo_name + "/subelements"];
    connectivity = subelements["connectivity"].value();
    sizes = subelements["sizes"].value();
    offsets = subelements["offsets"].value();
    polyhedral_shape = subelements["shape"].as_string();
    if(polyhedral_shape != "polygonal")
    {
      polyhedral_shape_size = get_num_vertices(polyhedral_shape);
    }
  }
  else if(shape == "polygonal")
  {
    connectivity = elements["connectivity"].value();
    sizes = elements["sizes"].value();
    offsets = elements["offsets"].value();
    num_cells = sizes.dtype().number_of_elements();
  }
  else
  {
    connectivity = elements["connectivity"].value();
    shape_size = get_num_vertices(shape);
    num_cells = connectivity.dtype().number_of_elements() / shape_size;
  }
}

template <typename T, size_t N>
size_t
UnstructuredTopology<T, N>::get_num_points() const
{
  // number of unique elements in connectivity
  const conduit::int32 *conn_begin = (conduit::int32 *)connectivity.data_ptr();
  const conduit::int32 *conn_end =
      conn_begin + connectivity.dtype().number_of_elements();
  // points used in the topology
  const size_t num_points = std::unordered_set<T>(conn_begin, conn_end).size();
  // points available in the coordset
  const size_t coords_size = domain["coordsets/" + coords_name + "/values"]
                                 .child(0)
                                 .dtype()
                                 .number_of_elements();
  if(num_points != coords_size)
  {
    ASCENT_ERROR("Unstructured topology '"
                 << topo_name << "' has " << coords_size
                 << " points in its associated coordset '" << coords_name
                 << "' but the connectivity "
                    "array only uses "
                 << num_points << " of them.");
  }
  return num_points;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UnstructuredTopology<T, N>::vertex_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] = coords[i][index];
  }
  return loc;
}

template <typename T, size_t N>
std::array<conduit::float64, 3>
UnstructuredTopology<T, N>::element_location(const size_t index) const
{
  std::array<conduit::float64, 3> loc{};
  size_t offset;
  size_t cur_shape_vertices;
  if(shape == "polygonal")
  {
    offset = offsets[index];
    cur_shape_vertices = sizes[index];
  }
  else if(shape == "polyhedral")
  {
    cur_shape_vertices = -1;
    ASCENT_ERROR("element_location for polyhedral shapes is not implemented.");
  }
  else
  {
    offset = index * shape_size;
    cur_shape_vertices = shape_size;
  }
  for(size_t i = 0; i < cur_shape_vertices; ++i)
  {
    const auto vert_loc = vertex_location(connectivity[offset + i]);
    for(size_t i = 0; i < N; ++i)
    {
      loc[i] += vert_loc[i];
    }
  }
  for(size_t i = 0; i < N; ++i)
  {
    loc[i] /= cur_shape_vertices;
  }
  return loc;
}

//-----------------------------------------------------------------------------
// -- topologyFactory
//-----------------------------------------------------------------------------

// make_unique is a c++14 feature
// this is not as general (e.g. doesn't work on array types)
template <typename T, typename... Args>
std::unique_ptr<T>
my_make_unique(Args &&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::unique_ptr<Topology>
topologyFactory(const std::string &topo_name, const conduit::Node &domain)
{
  const conduit::Node &n_topo = domain["topologies/" + topo_name];
  const std::string &topo_type = n_topo["type"].as_string();
  const size_t num_dims = topo_dim(topo_name, domain);
  const std::string type = detail::coord_dtype(topo_name, domain);
  if(topo_type == "uniform")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UniformTopology<conduit::float64, 1>>(topo_name,
                                                                    domain);
        break;
      case 2:
        return my_make_unique<UniformTopology<conduit::float64, 2>>(topo_name,
                                                                    domain);
        break;
      case 3:
        return my_make_unique<UniformTopology<conduit::float64, 3>>(topo_name,
                                                                    domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UniformTopology<conduit::float32, 1>>(topo_name,
                                                                    domain);
        break;
      case 2:
        return my_make_unique<UniformTopology<conduit::float32, 2>>(topo_name,
                                                                    domain);
        break;
      case 3:
        return my_make_unique<UniformTopology<conduit::float32, 3>>(topo_name,
                                                                    domain);
        break;
      }
    }
  }
  else if(topo_type == "rectilinear")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<RectilinearTopology<conduit::float64, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<RectilinearTopology<conduit::float64, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<RectilinearTopology<conduit::float64, 3>>(
            topo_name, domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<RectilinearTopology<conduit::float32, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<RectilinearTopology<conduit::float32, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<RectilinearTopology<conduit::float32, 3>>(
            topo_name, domain);
        break;
      }
    }
  }
  else if(topo_type == "structured")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<StructuredTopology<conduit::float64, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<StructuredTopology<conduit::float64, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<StructuredTopology<conduit::float64, 3>>(
            topo_name, domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<StructuredTopology<conduit::float32, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<StructuredTopology<conduit::float32, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<StructuredTopology<conduit::float32, 3>>(
            topo_name, domain);
        break;
      }
    }
  }
  else if(topo_type == "unstructured")
  {
    if(type == "double")
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UnstructuredTopology<conduit::float64, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<UnstructuredTopology<conduit::float64, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<UnstructuredTopology<conduit::float64, 3>>(
            topo_name, domain);
        break;
      }
    }
    else
    {
      switch(num_dims)
      {
      case 1:
        return my_make_unique<UnstructuredTopology<conduit::float32, 1>>(
            topo_name, domain);
        break;
      case 2:
        return my_make_unique<UnstructuredTopology<conduit::float32, 2>>(
            topo_name, domain);
        break;
      case 3:
        return my_make_unique<UnstructuredTopology<conduit::float32, 3>>(
            topo_name, domain);
        break;
      }
    }
  }
  else
  {
    ASCENT_ERROR("The Architect: Unsupported topology type '" << topo_type
                                                              << "'.");
  }
  ASCENT_ERROR("topologyFactory returning nullptr, this should never happen.");
  return nullptr;
}

int
get_num_vertices(const std::string &shape_type)
{
  int num = 0;
  if(shape_type == "tri")
  {
    num = 3;
  }
  else if(shape_type == "quad")
  {
    num = 4;
  }
  else if(shape_type == "tet")
  {
    num = 4;
  }
  else if(shape_type == "hex")
  {
    num = 8;
  }
  else if(shape_type == "point")
  {
    num = 1;
  }
  else
  {
    ASCENT_ERROR("Cannot get the number of vertices for the shape '"
                 << shape_type << "'.");
  }
  return num;
}

int
topo_dim(const std::string &topo_name, const conduit::Node &dom)
{
  if(!dom.has_path("topologies/" + topo_name))
  {
    ASCENT_ERROR("Topology '" << topo_name << "' not found in domain.");
  }

  const conduit::Node &n_topo = dom["topologies/" + topo_name];

  const std::string c_name = n_topo["coordset"].as_string();
  const conduit::Node &n_coords = dom["coordsets/" + c_name];
  const std::string c_type = n_coords["type"].as_string();

  int num_dims;
  if(c_type == "uniform")
  {
    num_dims = n_coords["dims"].number_of_children();
  }
  else if(c_type == "rectilinear" || c_type == "explicit")
  {
    num_dims = n_coords["values"].number_of_children();
  }
  else
  {
    num_dims = -1;
    ASCENT_ERROR("Unknown coordinate set type: '" << c_type << "'.");
  }
  if(num_dims <= 0 || num_dims > 3)
  {
    ASCENT_ERROR("The Architect: topology '"
                 << topo_name << "' with " << num_dims
                 << " dimensions is not supported.");
  }
  return num_dims;
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

