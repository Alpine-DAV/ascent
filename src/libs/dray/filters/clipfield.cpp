#include <dray/filters/clipfield.hpp>

#include <dray/dispatcher.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{


/*

a hex will make 6 tets. The number of edges is:

12 from the original hex
6 to divide each face
1 internal diagonal

Each cell can make 19 edges.


int cell_edge_count[ncells] = 0;
int cell_edge_start[ncells][19];
int cell_edge_end[ncells][19];

[parallel] for each hex cell:
    // look up the distances for the hex corners.
    dist[8] = ...

    for each tet in hex
        tetdist[] = the dist values that matter for this tet.
        for each edge in tet
            if edge appears in solid case
                edgecount = cell_edge_count[cellid]++;
                cell_edge_start[cellid][edgecount] = edge.start;
                cell_edge_end[cellid][edgecount] = edge.end;
                cell_edge_blend[cellid][edgecount] = t;
            
// We will have determined all of the edges for all of the cells
// at this point. The cell_edge arrays will contain some duplicates.
int lowest_mask[possible_edges] = 0
int lowest_index[possible_edges] = [0,1,2,3,4,...]

[parallel] foreach possible_edge:
    edge = determine_from_possible_edge_index()
    index = -1;
    foreach ce in cell_edge:
        if edge == ce:
            if index == -1:
                index = e
                lowest_mask[edgeindex] = 1
                
                break

// Can we do the exclusive scan trick to get the indices that we'll pull
// from the cell_edge...


*/

// Applies a clip field operation on a DataSet.
struct ClipFieldFunctor
{
  // Keep a handle to the original dataset because we need it to be able to
  // access the other fields.
  DataSet m_input;

  // Output dataset produced by the functor.
  DataSet m_output;

  // ClipField attributes.
  Float m_clip_value;
  std::string   m_field_name;
  bool m_invert;

  ClipFieldFunctor(DataSet &input, Float value, const std::string &field_name,
     bool invert) : m_input(input), m_output(), m_clip_value(value),
      m_field_name(field_name), m_invert(invert)
  {
  }

  // Execute the filter for the input mesh across all possible field types.
  void execute()
  {
    // This iterates over the product of possible mesh and scalar field types
    // to call the operator() function that can operate on concrete types.
    Field *field = m_input.field(m_field_name);
    if(field != nullptr && field->components() == 1)
    {
      dispatch(field, *this);
    }
  }

  // This method gets invoked by dispatch, which will have converted the field
  // into a concrete derived type so this method is able to call methods on
  // the derived type with no virtual calls.
  template<typename ScalarField>
  void operator()(ScalarField &field)
  {
    DRAY_LOG_OPEN("mesh_threshold");

    // If the field range contains the clip value then we have to process it.
    auto range = field.range();
    if(!m_invert && range[0].max() < m_clip_value)
    {
      // The field values are all less than the clip value. Keep intact.
      m_output = m_input;
    }
    else if(m_invert && range[0].min() > m_clip_value)
    {
      // The field values are all more than the clip value - but we're
      // inverting so we want that. Keep intact.
      m_output = m_input;
    }
    else if(range[0].contains(m_clip_value))
    {
      // The range contains the clip value so figure out which parts we'll keep.
#if 0
      // Figure out which elements to keep based on the input field.
      Array<int32> elem_mask;
      determine_elements_to_keep(field, m_range, m_return_all_in_range, elem_mask);

      std::cout << "elem_mask={";
      for(size_t i = 0; i < elem_mask.size(); i++)
          std::cout << ", " << elem_mask.get_value(i);
      std::cout << "}" << std::endl;

      // Use the element mask to subset the data.
      dray::Subset subset;
      m_output = subset.execute(m_input, elem_mask);
#endif
    }
    else
    {
       
    }

    DRAY_LOG_CLOSE();
  }
};

}//namespace detail

ClipField::ClipField() : m_clip_value(0.), m_field_name(), m_invert(false)
{
}

ClipField::~ClipField()
{
}

void
ClipField::set_clip_value(Float value)
{
  m_clip_value = value;
}

void
ClipField::set_field(const std::string &field)
{
  m_field_name = field;
}

void
ClipField::set_invert_clip(bool value)
{
  m_invert = value;
}

Float
ClipField::clip_value() const
{
  return m_clip_value;
}

const std::string &
ClipField::field() const
{
  return m_field_name;
}

bool
ClipField::invert() const
{
  return m_invert;
}

Collection
ClipField::execute(Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dom = collection.domain(i);
    detail::ClipFieldFunctor func(dom, m_clip_value, m_field_name, m_invert);
    func.execute();
    res.add_domain(func.m_output);
  }
  return res;
}

}//namespace dray
