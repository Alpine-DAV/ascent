
#include <vtkh/filters/UniformGrid.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/vtkm_filters/vtkmProbe.hpp>

#include <limits>

#ifdef VTKH_PARALLEL
#include <vtkh/utils/vtkm_array_utils.hpp>

#include <vtkm/thirdparty/diy/diy.h>
#include <vtkm/thirdparty/diy/mpi-cast.h>
#include <vtkm/cont/EnvironmentTracker.h>

#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

std::vector<vtkm::Id> UnionDomainIds(vtkh::DataSet *m_data_set)
{
  std::vector<vtkm::Id> domain_ids;
  const size_t num_dsets = m_data_set;
  for(size_t i = 0; i < num_dsets; ++i)
  {
    std::vector<vtkm::Id> add = m_data_sets[i]->GetDomainIds();
    domain_ids.insert(domain_ids.end(), add.begin(), add.end());
  }

  std::sort(domain_ids.begin(), domain_ids.end());
  auto last = std::unique(domain_ids.begin(), domain_ids.end());
  domain_ids.erase(last, domain_ids.end());
  return domain_ids;
}

class Offset : public vtkm::worklet::WorkletMapField
{
protected:
  vtkm::Id m_offset;

public:
  VTKM_CONT
  Offset(const vtkm::Id offset)
    : m_offset(offset)
  {
  }

  typedef void ControlSignature(FieldIn, WholeArrayInOut);
  typedef void ExecutionSignature(_1, _2);

  template<typename PortalType>
  VTKM_EXEC
  void operator()(const vtkm::Id &index, PortalType values) const
  {
    vtkm::Id value = values.Get(index);
    values.Set(index, value + m_offset);
  }
}; //class Offset

class MergeDomains
{
  std::vector<vtkm::cont::DataSet> &m_data_sets;
  std::vector<vtkm::Id> m_domain_ids; // we skip the slice field
public:
  MergeContours(std::vector<vtkm::cont::DataSet> &data_sets, std::vector<vtkm::Id> domain_ids)
    : m_data_sets(data_sets),
      m_domain_ids(domain_ids)
  {}

  ~MergeContours()
  {}


  template<typename U>
  struct CopyFunctor
  {
    vtkm::cont::ArrayHandle<vtkm::Vec<U,3>> output;
    vtkm::Id offset;

    template<typename Type, typename S>
    void operator()(vtkm::cont::ArrayHandle<Type,S> &input)
    {
      vtkm::Id copy_size = input.GetNumberOfValues();
      vtkm::Id start = 0;
      vtkm::cont::Algorithm::CopySubRange(input, start, copy_size, output, offset);
    }
  };
  template<typename T, typename S, typename U>
  void CopyCoords(vtkm::cont::UncertainArrayHandle<T,S> &input,
                  vtkm::cont::ArrayHandle<vtkm::Vec<U,3>> &output,
                  vtkm::Id offset)
  {
    CopyFunctor<U> func{output,offset};
    input.CastAndCall(func);
  }

  struct CopyField
  {
    vtkm::cont::DataSet &m_data_set;
    std::vector<vtkm::cont::DataSet> m_in_data_sets;
    vtkm::Id *m_point_offsets;
    vtkm::Id *m_cell_offsets;
    vtkm::Id  m_field_index;
    vtkm::Id  m_num_points;
    vtkm::Id  m_num_cells;

    CopyField(vtkm::cont::DataSet &data_set,
              std::vector<vtkm::cont::DataSet> in_data_sets,
              vtkm::Id *point_offsets,
              vtkm::Id *cell_offsets,
              vtkm::Id num_points,
              vtkm::Id num_cells,
              vtkm::Id field_index)
      : m_data_set(data_set),
        m_in_data_sets(in_data_sets),
        m_point_offsets(point_offsets),
        m_cell_offsets(cell_offsets),
        m_field_index(field_index),
        m_num_points(num_points),
        m_num_cells(num_cells)
    {}

    template<typename T, typename S>
    void operator()(const vtkm::cont::ArrayHandle<T,S> &vtkmNotUsed(field)) const
    {
      //check to see if this is a supported field ;
      const vtkm::cont::Field &scalar_field = m_in_data_sets[0].GetField(m_field_index);
      bool is_supported = (scalar_field.GetAssociation() == vtkm::cont::Field::Association::Points ||
                           scalar_field.GetAssociation() == vtkm::cont::Field::Association::Cells);

      if(!is_supported) return;

      bool assoc_points = scalar_field.GetAssociation() == vtkm::cont::Field::Association::Points;
      vtkm::cont::ArrayHandle<T> out;
      if(assoc_points)
      {
        out.Allocate(m_num_points);
      }
      else
      {
        out.Allocate(m_num_cells);
      }

      for(size_t i = 0; i < m_in_data_sets.size(); ++i)
      {
        const vtkm::cont::Field &f = m_in_data_sets[i].GetField(m_field_index);
        vtkm::cont::ArrayHandle<T,S> in = f.GetData().AsArrayHandle<vtkm::cont::ArrayHandle<T,S>>();
        vtkm::Id start = 0;
        vtkm::Id copy_size = in.GetNumberOfValues();
        vtkm::Id offset = assoc_points ? m_point_offsets[i] : m_cell_offsets[i];

        vtkm::cont::Algorithm::CopySubRange(in, start, copy_size, out, offset);
      }

      vtkm::cont::Field out_field(scalar_field.GetName(),
                                  scalar_field.GetAssociation(),
                                  out);
      m_data_set.AddField(out_field);

    }
  };

  vtkm::cont::DataSet MergeDomains(std::vector<vtkm::cont::DataSet> &doms)
  {
    vtkm::cont::DataSet res;

    vtkm::Id num_cells = 0;
    vtkm::Id num_points = 0;
    std::vector<vtkm::Id> cell_offsets(doms.size());
    std::vector<vtkm::Id> point_offsets(doms.size());

    for(size_t dom = 0; dom < doms.size(); ++dom)
    {
      auto cell_set = doms[dom].GetCellSet();

      // In the past, we were making assumptions that the output of contour
      // was a cell set single type. Because of difficult vtkm reasons, the output
      // of contour is now explicit cell set,but we can still assume that
      // this output will be all triangles.
      // this becomes more complicated if we want to support mixed types
      //if(!cell_set.IsType(vtkm::cont::CellSetSingleType<>())) continue;
      if(!cell_set.IsType<vtkm::cont::CellSetExplicit<>>())
      {
        std::cout<<"expected explicit cell set as the result of contour\n";

        continue;
      }

      cell_offsets[dom] = num_cells;
      num_cells += cell_set.GetNumberOfCells();

      auto coords = doms[dom].GetCoordinateSystem();
      point_offsets[dom] = num_points;
      num_points += coords.GetData().GetNumberOfValues();

    }

    const vtkm::Id conn_size = num_cells * 3;

    // calculate merged offsets for all domains
    vtkm::cont::ArrayHandle<vtkm::Id> conn;
    conn.Allocate(conn_size);

    // handle coordinate merging
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>> out_coords;
    out_coords.Allocate(num_points);
    // coordinate type that contour produces
    //using CoordsType3f = vtkm::cont::ArrayHandleVirtual<vtkm::Vec<vtkm::Float32,3>>;
    //using CoordsType3d = vtkm::cont::ArrayHandleVirtual<vtkm::Vec<vtkm::Float64,3>>;

    for(size_t dom = 0; dom < doms.size(); ++dom)
    {
      auto cell_set = doms[dom].GetCellSet();

      //if(!cell_set.IsType(vtkm::cont::CellSetSingleType<>())) continue;
      if(!cell_set.IsType<vtkm::cont::CellSetExplicit<>>())
      {
        std::cout<<"expected explicit cell set as the result of contour\n";
        continue;
      }

      // grab the connectivity and copy it into the larger array
      //vtkm::cont::CellSetSingleType<> single_type = cell_set.Cast<vtkm::cont::CellSetSingleType<>>();
      vtkm::cont::CellSetExplicit<> single_type =
        cell_set.AsCellSet<vtkm::cont::CellSetExplicit<>>();
      const vtkm::cont::ArrayHandle<vtkm::Id> dconn = single_type.GetConnectivityArray(
        vtkm::TopologyElementTagCell(),
        vtkm::TopologyElementTagPoint());

      vtkm::Id copy_size = dconn.GetNumberOfValues();
      vtkm::Id start = 0;

      vtkm::cont::Algorithm::CopySubRange(dconn, start, copy_size, conn, cell_offsets[dom]*3);
      // now we offset the connectiviy we just copied in so we references the
      // correct points
      if(cell_offsets[dom] != 0)
      {
        vtkm::cont::ArrayHandleCounting<vtkm::Id> indexes(cell_offsets[dom]*3, 1, copy_size);
        vtkm::worklet::DispatcherMapField<detail::Offset>(detail::Offset(point_offsets[dom]))
          .Invoke(indexes, conn);
      }

      // merge coodinates
      auto coords = doms[dom].GetCoordinateSystem().GetData();
      this->CopyCoords(coords, out_coords, point_offsets[dom]);
      //if(coords == CoordsType3f())
      //{
      //  CoordsType3f in = coords.Cast<CoordsType3f>();
      //  this->CopyCoords(in, out_coords, point_offsets[dom]);
      //}
      //if(coords.IsType<CoordsType3d>())
      //{
      //  CoordsType3d in = coords.Cast<CoordsType3d>();
      //  this->CopyCoords(in, out_coords, point_offsets[dom]);
      //}
      //else
      //{
      //  throw Error("Merge contour: unknown coordinate type");
      //}

    } // for each domain


    vtkm::cont::CellSetSingleType<> cellSet;
    cellSet.Fill(num_points, vtkm::CELL_SHAPE_TRIANGLE, 3, conn);
    res.SetCellSet(cellSet);

    res.AddCoordinateSystem(vtkm::cont::CoordinateSystem("coords", out_coords));

    // handle fields, they are all the same since they came from the same data set
    const int num_fields = doms[0].GetNumberOfFields();

    for(int f = 0; f < num_fields; ++f)
    {
      const vtkm::cont::Field &field = doms[0].GetField(f);

      if(field.GetName() == m_skip_field) continue;

      CopyField copier(res,
                       doms,
                       &point_offsets[0],
                       &cell_offsets[0],
                       num_points,
                       num_cells,
                       f);

      auto full = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
      full.CastAndCall(copier);
    }
    return res;
  }

  vtkm::cont::DataSet Merge()
  {
    vtkm::cont::DataSet *res = new vtkm::cont::DataSet();
    for(size_t dom = 0; dom < domain_ids.size(); ++dom)
    {
      // gather domain
      std::vector<vtkm::cont::DataSet> doms;
      vtkm::Id domain_id = domain_ids[dom];
      for(size_t i = 0; i < m_data_sets.size(); ++i)
      {
        if(m_data_sets[i]->HasDomainId(domain_id))
        {
          doms.push_back(m_data_sets[i]->GetDomainById(domain_id));
        }

      } // for each data set
      res->AddDomain(this->MergeDomains(doms), domain_id);
    } // for each domain id

    return res;
  }

};// end MergeDomains

#ifdef VTKH_PARALLEL
class GlobalReduceFields
{
  vtkm::cont::DataSet &m_dataset;
  vtkm::Float64       m_invalid_value;

public:
  GlobalReduceFields(vtkm::cont::DataSet &dataset, vtkm::Float64 invalid_value)
    : m_dataset(dataset),
      m_invalid_value(invalid_value)
  {}
  ~GlobalReduceFields()
  {}

  vtkm::cont::DataSet Reduce()
  {
    vtkm::cont::DataSet res;
    res.CopyStructure(m_dataset);
    int num_fields = m_dataset.GetNumberOfFields();
    for(int i = 0; i < num_fields; ++i)
    { 
      vtkm::cont::Field field = m_dataset.GetField(i);
      ReduceField r_field(field, m_dataset, m_invalid_value);
      vtkm::cont::Field res_field = r_field.reduce();
      //auto reduce_field = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
      //reduce_field.CastAndCall(r_field);
      res.AddField(res_field);
    } 

    return res;
  }

  struct ReduceField
  {
    vtkm::cont::Field &m_input_field;
    vtkm::cont::DataSet &m_data_set;
    vtkm::Float64 &m_invalid_value;
  
    ReduceField(vtkm::cont::Field &input_field, vtkm::cont::DataSet &data_set, vtkm::Float64 &invalid_value)
      : m_input_field(input_field),
        m_data_set(data_set),
	m_invalid_value(invalid_value)
    {}
  
    vtkm::cont::Field 
    reduce()
    {
      if(m_input_field.GetName() == "mask")
        return m_input_field;
      vtkm::cont::Field res;
      MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
      vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
      int par_rank;
      int par_size;
      MPI_Comm_rank(mpi_comm, &par_rank);
      MPI_Comm_size(mpi_comm, &par_size);  

      vtkm::cont::UnknownArrayHandle uah_field = m_input_field.GetData();

      using scalarI = vtkm::cont::ArrayHandle<vtkm::Int32>;
      using scalarF = vtkm::cont::ArrayHandle<vtkm::Float32>;
      using scalarD = vtkm::cont::ArrayHandle<vtkm::Float64>;
      using vec3_32  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>; 
      using vec3_64  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>; 
      using vec2_32  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>; 
      using vec2_64  = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>; 

      vtkm::cont::ArrayHandle<vtkm::Float32> ah_mask;
      m_data_set.GetField("mask").GetData().AsArrayHandle(ah_mask);
      auto mask_portal = ah_mask.ReadPortal();
      int num_points = mask_portal.GetNumberOfValues();

      //create invalid mask where true == invalid
      std::vector<int> l_mask(num_points,1);
      std::vector<int> g_mask(num_points,1);
      std::vector<int> g_valid(num_points,0);
      std::vector<int> l_valid(num_points,0);
      for(int j = 0; j < num_points; ++j)
      {
        l_mask[j] = l_mask[j] && mask_portal.Get(j);
	if(l_mask[j] == 0)
	  l_valid[j] = 1;
      }
      
      MPI_Reduce(l_mask.data(), g_mask.data(), num_points, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
      MPI_Reduce(l_valid.data(), g_valid.data(), num_points, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

      ////send to root process
      if(uah_field.CanConvert<scalarI>())
      {
        //loop through field, zero out invalid value
	scalarI ah_field = m_input_field.GetData().AsArrayHandle<scalarI>();
        int * local_field = GetVTKMPointer(ah_field);
        std::vector<int> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
            ah_field.WritePortal().Set(i,0);
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_field[i] = global_field[i]/g_valid[i];
	    }
	  }
	  scalarI ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end scalarI
      else if(uah_field.CanConvert<scalarF>())
      {
        //loop through field, zero out invalid value
	scalarF ah_field = m_input_field.GetData().AsArrayHandle<scalarF>();
        float * local_field = GetVTKMPointer(ah_field);
        std::vector<float> global_field(num_points,0);

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
            ah_field.WritePortal().Set(i,0);
        }

        MPI_Reduce(local_field, global_field.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_field[i] = global_field[i]/g_valid[i];
	    }
	  }
	  scalarF ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end scalarF
      else if(uah_field.CanConvert<scalarD>())
      {
	scalarD ah_field = uah_field.AsArrayHandle<scalarD>();
        //loop through field, zero out invalid value
        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1.0)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
        }
        double * local_field = GetVTKMPointer(ah_field);
        std::vector<double> global_field(num_points,0);
        MPI_Reduce(local_field, global_field.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(par_rank == 0)
	{
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_field[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_field[i] = global_field[i]/g_valid[i];
	    }
	  }
	  scalarD ah_out = vtkm::cont::make_ArrayHandle(global_field.data(),num_points,vtkm::CopyFlag::On);
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      } //end scalarD
      else if(uah_field.CanConvert<vec2_32>())
      {
        //loop through field, zero out invalid value
	vec2_32 ah_field = m_input_field.GetData().AsArrayHandle<vec2_32>();
        std::vector<float> local_x_points(num_points,0);
        std::vector<float> local_y_points(num_points,0);
        std::vector<float> global_x_points(num_points,0);
        std::vector<float> global_y_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
	  vec2_32 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,2>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
	    }
            vtkm::Vec<vtkm::Float32,2> points_vec = vtkm::make_Vec(global_x_points[i],global_y_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec2_32
      else if(uah_field.CanConvert<vec2_64>())
      {
        //loop through field, zero out invalid value
	vec2_64 ah_field = m_input_field.GetData().AsArrayHandle<vec2_64>();
        std::vector<double> local_x_points(num_points,0);
        std::vector<double> local_y_points(num_points,0);
        std::vector<double> global_x_points(num_points,0);
        std::vector<double> global_y_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
	  vec2_64 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,2>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
	    }
            vtkm::Vec<vtkm::Float64,2> points_vec = vtkm::make_Vec(global_x_points[i],global_y_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec2_64
      else if(uah_field.CanConvert<vec3_32>())
      {
        //loop through field, zero out invalid value
	vec3_32 ah_field = m_input_field.GetData().AsArrayHandle<vec3_32>();
        std::vector<float> local_x_points(num_points,0);
        std::vector<float> local_y_points(num_points,0);
        std::vector<float> local_z_points(num_points,0);
        std::vector<float> global_x_points(num_points,0);
        std::vector<float> global_y_points(num_points,0);
        std::vector<float> global_z_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
          local_z_points[i] = ah_field.ReadPortal().Get(i)[2];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), num_points, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
	  vec3_32 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
              global_z_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
              global_z_points[i] = global_z_points[i]/g_valid[i];
	    }

            vtkm::Vec<vtkm::Float32,3> points_vec = vtkm::make_Vec(global_x_points[i],
                                                                   global_y_points[i],
                                                                   global_z_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec3_32
      else if(uah_field.CanConvert<vec3_64>())
      {
        //loop through field, zero out invalid value
	vec3_64 ah_field = m_input_field.GetData().AsArrayHandle<vec3_64>();
        std::vector<double> local_x_points(num_points,0);
        std::vector<double> local_y_points(num_points,0);
        std::vector<double> local_z_points(num_points,0);
        std::vector<double> global_x_points(num_points,0);
        std::vector<double> global_y_points(num_points,0);
        std::vector<double> global_z_points(num_points,0);

	//std::cerr <<  	ah_field.ReadPortal().Get(i) << ": " << ah_field.ReadPortal().Get(i)[0] << " " << ah_field.ReadPortal().Get(i)[1] << " | ";

        for(int i = 0; i < num_points; ++i)
        {
          if(l_mask[i] == 1)
	  {
            ah_field.WritePortal().Set(i,0);
	  }
          local_x_points[i] = ah_field.ReadPortal().Get(i)[0];
          local_y_points[i] = ah_field.ReadPortal().Get(i)[1];
          local_z_points[i] = ah_field.ReadPortal().Get(i)[2];
        }

        MPI_Reduce(local_x_points.data(), global_x_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_y_points.data(), global_y_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_z_points.data(), global_z_points.data(), num_points, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(par_rank == 0)
	{
          vec3_64 ah_out = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>();
          ah_out.Allocate(num_points);
          for(int i = 0; i < num_points; ++i)
	  {
            if(g_mask[i] == 1)
	    {
              global_x_points[i] = m_invalid_value;
              global_y_points[i] = m_invalid_value;
              global_z_points[i] = m_invalid_value;
	    }
	    if(g_valid[i] > 1)
	    {
              global_x_points[i] = global_x_points[i]/g_valid[i];
              global_y_points[i] = global_y_points[i]/g_valid[i];
              global_z_points[i] = global_z_points[i]/g_valid[i];
	    }

            vtkm::Vec<vtkm::Float64,3> points_vec = vtkm::make_Vec(global_x_points[i],
                                                                   global_y_points[i],
                                                                   global_z_points[i]);
            ah_out.WritePortal().Set(i,points_vec);
	  }
	  vtkm::cont::Field out_field(m_input_field.GetName(),
			              m_input_field.GetAssociation(),
				      ah_out);
				      
	  res = out_field;
	}
	else
	  res = m_input_field;
      }//end vec3_64
      else
        return m_input_field;

      return res;
  
    }
  }; //struct reduceFields

};//class globalReduceFields
#endif

} //namespace detail

UniformGrid::UniformGrid()
	: m_invalid_value(std::numeric_limits<double>::min())
{

}

UniformGrid::~UniformGrid()
{

}

void
UniformGrid::PreExecute()
{
  Filter::PreExecute();
}

void
UniformGrid::DoExecute()
{
#ifdef VTKH_PARALLEL
  // Setup VTK-h and VTK-m comm.
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  vtkm::cont::EnvironmentTracker::SetCommunicator(vtkmdiy::mpi::communicator(vtkmdiy::mpi::make_DIY_MPI_Comm(mpi_comm)));
  int par_rank;
  int par_size;
  MPI_Comm_rank(mpi_comm, &par_rank);
  MPI_Comm_size(mpi_comm, &par_size);  
#endif
  this->m_output = new DataSet();

  std::vector<vtkm::Id> domain_ids = UnionDomainIds(this-m_input);
  const int num_domains = domain_ids.size();
  this->m_input->AddConstantPointField(0.0, "mask");
  std::vector<vtkm::cont::DataSet> sampled_doms;
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::cont::DataSet dom;
    
    if(this->m_input->HasDomainId(domain_ids[i])
    {
      dom = this->m_input->GetDomainById(domain_ids[i]);

      vtkh::vtkmProbe probe;
      probe.dims(m_dims);
      probe.origin(m_origin);
      probe.spacing(m_spacing);
      probe.invalidValue(m_invalid_value);
      auto dataset = probe.Run(dom);
      sampled_doms.push_back(dataset);
    }
  }
  //reduce across domains local domain
  //ie merge domains togeter
  detail::MergeDomains merge_doms(sampled_doms, domain_ids);
  auto merged = merge_doms.Merge();
  
  //take uniform sampled grid and reduce to root process
#ifdef VTKH_PARALLEL
  vtkh::detail::GlobalReduceFields g_reducefields(merged,m_invalid_value);
  auto output = g_reducefields.Reduce();
  //auto full = field.GetData().ResetTypes(vtkm::TypeListCommon(),VTKM_DEFAULT_STORAGE_LIST{});
  //full.CastAndCall(g_reducefields);
  this->m_output->AddDomain(output, domain_id);
#else
  this->m_output->AddDomain(merged, domain_id);
#endif

}

void
UniformGrid::PostExecute()
{
  Filter::PostExecute();
}

std::string
UniformGrid::GetName() const
{
  return "vtkh::UniformGrid";
}

void
UniformGrid::Dims(const Vec3f dims)
{
  m_dims = dims;
}

void
UniformGrid::Origin(const Vec3f origin)
{
  m_origin = origin;
}

void
UniformGrid::Spacing(const Vec3f spacing)
{
  m_spacing = spacing;
}

void
UniformGrid::InvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

} // namespace vtkh
