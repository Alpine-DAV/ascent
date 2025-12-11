#include <vtkh/filters/Histogram.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/utils/viskores_array_utils.hpp>
#include <vtkh/viskores_filters/viskoresHistogram.hpp>
#include <viskores/filter/density_estimate/worklet/FieldHistogram.h>
#include <viskores/cont/PartitionedDataSet.h>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

struct HistoFunctor
{

  viskores::Range m_range;
  viskores::Id m_num_bins;

  viskores::cont::ArrayHandle<viskores::Id> m_bins;
  viskores::Float64 m_bin_delta;

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<T,S> &array)
  {
    T bin_delta;
    T min_range = static_cast<T>(m_range.Min);
    T max_range = static_cast<T>(m_range.Max);

    //TODO:Rewrite using viskores::filter::density_estimate::Histogram
    viskores::worklet::FieldHistogram worklet;
    worklet.Run(array,m_num_bins,min_range,max_range,bin_delta,m_bins);
    m_bin_delta = static_cast<viskores::Float64>(bin_delta);
  }
};

template<typename T>
void reduce(T *array, int size);

template<>
void reduce<viskores::Int32>(viskores::Int32 *array, int size)
{
#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce(MPI_IN_PLACE,array,size, MPI_INT,MPI_SUM,mpi_comm);
#else
  (void) array;
  (void) size;
#endif
}

template<>
void reduce<viskores::Int64>(viskores::Int64 *array, int size)
{
#ifdef VTKH_PARALLEL
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce(MPI_IN_PLACE,array,size, MPI_LONG_LONG,MPI_SUM,mpi_comm);
#else
  (void) array;
  (void) size;
#endif
}


} // namespace detail

Histogram::Histogram()
  : m_num_bins(256)
{

}

Histogram::~Histogram()
{

}

void
Histogram::SetRange(const viskores::Range &range)
{
  m_range = range;
}

void
Histogram::SetNumBins(const int num_bins)
{
  m_num_bins = num_bins;
}

void 
Histogram::PreExecute()
{
  Filter::PreExecute();
}


void 
Histogram::PostExecute()
{
  Filter::PostExecute();
}

void 
Histogram::DoExecute()
{
  VTKH_DATA_OPEN("histogram");
  VTKH_DATA_ADD("device", GetCurrentDevice());
  VTKH_DATA_ADD("bins", m_num_bins);
  VTKH_DATA_ADD("input_cells", this->m_input->GetNumberOfCells());
  VTKH_DATA_ADD("input_domains", this->m_input->GetNumberOfDomains());

  const int global_domains = this->m_input->GetGlobalNumberOfDomains();
  if(global_domains == 0)
  {
    throw Error("Histogram: can't run since there is no data!");
  }

  if(!this->m_input->GlobalFieldExists(m_field_name))
  {
    throw Error("Histogram: field '"+m_field_name+"' does not exist");
  }

  this->m_output = new DataSet();

  viskores::Range range;
  if(m_range.IsNonEmpty())
  {
    range = m_range;
  }
  else
  {
    viskores::cont::ArrayHandle<viskores::Range> ranges = this->m_input->GetGlobalRange(m_field_name);

    if(ranges.GetNumberOfValues() != 1)
    {
      throw Error("Histogram: field must have a single component");
    }
    range = ranges.ReadPortal().Get(0);
  }

  const int num_domains = this->m_input->GetNumberOfDomains();
  std::vector<HistogramResult> local_histograms;
  viskores::cont::PartitionedDataSet p_dataset;

  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    if(!dom.HasField(m_field_name)) continue;

    viskores::cont::Field field = dom.GetField(m_field_name);
    p_dataset.AddField(field);
  }

  viskoresHistogram hist;
  auto result = hist.Run(p_dataset, m_num_bins, range);

  std::vector<viskores::cont::DataSet> v_datasets = result.GetPartitions();
  int size = v_datasets.size();
  for(int i  = 0; i < size; i++)
  {
    this->m_output->AddDomain(v_datasets[i],i);
  }
  

  VTKH_DATA_CLOSE();
}

std::string
Histogram::GetName() const
{
  return "vtkh::Histogram";
}

//Needed for HistSampling
//Will remove once HistSampling Viskores filter is written

void
Histogram::HistogramResult::Print(std::ostream &out)
{
  auto binPortal = m_bins.ReadPortal();
  const int num_bins = m_bins.GetNumberOfValues();
  viskores::Id sum = 0;
  for (viskores::Id i = 0; i < num_bins; i++)
  {
    viskores::Float64 lo = m_range.Min + (static_cast<viskores::Float64>(i) * m_bin_delta);
    viskores::Float64 hi = lo + m_bin_delta;
    sum += binPortal.Get(i);
    out << " Bin [" << i << "] Range[" << lo
    << ", " << hi << "] = " << binPortal.Get(i)
    << "\n";
    }
  out<<"total points: "<<sum<<"\n";
}

viskores::Id
Histogram::HistogramResult::totalCount()
{
  auto binPortal = m_bins.ReadPortal();
  const int num_bins = m_bins.GetNumberOfValues();
  viskores::Id sum = 0;
  for (viskores::Id i = 0; i < num_bins; i++)
  {
    sum += binPortal.Get(i);
  }
  return sum;
}

Histogram::HistogramResult
Histogram::merge_histograms(std::vector<Histogram::HistogramResult> &histograms)
{
  Histogram::HistogramResult res;
  const int size = histograms.size();
  if(size < 1)
  {
    // we have data globally so we need to create a dummy result
    //     // and pass that off to mpi
    res.m_bins.Allocate(m_num_bins);
    res.m_range = m_range;
    res.m_bin_delta = m_range.Length() / double(m_num_bins);
    const int num_bins = res.m_bins.GetNumberOfValues();

    auto bins = res.m_bins.WritePortal();
    for(int n = 0; n < num_bins; ++n)
    {
      bins.Set(n, 0.);
    }
    return res;
  }

  res = histograms[0];
  auto bins1 = res.m_bins.WritePortal();
  const int num_bins = res.m_bins.GetNumberOfValues();
  for(int i = 1; i < size; ++i)
  {
    auto bins2 = histograms[i].m_bins.WritePortal();
    for(int n = 0; n < num_bins; ++n)
    {
      bins1.Set(n, bins1.Get(n) + bins2.Get(n));
    }
  }

  return res;
}

Histogram::HistogramResult
Histogram::Run(vtkh::DataSet &data_set, const std::string &field_name)
{
  VTKH_DATA_OPEN("histogram");
  VTKH_DATA_ADD("device", GetCurrentDevice());
  VTKH_DATA_ADD("bins", m_num_bins);
  VTKH_DATA_ADD("input_cells", data_set.GetNumberOfCells());
  VTKH_DATA_ADD("input_domains", data_set.GetNumberOfDomains());

  if(!data_set.GetGlobalNumberOfDomains())
  {
    throw Error("Histogram: can't run since there is no data!");
  }

  if(!data_set.GlobalFieldExists(field_name))
  {
    throw Error("Histogram: field '"+field_name+"' does not exist");
  }


  viskores::Range range;
  if(m_range.IsNonEmpty())
  {
    range = m_range;
  }
  else
  {
    viskores::cont::ArrayHandle<viskores::Range> ranges = data_set.GetGlobalRange(field_name);

    if(ranges.GetNumberOfValues() != 1)
    {
      throw Error("Histogram: field must have a single component");
    }
      range = ranges.ReadPortal().Get(0);
  }

  const int num_domains = data_set.GetNumberOfDomains();
  std::vector<HistogramResult> local_histograms;
  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    data_set.GetDomain(i, dom, domain_id);
    if(!dom.HasField(field_name)) continue;

    viskores::cont::Field field = dom.GetField(field_name);

    detail::HistoFunctor hist;
    hist.m_num_bins = m_num_bins;
    hist.m_range = range;

    field.GetData().ResetTypes(viskores::TypeListFieldScalar(), VISKORES_DEFAULT_STORAGE_LIST{}).CastAndCall(hist);
    HistogramResult dom_hist;
    dom_hist.m_bins = hist.m_bins;
    dom_hist.m_bin_delta = hist.m_bin_delta;
    dom_hist.m_range = range;
    local_histograms.push_back(dom_hist);
  }

  HistogramResult local = merge_histograms(local_histograms);
  viskores::Id * bin_ptr = GetVISKORESPointer(local.m_bins);
  detail::reduce(bin_ptr, m_num_bins);

  VTKH_DATA_CLOSE();
  return local;
}

} //  namespace vtkh
