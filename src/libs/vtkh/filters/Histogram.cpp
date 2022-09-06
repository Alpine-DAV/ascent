#include <vtkh/filters/Histogram.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/utils/vtkm_array_utils.hpp>

//TODO: Header for new Histogram filter
//#include <vtkm/filter/density_estimate/Histogram.h>
#include <vtkm/filter/density_estimate/worklet/FieldHistogram.h>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

struct HistoFunctor
{

  vtkm::Range m_range;
  vtkm::Id m_num_bins;

  vtkm::cont::ArrayHandle<vtkm::Id> m_bins;
  vtkm::Float64 m_bin_delta;

  template<typename T, typename S>
  void operator()(const vtkm::cont::ArrayHandle<T,S> &array)
  {
    T bin_delta;
    T min_range = static_cast<T>(m_range.Min);
    T max_range = static_cast<T>(m_range.Max);

    //TODO:Rewrite using vtkm::filter::density_estimate::Histogram
    vtkm::worklet::FieldHistogram worklet;
    worklet.Run(array,m_num_bins,min_range,max_range,bin_delta,m_bins);
    m_bin_delta = static_cast<vtkm::Float64>(bin_delta);
  }
};

template<typename T>
void reduce(T *array, int size);

template<>
void reduce<vtkm::Int32>(vtkm::Int32 *array, int size)
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
void reduce<vtkm::Int64>(vtkm::Int64 *array, int size)
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
Histogram::SetRange(const vtkm::Range &range)
{
  m_range = range;
}

void
Histogram::SetNumBins(const int num_bins)
{
  m_num_bins = num_bins;
}

Histogram::HistogramResult
Histogram::merge_histograms(std::vector<Histogram::HistogramResult> &histograms)
{
  Histogram::HistogramResult res;
  const int size = histograms.size();
  if(size < 1)
  {
    // we have data globally so we need to create a dummy result
    // and pass that off to mpi
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


  vtkm::Range range;
  if(m_range.IsNonEmpty())
  {
    range = m_range;
  }
  else
  {
    vtkm::cont::ArrayHandle<vtkm::Range> ranges = data_set.GetGlobalRange(field_name);

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
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    data_set.GetDomain(i, dom, domain_id);
    if(!dom.HasField(field_name)) continue;

    vtkm::cont::Field field = dom.GetField(field_name);

    detail::HistoFunctor hist;
    hist.m_num_bins = m_num_bins;
    hist.m_range = range;

    field.GetData().ResetTypes(vtkm::TypeListFieldScalar(), VTKM_DEFAULT_STORAGE_LIST{}).CastAndCall(hist);
    HistogramResult dom_hist;
    dom_hist.m_bins = hist.m_bins;
    dom_hist.m_bin_delta = hist.m_bin_delta;
    dom_hist.m_range = range;
    local_histograms.push_back(dom_hist);
  }

  HistogramResult local = merge_histograms(local_histograms);
  vtkm::Id * bin_ptr = GetVTKMPointer(local.m_bins);
  detail::reduce(bin_ptr, m_num_bins);

  VTKH_DATA_CLOSE();
  return local;
}

void
Histogram::HistogramResult::Print(std::ostream &out)
{
  auto binPortal = m_bins.ReadPortal();
  const int num_bins = m_bins.GetNumberOfValues();
  vtkm::Id sum = 0;
  for (vtkm::Id i = 0; i < num_bins; i++)
  {
    vtkm::Float64 lo = m_range.Min + (static_cast<vtkm::Float64>(i) * m_bin_delta);
    vtkm::Float64 hi = lo + m_bin_delta;
    sum += binPortal.Get(i);
    out << " Bin [" << i << "] Range[" << lo
        << ", " << hi << "] = " << binPortal.Get(i)
        << "\n";
  }
  out<<"total points: "<<sum<<"\n";
}

vtkm::Id
Histogram::HistogramResult::totalCount()
{
  auto binPortal = m_bins.ReadPortal();
  const int num_bins = m_bins.GetNumberOfValues();
  vtkm::Id sum = 0;
  for (vtkm::Id i = 0; i < num_bins; i++)
  {
    sum += binPortal.Get(i);
  }
  return sum;
}

} //  namespace vtkh
