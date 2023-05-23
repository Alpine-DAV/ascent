#include <vtkh/filters/HistSampling.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/filters/Histogram.hpp>
#include <vtkh/Error.hpp>

#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>

#include <vtkm/worklet/FieldStatistics.h>
//#include <vtkm/filter/CreateResult.h>
#include <vtkm/cont/ArrayHandleTransform.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <iostream>
#include <algorithm>
#include <vtkm/worklet/WorkletMapField.h>
#include <iostream>

namespace vtkh
{

namespace detail
{

class RandomGenerate : public vtkm::worklet::WorkletMapField
{
protected:
  vtkm::Int32 m_seed;
public:
  VTKM_CONT
  RandomGenerate(vtkm::Int32 seed)
   : m_seed(seed)
  {}

  typedef void ControlSignature(FieldOut);
  typedef void ExecutionSignature(WorkIndex, _1);

  VTKM_EXEC
  void operator()(const vtkm::Id &index, vtkm::Float32 &value) const
  {
    const vtkm::Int32 sample = static_cast<vtkm::UInt32>(m_seed + index);
    vtkm::Float32 y = 0.0f;
    vtkm::Float32 yadd = 1.0f;
    vtkm::Int32 bn = sample;
    const vtkm::Int32 base = 7;
    while (bn != 0)
    {
      yadd *= 1.0f / (vtkm::Float32)base;
      y += (vtkm::Float32)(bn % base) * yadd;
      bn /= base;
    }

    value = y;
  }
}; //class RandomGenerate


vtkm::cont::ArrayHandle<vtkm::Float32>
calculate_pdf(const vtkm::Int32 tot_points,
              const vtkm::Int32 num_bins,
              const vtkm::Float32 sample_percent,
              vtkm::cont::ArrayHandle<vtkm::Id> mybins)
{
  vtkm::cont:: ArrayHandle <vtkm::Id > bins;
  vtkm::cont:: Algorithm ::Copy(mybins , bins);
  vtkm::cont::ArrayHandleIndex indexArray (num_bins);
  vtkm::cont::ArrayHandle<vtkm::Id> indices;
  vtkm::cont::Algorithm::Copy(indexArray, indices);

  vtkm::cont:: ArrayHandleZip <vtkm::cont:: ArrayHandle <vtkm::Id >,
                               vtkm::cont:: ArrayHandle <vtkm::Id >>
                                 zipArray(bins, indices );

  vtkm::cont::Algorithm::Sort(zipArray);

  auto binPortal = zipArray.ReadPortal();

  vtkm::Float32 remainingSamples = sample_percent*tot_points;

  vtkm::Float32 remainingBins = num_bins;
  std::vector<vtkm::Float32> targetSamples;

  for (int i = 0; i < num_bins; ++i)
  {
    vtkm::Float32 targetNeededSamples = remainingSamples / (1.0f*remainingBins);
    vtkm::Float32 curCount = (vtkm::Float32)binPortal.Get(i).first;
    vtkm::Float32 samplesTaken;

    if(curCount < targetNeededSamples)
    {
      samplesTaken = curCount;
    }
    else // for speed up, this else loop can be used to set the rest of the samples
    {
      samplesTaken = targetNeededSamples;
    }
    targetSamples.push_back(samplesTaken);
    remainingBins = remainingBins-1;
    remainingSamples = remainingSamples - samplesTaken;
  }

  vtkm::cont::ArrayHandle<vtkm::Float32> acceptanceProbsVec;
  acceptanceProbsVec.Allocate(num_bins);
  auto acceptance_portal = acceptanceProbsVec.WritePortal();
  for(int i = 0; i < num_bins; ++i)
  {
    acceptance_portal.Set(i, -1.f);
  }

  vtkm::Float32 sum=0.0;
  int counter=0;
  for(vtkm::Float32 n : targetSamples)
  {
    acceptance_portal.Set(binPortal.Get(counter).second,n/binPortal.Get(counter).first);
    if (binPortal.Get(counter).first < 0.00000000000001f)
    {
    	acceptance_portal.Set(binPortal.Get(counter).second,0.0);
    }
    else
    {
      acceptance_portal.Set(binPortal.Get(counter).second,n/binPortal.Get(counter).first);
    }
    sum+=n;
    counter++;

  }
  counter = 0;

  return acceptanceProbsVec;
}


}

HistSampling::HistSampling()
  : m_sample_percent(0.1f),
    m_num_bins(128)
{

}

HistSampling::~HistSampling()
{

}

void
HistSampling::SetSamplingPercent(const float percent)
{
  if(percent <= 0.f || percent > 1.f)
  {
    throw Error("HistSampling: sampling percent must be in the range (0,1]");
  }
  m_sample_percent = percent;
}

void
HistSampling::SetNumBins(const int num_bins)
{
  if(num_bins <= 0)
  {
    throw Error("HistSampling: num_bins must be positive");
  }
  m_num_bins = num_bins;
}

void
HistSampling::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
HistSampling::SetGhostField(const std::string &field_name)
{
  m_ghost_field = field_name;
}

void HistSampling::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void HistSampling::PostExecute()
{
  Filter::PostExecute();
}

std::string
HistSampling::GetField() const
{
  return m_field_name;
}

struct LookupWorklet : public vtkm::worklet::WorkletMapField
{
protected:
  vtkm::Id m_num_bins;
  vtkm::Float64 m_min;
  vtkm::Float64 m_bin_delta;
public:
  LookupWorklet(const vtkm::Id num_bins,
                const vtkm::Float64 min_value,
                const vtkm::Float64 bin_delta)
    : m_num_bins(num_bins),
      m_min(min_value),
      m_bin_delta(bin_delta)
  {}

  using ControlSignature = void(FieldIn, FieldOut, WholeArrayIn, FieldIn);
  using ExecutionSignature = _2(_1, _3, _4);

  template <typename TablePortal>
  VTKM_EXEC vtkm::UInt8 operator()(const vtkm::Float64 &field_value,
                                   TablePortal table,
                                   const vtkm::Float32 &random) const
  {
    vtkm::Id bin = static_cast<vtkm::Id>((field_value - m_min) / m_bin_delta);
    if(bin < 0)
    {
      bin = 0;
    }
    if(bin >= m_num_bins)
    {
      bin = m_num_bins - 1;
    }

    return random < table.Get(bin);
  }
};


void PrintStatInfo(vtkm::worklet::FieldStatistics<vtkm::Float64>::StatInfo statinfo)
{
  std::cout << "   Median " << statinfo.median << std::endl;
  std::cout << "   Minimum " << statinfo.minimum << std::endl;
  std::cout << "   Maximum " << statinfo.maximum << std::endl;
  std::cout << "   Mean " << statinfo.mean << std::endl;
  std::cout << "   Variance " << statinfo.variance << std::endl;
  std::cout << "   Standard Deviation " << statinfo.stddev << std::endl;
  std::cout << "   Skewness " << statinfo.skewness << std::endl;
  std::cout << "   Kurtosis " << statinfo.kurtosis << std::endl;
  std::cout << "   Raw Moment 1-4 [ ";
  for (vtkm::Id i = 0; i < 4; i++)
    std::cout << statinfo.rawMoment[i] << " ";
  std::cout << "]" << std::endl;
  std::cout << "   Central Moment 1-4 [ ";
  for (vtkm::Id i = 0; i < 4; i++)
    std::cout << statinfo.centralMoment[i] << " ";
  std::cout << "]" << std::endl;
}

void HistSampling::DoExecute()
{

  vtkh::DataSet *input = this->m_input;
  bool has_ghosts = m_ghost_field != "";

  if(has_ghosts)
  {
    vtkh::GhostStripper stripper;

    stripper.SetInput(this->m_input);
    stripper.SetField(m_ghost_field);
    stripper.SetMinValue(0);
    stripper.SetMaxValue(0);
    stripper.Update();
    input = stripper.GetOutput();
  }

  const int num_domains = input->GetNumberOfDomains();

  Histogram histogrammer;
  histogrammer.SetNumBins(m_num_bins);
  Histogram::HistogramResult histogram = histogrammer.Run(*input,m_field_name);
  //histogram.Print(std::cout);

  vtkm::Id numberOfBins = histogram.m_bins.GetNumberOfValues();

  bool valid_field;
  vtkm::cont::Field::Association assoc = input->GetFieldAssociation(m_field_name,
                                                                    valid_field);


  vtkm::Id global_num_values = histogram.totalCount();
  vtkm::cont:: ArrayHandle <vtkm::Id > globCounts = histogram.m_bins;

  vtkm::cont::ArrayHandle <vtkm::Float32 > probArray;
  probArray = detail::calculate_pdf(global_num_values, numberOfBins, m_sample_percent, globCounts);

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Range range;
    vtkm::Float64 delta;
    vtkm::cont::DataSet &dom = input->GetDomain(i);

    if(!dom.HasField(m_field_name))
    {
      // We have already check to see if the field exists globally,
      // so just skip if this particular domain doesn't have the field
      continue;
    }

    vtkm::cont::ArrayHandle<vtkm::Float64> data;
    dom.GetField(m_field_name).GetData().AsArrayHandle(data);


    //vtkm::worklet::FieldStatistics<vtkm::Float64>::StatInfo statinfo;
    //vtkm::worklet::FieldStatistics<vtkm::Float64>().Run(data, statinfo);

    //std::cout << "Statistics for CELL data:" << std::endl;
    //PrintStatInfo(statinfo);


    // start doing sampling

    vtkm::Int32 tot_points = data.GetNumberOfValues();

    // use the acceptance probabilities to create a stencil buffer
    vtkm::cont::ArrayHandle<vtkm::Float32> randArray;

    randArray.Allocate(tot_points);

    const vtkm::Int32 seed = 0;

    vtkm::worklet::DispatcherMapField<detail::RandomGenerate>(seed).Invoke(randArray);



    vtkm::cont::ArrayHandle <vtkm::UInt8> stencilBool;
    vtkm::worklet::DispatcherMapField<LookupWorklet>(LookupWorklet{numberOfBins,
                                                     histogram.m_range.Min,
                                                     histogram.m_bin_delta}).Invoke(data,
                                                                                    stencilBool,
                                                                                    probArray,
                                                                                    randArray);


    vtkm::cont::ArrayHandle <vtkm::Float32> output;
    vtkm::cont::Algorithm ::Copy(stencilBool , output );

    if(assoc == vtkm::cont::Field::Association::Points)
    {
      dom.AddPointField("valSampled", output);
    }
    else
    {
      dom.AddCellField("valSampled", output);
    }
  }

  vtkh::Threshold thresher;
  thresher.SetInput(input);
  thresher.SetField("valSampled");

  double upper_bound = 1.;
  double lower_bound = 1.;

  thresher.SetUpperThreshold(upper_bound);
  thresher.SetLowerThreshold(lower_bound);
  thresher.Update();
  this->m_output = thresher.GetOutput();

  if(has_ghosts)
  {
    delete input;
  }
}

std::string
HistSampling::GetName() const
{
  return "vtkh::HistSampling";
}

} //  namespace vtkh
