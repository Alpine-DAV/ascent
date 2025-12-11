#include <vtkh/filters/HistSampling.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/filters/Histogram.hpp>
#include <vtkh/Error.hpp>

#include <viskores/worklet/DispatcherMapField.h>
#include <viskores/worklet/WorkletMapField.h>

#include <viskores/worklet/DescriptiveStatistics.h>
//#include <viskores/filter/CreateResult.h>
#include <viskores/cont/ArrayHandleTransform.h>
#include <viskores/worklet/DispatcherMapField.h>
#include <iostream>
#include <algorithm>
#include <viskores/worklet/WorkletMapField.h>
#include <iostream>

namespace vtkh
{

namespace detail
{

class RandomGenerate : public viskores::worklet::WorkletMapField
{
protected:
  viskores::Int32 m_seed;
public:
  VISKORES_CONT
  RandomGenerate(viskores::Int32 seed)
   : m_seed(seed)
  {}

  typedef void ControlSignature(FieldOut);
  typedef void ExecutionSignature(WorkIndex, _1);

  VISKORES_EXEC
  void operator()(const viskores::Id &index, viskores::Float32 &value) const
  {
    const viskores::Int32 sample = static_cast<viskores::UInt32>(m_seed + index);
    viskores::Float32 y = 0.0f;
    viskores::Float32 yadd = 1.0f;
    viskores::Int32 bn = sample;
    const viskores::Int32 base = 7;
    while (bn != 0)
    {
      yadd *= 1.0f / (viskores::Float32)base;
      y += (viskores::Float32)(bn % base) * yadd;
      bn /= base;
    }

    value = y;
  }
}; //class RandomGenerate


viskores::cont::ArrayHandle<viskores::Float32>
calculate_pdf(const viskores::Int32 tot_points,
              const viskores::Int32 num_bins,
              const viskores::Float32 sample_percent,
              viskores::cont::ArrayHandle<viskores::Id> mybins)
{
  viskores::cont:: ArrayHandle <viskores::Id > bins;
  viskores::cont:: Algorithm ::Copy(mybins , bins);
  viskores::cont::ArrayHandleIndex indexArray (num_bins);
  viskores::cont::ArrayHandle<viskores::Id> indices;
  viskores::cont::Algorithm::Copy(indexArray, indices);

  viskores::cont:: ArrayHandleZip <viskores::cont:: ArrayHandle <viskores::Id >,
                               viskores::cont:: ArrayHandle <viskores::Id >>
                                 zipArray(bins, indices );

  viskores::cont::Algorithm::Sort(zipArray);

  auto binPortal = zipArray.ReadPortal();

  viskores::Float32 remainingSamples = sample_percent*tot_points;

  viskores::Float32 remainingBins = num_bins;
  std::vector<viskores::Float32> targetSamples;

  for (int i = 0; i < num_bins; ++i)
  {
    viskores::Float32 targetNeededSamples = remainingSamples / (1.0f*remainingBins);
    viskores::Float32 curCount = (viskores::Float32)binPortal.Get(i).first;
    viskores::Float32 samplesTaken;

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

  viskores::cont::ArrayHandle<viskores::Float32> acceptanceProbsVec;
  acceptanceProbsVec.Allocate(num_bins);
  auto acceptance_portal = acceptanceProbsVec.WritePortal();
  for(int i = 0; i < num_bins; ++i)
  {
    acceptance_portal.Set(i, -1.f);
  }

  viskores::Float32 sum=0.0;
  int counter=0;
  for(viskores::Float32 n : targetSamples)
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

struct LookupWorklet : public viskores::worklet::WorkletMapField
{
protected:
  viskores::Id m_num_bins;
  viskores::Float64 m_min;
  viskores::Float64 m_bin_delta;
public:
  LookupWorklet(const viskores::Id num_bins,
                const viskores::Float64 min_value,
                const viskores::Float64 bin_delta)
    : m_num_bins(num_bins),
      m_min(min_value),
      m_bin_delta(bin_delta)
  {}

  using ControlSignature = void(FieldIn, FieldOut, WholeArrayIn, FieldIn);
  using ExecutionSignature = _2(_1, _3, _4);

  template <typename TablePortal>
  VISKORES_EXEC viskores::UInt8 operator()(const viskores::Float64 &field_value,
                                   TablePortal table,
                                   const viskores::Float32 &random) const
  {
    viskores::Id bin = static_cast<viskores::Id>((field_value - m_min) / m_bin_delta);
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


void PrintStatInfo(viskores::worklet::DescriptiveStatistics::StatState<viskores::Float64> statinfo)
{

  std::cout << "   Minimum " << statinfo.Min() << std::endl;
  std::cout << "   Maximum " << statinfo.Max() << std::endl;
  std::cout << "   Mean " << statinfo.Mean() << std::endl;
  std::cout << "   Variance " << statinfo.PopulationVariance() << std::endl;
  std::cout << "   Standard Deviation " << statinfo.PopulationStddev() << std::endl;
  std::cout << "   Skewness " << statinfo.Skewness() << std::endl;
  std::cout << "   Kurtosis " << statinfo.Kurtosis() << std::endl;
  
  // Not supported by Viskores 2.1
  // std::cout << "   Median " << statinfo.median << std::endl;
  // std::cout << "   Raw Moment 1-4 [ ";
  // for (viskores::Id i = 0; i < 4; i++)
  //   std::cout << statinfo.rawMoment[i] << " ";
  // std::cout << "]" << std::endl;
  // std::cout << "   Central Moment 1-4 [ ";
  // for (viskores::Id i = 0; i < 4; i++)
  //   std::cout << statinfo.centralMoment[i] << " ";
  // std::cout << "]" << std::endl;
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

  viskores::Id numberOfBins = histogram.m_bins.GetNumberOfValues();

  bool valid_field;
  viskores::cont::Field::Association assoc = input->GetFieldAssociation(m_field_name,
                                                                    valid_field);


  viskores::Id global_num_values = histogram.totalCount();
  viskores::cont:: ArrayHandle <viskores::Id > globCounts = histogram.m_bins;

  viskores::cont::ArrayHandle <viskores::Float32 > probArray;
  probArray = detail::calculate_pdf(global_num_values, numberOfBins, m_sample_percent, globCounts);

  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Range range;
    viskores::Float64 delta;
    viskores::cont::DataSet &dom = input->GetDomain(i);

    if(!dom.HasField(m_field_name))
    {
      // We have already check to see if the field exists globally,
      // so just skip if this particular domain doesn't have the field
      continue;
    }

    viskores::cont::ArrayHandle<viskores::Float64> data;
    dom.GetField(m_field_name).GetData().AsArrayHandle(data);
 
    //auto viskores::worklet::DescriptiveStatistics::Run(data);

    //std::cout << "Statistics for CELL data:" << std::endl;
    //PrintStatInfo(statinfo);


    // start doing sampling

    viskores::Int32 tot_points = data.GetNumberOfValues();

    // use the acceptance probabilities to create a stencil buffer
    viskores::cont::ArrayHandle<viskores::Float32> randArray;

    randArray.Allocate(tot_points);

    const viskores::Int32 seed = 0;

    viskores::worklet::DispatcherMapField<detail::RandomGenerate>(seed).Invoke(randArray);



    viskores::cont::ArrayHandle <viskores::UInt8> stencilBool;
    viskores::worklet::DispatcherMapField<LookupWorklet>(LookupWorklet{numberOfBins,
                                                     histogram.m_range.Min,
                                                     histogram.m_bin_delta}).Invoke(data,
                                                                                    stencilBool,
                                                                                    probArray,
                                                                                    randArray);


    viskores::cont::ArrayHandle <viskores::Float32> output;
    viskores::cont::Algorithm ::Copy(stencilBool , output );

    if(assoc == viskores::cont::Field::Association::Points)
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

  thresher.SetFieldUpperThreshold(upper_bound);
  thresher.SetFieldLowerThreshold(lower_bound);
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
