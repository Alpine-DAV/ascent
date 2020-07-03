//
// Created by Sergei Shudler on 2020-06-09.
//

#ifndef ASCENT_ASCENT_RUNTIME_BABELFLOW_VOL_UTILS_H
#define ASCENT_ASCENT_RUNTIME_BABELFLOW_VOL_UTILS_H


#include "BabelFlow/TypeDefinitions.h"
#include "BabelFlow/mpi/Controller.h"
#include "BabelFlow/reduce/BinarySwap.h"
#include "BabelFlow/reduce/BinarySwapTaskMap.h"
#include "BabelFlow/reduce/KWayReduction.h"
#include "BabelFlow/reduce/KWayReductionTaskMap.h"
#include "BabelFlow/reduce/RadixKExchange.h"
#include "BabelFlow/reduce/RadixKExchangeTaskMap.h"
#include "BabelFlow/PreProcessInputTaskGraph.hpp"
#include "BabelFlow/ModTaskMap.hpp"

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin bflow_volume:: --
//-----------------------------------------------------------------------------
namespace bflow_volume
{

//-----------------------------------------------------------------------------

class BabelGraphWrapper
{
public:
  BabelGraphWrapper(BabelFlow::FunctionType* data_ptr, int32_t task_id, 
                    const int32_t* data_size, const int32_t* n_blocks,
                    const int32_t* low, const int32_t* high, int32_t fanin, 
                    BabelFlow::FunctionType extra_val, MPI_Comm mpi_comm);
  virtual ~BabelGraphWrapper() {}
  virtual void Initialize() = 0;
  virtual void Execute();
  
protected:
  BabelFlow::FunctionType* m_dataPtr;
  uint32_t m_taskId;
  uint32_t m_dataSize[3];
  uint32_t m_nBlocks[3];
  BabelFlow::GlobalIndexType m_low[3];
  BabelFlow::GlobalIndexType m_high[3];
  uint32_t m_fanin;
  BabelFlow::FunctionType m_extraVal;
  MPI_Comm m_comm;

  std::map<BabelFlow::TaskId, BabelFlow::Payload> m_inputs;
  BabelFlow::mpi::Controller m_master;
  BabelFlow::ControllerMap m_contMap;
};

//-----------------------------------------------------------------------------

class BabelVolRenderingReduce : public BabelGraphWrapper
{
public:
  BabelVolRenderingReduce(BabelFlow::FunctionType* data_ptr, int32_t task_id, 
                          const int32_t* data_size, const int32_t* n_blocks,
                          const int32_t* low, const int32_t* high, int32_t fanin, 
                          BabelFlow::FunctionType isoval, MPI_Comm mpi_comm);
  virtual ~BabelVolRenderingReduce() {}
  virtual void Initialize() override;

private:
  BabelFlow::KWayReduction m_graph;
  BabelFlow::KWayReductionTaskMap m_taskMap; 
  BabelFlow::PreProcessInputTaskGraph m_modGraph;
  BabelFlow::ModTaskMap m_modMap;
};

//-----------------------------------------------------------------------------

class BabelVolRenderingBinswap : public BabelGraphWrapper
{
public:
  static uint32_t TOTAL_NUM_BLOCKS;
  
  BabelVolRenderingBinswap(BabelFlow::FunctionType* data_ptr, int32_t task_id, 
                           const int32_t* data_size, const int32_t* n_blocks,
                           const int32_t* low, const int32_t* high, int32_t fanin, 
                           BabelFlow::FunctionType isoval, MPI_Comm mpi_comm);
  virtual ~BabelVolRenderingBinswap() {}
  virtual void Initialize() override;

private:
  BabelFlow::BinarySwap m_graph;
  BabelFlow::BinarySwapTaskMap m_taskMap; 
  BabelFlow::PreProcessInputTaskGraph m_modGraph;
  BabelFlow::ModTaskMap m_modMap;
};

//-----------------------------------------------------------------------------

class BabelVolRenderingRadixK : public BabelGraphWrapper
{
public:
  static uint32_t TOTAL_NUM_BLOCKS;
  static std::vector<uint32_t> RADICES_VEC;
  
  BabelVolRenderingRadixK(BabelFlow::FunctionType* data_ptr, int32_t task_id, 
                          const int32_t* data_size, const int32_t* n_blocks,
                          const int32_t* low, const int32_t* high, int32_t fanin, 
                          BabelFlow::FunctionType isoval, const std::vector<uint32_t>& radix_v,
                          MPI_Comm mpi_comm);
  virtual ~BabelVolRenderingRadixK() {}
  virtual void Initialize() override;

private:
  std::vector<uint32_t> m_Radices;
  
  BabelFlow::RadixKExchange m_graph;
  BabelFlow::RadixKExchangeTaskMap m_taskMap; 
  BabelFlow::PreProcessInputTaskGraph m_modGraph;
  BabelFlow::ModTaskMap m_modMap;
};

//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end bflow_volume --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent --
//-----------------------------------------------------------------------------







#endif    // ASCENT_ASCENT_RUNTIME_BABELFLOW_VOL_UTILS_H
