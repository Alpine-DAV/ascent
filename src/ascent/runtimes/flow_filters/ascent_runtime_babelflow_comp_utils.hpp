//
// Created by Sergei Shudler on 2020-06-09.
//

#ifndef ASCENT_ASCENT_RUNTIME_BABELFLOW_COMP_UTILS_H
#define ASCENT_ASCENT_RUNTIME_BABELFLOW_COMP_UTILS_H

#include <string>
#include <vector>

#include "BabelFlow/TypeDefinitions.h"
// #include "BabelFlow/mpi/Controller.h"
// #include "BabelFlow/charm/Controller.h"
#include "BabelFlow/reduce/BinarySwap.h"
#include "BabelFlow/reduce/BinarySwapTaskMap.h"
#include "BabelFlow/reduce/KWayReduction.h"
#include "BabelFlow/reduce/KWayReductionTaskMap.h"
#include "BabelFlow/reduce/RadixKExchange.h"
#include "BabelFlow/reduce/RadixKExchangeTaskMap.h"
#include "BabelFlow/reduce/SingleTaskGraph.h"
#include "BabelFlow/ComposableTaskGraph.h"
#include "BabelFlow/ComposableTaskMap.h"
#include "BabelFlow/DefGraphConnector.h"
#include "BabelFlow/ModuloMap.h"

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#define _NOMPI
#endif


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin bflow_comp:: --
//-----------------------------------------------------------------------------
namespace bflow_comp
{

int volume_render_radixk(std::vector<BabelFlow::Payload>& inputs, 
                         std::vector<BabelFlow::Payload>& outputs, 
                         BabelFlow::TaskId task_id);

int composite_radixk(std::vector<BabelFlow::Payload>& inputs, 
                     std::vector<BabelFlow::Payload>& outputs, 
                     BabelFlow::TaskId task_id);

int write_results_radixk(std::vector<BabelFlow::Payload>& inputs,
                         std::vector<BabelFlow::Payload>& outputs, 
                         BabelFlow::TaskId task_id);

int gather_results_radixk(std::vector<BabelFlow::Payload>& inputs,
                          std::vector<BabelFlow::Payload>& outputs, 
                          BabelFlow::TaskId task_id);

struct ImageData
{
  using PixelType = float;

  static const uint32_t sNUM_CHANNELS = 4;
  constexpr static const PixelType sOPAQUE = 0.f;
  
  PixelType* image; 
  PixelType* zbuf;
  uint32_t* bounds;
  uint32_t* rend_bounds;     // Used only for binswap and k-radix
  
  ImageData() : image( nullptr ), zbuf( nullptr ), bounds( nullptr ), rend_bounds( nullptr ) {}
  
  void writeImage(const char* filename, uint32_t* extent);
  void writeDepth(const char* filename, uint32_t* extent);
  BabelFlow::Payload serialize() const;
  void deserialize(BabelFlow::Payload buffer);
  void delBuffers();
};
  
void compose_images(const std::vector<ImageData>& input_images, 
                    std::vector<ImageData>& out_images, 
                    int id,
                    bool flip_split_side,
                    bool skip_z_check);
                        
void split_and_blend(const std::vector<ImageData>& input_images,
                     std::vector<ImageData>& out_images,
                     uint32_t* union_box,
                     bool flip_split_side,
                     bool skip_z_check);

//-----------------------------------------------------------------------------

class BabelGraphWrapper
{
public:
  static std::string sIMAGE_NAME;
   
  BabelGraphWrapper(const ImageData& input_img,
                    const std::string& img_name,
                    int32_t rank_id,
                    int32_t n_ranks,
                    int32_t fanin);
                    // MPI_Comm mpi_comm);
  virtual ~BabelGraphWrapper() {}
  virtual void Initialize() = 0;
  virtual void Execute();
  
protected:
  ImageData m_inputImg;
  uint32_t m_rankId;
  uint32_t m_nRanks;
  uint32_t m_fanin;
  // MPI_Comm m_comm;

  std::map<BabelFlow::TaskId, BabelFlow::Payload> m_inputs;
  // BabelFlow::mpi::Controller m_master;
  // BabelFlow::ControllerMap m_contMap;

  ///// Charm++
  BabelFlow::charm::Controller m_controller;  
  BabelFlow::charm::Controller::ProxyType m_proxy;
  /////
};

//-----------------------------------------------------------------------------

class BabelCompReduce : public BabelGraphWrapper
{
public:
  BabelCompReduce(const ImageData& input_img,
                  const std::string& img_name,
                  int32_t rank_id,
                  int32_t n_blocks,
                  int32_t fanin);
                  // MPI_Comm mpi_comm);
  virtual ~BabelCompReduce() {}
  virtual void Initialize() override;

private:
  BabelFlow::SingleTaskGraph m_preProcTaskGr;
  BabelFlow::ModuloMap m_preProcTaskMp;

  BabelFlow::KWayReduction m_reduceTaskGr;
  BabelFlow::KWayReductionTaskMap m_reduceTaskMp; 

  BabelFlow::ComposableTaskGraph m_reduceGraph;
  BabelFlow::ComposableTaskMap m_reduceTaskMap;

  BabelFlow::DefGraphConnector m_defGraphConnector;
};

//-----------------------------------------------------------------------------

class BabelCompBinswap : public BabelGraphWrapper
{
public:
  BabelCompBinswap(const ImageData& input_img,
                   const std::string& img_name,
                   int32_t rank_id,
                   int32_t n_blocks,
                   int32_t fanin);
                  //  MPI_Comm mpi_comm);
  virtual ~BabelCompBinswap() {}
  virtual void Initialize() override;

private:
  BabelFlow::SingleTaskGraph m_preProcTaskGr;
  BabelFlow::ModuloMap m_preProcTaskMp;

  BabelFlow::BinarySwap m_binSwapTaskGr;
  BabelFlow::BinarySwapTaskMap m_binSwapTaskMp; 

  BabelFlow::ComposableTaskGraph m_binSwapGraph;
  BabelFlow::ComposableTaskMap m_binSwapTaskMap;

  BabelFlow::DefGraphConnector m_defGraphConnector;
};

//-----------------------------------------------------------------------------

class BabelCompRadixK : public BabelGraphWrapper
{
public:
  BabelCompRadixK(const ImageData& input_img,
                  const std::string& img_name,
                  int32_t rank_id,
                  int32_t n_blocks,
                  int32_t fanin,
                  // MPI_Comm mpi_comm,
                  const std::vector<uint32_t>& radix_v);
  virtual ~BabelCompRadixK();
  
  virtual void InitRadixKGraph();
  virtual void InitGatherGraph();
  virtual void Initialize() override;
  
protected:
  std::vector<uint32_t> m_Radices;
  
  BabelFlow::SingleTaskGraph m_preProcTaskGr;
  BabelFlow::ModuloMap m_preProcTaskMp;

  BabelFlow::RadixKExchange m_radixkGr;
  BabelFlow::RadixKExchangeTaskMap m_radixkMp; 

  BabelFlow::KWayReduction m_gatherTaskGr;
  BabelFlow::KWayReductionTaskMap m_gatherTaskMp; 

  BabelFlow::ComposableTaskGraph m_radGatherGraph;
  BabelFlow::ComposableTaskMap m_radGatherTaskMap;

  BabelFlow::DefGraphConnector m_defGraphConnector;
};

//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end bflow_comp --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent --
//-----------------------------------------------------------------------------







#endif    // ASCENT_ASCENT_RUNTIME_BABELFLOW_COMP_UTILS_H
