//
// Created by Sergei Shudler on 2020-06-09.
//

#ifndef ASCENT_ASCENT_RUNTIME_BABELFLOW_COMP_UTILS_H
#define ASCENT_ASCENT_RUNTIME_BABELFLOW_COMP_UTILS_H

#include <string>
#include <vector>

#include "BabelFlow/TypeDefinitions.h"
#include "BabelFlow/mpi/Controller.h"
#include "BabelFlow/reduce/BinarySwap.h"
#include "BabelFlow/reduce/BinarySwapTaskMap.h"
#include "BabelFlow/reduce/KWayReduction.h"
#include "BabelFlow/reduce/KWayReductionTaskMap.h"
#include "BabelFlow/reduce/RadixKExchange.h"
#include "BabelFlow/reduce/RadixKExchangeTaskMap.h"
#include "BabelFlow/reduce/SingleTaskGraph.h"
#include "BabelFlow/reduce/SingleTaskMap.h"
#include "BabelFlow/ComposableTaskGraph.h"
#include "BabelFlow/ComposableTaskMap.h"
#include "BabelFlow/DefGraphConnector.h"
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
// -- begin bflow_comp:: --
//-----------------------------------------------------------------------------
namespace bflow_comp
{

struct ImageData
{
  static const uint32_t sNUM_CHANNELS = 4;
  
  unsigned char* image; 
  unsigned char* zbuf;
  uint32_t* bounds;
  uint32_t* rend_bounds;     // Used only for binswap and k-radix
  
  ImageData() : image( nullptr ), zbuf( nullptr ), bounds( nullptr ), rend_bounds( nullptr ) {}
  
  void writeImage(const char* filename, uint32_t* extent);
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
                    int32_t n_blocks,
                    int32_t fanin,
                    MPI_Comm mpi_comm);
  virtual ~BabelGraphWrapper() {}
  virtual void Initialize() = 0;
  virtual void Execute();
  
protected:
  ImageData m_inputImg;
  uint32_t m_rankId;
  uint32_t m_numBlocks;
  uint32_t m_fanin;
  MPI_Comm m_comm;

  std::map<BabelFlow::TaskId, BabelFlow::Payload> m_inputs;
  BabelFlow::mpi::Controller m_master;
  BabelFlow::ControllerMap m_contMap;
};

//-----------------------------------------------------------------------------

class BabelCompReduce : public BabelGraphWrapper
{
public:
  BabelCompReduce(const ImageData& input_img,
                  const std::string& img_name,
                  int32_t rank_id,
                  int32_t n_blocks,
                  int32_t fanin,
                  MPI_Comm mpi_comm,
                  const int32_t* blk_arr);
  virtual ~BabelCompReduce() {}
  virtual void Initialize() override;

private:
  uint32_t m_nBlocks[3];

  BabelFlow::KWayReduction m_graph;
  BabelFlow::KWayReductionTaskMap m_taskMap; 
  BabelFlow::PreProcessInputTaskGraph m_modGraph;
  BabelFlow::ModTaskMap m_modMap;
};

//-----------------------------------------------------------------------------

class BabelCompBinswap : public BabelGraphWrapper
{
public:
  BabelCompBinswap(const ImageData& input_img,
                   const std::string& img_name,
                   int32_t rank_id,
                   int32_t n_blocks,
                   int32_t fanin,
                   MPI_Comm mpi_comm);
  virtual ~BabelCompBinswap() {}
  virtual void Initialize() override;

private:
  BabelFlow::BinarySwap m_graph;
  BabelFlow::BinarySwapTaskMap m_taskMap; 
  BabelFlow::PreProcessInputTaskGraph m_modGraph;
  BabelFlow::ModTaskMap m_modMap;
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
                  MPI_Comm mpi_comm,
                  const std::vector<uint32_t>& radix_v);
  virtual ~BabelCompRadixK();
  virtual void Initialize() override;

private:
  std::vector<uint32_t> m_Radices;
  
  BabelFlow::RadixKExchange m_radixkGr;
  BabelFlow::RadixKExchangeTaskMap m_radixkMp; 

  BabelFlow::SingleTaskGraph m_gatherTaskGr;
  BabelFlow::SingleTaskMap m_gatherTaskMp;

  BabelFlow::ComposableTaskGraph* m_ptrGraph;
  BabelFlow::ComposableTaskMap* m_ptrTaskMap;

  BabelFlow::DefGraphConnector* m_ptrDefGraphConnector;

  //BabelFlow::PreProcessInputTaskGraph m_modGraph;
  //BabelFlow::ModTaskMap m_modMap;
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
