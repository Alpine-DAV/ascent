//
// Created by Li, Jixian on 2019-06-04.
//

#include "ascent_runtime_babelflow_filters.hpp"

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_workspace.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif

//-----------------------------------------------------------------------------
// -- begin ParallelMergeTree --
//-----------------------------------------------------------------------------

#include "BabelFlow/mpi/Controller.h"
#include "PMT/KWayMerge.h"
#include "PMT/KWayTaskMap.h"
#include "PMT/SortedUnionFindAlgorithm.h"
#include "PMT/SortedJoinAlgorithm.h"
#include "PMT/LocalCorrectionAlgorithm.h"
#include "PMT/MergeTree.h"
#include "PMT/AugmentedMergeTree.h"
#include "BabelFlow/reduce/SingleTaskGraph.h"
#include "BabelFlow/reduce/RadixKExchange.h"
#include "BabelFlow/reduce/RadixKExchangeTaskMap.h"
#include "BabelFlow/ComposableTaskGraph.h"
#include "BabelFlow/ComposableTaskMap.h"
#include "BabelFlow/DefGraphConnector.h"
#include "BabelFlow/ModuloMap.h"
#include "BabelFlow/RelayTask.h"

#include "StreamStat/StreamingStatistic.h"
#include "StreamStat/StreamingStatisticFactory.h"

#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <float.h>
#include <climits>
#include <memory>


// #define BFLOW_PMT_DEBUG
// #define BFLOW_PMT_WRITE_RES
#define EPSILON     0.00001


class ParallelMergeTree {
public:
  ParallelMergeTree(FunctionType *data_ptr, int32_t task_id, const int32_t *data_size, const int32_t *n_blocks,
                    const int32_t *low, const int32_t *high, int32_t fanin,
                    FunctionType threshold, MPI_Comm mpi_comm,
                    const std::vector<uint32_t>& radix_v,
                    const std::vector<uint32_t>& stat_types_v);

  void Initialize();

  void Execute();

  void ExtractSegmentation(FunctionType* output_data_ptr);

  static int DownSizeGhosts(std::vector<BabelFlow::Payload> &inputs, std::vector<BabelFlow::Payload> &output,
                            BabelFlow::TaskId task);
  static void ComputeGhostOffsets(uint32_t low[3], uint32_t high[3],
                                  uint32_t& dnx, uint32_t& dny, uint32_t& dnz,
                                  uint32_t& dpx, uint32_t& dpy, uint32_t& dpz);
  static uint32_t o_ghosts[6];
  static uint32_t n_ghosts[6];
  static uint32_t s_data_size[3];

private:
  FunctionType *m_dataPtr;
  uint32_t m_taskId;
  uint32_t m_dataSize[3];
  uint32_t m_nBlocks[3];
  uint32_t m_low[3];
  uint32_t m_high[3];
  uint32_t m_fanin;

  FunctionType m_threshold;
  std::map<BabelFlow::TaskId, BabelFlow::Payload> m_inputs;

  MPI_Comm m_comm;
  BabelFlow::mpi::Controller m_master;
  BabelFlow::ControllerMap m_cMap;
  
  BabelFlow::SingleTaskGraph m_preProcTaskGr;
  BabelFlow::ModuloMap m_preProcTaskMp;

  KWayMerge m_kWayMergeGr;
  KWayTaskMap m_kWayTaskMp;

  BabelFlow::RadixKExchange m_treeStatsGr;
  BabelFlow::RadixKExchangeTaskMap m_treeStatsMp;

  BabelFlow::ComposableTaskGraph m_fullGraph;
  BabelFlow::ComposableTaskMap m_fullTaskMap;

  BabelFlow::DefGraphConnector m_defGraphConnector_1;
  BabelFlow::DefGraphConnector m_defGraphConnector_2;

  std::vector<uint32_t> m_radices;
};


// CallBack Functions
uint32_t ParallelMergeTree::o_ghosts[6] = {1, 1, 1, 1, 1, 1};
uint32_t ParallelMergeTree::n_ghosts[6] = {1, 1, 1, 1, 1, 1};
uint32_t ParallelMergeTree::s_data_size[3];

// Static ptr for local data to be passed to computeSegmentation()
FunctionType* sLocalData = NULL;


int local_compute(std::vector<BabelFlow::Payload> &inputs,
                  std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  sorted_union_find_algorithm(inputs, output, task);
  /*
  MergeTree t;

  //fprintf(stderr,"LOCAL COMPUTE performed by task %d\n", task);
  t.decode(output[0]);

  t.writeToFile(task);
*/
  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )  
    payl.reset();

  return 1;
}


int join(std::vector<BabelFlow::Payload> &inputs,
         std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {


  //fprintf(stderr, "Task : %d : Started with join algorithm\n", task);
  sorted_join_algorithm(inputs, output, task);
  //fprintf(stderr, "Task : %d : Done with join algorithm\n", task);
/*
  MergeTree join_tree;

  join_tree.decode(output[0]);
  join_tree.writeToFile(task+1000);
  */
  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )  
    payl.reset();
  
  return 0;
}

int local_correction(std::vector<BabelFlow::Payload> &inputs,
                     std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  //if ((task & ~sPrefixMask) == 237)
  local_correction_algorithm(inputs, output, task);

  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )  
    payl.reset();

  //fprintf(stderr,"CORRECTION performed by task %d\n", task & ~sPrefixMask);
  return 1;
}

int write_results(std::vector<BabelFlow::Payload> &inputs,
                  std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  AugmentedMergeTree t;
  t.decode(inputs[0]);
  t.id(task & ~KWayMerge::sPrefixMask);
  //t.writeToFile(task & ~sPrefixMask);
  t.persistenceSimplification(1.f);
  t.computeSegmentation(sLocalData);
#ifdef BFLOW_PMT_WRITE_RES
  t.writeToFileBinary(task & ~KWayMerge::sPrefixMask);
  t.writeToFile(task & ~KWayMerge::sPrefixMask);
  //t.writeToHtmlFile(task & ~sPrefixMask);
#endif

  t.writeToHtmlFile(task & ~KWayMerge::sPrefixMask);

  // Set the final tree as an output so that it could be extracted later
  output[0] = t.encode();

  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )  
    payl.reset();

  //fprintf(stderr,"WRITING RESULTS performed by %d\n", task & ~sPrefixMask);
  return 1;
}

//////////// Streaming statistics //////////////////

template<typename T>
static inline void serialize_elem( char*& buff, T val )
{
  memcpy( buff, &val, sizeof(T) );
  buff += sizeof(T);
}

template<typename T>
static inline void deserialize_elem( char*& buff, T& val )
{
  memcpy( &val, buff, sizeof(T) );
  buff += sizeof(T);
}

class StatisticsMap
{
public:
  using StreamingStatisticPtr = std::shared_ptr<Statistics::StreamingStatistic>;
  using StatisticsVec = std::vector<StreamingStatisticPtr>;

  static std::vector<Statistics::StreamingStatisticType> sRequestedStats;

  StatisticsMap() 
  {
    mLow[0] = mLow[1] = mLow[2] = 0;
    mHigh[0] = mHigh[1] = mHigh[2] = 0;
  }

  StatisticsMap( GlobalIndexType low[3], GlobalIndexType high[3] ) 
  {
    setLow( low );
    setHigh( high );
  }

  ~StatisticsMap() {}

  StatisticsVec& getAndInitStatsVec( GlobalIndexType id ) 
  {
    StatisticsVec& stats_vec = mStats[id];
    if( stats_vec.size() == 0 )
    {
      stats_vec.resize( StatisticsMap::sRequestedStats.size() );
      for( unsigned int j = 0; j < StatisticsMap::sRequestedStats.size(); ++j )
        stats_vec[j] = 
          StreamingStatisticPtr( mStatsFac.make_statistic( StatisticsMap::sRequestedStats[j] ) );
    }
    return stats_vec;
  }

  StatisticsVec& getStatsVec( GlobalIndexType id ) 
  {
    return mStats[id];
  }

  void setStatsVec( GlobalIndexType id, const StatisticsVec& stats_vec )
  {
    mStats[id] = stats_vec;
  }

  std::unordered_map<GlobalIndexType, StatisticsVec>::const_iterator beginIter() const
  {
    return mStats.begin();
  }

  std::unordered_map<GlobalIndexType, StatisticsVec>::const_iterator endIter() const
  {
    return mStats.end();
  }

  void setLow( GlobalIndexType low[3] )
  {
    mLow[0] = low[0];
    mLow[1] = low[1];
    mLow[2] = low[2];
  }

  void setHigh( GlobalIndexType high[3] )
  {
    mHigh[0] = high[0];
    mHigh[1] = high[1];
    mHigh[2] = high[2];
  }

  void getBoundary( GlobalIndexType low[3], GlobalIndexType high[3] ) const
  {
    low[0] = mLow[0];
    low[1] = mLow[1];
    low[2] = mLow[2];

    high[0] = mHigh[0];
    high[1] = mHigh[1];
    high[2] = mHigh[2];
  }

  void mergeBoundary( GlobalIndexType low[3], GlobalIndexType high[3] )
  {
    mLow[0] = std::min( mLow[0], low[0] );
    mLow[1] = std::min( mLow[1], low[1] );
    mLow[2] = std::min( mLow[2], low[2] );

    mHigh[0] = std::max( mHigh[0], high[0] );
    mHigh[1] = std::max( mHigh[1], high[1] );
    mHigh[2] = std::max( mHigh[2], high[2] );
  }

  bool coordsWithinBoundary( GlobalIndexType coords[3] ) const
  {
    bool within[3] = {
      mLow[0] <= coords[0] && coords[0] <= mHigh[0],
      mLow[1] <= coords[1] && coords[1] <= mHigh[1],
      mLow[2] <= coords[2] && coords[2] <= mHigh[2]
    };

    return within[0] && within[1] && within[2];
  }

  //! Serialize the statistics map data as a Payload
  BabelFlow::Payload serialize()
  {
    // Calculate total size
    int32_t buff_sz = 6 * sizeof(GlobalIndexType);    // mLow[3] and mHigh[3]

    auto iter = mStats.begin();
    auto end_iter = mStats.end();
    
    buff_sz += sizeof(uint32_t);                      // num elems in map

    for( ; iter != end_iter; ++iter )
    {
        buff_sz += sizeof(GlobalIndexType);           // map key
        buff_sz += sizeof(uint32_t);                  // num elems in stat vec

        for( auto& stat_ptr : iter->second )
        {
          buff_sz += sizeof(uint32_t);                // elem buff size
          buff_sz += sizeof(uint32_t);                // elem type

          int32_t stat_buff_sz = 0;
          char* stat_buff_ptr = nullptr;
          stat_ptr->serialize( stat_buff_sz, stat_buff_ptr, false );
          buff_sz += stat_buff_sz;                    // elem buff
        }
    }

    char* full_buff = new char[buff_sz];

    GlobalIndexType* tbuff = (GlobalIndexType*)full_buff;
    tbuff[0] = mLow[0];
    tbuff[1] = mLow[1];
    tbuff[2] = mLow[2];
    tbuff[3] = mHigh[0];
    tbuff[4] = mHigh[1];
    tbuff[5] = mHigh[2];

    char* buff_ptr = full_buff + 6 * sizeof(GlobalIndexType);
    
    serialize_elem<uint32_t>( buff_ptr, mStats.size() );

    iter = mStats.begin();
    end_iter = mStats.end();

    for( ; iter != end_iter; ++iter )
    {
        serialize_elem<GlobalIndexType>( buff_ptr, iter->first );
        serialize_elem<uint32_t>( buff_ptr, iter->second.size() );

        for( auto& stat_ptr : iter->second )
        {
          int32_t stat_buff_sz = 0;
          char* stat_buff_ptr = nullptr;
          stat_ptr->serialize( stat_buff_sz, stat_buff_ptr, false );

          serialize_elem<uint32_t>( buff_ptr, stat_buff_sz );
          serialize_elem<uint32_t>( buff_ptr, stat_ptr->type() );

          stat_ptr->serialize( stat_buff_sz, buff_ptr, false );
          buff_ptr += stat_buff_sz;
        }
    }

    return BabelFlow::Payload( buff_sz, full_buff );
  }

  //! Deserialize the statistics data from the given Payload
  void deserialize(const BabelFlow::Payload& pl)
  {
    uint32_t buff_sz;
    GlobalIndexType* tbuff = (GlobalIndexType*)pl.buffer();
    mLow[0] = tbuff[0];
    mLow[1] = tbuff[1];
    mLow[2] = tbuff[2];
    mHigh[0] = tbuff[3];
    mHigh[1] = tbuff[4];
    mHigh[2] = tbuff[5];

    char* buff_ptr = pl.buffer() + 6 * sizeof(GlobalIndexType);

    uint32_t map_sz = 0;
    deserialize_elem<uint32_t>( buff_ptr, map_sz );

    for( uint32_t i = 0; i < map_sz; ++i )
    {
      GlobalIndexType feat_id;
      deserialize_elem<GlobalIndexType>( buff_ptr, feat_id );

      StatisticsVec& stats_vec = mStats[feat_id];

      uint32_t stat_vec_sz = 0;
      deserialize_elem<uint32_t>( buff_ptr, stat_vec_sz );
      
      stats_vec.resize( stat_vec_sz );
      for( uint32_t j = 0; j < stats_vec.size(); ++j )
      {
        uint32_t stat_buff_sz = 0;
        deserialize_elem<uint32_t>( buff_ptr, stat_buff_sz );

        uint32_t stat_type = 0;
        deserialize_elem<uint32_t>( buff_ptr, stat_type );

        stats_vec[j] = 
          StreamingStatisticPtr( mStatsFac.make_statistic( Statistics::StreamingStatisticType( stat_type ) ) );

        stats_vec[j]->deserialize( stat_buff_sz, buff_ptr );
        buff_ptr += stat_buff_sz;
      }

    }
  }

private:
  GlobalIndexType mLow[3];
  GlobalIndexType mHigh[3];
  std::unordered_map<GlobalIndexType, StatisticsVec> mStats;

  Statistics::Factory mStatsFac;
};

std::vector<Statistics::StreamingStatisticType> StatisticsMap::sRequestedStats;

void split_stats( const StatisticsMap& stats_map, std::vector<StatisticsMap>& out_stats_map )
{
  // Got over stats map, for each feature:
  //   Translate feature label (id) to location
  //   Determine in which part of the full extent the feature location belongs
  //   Add the statistic to that part stat map

  // Split extent
  GlobalIndexType low[3];
  GlobalIndexType high[3];

  stats_map.getBoundary( low, high );

  // Compute splitting delta
  uint32_t num_outs = out_stats_map.size();
  GlobalIndexType extent[3] = 
  {
    (high[0] - low[0]) + 1,
    (high[1] - low[1]) + 1,
    (high[2] - low[2]) + 1
  };
  int delta[3] = {0, 0, 0};
  GlobalIndexType max_extent = std::max( std::max( extent[0], extent[1] ), extent[2] );
  if( extent[0] == max_extent )
  {
    delta[0] = max_extent / num_outs;
  }
  else if( extent[1] == max_extent )
  {
    delta[1] = max_extent / num_outs;
  }
  else
  {
    delta[2] = max_extent / num_outs;
  }

  // Set extents for each output map according to the delta
  for( unsigned int i = 0; i < num_outs; ++i )
  {
    GlobalIndexType l[3] = {low[0] + delta[0]*i, low[1] + delta[1]*i, low[2] + delta[2]*i};
    GlobalIndexType h[3] = {high[0], high[1], high[2]};
    if( delta[0] )
    {
      h[0] = low[0] + delta[0]*(i + 1) - 1;
    }
    else if( delta[1] )
    {
      h[1] = low[1] + delta[1]*(i + 1) - 1;
    }
    else
    {
      h[2] = low[2] + delta[2]*(i + 1) - 1;
    }

    out_stats_map[i].setLow(l);
    out_stats_map[i].setHigh(h);
  }

  // Split given statistic map according to feature locations -- belonging to each one of
  // the output extents
  auto iter = stats_map.beginIter();
  auto end_iter = stats_map.endIter();

  for( ; iter != end_iter; ++iter )
  {
    GlobalIndexType feat_label = iter->first;

    // Convert the feature id (index in the global data) to coords
    GlobalIndexType coords[3];
    coords[0] = feat_label % MergeTree::sDimension[0];
    coords[1] = (feat_label / MergeTree::sDimension[0]) % MergeTree::sDimension[1];
    coords[2] = feat_label / (MergeTree::sDimension[0] * MergeTree::sDimension[1]);
    
    for( unsigned int i = 0; i < num_outs; ++i )
    {
      if( out_stats_map[i].coordsWithinBoundary( coords ) )
      {
        out_stats_map[i].getStatsVec( feat_label ) = iter->second;
        break;
      }
    }
  }
}

void merge_stat_maps( StatisticsMap& merged_stats_map, std::vector<BabelFlow::Payload> &inputs )
{
  if( inputs.size() == 1 )
  {
    merged_stats_map.deserialize( inputs[0] );
    return;
  } 

  for( unsigned int i = 0; i < inputs.size(); ++i )
  {
    StatisticsMap in_map;
    in_map.deserialize( inputs[i] );

    GlobalIndexType low[3];
    GlobalIndexType high[3];

    in_map.getBoundary( low, high );

    if( i == 0 )
    {
      merged_stats_map.setLow( low );
      merged_stats_map.setHigh( high );
    }
    else
    {
      merged_stats_map.mergeBoundary( low, high );
    }

    auto iter = in_map.beginIter();
    auto end_iter = in_map.endIter();

    for( ; iter != end_iter; ++iter )
    {
      GlobalIndexType feat_label = iter->first;

      StatisticsMap::StatisticsVec& sv = merged_stats_map.getStatsVec( feat_label );
      if( sv.size() == 0 )
      {
        sv = iter->second;
      }
      else
      {
        assert( sv.size() == iter->second.size() );
        // Merge statistics
        for( unsigned int i = 0; i < sv.size(); ++i )
          sv[i]->add( iter->second[i].get() );
      }
    }
  }
}

int compute_loc_stats( std::vector<BabelFlow::Payload> &inputs,
                       std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task )
{
  // Decode the tree with the segmentation
  AugmentedMergeTree aug_t;
  aug_t.decode(inputs[0]);

  // Construct a regular merge tree, without the segmentation field
  // MergeTree mg_t( aug_t );

  GlobalIndexType low[3];
  GlobalIndexType high[3];

  aug_t.blockBoundary( low, high );

  GlobalIndexType extent[3] = 
  {
    (high[0] - low[0]) + 1,
    (high[1] - low[1]) + 1,
    (high[2] - low[2]) + 1
  };

  // Compute statistics on the local data
  StatisticsMap stats_map;
  stats_map.setHigh( MergeTree::sDimension );
  
  for( unsigned int i = 0; i < aug_t.sampleCount(); ++i )
  {
    GlobalIndexType coords[3];

    // Set the spatial coords
    coords[0] = i % extent[0];
    coords[1] = (i / extent[0]) % extent[1];
    coords[2] = i / (extent[0] * extent[1]);

    coords[0] += low[0];
    coords[1] += low[1];
    coords[2] += low[2];

    // TODO: Skip data on the boundary (coords[0] == low[0] or high[0])

    StatisticsMap::StatisticsVec& stats_vec = stats_map.getAndInitStatsVec( aug_t.label(i) );

    // Add the feature to all the statistics
    for( unsigned int j = 0; j < stats_vec.size(); ++j )
      stats_vec[j]->add( sLocalData[i] );
  }

  // Write out stats_map (debug)
#ifdef BFLOW_PMT_DEBUG
  {
    std::stringstream ss;
    ss << "stream_stat_comp_" << task << ".txt";
    std::ofstream bofs(ss.str(), std::ios::out | std::ios::binary);

    auto iter = stats_map.beginIter();
    auto end_iter = stats_map.endIter();

    for( ; iter != end_iter; ++iter )
    {
      bofs << iter->first << "  --  ";
      const StatisticsMap::StatisticsVec& stats_vec = iter->second;
      for( auto& stat_ptr : stats_vec )
        bofs << stat_ptr->value() << "  ";
      bofs << std::endl;
    }
    bofs.close();
  }
#endif

  std::vector<StatisticsMap> out_stats_map( output.size() );
  split_stats( stats_map, out_stats_map );

  // Got over outputs, serialize stats map
  for( unsigned int i = 0; i < output.size(); ++i )
    output[i] = out_stats_map[i].serialize();

  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )
    payl.reset();

  return 1;
}

int merge_stats( std::vector<BabelFlow::Payload> &inputs,
                 std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task )
{
  // --Next layer:
  // Go over input stats map and merge the statistics together
  // Go over the merged stats map and repeat step from layer #1

  StatisticsMap merged_stats_map;
  merge_stat_maps( merged_stats_map, inputs );

  // Write out merged_stats_map
#ifdef BFLOW_PMT_DEBUG
  {
    std::stringstream ss;
    ss << "stream_stat_merge_" << task << ".txt";
    std::ofstream bofs(ss.str(), std::ios::out | std::ios::binary);

    auto iter = merged_stats_map.beginIter();
    auto end_iter = merged_stats_map.endIter();

    for( ; iter != end_iter; ++iter )
    {
      bofs << iter->first << "  --  ";
      const StatisticsMap::StatisticsVec& stats_vec = iter->second;
      for( auto& stat_ptr : stats_vec )
        bofs << stat_ptr->value() << "  ";
      bofs << std::endl;
    }
    bofs.close();
  }
#endif

  std::vector<StatisticsMap> out_stats_map( output.size() );
  split_stats( merged_stats_map, out_stats_map );

  // Got over outputs, serialize stats map
  for( unsigned int i = 0; i < output.size(); ++i )
    output[i] = out_stats_map[i].serialize();

  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )
    payl.reset();
  
  return 1;  
}

int write_stats( std::vector<BabelFlow::Payload> &inputs,
                 std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task )
{
  // --Final layer:
  // Merge input stats maps
  // Write them out to output (HDF5?)
  assert( inputs.size() > 0 );

  StatisticsMap merged_stats_map;
  merge_stat_maps( merged_stats_map, inputs );

  // Write out merged_stats_map
  std::stringstream ss;
  ss << "stream_stat_" << task << ".txt";
  std::ofstream bofs(ss.str(), std::ios::out | std::ios::binary);

  auto iter = merged_stats_map.beginIter();
  auto end_iter = merged_stats_map.endIter();

  for( ; iter != end_iter; ++iter )
  {
    bofs << iter->first << "  --  ";
    const StatisticsMap::StatisticsVec& stats_vec = iter->second;
    for( auto& stat_ptr : stats_vec )
      bofs << stat_ptr->value() << "  ";
    bofs << std::endl;
  }

  bofs.close();

  // Deleting input data
  for( BabelFlow::Payload& payl : inputs )
    payl.reset();
  
  return 1;
}

/////////////////////////////////////////////////

void ParallelMergeTree::ComputeGhostOffsets(uint32_t low[3], uint32_t high[3],
                                            uint32_t& dnx, uint32_t& dny, uint32_t& dnz,
                                            uint32_t& dpx, uint32_t& dpy, uint32_t& dpz) {
  dnx = (low[0] == 0) ? 0 : o_ghosts[0] - n_ghosts[0];
  dny = (low[1] == 0) ? 0 : o_ghosts[1] - n_ghosts[1];
  dnz = (low[2] == 0) ? 0 : o_ghosts[2] - n_ghosts[2];
  dpx = (high[0] == s_data_size[0] - 1) ? 0 : o_ghosts[3] - n_ghosts[3];
  dpy = (high[1] == s_data_size[1] - 1) ? 0 : o_ghosts[4] - n_ghosts[4];
  dpz = (high[2] == s_data_size[2] - 1) ? 0 : o_ghosts[5] - n_ghosts[5];
}

int ParallelMergeTree::DownSizeGhosts(std::vector<BabelFlow::Payload> &inputs, std::vector<BabelFlow::Payload> &output,
                                      BabelFlow::TaskId task) {
  FunctionType *block_data;
  FunctionType threshold;
  GlobalIndexType low[3];
  GlobalIndexType high[3];

  decode_local_block(inputs[0], &block_data, low, high, threshold);

  GlobalIndexType xsize = high[0] - low[0] + 1;
  GlobalIndexType ysize = high[1] - low[1] + 1;
  GlobalIndexType zsize = high[2] - low[2] + 1;

  uint32_t dnx, dny, dnz, dpx, dpy, dpz;
  ParallelMergeTree::ComputeGhostOffsets( low, high, dnx, dny, dnz, dpx, dpy, dpz );

  uint32_t numx = xsize - dnx - dpx;
  uint32_t numy = ysize - dny - dpy;
  uint32_t numz = zsize - dnz - dpz;

  char *n_block_data = new char[numx * numy * numz * sizeof(FunctionType)];
  size_t offset = 0;
  for (uint32_t z = 0; z < zsize; ++z) {
    if (z >= dnz && z < zsize - dpz) {
      for (uint32_t y = 0; y < ysize; ++y) {
        if (y >= dny && y < ysize - dpy) {
          FunctionType *data_ptr = block_data + dnx + y * xsize + z * ysize * xsize;
          memcpy(n_block_data + offset, (char *) data_ptr, numx * sizeof(FunctionType));
          offset += numx * sizeof(FunctionType);
        }
      }
    }
  }

  GlobalIndexType n_low[3];
  GlobalIndexType n_high[3];
  n_low[0] = low[0] + dnx;
  n_low[1] = low[1] + dny;
  n_low[2] = low[2] + dnz;

  n_high[0] = high[0] - dpx;
  n_high[1] = high[1] - dpy;
  n_high[2] = high[2] - dpz;

  output.resize(1);
  output[0] = make_local_block((FunctionType*)n_block_data, n_low, n_high, threshold);
  
  sLocalData = (FunctionType*)n_block_data;

  delete[] inputs[0].buffer();
  return 0;
}

int pre_proc(std::vector<BabelFlow::Payload> &inputs,
             std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  return ParallelMergeTree::DownSizeGhosts(inputs, output, task);
  //printf("this is where the preprocessing supposed to happend for Task %d\n", task);
  // output = inputs;
  // return 1;
}


void ParallelMergeTree::Initialize() 
{
  int my_rank = 0;
  int mpi_size = 1;

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm_rank( m_comm, &my_rank );
  MPI_Comm_size( m_comm, &mpi_size );
#endif

  uint32_t num_blocks = m_nBlocks[0] * m_nBlocks[1] * m_nBlocks[2];

#ifdef BFLOW_PMT_DEBUG
  if( my_rank == 0 ) 
  {
    std::cout << "Num blocks: " << num_blocks << std::endl;
  }
#endif

  m_preProcTaskGr = BabelFlow::SingleTaskGraph( mpi_size );
  m_preProcTaskMp = BabelFlow::ModuloMap( mpi_size, m_nBlocks[0] * m_nBlocks[1] * m_nBlocks[2] );

  m_kWayMergeGr = KWayMerge( m_nBlocks, m_fanin );
  m_kWayTaskMp = KWayTaskMap( mpi_size, &m_kWayMergeGr );

  m_treeStatsGr = BabelFlow::RadixKExchange( mpi_size, m_radices );
  m_treeStatsMp = BabelFlow::RadixKExchangeTaskMap( mpi_size, &m_treeStatsGr );

  m_preProcTaskGr.setGraphId( 0 );
  m_kWayMergeGr.setGraphId( 1 );
  m_treeStatsGr.setGraphId( 2 );

  BabelFlow::TaskGraph::registerCallback( 0, BabelFlow::SingleTaskGraph::SINGLE_TASK_CB, pre_proc );
  
  BabelFlow::TaskGraph::registerCallback( 1, KWayMerge::LOCAL_COMP_CB, local_compute );
  BabelFlow::TaskGraph::registerCallback( 1, KWayMerge::JOIN_COMP_CB, join );
  BabelFlow::TaskGraph::registerCallback( 1, KWayMerge::LOCAL_CORR_CB, local_correction );
  BabelFlow::TaskGraph::registerCallback( 1, KWayMerge::WRITE_RES_CB, write_results );
  BabelFlow::TaskGraph::registerCallback( 1, KWayMerge::RELAY_CB, BabelFlow::relay_message );

  BabelFlow::TaskGraph::registerCallback( 2, BabelFlow::RadixKExchange::LEAF_TASK_CB, compute_loc_stats );
  BabelFlow::TaskGraph::registerCallback( 2, BabelFlow::RadixKExchange::MID_TASK_CB, merge_stats );
  BabelFlow::TaskGraph::registerCallback( 2, BabelFlow::RadixKExchange::ROOT_TASK_CB, write_stats );

#ifdef BFLOW_PMT_DEBUG
  if( my_rank == 0 ) 
  {
    m_kWayMergeGr.outputGraphHtml( mpi_size, &m_kWayTaskMp, "orig_pmt_gr.html" );
  }
#endif

  if( StatisticsMap::sRequestedStats.size() )
  {
    m_defGraphConnector_1 = BabelFlow::DefGraphConnector( &m_preProcTaskGr, 0, &m_kWayMergeGr, 1 );
    m_defGraphConnector_2 = BabelFlow::DefGraphConnector( &m_kWayMergeGr, 1, &m_treeStatsGr, 2 );

    std::vector<BabelFlow::TaskGraphConnector*> gr_connectors{ &m_defGraphConnector_1, &m_defGraphConnector_2 };
    std::vector<BabelFlow::TaskGraph*> gr_vec{ &m_preProcTaskGr, &m_kWayMergeGr, &m_treeStatsGr };
    std::vector<BabelFlow::TaskMap*> task_maps{ &m_preProcTaskMp, &m_kWayTaskMp, &m_treeStatsMp }; 

    m_fullGraph = BabelFlow::ComposableTaskGraph( gr_vec, gr_connectors );
    m_fullTaskMap = BabelFlow::ComposableTaskMap( task_maps );
  }
  else
  {
    m_defGraphConnector_1 = BabelFlow::DefGraphConnector( &m_preProcTaskGr, 0, &m_kWayMergeGr, 1 );

    std::vector<BabelFlow::TaskGraphConnector*> gr_connectors{ &m_defGraphConnector_1 };
    std::vector<BabelFlow::TaskGraph*> gr_vec{ &m_preProcTaskGr, &m_kWayMergeGr };
    std::vector<BabelFlow::TaskMap*> task_maps{ &m_preProcTaskMp, &m_kWayTaskMp }; 

    m_fullGraph = BabelFlow::ComposableTaskGraph( gr_vec, gr_connectors );
    m_fullTaskMap = BabelFlow::ComposableTaskMap( task_maps );
  }
  
  MergeTree::setDimension( m_dataSize );
#ifdef BFLOW_PMT_DEBUG
  if( my_rank == 0 ) 
  {
    m_fullGraph.outputGraphHtml( mpi_size, &m_fullTaskMap, "pmt_gr.html" );
  }
#endif

  m_master.initialize( m_fullGraph, &m_fullTaskMap, m_comm, &m_cMap );
 
  m_inputs[m_taskId] = make_local_block( m_dataPtr, m_low, m_high, m_threshold );
}

void ParallelMergeTree::Execute() 
{
  m_master.run( m_inputs );
}

void ParallelMergeTree::ExtractSegmentation(FunctionType* output_data_ptr) 
{
  
  // Sizes of the output data
  GlobalIndexType xsize = m_high[0] - m_low[0] + 1;
  GlobalIndexType ysize = m_high[1] - m_low[1] + 1;
  GlobalIndexType zsize = m_high[2] - m_low[2] + 1;
  
  // Output data includes ghost cells so we have to embed the smaller segmentation
  // part in the output data. The first step is to intialize all output
  std::fill( output_data_ptr, output_data_ptr + xsize*ysize*zsize, (FunctionType)GNULL );
  
  uint32_t dnx, dny, dnz, dpx, dpy, dpz;
  ParallelMergeTree::ComputeGhostOffsets( m_low, m_high, dnx, dny, dnz, dpx, dpy, dpz );
  
  // Get the outputs map (maps task IDs to outputs) from the controller
  std::map<BabelFlow::TaskId,std::vector<BabelFlow::Payload> >& outputs = m_master.getAllOutputs();
  
  // Only one task per rank should have output
  assert(outputs.size() == 1);
  
  auto iter = outputs.begin();
  // Output task should be only 'write_results' task
  // assert( m_fullGraph.task(m_fullGraph.gId(iter->first)).callbackId() == KWayMerge::WRITE_RES_CB );
  
  AugmentedMergeTree t;
  t.decode((iter->second)[0]);    // only one output per task

  // Copy the segmentation labels into the provided data array -- assume it
  // has enough space (sampleCount() -- the local data size w/o ghost cells)
  uint32_t label_idx = 0;   // runs from 0 to t.sampleCount()
  for (uint32_t z = dnz; z < zsize - dpz; ++z) {
    for (uint32_t y = dny; y < ysize - dpy; ++y) {
      for (uint32_t x = dnx; x < xsize - dpx; ++x) {
        uint32_t out_data_idx = x + y * xsize + z * ysize * xsize;
        output_data_ptr[out_data_idx] = (FunctionType)t.label(label_idx);
        ++label_idx;
      }
    }
  }
}

ParallelMergeTree::ParallelMergeTree( FunctionType *data_ptr, 
                                      int32_t task_id, 
                                      const int32_t *data_size,
                                      const int32_t *n_blocks,
                                      const int32_t *low, 
                                      const int32_t *high, 
                                      int32_t fanin,
                                      FunctionType threshold, 
                                      MPI_Comm mpi_comm,
                                      const std::vector<uint32_t>& radix_v,
                                      const std::vector<uint32_t>& stat_types_v )
  : m_dataPtr( data_ptr ), m_threshold( threshold ), m_comm( mpi_comm )
{
  m_taskId = static_cast<uint32_t>(task_id);
  memcpy( m_dataSize, reinterpret_cast<const uint32_t*>( data_size ), 3 * sizeof(uint32_t) );
  memcpy( m_nBlocks, reinterpret_cast<const uint32_t*>( n_blocks ), 3 * sizeof(uint32_t) );
  memcpy( m_low, reinterpret_cast<const uint32_t*>( low ), 3 * sizeof(uint32_t) );
  memcpy( m_high, reinterpret_cast<const uint32_t*>( high ), 3 * sizeof(uint32_t) );
  m_fanin = static_cast<uint32_t>( fanin );

  // Store the local data as a static pointer so that it could be passed later to
  // computeSegmentation function -- probably a better solution is needed
  sLocalData = data_ptr;

  m_radices = radix_v;

  StatisticsMap::sRequestedStats.resize( stat_types_v.size() );
  for( uint32_t i = 0; i < StatisticsMap::sRequestedStats.size(); ++i )
  {
    StatisticsMap::sRequestedStats[i] = Statistics::StreamingStatisticType( stat_types_v[i] );
  }
}


//-----------------------------------------------------------------------------
// -- end ParallelMergeTree --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
///
/// BFlowPmt Filter
///
//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowPmt::declare_interface(conduit::Node &i) 
{
  i["type_name"] = "bflow_pmt";
  i["port_names"].append() = "in";
  i["output_port"] = "true";  // true -- means filter, false -- means extract
}

//-----------------------------------------------------------------------------

//#define INPUT_SCALAR

void ascent::runtime::filters::BFlowPmt::execute() 
{
  if(!input(0).check_type<DataObject>())
  {
      ASCENT_ERROR("BabelFlow filter requires a DataObject");
  }

  int world_rank = 0;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm world_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(world_comm, &world_rank);
#endif

#ifdef BFLOW_PMT_DEBUG     // DEBUG printouts
  {
    auto in = input<DataObject>(0)->as_node();
    auto itr_dnode = in->children();
    while(itr_dnode.has_next())
    {
      auto& data_node = itr_dnode.next();
      std::string cld_dname = data_node.name();
      //std::cout << " dnode name " <<cld_dname  << std::endl; //<< ": " << cld.to_json()

      conduit::NodeIterator itr = data_node["fields/"].children();
      while(itr.has_next())
      {
            conduit::Node &cld = itr.next();
            std::string cld_name = itr.name();
            std::cout << "\tname " <<cld_name  << std::endl; //<< ": " << cld.to_json()
      }
      
      std::cout << world_rank<< ": dnode name " <<cld_dname<<" coordtype " << data_node["coordsets/coords/type"].as_string() 
      <<  " uniform " << data_node.has_path("coordsets/coords/spacing/x") << std::endl;
      if (data_node.has_path("coordsets/coords/spacing/dx"))
        data_node["coordsets/coords/spacing"].print();
      
    }
  }
  // MPI_Barrier(world_comm);
  // return;
#endif

  // connect to the input port and get the parameters
  DataObject *d_input = input<DataObject>(0);
  std::shared_ptr<conduit::Node> n_input = d_input->as_node();

  conduit::Node p = params();
  auto *in = n_input.get();
  
  auto &data_node = in->children().next();

  int color = 0;

  int uniform_color = 0;

  // check if coordset uniform
  if(data_node.has_path("coordsets/coords/type"))
  {
    std::string coordSetType = data_node["coordsets/coords/type"].as_string();
    if (coordSetType != "uniform")
    {
        uniform_color=0;
        // error
        //ASCENT_ERROR("BabelFlow filter currenlty only works with uniform grids");
    }
    else{
      uniform_color = 1;
    }
  }
  else
    ASCENT_ERROR("BabelFlow filter could not find coordsets/coords/type");

  //std::cout << world_rank << ": " << data_node["coordsets/coords/type"].as_string() << " color " << color <<std::endl;

  // Decide which uniform grid to work on (default 0, the finest spacing)
  double selected_spacing = 0;

  MPI_Comm uniform_comm;
  MPI_Comm_split(world_comm, uniform_color, world_rank, &uniform_comm);
  int uniform_rank, uniform_comm_size;
  MPI_Comm_rank(uniform_comm, &uniform_rank);
  MPI_Comm_size(uniform_comm, &uniform_comm_size);

  if(uniform_color){
    double myspacing = 0;
    
    // uniform grid should not have spacing as {x,y,z}
    // this is a workaround to support old Ascent dataset using {x,y,z}
    if(data_node.has_path("coordsets/coords/spacing/x"))
      myspacing = data_node["coordsets/coords/spacing/x"].to_float64();
    else if(data_node.has_path("coordsets/coords/spacing/dx"))
      myspacing = data_node["coordsets/coords/spacing/dx"].to_float64();
    
    std::vector<double> uniform_spacing(uniform_comm_size);
    
    MPI_Allgather(&myspacing, 1, MPI_DOUBLE, uniform_spacing.data(), 1, MPI_DOUBLE, uniform_comm);
    
    std::sort(uniform_spacing.begin(), uniform_spacing.end());
    std::unique(uniform_spacing.begin(), uniform_spacing.end());
    
    if(p.has_path("ugrid_select")) 
      selected_spacing = *std::next(uniform_spacing.begin(), p["ugrid_select"].as_int64());
    else
      selected_spacing = *std::next(uniform_spacing.begin(), 0);
    
    color = fabs(myspacing - selected_spacing) < EPSILON;
    
    //std::cout << "Selected spacing "<< selected_spacing << " rank " << world_rank << " contributing " << color <<"\n";
  }

  MPI_Barrier(uniform_comm);

  MPI_Comm comm;
  MPI_Comm_split(uniform_comm, color, uniform_rank, &comm);

  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  conduit::Node& fields_root_node = data_node["fields"];
  conduit::Node& field_node = fields_root_node[p["field"].as_string()];

  conduit::DataArray<double> array_mag = field_node["values"].as_float64_array();

  if(color) {
    //std::cout << rank << ": comm size " << comm_size << " color " << color << std::endl;
    //data_node["coordsets/coords"].print();
    //data_node["topologies"].print();

    const int ndims = data_node.has_path("coordsets/coords/dims/k") ? 3 : 2;

    // NOTE: when field is a vector the coords/spacing has dx/dy/dz
    int32_t dims[3] = {1, 1, 1};
    double spacing[3] = {1, 1, 1};
    double origin[3] = {0, 0, 0};
    
    dims[0] = data_node["coordsets/coords/dims/i"].value();
    dims[1] = data_node["coordsets/coords/dims/j"].value();
    if(ndims > 2)
      dims[2] = data_node["coordsets/coords/dims/k"].value();

    if(data_node.has_path("coordsets/coords/spacing")){

      // uniform grid should not have spacing as {x,y,z}
      // this is a workaround to support old Ascent dataset using {x,y,z}
      // TODO: we should probably remove {x,y,z} from the dataset
      if(data_node.has_path("coordsets/coords/spacing/x")){
        spacing[0] = data_node["coordsets/coords/spacing/x"].to_float64();
        spacing[1] = data_node["coordsets/coords/spacing/y"].to_float64();
        if(ndims > 2)
          spacing[2] = data_node["coordsets/coords/spacing/z"].to_float64();

        data_node["coordsets/coords/spacing/dx"] = spacing[0];
        data_node["coordsets/coords/spacing/dy"] = spacing[1];
        data_node["coordsets/coords/spacing/dz"] = spacing[2];
      }
      else if(data_node.has_path("coordsets/coords/spacing/dx")){
        spacing[0] = data_node["coordsets/coords/spacing/dx"].to_float64();
        spacing[1] = data_node["coordsets/coords/spacing/dy"].to_float64();
        if(ndims > 2)
          spacing[2] = data_node["coordsets/coords/spacing/dz"].to_float64();
      }
      
    }


    origin[0] = data_node["coordsets/coords/origin/x"].to_float64();
    origin[1] = data_node["coordsets/coords/origin/y"].to_float64();
    if(ndims > 2)
      origin[2] = data_node["coordsets/coords/origin/z"].to_float64();

    // Inputs of PMT assume 3D dataset
    int32_t low[3] = {0,0,0};
    int32_t high[3] = {0,0,0};

    int32_t global_low[3] = {0,0,0};
    int32_t global_high[3] = {0,0,0};
    int32_t data_size[3] = {1,1,1};

    int32_t n_blocks[3] = {1,1,1};
    
    if(p.has_path("in_ghosts")) {
      int64_t* in_ghosts = p["in_ghosts"].as_int64_ptr();
      for(int i=0;i< 6;i++) {
        ParallelMergeTree::o_ghosts[i] = (uint32_t)in_ghosts[i];
      }
    }

    for(int i=0; i<ndims; i++){
      low[i] = origin[i]/spacing[i];
      high[i] = low[i] + dims[i] -1;
      
#ifdef ASCENT_MPI_ENABLED
      MPI_Allreduce(&low[i], &global_low[i], 1, MPI_INT, MPI_MIN, comm);
      MPI_Allreduce(&high[i], &global_high[i], 1, MPI_INT, MPI_MAX, comm);
#else
      global_low[i] = low[i];
      global_high[i] = high[i];
#endif
      data_size[i] = global_high[i]-global_low[i]+1;
      // normalize box
      low[i] -= global_low[i];
      high[i] = low[i] + dims[i] -1;

      n_blocks[i] = std::ceil(data_size[i]*1.0/dims[i]);
    }
    // int32_t dn[3], dp[3];
    // dn[0] = (low[0] == data_size[0]) ? 0 : ParallelMergeTree::o_ghosts[0] - ParallelMergeTree::n_ghosts[0];
    // dn[1] = (low[1] == data_size[1]) ? 0 : ParallelMergeTree::o_ghosts[1] - ParallelMergeTree::n_ghosts[1];
    // dn[2] = (low[2] == data_size[2]) ? 0 : ParallelMergeTree::o_ghosts[2] - ParallelMergeTree::n_ghosts[2];
    // dp[0] = (high[0] == data_size[0] - 1) ? 0 : ParallelMergeTree::o_ghosts[3] - ParallelMergeTree::n_ghosts[3];
    // dp[1] = (high[1] == data_size[1] - 1) ? 0 : ParallelMergeTree::o_ghosts[4] - ParallelMergeTree::n_ghosts[4];
    // dp[2] = (high[2] == data_size[2] - 1) ? 0 : ParallelMergeTree::o_ghosts[5] - ParallelMergeTree::n_ghosts[5];

    // for(int i=0; i<ndims; i++){
    //   low[i] += dn[i];
    //   high[i] -= dp[i];
    // }

    //std::cout << p["field"].as_string() <<std::endl;

    // get the data handle
    FunctionType* array = reinterpret_cast<FunctionType *>(array_mag.data_ptr());

    //conduit::DataArray<float> array = data_node[p["data_path"].as_string()].as_float32_array();

#ifdef BFLOW_PMT_DEBUG
    {
      std::stringstream ss;
      ss << "block_" << dims[0] << "_" << dims[1] << "_" << dims[2] <<"_low_"<< low[0] << "_"<< low[1] << "_"<< low[2] << ".raw";
      std::fstream fil;
      fil.open(ss.str().c_str(), std::ios::out | std::ios::binary);
      fil.write(reinterpret_cast<char *>(array), (dims[0]*dims[1]*dims[2])*sizeof(FunctionType));
      fil.close();
      //MPI_Barrier(comm);
    }
#endif

    // int32_t *data_size = p["data_size"].as_int32_ptr();
    // int32_t *low = p["low"].as_int32_ptr();
    // int32_t *high = p["high"].as_int32_ptr();
    // int32_t *n_blocks = p["n_blocks"].as_int32_ptr();
    int32_t task_id = rank;
    // if (!p.has_child("task_id") || p["task_id"].as_int32() == -1) {
    //   MPI_Comm_rank(comm, &task_id);
    // } else {
    //   task_id = p["task_id"].as_int32();
    // }
    int64_t fanin = p["fanin"].as_int64();
    FunctionType threshold = p["threshold"].as_float64();
    int64_t gen_field = p["gen_segment"].as_int64();

    std::vector<uint32_t> radix_v(1);
    radix_v[0] = comm_size;
    if( p.has_path("radices") )
    {
      conduit::DataArray<int64_t> radices_arr = p["radices"].as_int64_array();
      radix_v.resize(radices_arr.number_of_elements());
      for( uint32_t i = 0; i < radix_v.size(); ++i ) radix_v[i] = (uint32_t)radices_arr[i];
    }

    std::vector<uint32_t> stream_stat_v;
    if( p.has_path("stream_stat_types") )
    {
      assert( !gen_field );
      conduit::DataArray<int64_t> stat_types_arr = p["stream_stat_types"].as_int64_array();
      stream_stat_v.resize( stat_types_arr.number_of_elements() );
      for( uint32_t i = 0; i < stream_stat_v.size(); ++i ) stream_stat_v[i] = (uint32_t)stat_types_arr[i];
    }

    // create ParallelMergeTree instance and run
    ParallelMergeTree pmt(array, 
                          task_id,
                          data_size,
                          n_blocks,
                          low, high,
                          fanin, threshold, comm,
                          radix_v, stream_stat_v);

    ParallelMergeTree::s_data_size[0] = data_size[0];
    ParallelMergeTree::s_data_size[1] = data_size[1];
    ParallelMergeTree::s_data_size[2] = data_size[2];

#ifdef BFLOW_PMT_DEBUG
    // Reduce all of the local sums into the global sum
    {
      std::stringstream ss;
      ss << "data_params_" << rank << ".txt";
      std::ofstream ofs(ss.str());
      ofs << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;
      ofs << "spacing " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl;
      ofs << "dims " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
      ofs << "low " << low[0] << " " << low[1] << " " << low[2] << std::endl;
      ofs << "high " << high[0] << " " << high[1] << " " << high[2] << std::endl;
      
      uint32_t dnx, dny, dnz, dpx, dpy, dpz;
      uint32_t loc_low[3] = {(uint32_t)low[0], (uint32_t)low[1], (uint32_t)low[2]};
      uint32_t loc_high[3] = {(uint32_t)high[0], (uint32_t)high[1], (uint32_t)high[2]};
      ParallelMergeTree::ComputeGhostOffsets(loc_low, loc_high, dnx, dny, dnz, dpx, dpy, dpz);
      
      ofs << "ghosts offsets " << dnx << " " << dny << " " << dnz << " "
                               << dpx << " " << dpy << " " << dpz << std::endl;
                                                
      if(rank==0){
        ofs << "*data_size " << data_size[0] << " " << data_size[1] << " " << data_size[2] << std::endl;
        ofs << "*global_low " << global_low[0] << " " << global_low[1] << " " << global_low[2] << std::endl;
        ofs << "*global_high " << global_high[0] << " " << global_high[1] << " " << global_high[2] << std::endl;
        ofs << "*n_blocks " << n_blocks[0] << " " << n_blocks[1] << " " << n_blocks[2] << std::endl;
      }

      ofs.close();
    }
    MPI_Barrier(comm);
    //data_node["fields/"].print();
    //std::cout<<"----------------------"<<std::endl;
#endif
    
    pmt.Initialize();
    pmt.Execute();

    MPI_Barrier(comm);

    if (gen_field) {
      // Generate new field 'segment'
      data_node["fields/segment/association"] = field_node["association"].as_string();
      data_node["fields/segment/topology"] = field_node["topology"].as_string();

      // New field data
      std::vector<FunctionType> seg_data(dims[0]*dims[1]*dims[2], 0.f);

      pmt.ExtractSegmentation(seg_data.data());

      data_node["fields/segment/values"].set(seg_data);

      // DEBUG -- write raw segment data to disk
#ifdef BFLOW_PMT_DEBUG
      {
        std::stringstream ss;
        ss << "segment_data_" << rank << "_" << dims[0] << "_" << dims[1] << "_" << dims[2] <<"_low_"<< low[0] << "_"<< low[1] << "_"<< low[2] << ".raw";
        std::ofstream bofs(ss.str(), std::ios::out | std::ios::binary);
        bofs.write(reinterpret_cast<char *>(seg_data.data()), dims[0]*dims[1]*dims[2]*sizeof(FunctionType));
        bofs.close();
      }
#endif

      // DEBUG -- verify modified BP node with 'segment' field
      // conduit::Node info;
      // if (conduit::blueprint::verify("mesh", *in, info))
      //   std::cout << "BP with new field verify -- successful" << std::endl;
      // else
      //   std::cout << "BP with new field verify -- failed" << std::endl;
      
      d_input->reset_vtkh_collection();
    }
  }
  else {
    // If needed add the new field with GNULL data so that 
    // we're consistent with other ranks that do have data
    int64_t gen_field = p["gen_segment"].as_int64();
    if (gen_field) {
      data_node["fields/segment/association"] = field_node["association"].as_string();
      data_node["fields/segment/topology"] = field_node["topology"].as_string();
      conduit::DataArray<double> array_mag = field_node["values"].as_float64_array();

      std::vector<FunctionType> seg_data(array_mag.number_of_elements(), (FunctionType)GNULL);

      data_node["fields/segment/values"].set(seg_data);
      d_input->reset_vtkh_collection();
    }
  }
  
#ifdef ASCENT_MPI_ENABLED
  MPI_Barrier(world_comm);
#endif
  
  set_output<DataObject>(d_input);
}

//-----------------------------------------------------------------------------

bool ascent::runtime::filters::BFlowPmt::verify_params(const conduit::Node &params, conduit::Node &info) 
{
  info.reset();

  bool res = true;

  res &= check_string("field", params, info, true);
  res &= check_numeric("fanin", params, info, true);
  res &= check_numeric("threshold", params, info, true);
  res &= check_numeric("gen_segment", params, info, true);
  res &= check_numeric("ugrid_select", params, info, false);
  
  return res;
}
