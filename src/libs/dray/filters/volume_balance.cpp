#include <dray/filters/volume_balance.hpp>

#include <dray/dray.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/error_check.hpp>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdlib>

#include <dray/filters/subset.hpp>
#include <dray/filters/redistribute.hpp>

#include <dray/data_model/device_mesh.hpp>
#include <dray/dispatcher.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#ifdef DRAY_MPI_ENABLED
#include<mpi.h>
#endif

namespace dray
{

namespace detail
{

template<typename MeshElement>
void mask_cells(UnstructuredMesh<MeshElement> &mesh,
                int32 comp,
                float32 min_coord,
                float32 max_coord,
                Array<int32> &mask)
{
  DRAY_LOG_OPEN("mask_cells");

  const int32 num_elems = mesh.cells();
  DeviceMesh<MeshElement> device_mesh(mesh);

  mask.resize(num_elems);
  int32 *mask_ptr = mask.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    int32 mask_val = 0;
    MeshElement element = device_mesh.get_elem(i);
    AABB<3> elem_bounds;
    element.get_bounds(elem_bounds);
    float32 center = elem_bounds.m_ranges[comp].center();
    if(center >= min_coord && center < max_coord)
    {
      mask_val = 1;
    }

    mask_ptr[i] = mask_val;
  });


  DRAY_LOG_CLOSE();
}

template<typename MeshElement>
void aabb_cells(UnstructuredMesh<MeshElement> &mesh,
                Array<AABB<3>> &aabbs)
{
  DRAY_LOG_OPEN("aabb_cells");

  const int32 num_elems = mesh.cells();
  DeviceMesh<MeshElement> device_mesh(mesh);

  aabbs.resize(num_elems);
  AABB<3> *aabb_ptr = aabbs.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    int32 mask_val = 0;
    MeshElement element = device_mesh.get_elem(i);
    AABB<3> elem_bounds;
    element.get_bounds(elem_bounds);
    aabb_ptr[i] = elem_bounds;
  });

  DRAY_LOG_CLOSE();
}

template<typename MeshElement>
void volume_sum(UnstructuredMesh<MeshElement> &mesh,
                float32 &sum)
{
  DRAY_LOG_OPEN("volume_sum");

  const int32 num_elems = mesh.cells();
  DeviceMesh<MeshElement> device_mesh(mesh);
  RAJA::ReduceSum<reduce_policy, float32> vsum(0.);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    int32 mask_val = 0;
    MeshElement element = device_mesh.get_elem(i);
    AABB<3> elem_bounds;
    element.get_bounds(elem_bounds);
    vsum += elem_bounds.volume();
  });

  sum = vsum.get();
  DRAY_LOG_CLOSE();
}

struct MaskFunctor
{
  Array<int32> m_mask;
  int32 m_dim;
  float32 m_min;
  float32 m_max;
  MaskFunctor(const int32 dim,
              const float32 min,
              const float32 max)
    : m_dim(dim),
      m_min(min),
      m_max(max)
  {
  }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    mask_cells(mesh, m_dim, m_min, m_max, m_mask);
  }
};

struct AABBFunctor
{
  Array<AABB<3>> m_aabbs;
  AABBFunctor()
  {
  }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    aabb_cells(mesh, m_aabbs);
  }
};

struct VolumeSumFunctor
{
  float32 m_sum;
  VolumeSumFunctor()
  {
  }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    volume_sum(mesh, m_sum);
  }
};


void split(std::vector<float32> &pieces,
           DataSet &dataset,
           Collection &col)
{
  DRAY_LOG_OPEN("split");

  AABB<3> bounds = dataset.mesh()->bounds();
  const int32 max_comp = bounds.max_dim();
  float32 length = bounds.m_ranges[max_comp].length();
  const float32 volume = bounds.volume();
  DRAY_LOG_ENTRY("box_length", length);
  DRAY_LOG_ENTRY("box_volume", bounds.volume());
  DRAY_LOG_ENTRY("box_min", bounds.m_ranges[max_comp].min());


  const int32 num_pieces = pieces.size();
  float32 total = 0;
  for(int32 i = 0; i < num_pieces; ++i)
  {
    total += pieces[i];
    DRAY_LOG_ENTRY("split", pieces[i]);
  }

  std::vector<float32> normalized_volume;
  normalized_volume.resize(num_pieces);

  // normalize
  for(int32 i = 0; i < num_pieces; ++i)
  {
    pieces[i] /= total;
    normalized_volume[i] = pieces[i];
    DRAY_LOG_ENTRY("normalized_length", pieces[i]);
  }
  // prefix sum
  for(int32 i = 1; i < num_pieces; ++i)
  {
    normalized_volume[i] += normalized_volume[i-1];
    DRAY_LOG_ENTRY("normalized_sum_length", normalized_volume[i]);
  }

  // scale
  for(int32 i = 0; i < num_pieces; ++i)
  {
    pieces[i] *= length;
    DRAY_LOG_ENTRY("piece_length", pieces[i]);
  }

  AABBFunctor aabb_func;
  dispatch(dataset.mesh(), aabb_func);
  Array<AABB<3>> aabbs = aabb_func.m_aabbs;
  AABB<3> *aabbs_ptr = aabbs.get_host_ptr();

  std::stable_sort(aabbs_ptr, aabbs_ptr + aabbs.size(),
       [&max_comp](const AABB<3> &i1, const AABB<3> &i2)
       {
         return i1.center()[max_comp] < i2.center()[max_comp];
       });

  float32 aabb_tot_vol = 0;
  for(int i = 0; i < aabbs.size(); ++i)
  {
    aabb_tot_vol += aabbs_ptr[i].volume();
  }

  DRAY_LOG_ENTRY("aabbs_vol", aabb_tot_vol);
  DRAY_LOG_ENTRY("aabbs_size", aabbs.size());

  std::vector<int32> divs;
  divs.resize(num_pieces);
  float32 curr_vol = 0;
  float32 piece_idx = 0;

  for(int32 i = 0; i < aabbs.size(); ++i)
  {
    curr_vol += aabbs_ptr[i].volume() / aabb_tot_vol;
    if(curr_vol >= normalized_volume[piece_idx])
    {
      divs[piece_idx] = i;
      DRAY_LOG_ENTRY("div_normal", curr_vol);
      DRAY_LOG_ENTRY("div_normal_idx", i);
      piece_idx++;
    }
  }

  divs[num_pieces - 1] = aabbs.size() - 1;

  for(int32 i = 0; i < num_pieces; ++i)
  {
    DRAY_LOG_ENTRY("div", divs[i]);;
  }

  std::vector<float32> ranges;
  ranges.resize(num_pieces+1);
  ranges[0] = bounds.m_ranges[max_comp].min();
  // bump it by an epsilon
  ranges[num_pieces] = bounds.m_ranges[max_comp].max() + length * 1e-3;

  //for(int32 i = 0; i < num_pieces - 1; ++i)
  //{
  //  ranges[i+1] = ranges[i] + pieces[i];
  //}

  for(int32 i = 0; i < num_pieces-1; ++i)
  {
    int32 idx = divs[i];
    ranges[i+1] = aabbs_ptr[idx].center()[max_comp] + length * 1e-3;
  }


  for(int32 i = 0; i < num_pieces+1; ++i)
  {
    DRAY_LOG_ENTRY("range", ranges[i]);
  }

  for(int32 i = 0; i < ranges.size() - 1; ++i)
  {
    MaskFunctor func(max_comp,ranges[i],ranges[i+1]);
    dispatch(dataset.mesh(), func);
    Subset subset;
    DataSet piece = subset.execute(dataset, func.m_mask);
    DRAY_LOG_ENTRY("piece_length",ranges[i+1] - ranges[i]);
    DRAY_LOG_ENTRY("target",normalized_volume[i]);
    DRAY_LOG_ENTRY("efficiency",normalized_volume[i] / (piece.mesh()->bounds().volume() / volume));

    if(piece.mesh()->cells() > 0)
    {
      col.add_domain(piece);
    }
  }
  DRAY_LOG_CLOSE();
}


}//namespace detail

VolumeBalance::VolumeBalance()
  : m_use_prefix(true),
    m_piece_factor(0.9f),
    m_threshold(2.0)
{
}

float32
VolumeBalance::schedule_prefix(std::vector<float32> &rank_volumes,
                               std::vector<int32> &global_counts,
                               std::vector<int32> &global_offsets,
                               std::vector<float32> &global_volumes,
                               std::vector<int32> &src_list,
                               std::vector<int32> &dest_list)
{
  const int32 size = rank_volumes.size();
  const int32 global_size = global_volumes.size();

  std::vector<int32> random;
  random.resize(global_size);
  for(int32 i = 0; i < global_size; ++i)
  {
    random[i] = i;
  }

  std::shuffle(random.begin(), random.end(), std::default_random_engine(0));
  

  std::vector<float32> prefix_sum;
  prefix_sum.resize(global_size);
  prefix_sum[0] = global_volumes[random[0]];
  for(int32 i = 1; i < global_size; ++i)
  {
    prefix_sum[i] = global_volumes[random[i]] + prefix_sum[i-1];
  }

  // init everthing to stay in place
  for(int i = 0; i < size; ++i)
  {
    for(int32 t = 0; t < global_counts[i]; ++t)
    {
      const int32 offset = global_offsets[i];
      src_list[offset+t] = i;
      dest_list[offset+t] = i;
    }
  }

  float32 sum = 0;
  float32 max_val = 0;
  for(int32 i = 0; i < size; ++i)
  {
    sum += rank_volumes[i];
    max_val = std::max(max_val, rank_volumes[i]);
    rank_volumes[i] = 0;
  }

  // target
  const float32 ave = sum / float32(size);
  DRAY_LOG_ENTRY("prefix_ave",ave);

  int32 pos = 0;
  for(int32 i = 0; i < size - 1; ++i)
  {
    int32 idx = pos;
    if(idx >= global_size - 1)
    {
      break;
    }

    float32 target = ave * float32(i+1);

    while(idx < global_size - 2 && prefix_sum[idx+1] < target)
    {
      idx++;
    }

    float32 d1 = std::abs(target - prefix_sum[idx]);
    float32 d2 = std::abs(target - prefix_sum[idx+1]);

    if(d2 < d1)
    {
      idx++;
    }

    for(int32 domain = pos; domain <= idx; ++domain)
    {
      rank_volumes[i] += global_volumes[random[domain]];
      dest_list[random[domain]] = i;
    }
    pos = idx + 1;
  }

  for(int32 i = pos; i < global_size; ++i)
  {
    dest_list[random[i]] = size - 1;
    rank_volumes[size -1] += global_volumes[random[i]];
  }


  float32 max_after = 0;
  for(int32 i = 0; i < size; ++i)
  {
    max_after = std::max(max_after,rank_volumes[i]);
  }

  return max_after / max_val;
}

float32
VolumeBalance::schedule_blocks(std::vector<float32> &rank_volumes,
                               std::vector<int32> &global_counts,
                               std::vector<int32> &global_offsets,
                               std::vector<float32> &global_volumes,
                               std::vector<int32> &src_list,
                               std::vector<int32> &dest_list)
{
  const int32 size = rank_volumes.size();
  const int32 global_size = global_volumes.size();

  float32 max_chunk = 0;
  float32 max_chunk_rank = 0;
  // init everthing to stay in place
  for(int i = 0; i < size; ++i)
  {
    for(int32 t = 0; t < global_counts[i]; ++t)
    {
      const int32 offset = global_offsets[i];
      src_list[offset+t] = i;
      dest_list[offset+t] = i;
      if(max_chunk < global_volumes[offset+t])
      {
        max_chunk = global_volumes[offset+t];
        max_chunk_rank = i;
      }
    }
  }

  std::vector<int32> idx(size);
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
       [&rank_volumes](int32 i1, int32 i2)
       {
         return rank_volumes[i1] < rank_volumes[i2];
       });

  float32 sum = 0;
  std::vector<int32> given_chunks;
  given_chunks.resize(size);
  for(int32 i = 0; i < size; ++i)
  {
    given_chunks[i] = 0;
    sum += rank_volumes[i];
  }
  // target
  const float32 ave = sum / float32(size);
  DRAY_LOG_ENTRY("average", ave);
  DRAY_LOG_ENTRY("max_chunk", max_chunk);
  DRAY_LOG_ENTRY("max_chunk_rank", max_chunk_rank);

  int32 giver = size - 1;
  int32 taker = 0;

  float32 eps = rank_volumes[idx[giver]] * 1e-3;
  float32 max_val = rank_volumes[idx[giver]];

  float32 target = ave * 1.1;
  conduit::Node rounds;
  while(giver > taker)
  {
    int32 giver_idx = idx[giver];
    int32 giver_chunks = global_counts[giver_idx];

    int32 taker_idx = idx[taker];

    float32 giver_work = rank_volumes[giver_idx];
    float32 taker_work = rank_volumes[taker_idx];

    float32 giver_dist = giver_work - target;
    float32 taker_dist = target - rank_volumes[taker_idx];

    int32 chunk_idx = given_chunks[giver_idx];
    int32 chunk_offset = global_offsets[giver_idx];
    float32 chunk_size = global_volumes[chunk_offset + chunk_idx];
    float32 giver_result = giver_work - chunk_size;
    float32 taker_result = taker_work + chunk_size;
    bool helps = std::max(giver_result, taker_result) < std::max(giver_work, taker_work);

    int32 take_chunks = given_chunks[taker_idx];
    bool do_it = take_chunks == 0 || taker_result < target + eps;

    if(helps && do_it)
    {
      rank_volumes[giver_idx] -= chunk_size;
      rank_volumes[taker_idx] += chunk_size;
      dest_list[chunk_offset + chunk_idx] = taker_idx;
      given_chunks[giver_idx]++;
      given_chunks[taker_idx]++;

    }
    else
    {
      // this give didn't help so move on
      taker++;
    }

    int32 remaining_chunks = giver_chunks - given_chunks[giver_idx];
    // does giver have more?
    if(rank_volumes[giver_idx] <= target + eps || remaining_chunks < 2)
    {
      giver--;
    }
    std::vector<float32> current;
    current.resize(size);
    for(int i = 0; i < size; ++i)
    {
      current[i] = rank_volumes[idx[i]];
    }
    conduit::Node &round = rounds.append();
    round.set(current);
  }
  if(dray::mpi_rank() == 0)
  {
    rounds.save("rounds", "json");
  }

  float32 max_after = 0;
  for(int32 i = 0; i < size; ++i)
  {
    max_after = std::max(max_after,rank_volumes[i]);
  }
  return max_after / max_val;
}

Collection
VolumeBalance::chopper(float32 piece_size,
                       std::vector<float32> &sizes,
                       Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    float32 psize = sizes[i];
    DataSet dataset = collection.domain(i);
    int32 num_pieces = psize / piece_size;
    DRAY_LOG_ENTRY("chopper_pieces", num_pieces);
    if(num_pieces > 1)
    {
      std::vector<float32> pieces;
      for(int32 p = 0; p < num_pieces; ++p)
      {
        pieces.push_back(piece_size);
      }
      float32 rem = psize - num_pieces * piece_size;
      if(rem > psize * 1.e-5 )
      {
        pieces.push_back(rem);
      }
      detail::split(pieces, dataset, res);
    }
    else
    {
      res.add_domain(dataset);
    }

  }

  return res;
}

void VolumeBalance::allgather(std::vector<float32> &local_volumes,
                              const int32 global_size,
                              std::vector<float32> &rank_volumes,
                              std::vector<int32> &global_counts,
                              std::vector<int32> &global_offsets,
                              std::vector<float32> &global_volumes)
{

#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  const int32 comm_size = dray::mpi_size();
  const int32 rank = dray::mpi_rank();

  const int32 local_size = local_volumes.size();
  global_counts.resize(comm_size);
  MPI_Allgather(&local_size, 1, MPI_INT, &global_counts[0], 1, MPI_INT, mpi_comm);

  global_offsets.resize(comm_size);
  global_offsets[0] = 0;

  for(int i = 1; i < comm_size; ++i)
  {
    global_offsets[i] = global_offsets[i-1] + global_counts[i-1];
  }

  global_volumes.resize(global_size);
  rank_volumes.resize(comm_size);

  float32 total_volume = 0;
  for(int32 i = 0; i < local_volumes.size(); ++i)
  {
    total_volume += local_volumes[i];
  }

  MPI_Allgather(&total_volume,1, MPI_FLOAT,&rank_volumes[0],1,MPI_FLOAT,mpi_comm);

  MPI_Allgatherv(&local_volumes[0],
                 local_size,
                 MPI_FLOAT,
                 &global_volumes[0],
                 &global_counts[0],
                 &global_offsets[0],
                 MPI_FLOAT,
                 mpi_comm);
#endif
}

float32
VolumeBalance::volumes(Collection &collection,
                       Camera &camera,
                       int32 samples,
                       std::vector<float32> &volumes)
{
  const int32 local_doms = collection.local_size();
  volumes.resize(local_doms);

  AABB<> sample_bounds = collection.bounds();
  float32 mag = (sample_bounds.max() - sample_bounds.min()).magnitude();
  const float32 sample_distance = mag / float32(samples);
  const float32 sample_volume = sample_distance * sample_distance * sample_distance;

  float32 total_volume = 0;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dataset = collection.domain(i);
    AABB<3> bounds = dataset.mesh()->bounds();

    //float32 samples = bounds.m_ranges[bounds.max_dim()].length() / sample_distance;

    float32 volume = bounds.volume();
    // alternative volume calculation
    //detail::VolumeSumFunctor vfunc;
    //dispatch(dataset.mesh(), vfunc);
    //float32 volume = vfunc.m_sum;

    float32 pixels = static_cast<float32>(camera.subset_size(bounds));
    float32 normalized_pixels = pixels / float32(camera.get_width() + camera.get_height());
    volumes[i] = volume * normalized_pixels;
    //volumes[i] = volume * pixels;
    volumes[i] = (volume/sample_volume) * pixels;
    total_volume += volumes[i];
  }
  return total_volume;
}

Collection
VolumeBalance::execute(Collection &collection, Camera &camera, int32 samples)
{
#ifndef DRAY_MPI_ENABLED
  // don't load balance if we don't have mpi
  return collection;
#endif
  DRAY_LOG_OPEN("volume_balance");
  Collection res;

  const int32 local_doms = collection.local_size();

  std::vector<float32> local_volumes;

  float32 total_volume = volumes(collection, camera, samples, local_volumes);

  DRAY_LOG_ENTRY("local_volume", total_volume);

#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  const int32 comm_size = dray::mpi_size();
  const int32 rank = dray::mpi_rank();
  const int32 global_size = collection.size();

  float32 global_volume = 0;
  MPI_Allreduce(&total_volume, &global_volume,1, MPI_FLOAT, MPI_SUM, mpi_comm);

  float32 global_ave = global_volume / float32(comm_size);


  float32 imbalance = total_volume / global_ave;
  float32 max_imbalance;
  MPI_Allreduce(&imbalance, &max_imbalance,1, MPI_FLOAT, MPI_MAX, mpi_comm);

  if(max_imbalance < m_threshold)
  {
    return collection;
  }

  DRAY_LOG_ENTRY("global_average", global_ave);
  float32 piece_size = m_piece_factor * global_ave;

  DRAY_LOG_ENTRY("piece_size", piece_size);

  Collection pre_chopped = chopper(piece_size, local_volumes, collection);

  const int32 chopped_size = pre_chopped.size();
  std::vector<float32> rank_volumes;
  std::vector<int32> global_counts;
  std::vector<int32> global_offsets;
  std::vector<float32> global_volumes;

  std::vector<float32> chopped_local;
  float32 total_chopped = volumes(pre_chopped, camera, samples, chopped_local);
  for(int32 i = 0; i < chopped_local.size(); ++i)
  {
    DRAY_LOG_ENTRY("chopped_volume ", chopped_local[i]);
  }

  allgather(chopped_local,
            chopped_size,
            rank_volumes,
            global_counts,
            global_offsets,
            global_volumes);

  std::vector<int32> src_list;
  std::vector<int32> dest_list;
  src_list.resize(chopped_size);
  dest_list.resize(chopped_size);

  float32 ratio;
  if(m_use_prefix)
  {
    ratio = schedule_prefix(rank_volumes,
                            global_counts,
                            global_offsets,
                            global_volumes,
                            src_list,
                            dest_list);
  }
  else
  {
    ratio = schedule_blocks(rank_volumes,
                            global_counts,
                            global_offsets,
                            global_volumes,
                            src_list,
                            dest_list);
  }

  DRAY_LOG_ENTRY("ratio",ratio);

  Redistribute redist;
  res = redist.execute(pre_chopped, src_list, dest_list);
  DRAY_LOG_ENTRY("result_local_domains", res.local_size());
#endif

  std::vector<float32> res_local;
  total_volume = volumes(res, camera, samples, res_local);
  for(int32 i = 0; i < res.local_size(); ++i)
  {
    DRAY_LOG_ENTRY("result_domain_volume", res_local[i]);
  }
  DRAY_LOG_ENTRY("result_local_volume", total_volume);
  DRAY_LOG_CLOSE();
  return res;
}

void VolumeBalance::prefix_balancing(bool on)
{
  m_use_prefix = on;
}

void VolumeBalance::piece_factor(float32 size)
{
  if(size <= 0)
  {
    DRAY_ERROR("piece_factor must be greater than zero (piece size is ave * piece_factor");
  }
  m_piece_factor = size;
}

void VolumeBalance::threshold(float32 value)
{
  m_threshold = value;
}

}//namespace dray
