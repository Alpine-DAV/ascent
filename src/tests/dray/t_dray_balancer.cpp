// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/filters/volume_balance.hpp>
#include <cmath>

constexpr int work_size = 144;

const float work[work_size] = {
  60.5193 ,24.4454 ,2.66277 ,1.32776 ,4.16441 ,4.45648
  ,1.77427 ,0.791729 ,1.285 ,0.925656 ,8.28403 ,11.8021
  ,2.05597 ,4.83323 ,33.6583 ,0.639632 ,53.5863 ,55.2289
  ,3.18005 ,1.26828 ,2.34962 ,32.791 ,30.3632 ,52.5045
  ,0.856731 ,0.671662 ,0.367876 ,84.7753 ,2.63224 ,0.489289
  ,4.65277 ,2.46773 ,2.34595 ,3.13008 ,1.70321 ,5.62021 ,16.1559
  ,4.10015 ,1.87419 ,8.14788 ,1.48811 ,4.76692 ,33.488 ,8.83777
  ,78.08 ,8.68155 ,7.57769 ,1.61049 ,60.4503 ,16.3925 ,0.161733
  ,8.74347 ,1.04369 ,4.56351 ,0.163543 ,12.6667 ,7.04864 ,8.62656
  ,0.179388 ,3.04785 ,1.58851 ,37.7146 ,6.85736 ,74.4122 ,17.9546
  ,4.3499 ,6.4384 ,5.41004 ,5.67192 ,0.77412 ,4.27266 ,4.05268
  ,5.35581 ,0.985918 ,3.38658 ,46.0065 ,1.94849 ,1.92502 ,3.86928
  ,0.829491 ,1.55729 ,1.32278 ,0.155082 ,0.889415 ,1.82209 ,1.9594
  ,0.18765 ,1.61068 ,0.265306 ,2.01387 ,1.6037 ,13.5489 ,1.35909
  ,20.2688 ,1.2734 ,1.63218 ,3.49358 ,5.47509 ,2.68348 ,2.79484
  ,2.17933 ,0.349412 ,1.67682 ,3.61832 ,0.43928 ,26.8999 ,0.244606
  ,0.638031 ,1.05137 ,1.22614 ,83.6862 ,4.29085 ,3.93329 ,0.51893
  ,0.198894 ,0.414152 ,9.11406 ,1.69461 ,6.01617 ,4.81977 ,0.880785
  ,0.235686 ,23.669 ,1.4565 ,0.358066 ,0.521558 ,133.848 ,0.548134
  ,17.8261 ,0.198939 ,1.13729 ,4.33412 ,0.508466 ,26.4711 ,121.223
  ,114.762 ,12.635 ,18.9218 ,39.1997 ,36.7212 ,4.1916 ,3.32114
  ,54.9441 ,8.47092 };

void fill_tasks(std::vector<float> &rank_volumes,
                std::vector<int> &global_counts,
                std::vector<int> &global_offsets,
                std::vector<float> &global_volumes,
                float piece_factor = 1)
{
  rank_volumes.resize(work_size);

  float sum = 0;
  for(int i = 0; i < work_size; ++i)
  {
    sum += work[i];
    rank_volumes[i] = work[i];
  }
  float ave = sum / float(work_size);
  float piece_size = piece_factor * ave;

  for(int i = 0; i < work_size; ++i)
  {
    global_offsets.push_back(global_volumes.size());

    float load = work[i];
    if(load < piece_size)
    {
      global_volumes.push_back(load);
      global_counts.push_back(1);
    }
    else
    {
      int num_pieces = load / piece_size;
      int count = num_pieces;
      for(int p = 0; p < num_pieces; ++p)
      {
        global_volumes.push_back(piece_size);
      }
      float rem = load - num_pieces * piece_size;
      if(rem > piece_size * 1e-6 )
      {
        global_volumes.push_back(rem);
        count++;
      }
      global_counts.push_back(count);
    }
  }
}

TEST (dray_balance, dray_balancing_prefix_sum)
{
  std::vector<float> rank_volumes;
  std::vector<int> offsets;
  std::vector<int> counts;
  std::vector<float> volumes;

  fill_tasks(rank_volumes,counts, offsets, volumes,0.75);

  dray::VolumeBalance balancer;

  std::vector<int> src_list;
  std::vector<int> dest_list;
  src_list.resize(volumes.size());
  dest_list.resize(volumes.size());

  float ratio  = balancer.schedule_prefix(rank_volumes,
                                          counts,
                                          offsets,
                                          volumes,
                                          src_list,
                                          dest_list);

  std::cout<<"Resulting ratio "<<ratio<<"\n";
}

TEST (dray_balance, dray_balancing_giver_taker)
{
  std::vector<float> rank_volumes;
  std::vector<int> offsets;
  std::vector<int> counts;
  std::vector<float> volumes;

  fill_tasks(rank_volumes,counts, offsets, volumes, 0.75);
  dray::VolumeBalance balancer;

  std::vector<int> src_list;
  std::vector<int> dest_list;
  src_list.resize(volumes.size());
  dest_list.resize(volumes.size());

  float ratio  = balancer.schedule_blocks(rank_volumes,
                                          counts,
                                          offsets,
                                          volumes,
                                          src_list,
                                          dest_list);
  std::cout<<"Resulting ratio "<<ratio<<"\n";
}
