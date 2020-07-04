//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: holo.cpp
///
//-----------------------------------------------------------------------------
#include <ascent.hpp>
#include <ascent_hola.hpp>

#include <conduit_relay.hpp>

#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef DRAY_ENABLED
#include "colors.hpp"
#endif

std::string output_name;
void usage()
{
  std::cout<<"holo usage: holo high_order.root low_order.root output_name\n";
}

bool validate_dims(const conduit::Node &i1,
                   const conduit::Node &i2,
                   int &width,
                   int &height)
{
  int w1 = i1["coordsets/coords/dims/i"].to_int32() - 1;
  int h1 = i1["coordsets/coords/dims/j"].to_int32() - 1;
  int w2 = i2["coordsets/coords/dims/i"].to_int32() - 1;
  int h2 = i2["coordsets/coords/dims/j"].to_int32() - 1;
  width = w1;
  height = h1;
  return (w1 == w2) && (h1 == h2);
}

std::vector<std::string>
common_fields(const conduit::Node &i1,
              const conduit::Node &i2)
{
  std::vector<std::string> fields1 = i1["fields"].child_names();
  std::vector<std::string> fields2 = i2["fields"].child_names();

  std::vector<std::string> res;
  for(auto name : fields1)
  {
    auto it = std::find(fields2.begin(), fields2.end(), name);
    if(it != fields2.end())
    {
      res.push_back(name);
      std::cout<<name<<"\n";
    }
  }
  return res;
}

void compare_diff(conduit::Node &info,
             const float *p1, //ho fiels
             const float *p2,
             float *pdiff,
             const int size)
{
  int count_both = 0;
  int count_either = 0;
  double rms_sum = 0;
  double mae_sum = 0;
  double mre_sum = 0;
  float vmin = std::numeric_limits<float>::max();
  float vmax = std::numeric_limits<float>::min();
  for(int i = 0; i < size; ++i)
  {
    const float lval = p1[i];
    const float rval = p2[i];
    bool lnan = std::isnan(lval);
    bool rnan = std::isnan(rval);

    if (!lnan || !rnan)
      count_either++;

    pdiff[i] = rval-lval;
    // Do this before nans test so we propagate nans.

    if(lnan || rnan)
    {
      // For now just skip these which
      // are misses on the edge of the data
      // (maybe)
      continue;
    }
    rms_sum += pow((rval - lval),2.f);
    vmin = std::min(vmin, lval);
    vmax = std::max(vmax, lval);
    mae_sum += fabs(rval-lval);
    mre_sum += fabs(rval-lval) / double(fabs(lval));
    count_both++;
  }
  double rms = sqrt(rms_sum/double(count_both));
  info["rms"] = rms;  //L2
  info["nrms"] = rms / (vmax-vmin);
  info["ho_max"] = vmax;
  info["ho_min"] = vmin;
  info["mae"] = mae_sum / (double(count_both));  //L1
  info["mre"] = mre_sum / (double(count_both));
  info["overlap"] = double(count_both)/double(count_either);
}




void compare(conduit::Node &info,
             const float *p1, //ho fiels
             const float *p2,
             const int size)
{
  int count_both = 0;
  int count_either = 0;
  double rms_sum = 0;
  double mae_sum = 0;
  double mre_sum = 0;
  float vmin = std::numeric_limits<float>::max();
  float vmax = std::numeric_limits<float>::min();
  for(int i = 0; i < size; ++i)
  {
    const float lval = p1[i];
    const float rval = p2[i];
    bool lnan = std::isnan(lval);
    bool rnan = std::isnan(rval);

    if (!lnan || !rnan)
      count_either++;

    if(lnan || rnan)
    {
      // For now just skip these which
      // are misses on the edge of the data
      // (maybe)
      continue;
    }
    rms_sum += pow((rval - lval),2.f);
    vmin = std::min(vmin, lval);
    vmax = std::max(vmax, lval);
    mae_sum += fabs(rval-lval);
    mre_sum += fabs(rval-lval) / double(fabs(lval));
    count_both++;
  }
  double rms = sqrt(rms_sum/double(count_both));
  info["rms"] = rms;  //L2
  info["nrms"] = rms / (vmax-vmin);
  info["ho_max"] = vmax;
  info["ho_min"] = vmin;
  info["mae"] = mae_sum / (double(count_both));  //L1
  info["mre"] = mre_sum / (double(count_both));
  info["overlap"] = double(count_both)/double(count_either);
}

conduit::Node
compare_fields(const std::vector<std::string> &names,
               const conduit::Node &i1,
               const conduit::Node &i2,
               const int width,
               const int height,
               conduit::Node &diff)
{
  conduit::Node res;
  for(auto field : names)
  {
    const conduit::Node &vals1 = i1["fields/"+field+"/values"];
    const conduit::Node &vals2 = i2["fields/"+field+"/values"];

    diff["fields/"+field+"/association"] = i1["fields/"+field+"/association"];
    diff["fields/"+field+"/topology"]    = i1["fields/"+field+"/topology"];

    const int size1 = vals1.dtype().number_of_elements();
    const int size2 = vals2.dtype().number_of_elements();
    if(size1 != size2)
    {
      std::cout<<"size mismatch "<<field<<" "<<size1<<" "<<size2<<"\n";
    }
    const float *p1 = vals1.value();
    const float *p2 = vals2.value();

    diff["fields/"+field+"/values"].set(conduit::DataType::float32(size1));
    float *pdiff = diff["fields/"+field+"/values"].value();

    compare_diff(res[field], p1, p2, pdiff, size1);
#ifdef DRAY_ENABLED
    conduit::Node color_table;
    conduit::Node color_info;
    color_table["name"] = "4w_bgTR";
    compare_colors(color_info,
                   color_table,
                   p1,
                   p2,
                   size1,
                   width,
                   height,
                   field,
                   output_name);
#endif
  }
  return res;
}

int main (int argc, char *argv[])
{

  if(argc != 4)
  {
    usage();
    return 1;
  }

  std::string file1(argv[1]);
  std::string file2(argv[2]);
  output_name = std::string(argv[3]);

  conduit::Node hola_opts, data1, data2;
  conduit::Node data_diff;

  hola_opts["root_file"] = file1;
  ascent::hola("relay/blueprint/mesh", hola_opts, data1);

  hola_opts["root_file"] = file2;
  ascent::hola("relay/blueprint/mesh", hola_opts, data2);

  // these are multi-domain data sets and
  // there should only be one inside each
  // so just grab it
  conduit::Node &image1 = data1.child(0);
  conduit::Node &image2 = data2.child(0);
  conduit::Node &image_diff = data_diff.append();

  int width, height;
  bool valid = validate_dims(image1, image2, width, height);
  if(!valid)
  {
    std::cout<<"image dims do not match\n";
    return 1;
  }

  int cycle = image1["state/cycle"].to_int32();

  image_diff["state/cycle"] = cycle;
  image_diff["coordsets"] = image1["coordsets"];
  image_diff["topologies"] = image1["topologies"];

  std::vector<std::string> fields = common_fields(image1,image2);
  conduit::Node info = compare_fields(fields,
                                      image1,
                                      image2,
                                      width,
                                      height,
                                      image_diff);
  info.print();

  char cycle_suffix[30];
  snprintf(cycle_suffix, 30, "%06d", cycle);
  std::string diff_name = output_name + "_holo_diff.cycle_"
                        + std::string(cycle_suffix)
                        + ".blueprint_root_hdf5";
  conduit::relay::io_blueprint::save(data_diff, diff_name);

  return 0;
}
