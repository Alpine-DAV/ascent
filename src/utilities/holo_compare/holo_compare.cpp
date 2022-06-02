//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: holo.cpp
///
//-----------------------------------------------------------------------------
#include <ascent.hpp>
#include <ascent_hola.hpp>

#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

void usage()
{
  std::cout<<"holo usage: holo high_order.root low_order.root\n";
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

void compare(conduit::Node &info,
             const float *p1, //ho fiels
             const float *p2,
             const int size)
{
  int count = 0;
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
    mae_sum += abs(rval-lval);
    mre_sum += abs(rval-lval) / lval;
    count++;
  }
  double rms = sqrt(rms_sum/double(count));
  info["rms"] = rms;
  info["nrms"] = rms / (vmax-vmin);
  info["ho_max"] = vmax;
  info["ho_min"] = vmin;
  info["mae"] = mae_sum / (double(count));
  info["mre"] = mre_sum / (double(count));
}

conduit::Node
compare_fields(const std::vector<std::string> &names,
               const conduit::Node &i1,
               const conduit::Node &i2)
{
  conduit::Node res;
  for(auto field : names)
  {
    const conduit::Node &vals1 = i1["fields/"+field+"/values"];
    const conduit::Node &vals2 = i2["fields/"+field+"/values"];
    const int size1 = vals1.dtype().number_of_elements();
    const int size2 = vals2.dtype().number_of_elements();
    if(size1 != size2)
    {
      std::cout<<"size mismatch "<<field<<" "<<size1<<" "<<size2<<"\n";
    }
    const float *p1 = vals1.value();
    const float *p2 = vals2.value();
    compare(res[field], p1, p2, size1);
  }
  return res;
}

int main (int argc, char *argv[])
{

  if(argc != 3)
  {
    usage();
    return 1;
  }

  std::string file1(argv[1]);
  std::string file2(argv[2]);

  conduit::Node hola_opts, data1, data2;

  hola_opts["root_file"] = file1;
  ascent::hola("relay/blueprint/mesh", hola_opts, data1);

  hola_opts["root_file"] = file2;
  ascent::hola("relay/blueprint/mesh", hola_opts, data2);

  // these are multi-domain data sets and
  // there should only be one inside each
  // so just grab it
  conduit::Node &image1 = data1.child(0);
  conduit::Node &image2 = data2.child(0);

  int width, height;
  bool valid = validate_dims(image1, image2, width, height);
  if(!valid)
  {
    std::cout<<"image dims do not match\n";
    return 1;
  }

  std::vector<std::string> fields = common_fields(image1,image2);
  conduit::Node info = compare_fields(fields, image1, image2);
  info.print();

  return 0;
}
