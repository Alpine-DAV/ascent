//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_test_utils.cpp
///
//-----------------------------------------------------------------------------
#include <apcomp/utils/png_compare.hpp>
#include <apcomp/utils/png_encoder.hpp>
#include <apcomp/utils/file_utils.hpp>
#include <apcomp/volume_partial.hpp>
#include "t_config.hpp"

#include <vector>
#include <sstream>

void
gen_float32_image(std::vector<float> &pixels,
                  std::vector<float> &depths,
                  int width,
                  int height,
                  float depth,
                  int bottom_x,
                  int bottom_y,
                  int size,
                  float color[4])
{
  pixels.resize(width * height * 4);
  depths.resize(width * height);
  for(int y = 0; y < height; ++y)
  {
    for(int x = 0; x < width; ++x)
    {
      pixels[(y * width + x) * 4 + 0] = 0.f;
      pixels[(y * width + x) * 4 + 1] = 0.f;
      pixels[(y * width + x) * 4 + 2] = 0.f;
      pixels[(y * width + x) * 4 + 3] = 0.f;
      depths[y * width + x] = 1.01f;
    }
  }

  for(int y = bottom_y ; y < bottom_y + size; ++y)
  {
    for(int x = bottom_x; x < bottom_x + size; ++x)
    {
      pixels[(y * width + x) * 4 + 0] = color[0];
      pixels[(y * width + x) * 4 + 1] = color[1];
      pixels[(y * width + x) * 4 + 2] = color[2];
      pixels[(y * width + x) * 4 + 3] = color[3];
      depths[y * width + x] = depth;
    }
  }

}

void
gen_float32_partials(std::vector<apcomp::VolumePartial<float>> &partials,
                     int width,
                     int height,
                     float depth,
                     int bottom_x,
                     int bottom_y,
                     int size,
                     float color[4])
{
  partials.resize(size * size);
  int count = 0;
  for(int y = bottom_y ; y < bottom_y + size; ++y)
  {
    for(int x = bottom_x; x < bottom_x + size; ++x)
    {
      apcomp::VolumePartial<float> partial;
      partial.m_pixel_id = y * width + x;
      partial.m_pixel[0] = color[0];
      partial.m_pixel[1] = color[1];
      partial.m_pixel[2] = color[2];
      partial.m_alpha = color[3];
      partial.m_depth = depth;
      partials[count] = partial;
      count++;
    }
  }

}

void partials_to_png(std::vector<apcomp::VolumePartial<float>> &partials,
                     const int width,
                     const int height,
                     const std::string file_name)
{
  std::vector<float> image;
  image.resize(width * height * 4);

  for(int y = 0; y < height; ++y)
  {
    for(int x = 0; x < width; ++x)
    {
      image[(y * width + x) * 4 + 0] = 0.f;
      image[(y * width + x) * 4 + 1] = 0.f;
      image[(y * width + x) * 4 + 2] = 0.f;
      image[(y * width + x) * 4 + 3] = 0.f;
    }
  }

  const int partials_size = partials.size();
  for(int i = 0; i < partials.size(); ++i)
  {
    apcomp::VolumePartial<float> p = partials[i];
    const int offset = p.m_pixel_id * 4;
    image[offset + 0] = p.m_pixel[0];
    image[offset + 1] = p.m_pixel[1];
    image[offset + 2] = p.m_pixel[2];
    image[offset + 3] = p.m_alpha;
  }

  apcomp::PNGEncoder encoder;
  encoder.Encode(&image[0], width, height);
  encoder.Save(file_name  + ".png");

}

inline std::string
t_apcomp_baseline_dir()
{
    std::string res = conduit::utils::join_file_path(ASCENT_T_SRC_DIR,"apcomp");
    res = conduit::utils::join_file_path(res,"baseline_images");
    std::cout << res << std::endl;
    return res;
}


// inline std::string
// output_dir()
// {
//   return apcomp::join_file_path(ASCENT_T_BIN_DIR,"_output");
// }
//
// inline void
// remove_test_file(const std::string &path)
// {
//   if(apcomp::is_file(path))
//   {
//     apcomp::remove_file(path);
//   }
// }
//
// inline std::string
// prepare_output_dir()
// {
//   std::string output_path = ASCENT_T_BIN_DIR;
//
//   output_path = apcomp::join_file_path(output_path,"_output");
//
//   if(!apcomp::is_directory(output_path))
//   {
//     apcomp::create_directory(output_path);
//   }
//
//   return output_path;
// }

// inline bool
// check_test_image(const std::string &path, const float tolerance = 0.01f)
// {
//   std::stringstream info;
//   std::string png_path = path + ".png";
//   // for now, just check if the file exists.
//   bool res = apcomp::is_file(png_path);
//   info<<"test_file/path: "<<png_path<<"\n";;
//   if(res)
//   {
//     info<<"test_file/exists: true\n";
//   }
//   else
//   {
//     info<<"test_file/exists: false\n";
//   }
//
//   std::string file_name;
//   std::string path_b;
//
//   apcomp::rsplit_file_path(png_path,
//                            file_name,
//                            path_b);
//
//   std::string baseline_dir = apcomp::join_file_path(ASCENT_T_SRC_DIR,"apcomp");
//   baseline_dir = apcomp::join_file_path(baseline_dir,"baseline_images");
//   std::string baseline = apcomp::join_file_path(baseline_dir,file_name);
//
//   info<<"baseline_file/path: "<<baseline<<"\n";
//   if(apcomp::is_file(baseline))
//   {
//     info<<"baseline_file/exists: true\n";
//   }
//   else
//   {
//     info<<"baseline_file/exists: false\n";
//     res = false;
//   }
//
//   if(res)
//   {
//     apcomp::PNGCompare compare;
//     float diff;
//     res &= compare.Compare(png_path, baseline, diff, tolerance);
//     if(!res)
//     {
//       info<<"image_diff: "<<diff<<"\n";
//     }
//   }
//
//   if(!res)
//   {
//     std::cout<<info.str();
//   }
//
//   return res;
// }
