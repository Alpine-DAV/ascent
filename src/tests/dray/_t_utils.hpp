#ifndef DRAY_TEST_UTILS
#define DRAY_TEST_UTILS

#include <iostream>
#include <math.h>

#include "test_config.h"
#include <dray/utils/png_compare.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
void remove_test_image (const std::string &path)
{
  if (conduit::utils::is_file (path + ".png"))
  {
    conduit::utils::remove_file (path + ".png");
  }

  if (conduit::utils::is_file (path + ".pnm"))
  {
    conduit::utils::remove_file (path + ".pnm");
  }
}

//-----------------------------------------------------------------------------
void remove_test_file (const std::string &path)
{
  if (conduit::utils::is_file (path))
  {
    conduit::utils::remove_file (path);
  }
}

//-----------------------------------------------------------------------------
std::string prepare_output_dir ()
{
  string output_path = DRAY_T_BIN_DIR;

  output_path = conduit::utils::join_file_path (output_path, "_output");

  if (!conduit::utils::is_directory (output_path))
  {
    conduit::utils::create_directory (output_path);
  }

  return output_path;
}

//----------------------------------------------------------------------------
std::string output_dir ()
{
  return conduit::utils::join_file_path (DRAY_T_BIN_DIR, "_output");
  ;
}

//-----------------------------------------------------------------------------
bool check_test_image (const std::string &path, const float tolerance = 0.2f)
{
  Node info;
  std::string png_path = path + ".png";
  // for now, just check if the file exists.
  bool res = conduit::utils::is_file (png_path);
  bool both_exist = true;
  info["test_file/path"] = png_path;
  if (res)
  {
    info["test_file/exists"] = "true";
  }
  else
  {
    info["test_file/exists"] = "false";
    both_exist = false;
    res = false;
  }

  std::string file_name;
  std::string path_b;

  conduit::utils::rsplit_file_path (png_path, file_name, path_b);

  string baseline_dir =
  conduit::utils::join_file_path (DRAY_T_SRC_DIR, "baseline_images");
  string baseline = conduit::utils::join_file_path (baseline_dir, file_name);

  info["baseline_file/path"] = baseline;
  if (conduit::utils::is_file (baseline))
  {
    info["baseline_file/exists"] = "true";
  }
  else
  {
    info["baseline_file/exists"] = "false";
    both_exist = false;
    res = false;
  }

  if (both_exist)
  {

    dray::PNGCompare compare;

    res &= compare.compare (png_path, baseline, info, tolerance);
  }

  if (!res)
  {
    info.print ();
  }

  std::string info_fpath = path + "_img_compare_results.json";
  info.save (info_fpath, "json");

  return res;
}

bool check_test_file (const std::string &path)
{
  // for now, just check if the file exists.
  return conduit::utils::is_file (path);
}

#endif
