#ifndef HOLO_COLORS_HPP
#define HOLO_COLORS_HPP

#include <dray/color_map.hpp>
#include <dray/utils/png_encoder.hpp>

//https://github.com/antimatter15/rgb-lab/blob/master/color.js
dray::Vec<float,4> lab2rgb(dray::Vec<float,3> lab)
{
  float y = (lab[0] + 16.f) / 116.f;
  float x = lab[1] / 500.f + y;
  float z = y - lab[2] / 200.f;
  float r, g, b;

  x = 0.95047f * ((x * x * x > 0.008856f)
      ? x * x * x : (x - 16.f/116.f) / 7.787f);
  y = 1.00000f * ((y * y * y > 0.008856f)
      ? y * y * y : (y - 16.f/116.f) / 7.787f);
  z = 1.08883f * ((z * z * z > 0.008856f) ?
      z * z * z : (z - 16.f/116.f) / 7.787f);

  r = x *  3.2406 + y * -1.5372f + z * -0.4986f;
  g = x * -0.9689 + y *  1.8758f + z *  0.0415f;
  b = x *  0.0557 + y * -0.2040f + z *  1.0570f;

  r = (r > 0.0031308) ? (1.055 * std::pow(r, 1.f/2.4f) - 0.055f) : 12.92f * r;
  g = (g > 0.0031308) ? (1.055 * std::pow(g, 1.f/2.4f) - 0.055f) : 12.92f * g;
  b = (b > 0.0031308) ? (1.055 * std::pow(b, 1.f/2.4f) - 0.055f) : 12.92f * b;

  r = std::max(std::min(r, 1.f),0.f);
  g = std::max(std::min(g, 1.f),0.f);
  b = std::max(std::min(b, 1.f),0.f);

  dray::Vec<float,4> color;
  color[0] = r;
  color[1] = g;
  color[2] = b;
  color[3] = 1.f;
  return color;

}

dray::Vec<float,3> rgb2lab(dray::Vec<float,4> rgb)
{
  float r = rgb[0];
  float g = rgb[1];
  float b = rgb[2];
  float x, y, z;

  r = (r > 0.04045f) ? std::pow((r + 0.055f) / 1.055f, 2.4f) : r / 12.92f;
  g = (g > 0.04045f) ? std::pow((g + 0.055f) / 1.055f, 2.4f) : g / 12.92f;
  b = (b > 0.04045f) ? std::pow((b + 0.055f) / 1.055f, 2.4f) : b / 12.92f;

  x = (r * 0.4124f + g * 0.3576f + b * 0.1805f) / 0.95047f;
  y = (r * 0.2126f + g * 0.7152f + b * 0.0722f) / 1.00000f;
  z = (r * 0.0193f + g * 0.1192f + b * 0.9505f) / 1.08883f;

  x = (x > 0.008856f) ? std::pow(x, 1.f/3.f) : (7.787f * x) + 16.f/116.f;
  y = (y > 0.008856f) ? std::pow(y, 1.f/3.f) : (7.787f * y) + 16.f/116.f;
  z = (z > 0.008856f) ? std::pow(z, 1.f/3.f) : (7.787f * z) + 16.f/116.f;

  dray::Vec<float,3> res;
  res[0] = 116.f * y - 16.f;
  res[1] = 500.f * (x - y);
  res[2] = 200.f * (y - z);
  return res;
}

// http://colormine.org/delta-e-calculator
// using cie94 method
// more info: http://zschuessler.github.io/DeltaE/learn/
// Basic guideline backed by no facts.
// and Delta E accuracy must be confirmed through the
// very tool it was meant to remove subjectivity from
// - a pair of human eyes.
// <= 1.0 Not perceptible by human eyes.
// 1 - 2 Perceptible through close observation.
// 2 - 10 Perceptible at a glance.
// 11 - 49 Colors are more similar than opposite
// 100 Colors are exact opposite
float delta_e(dray::Vec<float,3> lab_a, dray::Vec<float,3> lab_b)
{
  float delta_l = lab_a[0] - lab_b[0];
  float delta_a = lab_a[1] - lab_b[1];
  float delta_b = lab_a[2] - lab_b[2];
  float c1 = std::sqrt(lab_a[1] * lab_a[1] + lab_a[2] * lab_a[2]);
  float c2 = std::sqrt(lab_b[1] * lab_b[1] + lab_b[2] * lab_b[2]);
  float delta_c = c1 - c2;
  float delta_h = delta_a * delta_a + delta_b * delta_b - delta_c * delta_c;
  delta_h = delta_h < 0 ? 0 : std::sqrt(delta_h);
  float sc = 1.0f + 0.045f * c1;
  float sh = 1.0f + 0.015f * c1;
  float delta_lklsl = delta_l / 1.0f;
  float delta_ckcsc = delta_c / sc;
  float delta_hkhsh = delta_h / sh;
  float i = delta_lklsl * delta_lklsl + delta_ckcsc * delta_ckcsc + delta_hkhsh * delta_hkhsh;
  return i < 0 ? 0 : std::sqrt(i);

}

dray::Range scalar_range(const float *p1,
                         const float *p2,
                         const int size)
{
  // find the total range of both images
  // so we apply the same transform
  dray::Range range;
  for(int i = 0; i < size; ++i)
  {
    float val1 = p1[i];
    if(!std::isnan(val1))
    {
      range.include(val1);
    }
    float val2 = p1[i];
    if(!std::isnan(val2))
    {
      range.include(val2);
    }
  }
  return range;
}

dray::Array<dray::Vec<float,4>>
to_colors(const float *image,
          const int size,
          dray::ColorTable &color_table,
          dray::Range &range)
{
  dray::Array<dray::Vec<float,4>> colors;
  colors.resize(size*4);

  dray::Vec<float,4> *colors_ptr = colors.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {

    dray::Vec<float,4> color{{1.f, 1.f, 1.f, 1.f}};

    float val1 = image[i];
    if(!std::isnan(val1))
    {
      float normalized = (val1 - range.min()) / range.length();
      color = color_table.map_rgb(normalized);
    }
    colors_ptr[i] = color;
  }

  return colors;
}


dray::Array<dray::Vec<float,4>>
image_diff(dray::Array<dray::Vec<float,4>> p1,
           dray::Array<dray::Vec<float,4>> p2)
{
  dray::Array<dray::Vec<float,4>> res;
  const int size = p1.size();
  res.resize(size);

  dray::Vec<float,4> *p1_ptr = p1.get_host_ptr();
  dray::Vec<float,4> *p2_ptr = p2.get_host_ptr();
  dray::Vec<float,4> *dif_ptr = res.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {
    dray::Vec<float,4> color = p1_ptr[i];
    color -= p2_ptr[i];
    color[0] = std::abs(color[0]);
    color[1] = std::abs(color[1]);
    color[2] = std::abs(color[2]);
    color[3] = 1.f;
    dif_ptr[i] = color;
  }

  return res;
}

void compare_colors(conduit::Node &info,
                    const conduit::Node &settings,
                    const float *p1, //ho fiels
                    const float *p2,
                    const int size,
                    const int width,
                    const int height,
                    const std::string field_name)
{
  const std::string name = settings["name"].as_string();
  dray::ColorTable color_table(name);
  dray::Range range = scalar_range(p1,p2,size);
  dray::Array<dray::Vec<float,4>> p1_colors;
  dray::Array<dray::Vec<float,4>> p2_colors;

  p1_colors = to_colors(p1, size, color_table, range);
  p2_colors = to_colors(p2, size, color_table, range);

  dray::PNGEncoder encoder;
  encoder.encode((float*)p1_colors.get_host_ptr(), width, height);
  encoder.save("p1_baseline_"+field_name + ".png");

  encoder.encode((float*)p2_colors.get_host_ptr(), width, height);
  encoder.save("p2_baseline_"+field_name + ".png");

  // do a basic image diff
  dray::Array<dray::Vec<float,4>> diff_colors;
  diff_colors = image_diff(p1_colors,p2_colors);

  encoder.encode((float*)diff_colors.get_host_ptr(), width, height);
  encoder.save("diff_"+field_name + ".png");

}

#endif
