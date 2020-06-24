/*
 * VTKutils.h
 *
 *  Created on: Aug 28, 2016
 *      Author: spetruzza
 */

#ifndef ASCENT_VTK_UTILS_HPP
#define ASCENT_VTK_UTILS_HPP


#include <vtkMarchingCubes.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>

#define DEBUG_IMAGES 0

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class VTKutils
{
public:

  struct ImageData
  {
    unsigned char* image; 
    unsigned char* zbuf;
    uint32_t* bounds;
    uint32_t* rend_bounds;     // Used only for binswap and k-radix
  };

  static uint32_t IMAGE_X_DIM;
  static uint32_t IMAGE_Y_DIM;

  static uint32_t DATASET_DIMS[3];
  
  static inline void setImageDims(uint32_t x_dim, uint32_t y_dim) { IMAGE_X_DIM = x_dim; IMAGE_Y_DIM = y_dim; }

  static inline void setDatasetDims(const int32_t* data_size)
  {
    DATASET_DIMS[0] = data_size[0]; 
    DATASET_DIMS[1] = data_size[1]; 
    DATASET_DIMS[2] = data_size[2];
  }

  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static vtkSmartPointer<vtkImageData> getImageData(const char* data, uint32_t* dims);

  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static vtkSmartPointer<vtkMarchingCubes> getSurface(const char* data, double isoValue, uint32_t* dims);

  static void computeZBuffer(vtkSmartPointer<vtkRenderWindow> renWin, double shift, int id);

  static vtkSmartPointer<vtkRenderWindow> render(vtkSmartPointer<vtkMarchingCubes> surface, 
                                                                 uint32_t* trans, 
                                                                 int id = 0);
  
  static vtkSmartPointer<vtkRenderWindow> render(vtkSmartPointer<vtkImageData> volume, 
                                                 uint32_t* trans,  
                                                 vtkSmartPointer<vtkImageData> depthImage, 
                                                 vtkSmartPointer<vtkImageData> colorImage, 
                                                 int id = 0);

  static void composite(const std::vector<ImageData>& images, 
                        ImageData& out_image, 
                        const int id = 0);
  
  static void composite(const std::vector<ImageData>& images, 
                        std::vector<ImageData>& out_image, 
                        const int id, 
                        int x_factor, 
                        int y_factor, 
                        int n_out = 2);
                        
  static void compositeRadixK(const std::vector<ImageData>& input_images, 
                              std::vector<ImageData>& out_images, 
                              const int id);
                        
  static void splitAndBlend(const std::vector<ImageData>& input_images,
                            std::vector<ImageData>& out_images,
                            uint32_t* union_box,
                            bool skip_z_check);

  static void writeImage(unsigned char* image, uint32_t* bound, const char* filename);
  
  static void writeImageFixedSize(unsigned char* image, uint32_t* bound, const char* filename);

  static void compute2DBounds(const float* zBuf, uint32_t* in_bounds, uint32_t* out_bounds);
  
  static inline void computeUnion(const uint32_t* a,const uint32_t* b, uint32_t* c)
  {
    c[0] = std::min(a[0], b[0]); c[1] = std::max(a[1], b[1]);
    c[2] = std::min(a[2], b[2]); c[3] = std::max(a[3], b[3]);
  }

  static inline bool computeIntersection(const uint32_t* a,const uint32_t* b, uint32_t* c)
  {
    c[0] = std::max(a[0], b[0]); c[1] = std::min(a[1], b[1]);
    c[2] = std::max(a[2], b[2]); c[3] = std::min(a[3], b[3]);

    bool intersect = true;
    if( int(c[1]-c[0]) < 0 || int(c[3]-c[2]) < 0 ) 
    {
      c[0] = c[1] = c[2] = c[3] = 0;
      intersect = false;
    }

    return intersect;
  }

};



class VTKCompositeRender
{
public:

  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static int volumeRender(uint32_t* box, 
                          char* data, 
                          VTKutils::ImageData& image_data, 
                          int id);
                          
  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static int volumeRender(uint32_t* box, 
                          char* data, 
                          std::vector<VTKutils::ImageData>& image_data, 
                          int id, 
                          int x_factor, 
                          int y_factor);
                          
  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static int volumeRenderRadixK(uint32_t* box, 
                                char* data, 
                                std::vector<VTKutils::ImageData>& image_data, 
                                int id);

  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static int isosurfaceRender(uint32_t* box, 
                              char* data, 
                              float isovalue, 
                              VTKutils::ImageData& image_data, 
                              int id);

  // Template is here to allow for different types of data, i.e., float or double
  template<typename T>
  static int isosurfaceRender(uint32_t* box, 
                              char* data, 
                              float isovalue, 
                              std::vector<VTKutils::ImageData>& image_data, 
                              int id, 
                              int x_factor, 
                              int y_factor);
};


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
