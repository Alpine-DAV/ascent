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
		uint32_t* rend_bounds;
	};

	struct SimpleImageData
	{
		unsigned char* image; 
		unsigned char* zbuf;
		uint32_t* bounds;
	};

	static void setImageDims(uint32_t x_dim, uint32_t y_dim) { IMAGE_X_DIM = x_dim; IMAGE_Y_DIM = y_dim;}
	static void setDatasetDims(const uint32_t* data_size)
	{ DATASET_DIMS[0] = data_size[0]; DATASET_DIMS[1] = data_size[1]; DATASET_DIMS[2] = data_size[2];}
	static void setDataDecomp(const uint32_t* decomp)
	{ DATA_DECOMP[0] = decomp[0]; DATA_DECOMP[1] = decomp[1]; DATA_DECOMP[2] = decomp[2];}

	vtkSmartPointer<vtkImageData> getImageData(const char* data, uint32_t* dims);
	vtkSmartPointer<vtkMarchingCubes> getSurface(const char* data, double isoValue, uint32_t* dims);

	void computeZBuffer(vtkSmartPointer<vtkRenderWindow> renWin, double shift, int id);

	vtkSmartPointer<vtkRenderWindow> render(vtkSmartPointer<vtkMarchingCubes> surface, uint32_t* trans, int id = 0);
	vtkSmartPointer<vtkRenderWindow> render(vtkSmartPointer<vtkImageData> volume, uint32_t* trans,  
                                          vtkSmartPointer<vtkImageData> depthImage, 
                                          vtkSmartPointer<vtkImageData> colorImage, int id = 0);

	int composite(const std::vector<SimpleImageData>& images, SimpleImageData& out_image, const int id = 0);
	int composite(const std::vector<ImageData>& images, std::vector<ImageData>& out_image, 
		const int id, int x_factor, int y_factor, int n_out=2);

	static int arraytoImage(unsigned char* image, uint32_t* bound, char* filename);
	static int arraytoImageFixedSize(unsigned char* image, uint32_t* bound, char* filename);

  static void compute2DBounds(const float* zBuf, uint32_t* in_bounds, uint32_t* out_bounds);
  static inline void computeUnion(const uint32_t* a,const uint32_t* b, uint32_t* c) 
  {
	  c[0] = std::min(a[0], b[0]); c[1] = std::max(a[1], b[1]);
	  c[2] = std::min(a[2], b[2]); c[3] = std::max(a[3], b[3]);

	  // printf("a size %d %d\n", a[1]-a[0],a[3]-a[2]);
	  // printf("b size %d %d\n", b[1]-b[0],b[3]-b[2]);
	  // printf("union size %d %d\n", c[1]-c[0],c[3]-c[2]);
	}

	static inline bool computeIntersection(const uint32_t* a,const uint32_t* b, uint32_t* c) 
	{
	  
	  c[0] = std::max(a[0], b[0]); c[1] = std::min(a[1], b[1]);
	  c[2] = std::max(a[2], b[2]); c[3] = std::min(a[3], b[3]);

	  bool intersect = true;
	  if(int(c[1]-c[0]) < 0 || int(c[3]-c[2]) < 0) {
	  	for(int i=0; i<4; i++)
	  		c[i] = 0;
	  	intersect = false;
	  }

	  // printf("%d %d ^ %d %d a size %d %d\n", a[0],a[1],a[2],a[3], a[1]-a[0],a[3]-a[2]);
	  // printf("%d %d ^ %d %d b size %d %d\n", b[0],b[1],b[2],b[3], b[1]-b[0],b[3]-b[2]);
	  // printf("%d %d ^ %d %d intersect size %d %d\n", c[0],c[1],c[2],c[3], c[1]-c[0],c[3]-c[2]);

	  return intersect;
	}

	static uint32_t IMAGE_X_DIM;
	static uint32_t IMAGE_Y_DIM;

	static uint32_t DATASET_DIMS[3];
	static uint32_t DATA_DECOMP[3];
};



class VTKCompositeRender
{
public:
	static int writeImage(unsigned char* image, uint32_t* bound, const char* filename);
	
	static inline int composite(std::vector<VTKutils::SimpleImageData>& images, VTKutils::SimpleImageData& out_image, int id)
	{
	  utils.composite(images, out_image, id);
	  return true;
	};

	static inline int composite(std::vector<VTKutils::ImageData>& images, std::vector<VTKutils::ImageData>& out_image, 
		int id, int x_factor, int y_factor, int n_out=2)
	{
	  utils.composite(images, out_image, id, x_factor, y_factor, n_out);
	  return true;
	};
	
	static int volumeRender(uint32_t* box, char* data, float isovalue, 
      VTKutils::SimpleImageData& image_data, int id);

	static int isosurfaceRender(uint32_t* box, char* data, float isovalue, 
      VTKutils::SimpleImageData& image_data, int id);

	static int isosurfaceRender(uint32_t* box, char* data, float isovalue, 
      std::vector<VTKutils::ImageData>& image_data, int id, int x_factor, int y_factor);
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
