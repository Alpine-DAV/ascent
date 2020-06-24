#include <cmath>
#include <limits>

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkMarchingCubes.h>
#include <vtkVoxelModeller.h>
#include <vtkSphereSource.h>
#include <vtkImageData.h>
#include <vtkDICOMImageReader.h>

#include <vtkActor.h>
#include <vtkTransform.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkProperty.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkFixedPointVolumeRayCastMapper.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkOpenGLGPUVolumeRayCastMapper.h>

#include <vtkPNGWriter.h>
#include <vtkCamera.h>
#include <vtkBMPWriter.h>
#include <vtkImageShiftScale.h>

#include <vtkAutoInit.h>

#include "ascent_vtk_utils.hpp"

#include <mutex>


#define DISABLE_RENDERING 0
//#define BOUNDING_BOX_OPTIMIZATION


std::mutex mtx;

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2);


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

uint32_t VTKutils::IMAGE_X_DIM = 0;
uint32_t VTKutils::IMAGE_Y_DIM = 0;
uint32_t VTKutils::DATASET_DIMS[3] = {0,0,0};

// Forward declarations of template specialization to avoid linkage problems since we
// implement templated functions in a source file rather than a header
template vtkSmartPointer<vtkImageData> VTKutils::getImageData<double>(const char* data, uint32_t* dims);
template vtkSmartPointer<vtkMarchingCubes> VTKutils::getSurface<double>(const char* data, double isoValue, uint32_t* dims);
template int VTKCompositeRender::volumeRender<double>(uint32_t* box, 
                                                      char* data,
                                                      VTKutils::ImageData& image_data, 
                                                      int id);
template int VTKCompositeRender::volumeRender<double>(uint32_t* box, 
                                                      char* data, 
                                                      std::vector<VTKutils::ImageData>& image_data, 
                                                      int id, 
                                                      int x_factor, 
                                                      int y_factor);
template int VTKCompositeRender::volumeRenderRadixK<double>(uint32_t* box, 
                                                            char* data, 
                                                            std::vector<VTKutils::ImageData>& image_data, 
                                                            int id);
template int VTKCompositeRender::isosurfaceRender<double>(uint32_t* box, 
                                                          char* data, 
                                                          float isovalue, 
                                                          VTKutils::ImageData& image_data, 
                                                          int id);
template int VTKCompositeRender::isosurfaceRender<double>(uint32_t* box, 
                                                          char* data, 
                                                          float isovalue, 
                                                          std::vector<VTKutils::ImageData>& image_data, 
                                                          int id, 
                                                          int x_factor, 
                                                          int y_factor);
      

template<typename T>
vtkSmartPointer<vtkImageData> VTKutils::getImageData(const char* data, uint32_t* dims)
{
  vtkSmartPointer<vtkImageData> volume = vtkSmartPointer<vtkImageData>::New();
  
   // Specify the size of the image data
  volume->SetDimensions(dims[0], dims[1], dims[2]);
#if VTK_MAJOR_VERSION <= 5
  volume->SetNumberOfScalarComponents(1);
  volume->SetScalarTypeToFloat();
#else
  volume->AllocateScalars(VTK_DOUBLE,1);
#endif

  const T* data_arr = reinterpret_cast<const T*>(data);
  //std::cout << "Dims: " << " x: " << dims[0] << " y: " << dims[1] << " z: " << dims[2] << std::endl;
 
//  std::cout << "Number of points: " << volume->GetNumberOfPoints() << std::endl;
//  std::cout << "Number of cells: " << volume->GetNumberOfCells() << std::endl;
 
  for (int z = 0; z < dims[2]; z++)
  {
    for (int y = 0; y < dims[1]; y++)
      {
      for (int x = 0; x < dims[0]; x++)
        {
          uint32_t idx = x + y * dims[0] + z*dims[0]*dims[1];
          double* pixel = static_cast<double*>(volume->GetScalarPointer(x,y,z));
          pixel[0] = data_arr[idx];
        }
      }
  }

  return volume;
}

template<typename T>
vtkSmartPointer<vtkMarchingCubes> VTKutils::getSurface(const char* data, double isoValue, uint32_t* dims)
{
  vtkSmartPointer<vtkImageData> volume = getImageData<T>(data, dims);

  vtkSmartPointer<vtkMarchingCubes> surface = vtkSmartPointer<vtkMarchingCubes>::New();

#if VTK_MAJOR_VERSION <= 5
  surface->SetInput(volume);
#else
  surface->SetInputData(volume);
#endif
  surface->ComputeNormalsOn();
  surface->SetValue(0, isoValue);

  return surface;
}

void VTKutils::computeZBuffer(vtkSmartPointer<vtkRenderWindow> renWin, double shift, int id)
{
  vtkSmartPointer<vtkWindowToImageFilter> filter_z = vtkSmartPointer<vtkWindowToImageFilter>::New();
  vtkSmartPointer<vtkWindowToImageFilter> filter_d = vtkSmartPointer<vtkWindowToImageFilter>::New();
  vtkSmartPointer<vtkBMPWriter> imageWriter = vtkSmartPointer<vtkBMPWriter>::New();
  vtkSmartPointer<vtkImageShiftScale> scale = vtkSmartPointer<vtkImageShiftScale>::New();
  
  // Create Depth Map
  filter_z->SetInput(renWin);
  // filter->SetMagnification(1);
  filter_z->SetInputBufferTypeToZBuffer();        //Extract z buffer value
 
  scale->SetOutputScalarTypeToUnsignedChar();
  scale->SetInputConnection(filter_z->GetOutputPort());
  scale->SetShift(0);
  scale->SetScale(-255);
 
  char filename[128];
  sprintf(filename, "zmap_%d.bmp", id);
  // Write depth map as a .bmp image
  imageWriter->SetFileName(filename);
  imageWriter->SetInputConnection(scale->GetOutputPort());
  imageWriter->Write();

  filter_d->SetInput(renWin);
  filter_d->SetInputBufferTypeToRGBA();

  sprintf(filename, "zmap_%d.png", id);
  vtkSmartPointer<vtkPNGWriter> writer = 
  vtkSmartPointer<vtkPNGWriter>::New();
  writer->SetFileName(filename);
  writer->SetInputConnection(filter_d->GetOutputPort());
  writer->Write();

  std::ofstream out;
  sprintf(filename, "zmap_%d.raw", id);
  out.open(filename);
  float* zbuf0 = renWin->GetZbufferData(0,0,VTKutils::IMAGE_X_DIM-1,VTKutils::IMAGE_Y_DIM-1);

  out.write((const char*)zbuf0, VTKutils::IMAGE_X_DIM*VTKutils::IMAGE_Y_DIM*sizeof(float));
  out.close();
}

void VTKutils::compute2DBounds(const float* zBuf, uint32_t* in_bounds, uint32_t* out_bounds)
{
#ifdef BOUNDING_BOX_OPTIMIZATION

  out_bounds[0] = in_bounds[1];
  out_bounds[1] = 0;
  out_bounds[2] = in_bounds[3];
  out_bounds[3] = 0;

  uint32_t total_points = (in_bounds[1]-in_bounds[0]+1)*(in_bounds[3]-in_bounds[2]+1);

  uint32_t in_x_size = (in_bounds[1]-in_bounds[0]+1);

  #pragma omp for
  for(int y=in_bounds[2]; y < in_bounds[3]+1; y++){
     for(int x=in_bounds[0]; x < in_bounds[1]+1; x++){
        uint32_t idx = x + y*in_x_size;

        uint32_t imgidx = idx*3;
        
        if( zBuf[idx] < 1.0 ){ 
     //     count++;

          if(x < out_bounds[0])
            out_bounds[0] = x;
          else if(x > out_bounds[1])
            out_bounds[1] = x;

          if(y < out_bounds[2])
            out_bounds[2] = y;
          else if(y > out_bounds[3])
            out_bounds[3] = y;

        }
    }
  }

#else
  out_bounds[0] = in_bounds[0];
  out_bounds[1] = in_bounds[1];
  out_bounds[2] = in_bounds[2];
  out_bounds[3] = in_bounds[3];

#endif

  /*
  float ratio = (float)((out_bounds[1]-out_bounds[0])*(out_bounds[3]-out_bounds[2]))/(float)(in_x_size*(in_bounds[3]-in_bounds[2]));

  printf("box resize %f\n", ratio*100);//100.f*(float)count/(float)total_points);
  */
}

void toRgb(int mag, int cmin, int cmax, float* color)
{
  float n = 0.5;

  if (float(cmax-cmin) != 0)
    n = float(mag-cmin)/float(cmax-cmin);

  int b = 255*std::fmin(std::fmax(4*(0.75-n), 0.), 1.);
  int r = 255*std::fmin(std::fmax(4*(n-0.25), 0.), 1.);
  int g = 255*std::fmin(std::fmax(4*std::fabs(n-0.5)-1., 0.), 1.);

  color[0] = r/255.f;
  color[1] = g/255.f;
  color[2] = b/255.f;

  // printf("in %d col %f %f %f\n", mag, color[0],color[1],color[2]);
}


vtkSmartPointer<vtkRenderWindow> VTKutils::render(vtkSmartPointer<vtkMarchingCubes> surface, uint32_t* trans, int id)
{
  vtkSmartPointer<vtkRenderer> renderer =  vtkSmartPointer<vtkRenderer>::New();
  renderer->SetBackground(0,0,0);//.1, .2, .3);

  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  //renderWindow->OffScreenRenderingOn();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(VTKutils::IMAGE_X_DIM, VTKutils::IMAGE_Y_DIM);
  // vtkSmartPointer<vtkRenderWindowInteractor> interactor = 
  //   vtkSmartPointer<vtkRenderWindowInteractor>::New();
  // interactor->SetRenderWindow(renderWindow);

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(surface->GetOutputPort());
  mapper->ScalarVisibilityOff();

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();

  //1k cube
  // camera->SetFocalPoint(512,512,512);//300,400,140.0);
  //camera->SetPosition(-1000,-2000,3800);

  //2k cube
  //camera->SetFocalPoint(1024,1024,1024);
  //camera->SetPosition(-1000,-1500,7000);

  //flame
  // camera->SetFocalPoint(1012,800,400);
  // camera->SetPosition(-1000,-1000,4000);

  // camera->SetFocalPoint(DATASET_DIMS[0]/2,DATASET_DIMS[1]/2,(float)DATASET_DIMS[2]/3.0);
  // camera->SetPosition(-0.5*DATASET_DIMS[0],-0.5*DATASET_DIMS[1],DATASET_DIMS[2]*3.5);

  // camera->SetViewUp(0,1.0,0);
  
  // camera->SetViewAngle(30.0);
  // camera->SetRoll(45.0);
  // camera->SetClippingRange(0.1,DATASET_DIMS[2]*10);

  // camera->SetFocalPoint(279.5,279.5,140.0);
  // camera->SetViewUp(0,1.0,0);
  // camera->SetPosition(279.5,279.5,1760.179842);
  // camera->SetViewAngle(30.0);
  // camera->SetRoll(20.0);
  // camera->SetClippingRange(1300,1995.182540);

  camera->SetFocalPoint(DATASET_DIMS[0]/2,DATASET_DIMS[1]/2,(float)DATASET_DIMS[2]/3.0);
  camera->SetPosition(-0.5*DATASET_DIMS[0],-0.5*DATASET_DIMS[1],DATASET_DIMS[2]*3.5);

  camera->SetViewUp(0,1.0,0);
  
  camera->SetViewAngle(30.0);
  camera->SetRoll(45.0);
  camera->SetClippingRange(0.1,DATASET_DIMS[2]*10);

  renderer->SetActiveCamera(camera);
  //renderer->ResetCameraClippingRange(-10000,100000,-100000,100000,100000,0.0001);

  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->PostMultiply(); //this is the key line
  transform->Translate(trans[0],trans[1],trans[2]);
  actor->SetUserTransform(transform);  

  // float color[3];
  // toRgb(id,0,8-1,color);
  // actor->GetProperty()->SetColor(color[0], color[1], color[2]);

  //actor->GetProperty()->SetColor(1,0,0);
  renderer->AddActor(actor);

  renderWindow->SetOffScreenRendering( 1 );

  mtx.lock();
  renderWindow->Render();
  mtx.unlock();

  //computeZBuffer(renderWindow, trans[2]/560.f*255.f, id);

  return renderWindow;
}

vtkSmartPointer<vtkRenderWindow> VTKutils::render(vtkSmartPointer<vtkImageData> volume, uint32_t* trans,
                                                  vtkSmartPointer<vtkImageData> depthImage, 
                                                  vtkSmartPointer<vtkImageData> colorImage, int id)
{
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->SetBackground(0,0,0);//.1, .2, .3);

  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  //renderWindow->OffScreenRenderingOn();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(VTKutils::IMAGE_X_DIM, VTKutils::IMAGE_Y_DIM);
  
#if USE_POLYDATA
  vtkSmartPointer<vtkImageDataGeometryFilter> imageDataGeometryFilter = 
    vtkSmartPointer<vtkImageDataGeometryFilter>::New();
#if VTK_MAJOR_VERSION <= 5
  imageDataGeometryFilter->SetInputConnection(volume->GetProducerPort());
#else
  imageDataGeometryFilter->SetInputData(volume);
#endif 
  imageDataGeometryFilter->Update();

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

  mapper->SetInputConnection(imageDataGeometryFilter->GetOutputPort());

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
#else

  vtkSmartPointer<vtkSmartVolumeMapper> mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
  mapper->SetBlendModeToComposite();

// vtkSmartPointer<vtkVolumeTextureMapper3D> mapper = 
//     vtkSmartPointer<vtkVolumeTextureMapper3D>::New();
 
  // vtkSmartPointer<vtkFixedPointVolumeRayCastMapper> mapper = 
  //   vtkSmartPointer<vtkFixedPointVolumeRayCastMapper>::New();

  // vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper> mapper = 
  //    vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper>::New();

  // mapper->RenderToImageOn();
  
  //mapper->SetBlendModeToComposite();

  //mapper->SetBlendModeToAdditiveComposite();
  //mapper->SetBlendModeToMaximumIntensity();
#if VTK_MAJOR_VERSION <= 5
  mapper->SetInputConnection(volume->GetProducerPort());
#else
  mapper->SetInputData(volume);
#endif 

  vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
  volumeProperty->ShadeOff();
  volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);
 
  vtkSmartPointer<vtkPiecewiseFunction> compositeOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
  compositeOpacity->AddPoint(0.0, 0.0);
  compositeOpacity->AddPoint(0.25, 0.0);
  compositeOpacity->AddPoint(0.5, 0.0);
  compositeOpacity->AddPoint(0.55, 0.1);
  compositeOpacity->AddPoint(0.8, 0.2);
  compositeOpacity->AddPoint(1.8, 1.0);
  volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.
 
  vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
  color->AddRGBPoint(0.0, 0.0, 0.0, 1.0);
  color->AddRGBPoint(0.5, 1.0, 0.5, 0.25);
  color->AddRGBPoint(1.8, 1.0, 0.0, 1.0);
  color->SetColorSpaceToRGB();
  volumeProperty->SetColor(color);
 
  vtkSmartPointer<vtkVolume> volumep = vtkSmartPointer<vtkVolume>::New();
  volumep->SetMapper(mapper);
  volumep->SetProperty(volumeProperty);

  renderer->AddVolume(volumep);

// #if !defined(VTK_LEGACY_REMOVE) && !defined(VTK_OPENGL2)
//   mapper->SetRequestedRenderModeToRayCastAndTexture();
// #endif // VTK_LEGACY_REMOVE

  //mapper->SetRequestedRenderModeToRayCastAndTexture();// RayCast();

  // vtkSmartPointer<vtkActor> actor = 
  //   vtkSmartPointer<vtkActor>::New();
  // actor->SetMapper(mapper);
#endif

  vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();

  camera->SetFocalPoint(DATASET_DIMS[0]/2,DATASET_DIMS[1]/2,(float)DATASET_DIMS[2]/3.0);
  camera->SetPosition(-0.5*DATASET_DIMS[0],-0.5*DATASET_DIMS[1],DATASET_DIMS[2]*3.5);

  camera->SetViewUp(0,1.0,0);
  
//  camera->SetPosition(279.5,279.5,1760.179842);
  camera->SetViewAngle(30.0);
  camera->SetRoll(45.0);
  camera->SetClippingRange(0.1,DATASET_DIMS[2]*10);

//  camera->SetClippingRange(1300,1995.182540);

  renderer->SetActiveCamera(camera);
  //renderer->ResetCameraClippingRange(-10000,100000,-100000,100000,100000,0.0001);

  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  transform->PostMultiply(); //this is the key line
  transform->Translate(trans[0],trans[1],trans[2]);

#if USE_POLYDATA
  actor->SetUserTransform(transform);  
  renderer->AddActor(actor);
#else
  volumep->SetUserTransform(transform);
#endif

  // if ( !mapper->IsRenderSupported(renderWindow, volumeProperty) )
  // {
  //   cout << "This mapper is unsupported on this platform" << endl;
  //   exit(EXIT_FAILURE);
  // }
  
  // vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  // windowToImageFilter->SetInput(renderWindow);
  // windowToImageFilter->SetInputBufferTypeToRGB();
  // windowToImageFilter->ShouldRerenderOn();
  // windowToImageFilter->Update();

  // vtkSmartPointer<vtkPNGWriter> writer = 
  //   vtkSmartPointer<vtkPNGWriter>::New();
  // char imgfilename[128];
  // sprintf(imgfilename, "screen_%d.png", id);
  // writer->SetFileName(imgfilename);
  // writer->SetInputConnection(windowToImageFilter->GetOutputPort());
  // writer->Write();

  renderWindow->SetOffScreenRendering(1);
  // mtx.lock();
  renderWindow->Render();
  //mtx.unlock();
  
  // mapper->GetDepthImage(depthImage);
  // int* dim_depth = depthImage->GetDimensions ();
  // mapper->GetColorImage(colorImage);
  // int* dim_color = colorImage->GetDimensions ();
  // const char* type_depth = depthImage->GetScalarTypeAsString  (   );
  // const char* type_color = colorImage->GetScalarTypeAsString  (   );
  // int comp_depth = depthImage->GetNumberOfScalarComponents (   );
  // int comp_color = colorImage->GetNumberOfScalarComponents (   );

  // printf("depth dims %d %d %d type %s comp %d\n", dim_depth[0],dim_depth[1],dim_depth[2], type_depth, comp_depth);
  // printf("color dims %d %d %d type %s comp %d\n", dim_color[0],dim_color[1],dim_color[2], type_color, comp_color);

  // char filename[128];
  // std::ofstream out;
  // sprintf(filename, "zmap_%d.raw", id);
  // out.open(filename);
  // unsigned char* zbuf0 = static_cast< unsigned char* >( depthImage->GetScalarPointer () );

  // out.write((const char*)zbuf0, VTKutils::IMAGE_X_DIM*VTKutils::IMAGE_Y_DIM);
  // out.close();

 
  // computeZBuffer(renderWindow, trans[2]/560.f*255.f, id);

  return renderWindow;
}

void VTKutils::writeImage(unsigned char* image, uint32_t* bound, const char* filename)
{
  vtkImageData *blankImage = vtkImageData::New();
  blankImage->SetExtent(0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1,0,1);
  blankImage->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
  blankImage->SetDimensions(VTKutils::IMAGE_X_DIM,VTKutils::IMAGE_Y_DIM, 1);

  unsigned char* pPixel = static_cast< unsigned char* >( blankImage->GetScalarPointer () );//GetScalarPointerForExtent
  memset(pPixel, 0, VTKutils::IMAGE_X_DIM*VTKutils::IMAGE_Y_DIM*3);

  for(int y=0; y < VTKutils::IMAGE_Y_DIM-1; y++)
  {
    for(int x=0; x < VTKutils::IMAGE_X_DIM-1; x++)
    {
      uint32_t idx = x + y*VTKutils::IMAGE_X_DIM; 
      uint32_t imgidx = idx*3;

      if(x >= bound[0] && x < bound[1] && y >= bound[2] && y < bound[3])
      {
        uint32_t x_size = bound[1]-bound[0]+1;
        uint32_t y_size = bound[3]-bound[2]+1;

        uint32_t myidx = (x-bound[0]) + (y-bound[2])*x_size;
        uint32_t myimgidx = myidx*3;

        memcpy(pPixel+imgidx, image+myimgidx, 3*sizeof(unsigned char));
      }
    }
  }

  clock_t start,finish;
  start = clock();

  vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
  writer->SetFileName(filename);
  writer->SetInputData(blankImage);
  writer->Write();

  finish = clock();

  std::cout << "VTKutils::writeImage (" << filename << ") time[s]: " << (double(finish)-double(start))/CLOCKS_PER_SEC << std::endl;
}

void VTKutils::writeImageFixedSize(unsigned char* image, uint32_t* bound, const char* filename)
{
  vtkImageData *blankImage = vtkImageData::New();
  
  int x_size = bound[1]-bound[0]+1;
  int y_size = bound[3]-bound[2]+1;

  blankImage->SetExtent(bound[0],bound[1],bound[2],bound[3],0,1);
  blankImage->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
  blankImage->SetDimensions(x_size, y_size, 1);

  unsigned char* pPixel = static_cast< unsigned char* >( blankImage->GetScalarPointer () );
  memset(pPixel, 0, (x_size*y_size)*3);

  for(int y=0; y < y_size; y++){
      for(int x=0; x < x_size; x++){
        int idx = x + y*x_size; 

        int imgidx = idx*3;
        memcpy(pPixel+imgidx, image+imgidx, 3*sizeof(unsigned char));
        
      }
  }

  vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
  writer->SetFileName(filename);
  writer->SetInputData(blankImage);
  writer->Write();
}

//#ifdef BOUNDING_BOX_OPTIMIZATION
void VTKutils::composite(const std::vector<ImageData>& image_data, 
                         std::vector<VTKutils::ImageData>& out_image, 
                         int id, 
                         int x_factor, 
                         int y_factor, 
                         int n_out /* def: = 2 */)
{
  out_image.resize(n_out);
  uint32_t split_size[2] = {VTKutils::IMAGE_X_DIM/x_factor, VTKutils::IMAGE_Y_DIM/y_factor};
  
  uint32_t x_size = image_data[0].bounds[1]-image_data[0].bounds[0]+1;
  uint32_t y_size = image_data[0].bounds[3]-image_data[0].bounds[2]+1;

  int split_dir = 0;
  if (x_size < y_size) split_dir = 1;

  for(int i=0; i < out_image.size(); i++){
    out_image[i].bounds = new uint32_t[4];
    out_image[i].rend_bounds = new uint32_t[4];

    if(split_dir == 0){
      if(i == 0)
      {
        out_image[i].bounds[0] = image_data[0].bounds[0];
        out_image[i].bounds[1] = image_data[0].bounds[0]+split_size[0]-1; 
      }
      else
      {
        out_image[i].bounds[0] = image_data[0].bounds[0]+split_size[0];
        out_image[i].bounds[1] = image_data[0].bounds[1];
      }

      out_image[i].bounds[2] = image_data[0].bounds[2];
      out_image[i].bounds[3] = image_data[0].bounds[3];

    }else
    {
      if(i == 0)
      {
        out_image[i].bounds[2] = image_data[0].bounds[2];
        out_image[i].bounds[3] = image_data[0].bounds[2]+split_size[1]-1;
      }
      else
      {
        out_image[i].bounds[2] = image_data[0].bounds[2]+split_size[1];
        out_image[i].bounds[3] = image_data[0].bounds[3];
      }

      out_image[i].bounds[0] = image_data[0].bounds[0];
      out_image[i].bounds[1] = image_data[0].bounds[1];
    }
    
  }

  uint32_t rend_size = 0;

  uint32_t union_box[4];
  memcpy(union_box,image_data[0].rend_bounds,4*sizeof(uint32_t));

  for(int i=1; i < image_data.size(); i++)
  {
    computeUnion(union_box, image_data[i].rend_bounds, union_box);
  }

  //printf("union box %d %d ^ %d %d\n", union_box[0],union_box[1],union_box[2],union_box[3]);
  for(int j=0; j < out_image.size(); j++){
    //uint32_t intersection[4];
    bool intersect = computeIntersection(out_image[j].bounds, union_box, out_image[j].rend_bounds);

    rend_size = (out_image[j].rend_bounds[1]-out_image[j].rend_bounds[0]+1)*(out_image[j].rend_bounds[3]-out_image[j].rend_bounds[2]+1);
    out_image[j].image = new unsigned char[rend_size*3];
    memset(out_image[j].image,0,rend_size*3);
    out_image[j].zbuf = new unsigned char[rend_size];
    memset(out_image[j].zbuf,0,rend_size);

    if(rend_size == 0)
      printf("%d image is empty\n", j);
  }

  for(int i=0; i < out_image.size(); i++)
  {
    uint32_t* bound = out_image[i].rend_bounds;
    //uint32_t intersection[4];
   // uint32_t myx_size = bound[1]-bound[0]+1;

    for(int j=0; j < image_data.size(); j++)
    {
      uint32_t c_point = 0;
      // int32_t delta_x = image_data[j].rend_bounds[0]-image_data[j].bounds[0];
      // int32_t delta_y = image_data[j].rend_bounds[2]-image_data[j].bounds[2];
      uint32_t x_size = image_data[j].rend_bounds[1]-image_data[j].rend_bounds[0]+1;
      //uint32_t y_size = image_data[j].rend_bounds[3]-image_data[j].rend_bounds[2]+1;
      
      //bool intersect = computeIntersection(out_image[i].rend_bounds, image_data[j].rend_bounds, intersection);
      
      //if(id == 805306395){
       // printf("out %d %d ^ %d %d in %d %d ^ %d %d\n", bound[0],bound[1],bound[2],bound[3],image_data[j].rend_bounds[0],image_data[j].rend_bounds[1],image_data[j].rend_bounds[2],image_data[j].rend_bounds[3]);
        //printf("intersect %d %d ^ %d %d\n", intersection[0],intersection[1],intersection[2],intersection[3]);
      //}
      // if(!intersect)
      //   continue;
      for(uint32_t y=bound[2]; y < bound[3]+1; y++){
        for(uint32_t x=bound[0]; x < bound[1]+1; x++){

          if(x < image_data[j].rend_bounds[0] || x >= image_data[j].rend_bounds[1] || y < image_data[j].rend_bounds[2] || y >= image_data[j].rend_bounds[3]){
            c_point++;
            continue;
          }

          uint32_t rx = x-image_data[j].rend_bounds[0];//delta_x;
          uint32_t ry = y-image_data[j].rend_bounds[2];//delta_y;

          uint32_t idx = (rx + ry*x_size);
          uint32_t imgidx = idx*3;

          uint32_t myidx = c_point;//(myrx + myry*myx_size);//c_point;
          uint32_t imgmyidx = myidx*3;

          if(out_image[i].zbuf[myidx] < image_data[j].zbuf[idx])
          {
            out_image[i].zbuf[myidx] = image_data[j].zbuf[idx];
            memcpy(out_image[i].image+imgmyidx, image_data[j].image+imgidx, 3*sizeof(unsigned char));
          }

          c_point++;
        }
      }
    }
  }
}

#ifdef BOUNDING_BOX_OPTIMIZATION
void VTKutils::composite(const std::vector<ImageData>& image_data, 
                         ImageData& out_image, 
                         const int id)
{
  //uint32_t out_image.bounds[4] = {0,0,0,0};
  out_image.bounds = new uint32_t[4];
  std::fill_n(out_image.bounds, 4, 0);

  for(int i=0; i < image_data.size(); i++)
    computeUnion(out_image.bounds, image_data[i].bounds, out_image.bounds);
  
  const uint32_t union_x_size = out_image.bounds[1]-out_image.bounds[0]+1;
  const uint32_t union_y_size = out_image.bounds[3]-out_image.bounds[2]+1;

#if DEBUG_IMAGES 
  vtkImageData *blankImage = vtkImageData::New();
  blankImage->SetExtent(out_image.bounds[0],out_image.bounds[1],out_image.bounds[2],out_image.bounds[3],0,1);
  blankImage->SetDimensions(union_x_size,union_y_size,1);
  blankImage->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
  unsigned char* pPixel = static_cast< unsigned char* >( blankImage->GetScalarPointer () );
  memset(pPixel, 0, union_x_size*union_y_size*3);
#endif
  
  const uint32_t this_size = union_x_size*union_y_size;

  out_image.image = new unsigned char[this_size*3];
  memset(out_image.image, 0, this_size*3);
  out_image.zbuf = new unsigned char[this_size];
  memset(out_image.zbuf, 0, this_size);
  
  for(int y=out_image.bounds[2]; y < out_image.bounds[3]+1; y++){
     for(int x=out_image.bounds[0]; x < out_image.bounds[1]+1; x++){
        const uint32_t idx = x + y*union_x_size;

        const uint32_t imgidx = idx*3;

        for(int i=0; i < image_data.size(); i++){
          const uint32_t* bound = image_data[i].bounds;

          if(x >= bound[0] && x < bound[1] && y >= bound[2] && y < bound[3]){
            const uint32_t x_size = bound[1]-bound[0]+1;
            //const uint32_t y_size = bound[3]-bound[2]+1;

            const uint32_t myidx = (x-image_data[i].bounds[0]) + (y-image_data[i].bounds[2])*x_size;
            const uint32_t myimgidx = myidx*3;

            if(out_image.zbuf[idx] < image_data[i].zbuf[myidx]){
              memcpy(out_image.image+imgidx, image_data[i].image+myimgidx, sizeof(unsigned char)*3);

              out_image.zbuf[idx] = image_data[i].zbuf[myidx];

#if DEBUG_IMAGES 
              memcpy(pPixel+imgidx, image_data[i].image+myimgidx, sizeof(unsigned char)*3);
#endif
            }
          }
        }
      }
  }

#if DEBUG_IMAGES 
  vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
  char compimg[128];
  sprintf(compimg, "comp_%d.png", id);
  writer->SetFileName(compimg);
  writer->SetInputData(blankImage);
  writer->Write();
#endif

}

#else

void VTKutils::composite(const std::vector<ImageData>& image_data, 
                         ImageData& out_image, 
                         const int id)
{
  out_image.bounds = new uint32_t[4];
  out_image.rend_bounds = new uint32_t[4];

  memcpy(out_image.bounds, image_data[0].bounds, 4*sizeof(uint32_t));
  memcpy(out_image.rend_bounds, image_data[0].bounds, 4*sizeof(uint32_t));
  
  // const uint32_t union_x_size = out_image.bounds[1]-out_image.bounds[0]+1;
  // const uint32_t union_y_size = out_image.bounds[3]-out_image.bounds[2]+1;

#if DEBUG_IMAGES 
  vtkImageData *blankImage = vtkImageData::New();
  blankImage->SetExtent(0,IMAGE_X_DIM-1,0,IMAGE_Y_DIM-1,0,1);
  blankImage->SetDimensions(IMAGE_X_DIM,IMAGE_Y_DIM,1);
  blankImage->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
  unsigned char* pPixel = static_cast< unsigned char* >( blankImage->GetScalarPointer () );
  memset(pPixel, 0, IMAGE_X_DIM*IMAGE_Y_DIM*3);
#endif
  
  const uint32_t this_size = IMAGE_X_DIM*IMAGE_Y_DIM;

  out_image.image = new unsigned char[this_size*3];
  memset(out_image.image, 0, this_size*3);
  out_image.zbuf = new unsigned char[this_size];
  memset(out_image.zbuf, 0, this_size);
  
  for(int y=0; y < IMAGE_Y_DIM; y++){
     for(int x=0; x < IMAGE_X_DIM; x++){
        const uint32_t idx = x + y*IMAGE_X_DIM;

        const uint32_t imgidx = idx*3;

        for(int i=0; i < image_data.size(); i++){

          if(out_image.zbuf[idx] < image_data[i].zbuf[idx]){
            memcpy(out_image.image+imgidx, image_data[i].image+imgidx, sizeof(unsigned char)*3);

            out_image.zbuf[idx] = image_data[i].zbuf[idx];

#if DEBUG_IMAGES 
            memcpy(pPixel+imgidx, image_data[i].image+imgidx, sizeof(unsigned char)*3);
#endif
          }
        }
      }
  }

#if DEBUG_IMAGES 
  vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
  char compimg[128];
  sprintf(compimg, "comp_%d.png", id);
  writer->SetFileName(compimg);
  writer->SetInputData(blankImage);
  writer->Write();
#endif

}

#endif

//-----------------------------------------------------------------------------------------------

void VTKutils::compositeRadixK(const std::vector<ImageData>& in_images, 
                              std::vector<ImageData>& out_images, 
                              const int id)
{
  uint32_t union_box[4];
  memcpy( union_box, in_images[0].rend_bounds, 4*sizeof(uint32_t) );

  for( uint32_t i = 1; i < in_images.size(); ++i )
  {
    computeUnion( union_box, in_images[i].rend_bounds, union_box );
  }

  splitAndBlend( in_images, out_images, union_box, false );
}

//-----------------------------------------------------------------------------------------------

void VTKutils::splitAndBlend(const std::vector<ImageData>& input_images,
                             std::vector<ImageData>& out_images,
                             uint32_t* union_box,
                             bool skip_z_check)
{
  uint32_t x_size = input_images[0].bounds[1] - input_images[0].bounds[0] + 1;
  uint32_t y_size = input_images[0].bounds[3] - input_images[0].bounds[2] + 1;
  uint32_t split_size[2] = {x_size, y_size};
  
  // Split factor is the size of the out_images array
  int split_dir = 1;    // 0 -- x-axis, 1 -- y-axis
  // TODO: fractions
  //if( x_size < y_size )
    split_size[1] = y_size / out_images.size();
  //else
  //  split_size[0] = x_size / out_images.size();
  
  ////
  std::cout << "VTKutils::splitAndBlend, split_size = " << split_size[1] << std::endl;
  ////
  
  for( int i = 0; i < out_images.size(); ++i )
  {
    ImageData& outimg = out_images[i];
    
    outimg.bounds = new uint32_t[4];
    outimg.rend_bounds = new uint32_t[4];
    
    if( split_dir == 0 )    // Split along x-axis
    {
      outimg.bounds[0] = input_images[0].bounds[0] + i*split_size[0];
      outimg.bounds[1] = input_images[0].bounds[0] + (i+1)*split_size[0] - 1; 
      outimg.bounds[2] = input_images[0].bounds[2];
      outimg.bounds[3] = input_images[0].bounds[3];
    }
    else                    // Split along y-axis
    {
      outimg.bounds[0] = input_images[0].bounds[0];
      outimg.bounds[1] = input_images[0].bounds[1]; 
      outimg.bounds[2] = input_images[0].bounds[2] + i*split_size[1];
      outimg.bounds[3] = input_images[0].bounds[2] + (i+1)*split_size[1] - 1;
    }

    VTKutils::computeIntersection( outimg.bounds, union_box, outimg.rend_bounds );

    uint32_t zsize = 
      (outimg.rend_bounds[1] - outimg.rend_bounds[0] + 1) * (outimg.rend_bounds[3] - outimg.rend_bounds[2] + 1);
    outimg.image = new unsigned char[zsize*3]();
    outimg.zbuf = new unsigned char[zsize]();
    // TODO: memset to zero if zero initialization above doesn't work
    
    if( zsize == 0 )
      printf("%d image is empty\n", i);
  }
  
  ////
  std::cout << "VTKutils::splitAndBlend, before blending skip_z_check = " << skip_z_check << std::endl;
  ////
  
  for( uint32_t j = 0; j < input_images.size(); ++j )      // Blend every input image
  {
    const ImageData& inimg = input_images[j];
    
    uint32_t x_size = inimg.rend_bounds[1] - inimg.rend_bounds[0] + 1;
    
    ////
    std::cout << "VTKutils::splitAndBlend, input image xsize = " << x_size << std::endl;
    std::cout << "VTKutils::splitAndBlend, input image rend_bounds[0] = " << inimg.rend_bounds[0] << std::endl;
    std::cout << "VTKutils::splitAndBlend, input image rend_bounds[2] = " << inimg.rend_bounds[2] << std::endl;
    ////
    
    for( uint32_t i = 0; i < out_images.size(); ++i )
    {
      ImageData& outimg = out_images[i];
      
      uint32_t* bound = outimg.rend_bounds;
      
      std::cout << "VTKutils::splitAndBlend, out image " << i << " bounds = " << bound[0] << " " << bound[1] << " " << bound[2] << " " << bound[3] << std::endl;
      
      for( uint32_t y = bound[2]; y < bound[3] + 1; ++y )
      {
        for( uint32_t x = bound[0]; x < bound[1] + 1; ++x )
        {
          if( x < inimg.rend_bounds[0] || x > inimg.rend_bounds[1] || 
              y < inimg.rend_bounds[2] || y > inimg.rend_bounds[3] )
          {
            std::cout << "VTKutils::splitAndBlend, shouldn't be here, x = " << x << ", y = " << y << std::endl;
            continue;
          }
          
          uint32_t rx = x - inimg.rend_bounds[0];
          uint32_t ry = y - inimg.rend_bounds[2];

          uint32_t idx = (rx + ry*x_size);
          uint32_t imgidx = idx*3;

          uint32_t myidx = (x - bound[0]) + (y - bound[2]) * (bound[1] - bound[0] + 1);
          uint32_t imgmyidx = myidx*3;

          if( skip_z_check || outimg.zbuf[myidx] < inimg.zbuf[idx] )
          {
            outimg.zbuf[myidx] = inimg.zbuf[idx];
            outimg.image[imgmyidx + 0] = inimg.image[imgidx + 0];
            outimg.image[imgmyidx + 1] = inimg.image[imgidx + 1];
            outimg.image[imgmyidx + 2] = inimg.image[imgidx + 2];
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------------------------

template<typename T>
int VTKCompositeRender::volumeRender(uint32_t* box, 
                                     char* data, 
                                     VTKutils::ImageData& image_data, 
                                     int id)
{
#if DISABLE_RENDERING
  uint32_t this_size = VTKutils::IMAGE_X_DIM*VTKutils::IMAGE_Y_DIM;
  uint32_t zsize = this_size;
  uint32_t psize = this_size*3;
  image_data.image = new unsigned char[psize];
  image_data.zbuf = new unsigned char[zsize];
  image_data.bounds = new uint32_t[4];
  image_data.rend_bounds = new uint32_t[4];

  image_data.bounds[0] = 0;
  image_data.bounds[2] = 0;
  image_data.bounds[1] = VTKutils::IMAGE_X_DIM-1;
  image_data.bounds[3] = VTKutils::IMAGE_Y_DIM-1;
  memcpy(image_data.rend_bounds, image_data.bounds, 4*sizeof(uint32_t));
  memset(image_data.image, 0, psize);
  memset(image_data.zbuf, 0, zsize);

  return 0;
#else
  uint32_t dims[3] = {box[1]-box[0]+1,box[3]-box[2]+1,box[5]-box[4]+1};

  vtkSmartPointer<vtkImageData> surface = VTKutils::getImageData<T>(data,dims);

  vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();
  // depthImage->SetExtent(0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1,0,1);
  // depthImage->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
  // depthImage->SetDimensions(VTKutils::IMAGE_X_DIM,VTKutils::IMAGE_Y_DIM, 1);

  vtkSmartPointer<vtkImageData> colorImage = vtkSmartPointer<vtkImageData>::New();
  // colorImage->SetExtent(0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1,0,1);
  // colorImage->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
  // colorImage->SetDimensions(VTKutils::IMAGE_X_DIM,VTKutils::IMAGE_Y_DIM, 1);

  uint32_t trans[3] = {box[0],box[2],box[4]};
  vtkSmartPointer<vtkRenderWindow> ren = VTKutils::render(surface, trans, depthImage, colorImage, id);
  
 // unsigned char* zbuf0 = static_cast< unsigned char* >( depthImage->GetScalarPointer () );
 //ren->GetZbufferData(0,0,VTKutils::IMAGE_X_DIM-1,VTKutils::IMAGE_Y_DIM-1);

  uint32_t bounds_size = 4*sizeof(uint32_t);

  image_data.bounds = new uint32_t[4];
  image_data.rend_bounds = new uint32_t[4];

  uint32_t in_bounds[4] = {0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1};
  memcpy(image_data.bounds, in_bounds, bounds_size);
  memcpy(image_data.rend_bounds, in_bounds, bounds_size);

  //VTKutils::compute2DBounds(zbuf0, in_bounds, image_data.bounds);
 
  uint32_t x_size = image_data.bounds[1]-image_data.bounds[0]+1;
  uint32_t y_size = image_data.bounds[3]-image_data.bounds[2]+1;
  
  //printf("%d: bounds %d %d : %d %d size %dx%d\n",id, bounds[0],bounds[1],bounds[2],bounds[3], x_size, y_size);
  
  vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  windowToImageFilter->SetInput(ren);
  windowToImageFilter->SetInputBufferTypeToRGBA();
  windowToImageFilter->ShouldRerenderOn();
  windowToImageFilter->Update();
  vtkImageData* img0 = windowToImageFilter->GetOutput();

  uint32_t this_size = (image_data.bounds[1]-image_data.bounds[0]+1)*(image_data.bounds[3]-image_data.bounds[2]+1);

  unsigned char* pPixel = static_cast< unsigned char* >( img0->GetScalarPointer());//colorImage->GetScalarPointer());//
  // printf ("IMAGE SIZE %d %d\n", VTKutils::IMAGE_X_DIM, VTKutils::IMAGE_Y_DIM);
  
  image_data.image = new unsigned char[this_size*3];
  image_data.zbuf = new unsigned char[this_size];

  for(uint32_t y=image_data.bounds[2]; y < image_data.bounds[3]+1; y++){
    for(uint32_t x=image_data.bounds[0]; x < image_data.bounds[1]+1; x++){
       uint32_t idx = (x + y*VTKutils::IMAGE_X_DIM );
       uint32_t myidx = ((x-image_data.bounds[0]) + (y-image_data.bounds[2])*x_size);

       uint32_t imgidx = idx*4; // four components!!
       uint32_t imgmyidx = myidx*3;
       //printf("%d:%d myidx %d idx %d\n", x,y,myidx, idx);

       //image_data.zbuf[myidx] = ( unsigned char ) -255+(zbuf0[idx]*255);//zbuf0[idx];c_zbuf[idx];//
       image_data.zbuf[idx] = pPixel[imgidx+3];//( unsigned char ) -255+(zbuf0[idx]*255);//zbuf0[idx];c_zbuf[idx];//
       memcpy(image_data.image+imgmyidx, pPixel+imgidx, 3*sizeof(unsigned char));

     }
  }

  // char filename[128];
  // std::ofstream out;
  // sprintf(filename, "zmap_%d.raw", id);
  // out.open(filename);

  // out.write((const char*)image_data.zbuf, VTKutils::IMAGE_X_DIM*VTKutils::IMAGE_Y_DIM);
  // out.close();

  // char imgfilename[128];
  // sprintf(imgfilename, "image_%d.png", id);
  // utils.arraytoImage_fixed_size(image_data.image, image_data.bounds, imgfilename);
  
  return true;
#endif
}

//-----------------------------------------------------------------------------------------------

template<typename T>
int VTKCompositeRender::volumeRender(uint32_t* box, 
                                     char* data, 
                                     std::vector<VTKutils::ImageData>& images, 
                                     int id, 
                                     int x_factor, 
                                     int y_factor)
{
#if DISABLE_RENDERING
  for(int i=0; i<2; i++)
  { 
    VTKutils::ImageData image_data;
    uint32_t this_size = (VTKutils::IMAGE_X_DIM/x_factor)*(VTKutils::IMAGE_Y_DIM/y_factor);
    uint32_t zsize = this_size;
    uint32_t psize = this_size*3;
    image_data.image = new unsigned char[psize];
    image_data.zbuf = new unsigned char[zsize];
    image_data.bounds = new uint32_t[4];
    image_data.rend_bounds = new uint32_t[4];

    image_data.bounds[0] = 0;
    image_data.bounds[2] = 0;
    image_data.bounds[1] = VTKutils::IMAGE_X_DIM/x_factor-1;
    image_data.bounds[3] = VTKutils::IMAGE_Y_DIM/y_factor-1;
    memcpy(image_data.rend_bounds,image_data.bounds,sizeof(uint32_t)*4);
    memset(image_data.image, 0, psize);
    memset(image_data.zbuf, 0, zsize);

    images.push_back(image_data);
  }
  return 0;
#else

  uint32_t dims[3] = {box[1]-box[0]+1,box[3]-box[2]+1,box[5]-box[4]+1};
  uint32_t trans[3] = {box[0],box[2],box[4]};

  //vtkSmartPointer<vtkMarchingCubes> surface = VTKutils::getSurface<T>(data, isovalue, dims);
  //vtkSmartPointer<vtkRenderWindow> ren = VTKutils::render(surface, trans, id);

  vtkSmartPointer<vtkImageData> surface = VTKutils::getImageData<T>(data, dims);
  vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();  // unused
  vtkSmartPointer<vtkImageData> colorImage = vtkSmartPointer<vtkImageData>::New();  // unused

  vtkSmartPointer<vtkRenderWindow> ren = VTKutils::render(surface, trans, depthImage, colorImage, id);

  float* zbuf0 = ren->GetZbufferData(0,0,VTKutils::IMAGE_X_DIM-1,VTKutils::IMAGE_Y_DIM-1);
  uint32_t in_bounds[4] = {0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1};
  uint32_t rend_bounds[4];
  VTKutils::compute2DBounds(zbuf0, in_bounds, rend_bounds);
  
  vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  windowToImageFilter->SetInput(ren);
  windowToImageFilter->SetInputBufferTypeToRGBA();
  windowToImageFilter->ShouldRerenderOn();
  windowToImageFilter->Update();
  vtkImageData* img0 = windowToImageFilter->GetOutput();
  
  images.resize(2);

  uint32_t split_size[2] = {VTKutils::IMAGE_X_DIM/x_factor, VTKutils::IMAGE_Y_DIM/y_factor};
  //uint32_t this_size = split_size[0]*split_size[1];

  unsigned char* pPixel = static_cast< unsigned char* >( img0->GetScalarPointer());
  //printf ("IMAGE SIZE %d %d\n", split_size[0],split_size[1]);
  uint32_t this_size = 0;

  for(int i=0; i < images.size(); i++)
  {
    images[i].bounds = new uint32_t[4];
    images[i].rend_bounds = new uint32_t[4];
    
    if(i == 0)
    {
      images[i].bounds[0] = 0;
      images[i].bounds[1] = split_size[0] -1;
    }
    else
    {
      images[i].bounds[0] = split_size[0];
      images[i].bounds[1] = VTKutils::IMAGE_X_DIM -1;
    }

    images[i].bounds[2] = 0;
    images[i].bounds[3] = VTKutils::IMAGE_Y_DIM -1;

    VTKutils::computeIntersection(images[i].bounds, rend_bounds, images[i].rend_bounds);

    this_size = (images[i].rend_bounds[1]-images[i].rend_bounds[0]+1)*(images[i].rend_bounds[3]-images[i].rend_bounds[2]+1);
    images[i].image = new unsigned char[this_size*3];
    images[i].zbuf = new unsigned char[this_size];

    // printf("iso %d\n", id);
    // printf("iso bounds %d %d ^ %d %d\n", images[i].bounds[0],images[i].bounds[1],images[i].bounds[2],images[i].bounds[3]);
    // printf("iso rend_bounds %d %d ^ %d %d\n", images[i].rend_bounds[0],images[i].rend_bounds[1],images[i].rend_bounds[2],images[i].rend_bounds[3]);
  
  }

  for(int i=0; i < images.size(); i++)
  {
    uint32_t* bound = images[i].rend_bounds;
    //uint32_t x_size = bound[1]-bound[0]+1;
    //uint32_t y_size = images[0].bounds[3]-images[0].bounds[2]+1;
    uint32_t c_point = 0;

    for(int y=bound[2]; y < bound[3]+1; y++){
      for(int x=bound[0]; x < bound[1]+1; x++){
         uint32_t idx = (x + y*VTKutils::IMAGE_X_DIM);
         uint32_t imgidx = idx*4;  //idx*3;

         uint32_t myidx = c_point;
         uint32_t imgmyidx = myidx*3;

         images[i].zbuf[myidx] = pPixel[imgidx+3];  //( unsigned char ) -255+(zbuf0[idx]*255);
         memcpy(images[i].image + imgmyidx, pPixel + imgidx, 3*sizeof(unsigned char));

         c_point++;
       }
    }

  }
#endif
}

//-----------------------------------------------------------------------------------------------

template<typename T>
int VTKCompositeRender::volumeRenderRadixK(uint32_t* box, 
                                           char* data, 
                                           std::vector<VTKutils::ImageData>& out_images, 
                                           int id)
{
#if DISABLE_RENDERING
  uint32_t factor = out_images.size();
  for( uint32_t i = 0; i < out_images.size(); ++i )
  { 
    VTKutils::ImageData& image_data = out_images[i];
    
    uint32_t this_size = VTKutils::IMAGE_X_DIM * (VTKutils::IMAGE_Y_DIM / factor);
    uint32_t zsize = this_size;
    uint32_t psize = this_size*3;
    image_data.image = new unsigned char[psize];
    image_data.zbuf = new unsigned char[zsize];
    image_data.bounds = new uint32_t[4];
    image_data.rend_bounds = new uint32_t[4];

    image_data.bounds[0] = 0;
    image_data.bounds[2] = 0;
    image_data.bounds[1] = VTKutils::IMAGE_X_DIM - 1;
    image_data.bounds[3] = VTKutils::IMAGE_Y_DIM/factor - 1;
    memcpy(image_data.rend_bounds,image_data.bounds,sizeof(uint32_t)*4);
    memset(image_data.image, 0, psize);
    memset(image_data.zbuf, 0, zsize);
  }
  return 0;
#else

  uint32_t dims[3] = {box[1]-box[0]+1,box[3]-box[2]+1,box[5]-box[4]+1};
  uint32_t trans[3] = {box[0],box[2],box[4]};
  
  ////
  std::cout << "VTKCompositeRender::volumeRenderRadixK, dims = " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  ////

  vtkSmartPointer<vtkImageData> surface = VTKutils::getImageData<T>(data, dims);
  vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();  // unused
  vtkSmartPointer<vtkImageData> colorImage = vtkSmartPointer<vtkImageData>::New();  // unused

  vtkSmartPointer<vtkRenderWindow> ren = VTKutils::render(surface, trans, depthImage, colorImage, id);
  
  ////
  std::cout << "VTKCompositeRender::volumeRenderRadixK, after render" << std::endl;
  ////

  float* zbuf0 = ren->GetZbufferData(0, 0, VTKutils::IMAGE_X_DIM - 1, VTKutils::IMAGE_Y_DIM - 1);
  uint32_t in_bounds[4] = {0, VTKutils::IMAGE_X_DIM - 1, 0, VTKutils::IMAGE_Y_DIM - 1};
  uint32_t rend_bounds[4];
  VTKutils::compute2DBounds(zbuf0, in_bounds, rend_bounds);
  
  vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  windowToImageFilter->SetInput(ren);
  windowToImageFilter->SetInputBufferTypeToRGBA();
  windowToImageFilter->ShouldRerenderOn();
  windowToImageFilter->Update();
  vtkImageData* img0 = windowToImageFilter->GetOutput();
  
  unsigned char* pPixel = static_cast< unsigned char* >( img0->GetScalarPointer());

  std::vector<VTKutils::ImageData> input_images( 1 );   // Rendered data is input for the splitAndBlend function
  
  VTKutils::ImageData& image_data = input_images[0];
  uint32_t zsize = VTKutils::IMAGE_X_DIM * VTKutils::IMAGE_Y_DIM;
  uint32_t psize = zsize*3;
  image_data.image = new unsigned char[psize];
  image_data.zbuf = new unsigned char[zsize];
  image_data.bounds = new uint32_t[4];
  image_data.rend_bounds = new uint32_t[4];
  image_data.bounds[0] = 0;
  image_data.bounds[2] = 0;
  image_data.bounds[1] = VTKutils::IMAGE_X_DIM - 1;
  image_data.bounds[3] = VTKutils::IMAGE_Y_DIM - 1;
  memcpy( image_data.rend_bounds, image_data.bounds, sizeof(uint32_t)*4 );
  for( uint32_t i = 0; i < zsize; ++i )
  {
    uint32_t imgidx = i*4;
    uint32_t imgmyidx = i*3;
    image_data.zbuf[i] = pPixel[imgidx+3];
    image_data.image[imgmyidx + 0] = pPixel[imgidx + 0];
    image_data.image[imgmyidx + 1] = pPixel[imgidx + 1];
    image_data.image[imgmyidx + 2] = pPixel[imgidx + 2];
  }
  
  ////
  std::cout << "VTKCompositeRender::volumeRenderRadixK, before splitAndBend" << std::endl;
  ////
  
  VTKutils::splitAndBlend( input_images, out_images, rend_bounds, true );

  ////
  std::cout << "VTKCompositeRender::volumeRenderRadixK, after splitAndBend" << std::endl;
  ////
  
  delete[] input_images[0].zbuf;
  delete[] input_images[0].image;
  delete[] input_images[0].bounds;
  delete[] input_images[0].rend_bounds;

#endif
}

//-----------------------------------------------------------------------------------------------

template<typename T>
int VTKCompositeRender::isosurfaceRender(uint32_t* box, 
                                         char* data, 
                                         float isovalue, 
                                         VTKutils::ImageData& image_data, 
                                         int id)
{
#if DISABLE_RENDERING
  uint32_t this_size = VTKutils::IMAGE_X_DIM*VTKutils::IMAGE_Y_DIM;
  uint32_t zsize = this_size;
  uint32_t psize = this_size*3;
  image_data.image = new unsigned char[psize];
  image_data.zbuf = new unsigned char[zsize];
  image_data.bounds = new uint32_t[4];
  image_data.rend_bounds = new uint32_t[4];

  image_data.bounds[0] = 0;
  image_data.bounds[2] = 0;
  image_data.bounds[1] = VTKutils::IMAGE_X_DIM-1;
  image_data.bounds[3] = VTKutils::IMAGE_Y_DIM-1;
  memcpy(image_data.rend_bounds, image_data.bounds, 4*sizeof(uint32_t));
  memset(image_data.image, 0, psize);
  memset(image_data.zbuf, 0, zsize);

  return 0;
#else

  uint32_t dims[3] = {box[1]-box[0]+1,box[3]-box[2]+1,box[5]-box[4]+1};

  vtkSmartPointer<vtkMarchingCubes> surface = VTKutils::getSurface<T>(data, isovalue, dims);

  uint32_t trans[3] = {box[0],box[2],box[4]};
  vtkSmartPointer<vtkRenderWindow> ren = VTKutils::render(surface, trans, id);

  float* zbuf0 = ren->GetZbufferData(0,0,VTKutils::IMAGE_X_DIM-1,VTKutils::IMAGE_Y_DIM-1);

  uint32_t bounds_size = 4*sizeof(int);

  image_data.bounds = new uint32_t[4];
  image_data.rend_bounds = new uint32_t[4];

  uint32_t in_bounds[4] = {0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1};
  VTKutils::compute2DBounds(zbuf0, in_bounds, image_data.bounds);
  memcpy(image_data.rend_bounds, image_data.bounds, 4*sizeof(uint32_t));

  uint32_t x_size = image_data.bounds[1]-image_data.bounds[0]+1;
  uint32_t y_size = image_data.bounds[3]-image_data.bounds[2]+1;
  
  //printf("%d: bounds %d %d : %d %d size %dx%d\n",id, image_data.bounds[0],image_data.bounds[1],image_data.bounds[2],image_data.bounds[3], x_size, y_size);
  
  vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  windowToImageFilter->SetInput(ren);
  windowToImageFilter->SetInputBufferTypeToRGB();
  windowToImageFilter->ShouldRerenderOn();
  windowToImageFilter->Update();
  vtkImageData* img0 = windowToImageFilter->GetOutput();

  uint32_t this_size = (image_data.bounds[1]-image_data.bounds[0]+1)*(image_data.bounds[3]-image_data.bounds[2]+1);

  unsigned char* pPixel = static_cast< unsigned char* >( img0->GetScalarPointer());
  //printf ("IMAGE SIZE %d %d\n", VTKutils::IMAGE_X_DIM, VTKutils::IMAGE_Y_DIM);
  image_data.image = new unsigned char[this_size*3];
  image_data.zbuf = new unsigned char[this_size];

  for(uint32_t y=image_data.bounds[2]; y < image_data.bounds[3]+1; y++){
    for(uint32_t x=image_data.bounds[0]; x < image_data.bounds[1]+1; x++){
       uint32_t idx = (x + y*VTKutils::IMAGE_X_DIM );
       uint32_t myidx = ((x-image_data.bounds[0]) + (y-image_data.bounds[2])*x_size);

       uint32_t imgidx = idx*3;
       uint32_t imgmyidx = myidx*3;
       //printf("%d:%d myidx %d idx %d\n", x,y,myidx, idx);

       image_data.zbuf[myidx] = ( unsigned char ) -255+(zbuf0[idx]*255);//zbuf0[idx];c_zbuf[idx];//
       memcpy(image_data.image+imgmyidx, pPixel+imgidx, 3*sizeof(unsigned char));

     }
  }
  
  return true;
#endif
}

//-----------------------------------------------------------------------------------------------

template<typename T>
int VTKCompositeRender::isosurfaceRender(uint32_t* box, 
                                         char* data, 
                                         float isovalue, 
                                         std::vector<VTKutils::ImageData>& images, 
                                         int id, 
                                         int x_factor, 
                                         int y_factor)
{
#if DISABLE_RENDERING
  for(int i=0; i<2; i++){ 
  VTKutils::ImageData image_data;
  uint32_t this_size = (VTKutils::IMAGE_X_DIM/x_factor)*(VTKutils::IMAGE_Y_DIM/y_factor);
  uint32_t zsize = this_size;
  uint32_t psize = this_size*3;
  image_data.image = new unsigned char[psize];
  image_data.zbuf = new unsigned char[zsize];
  image_data.bounds = new uint32_t[4];
  image_data.rend_bounds = new uint32_t[4];

  image_data.bounds[0] = 0;
  image_data.bounds[2] = 0;
  image_data.bounds[1] = VTKutils::IMAGE_X_DIM/x_factor-1;
  image_data.bounds[3] = VTKutils::IMAGE_Y_DIM/y_factor-1;
  memcpy(image_data.rend_bounds,image_data.bounds,sizeof(uint32_t)*4);
  memset(image_data.image, 0, psize);
  memset(image_data.zbuf, 0, zsize);

  images.push_back(image_data);
}
  return 0;
#else

  uint32_t dims[3] = {box[1]-box[0]+1,box[3]-box[2]+1,box[5]-box[4]+1};

  vtkSmartPointer<vtkMarchingCubes> surface = VTKutils::getSurface<T>(data, isovalue, dims);

  uint32_t trans[3] = {box[0],box[2],box[4]};
  vtkSmartPointer<vtkRenderWindow> ren = VTKutils::render(surface, trans, id);

  float* zbuf0 = ren->GetZbufferData(0,0,VTKutils::IMAGE_X_DIM-1,VTKutils::IMAGE_Y_DIM-1);

  uint32_t bounds_size = 4*sizeof(int);

  uint32_t in_bounds[4] = {0,VTKutils::IMAGE_X_DIM-1,0,VTKutils::IMAGE_Y_DIM-1};
  uint32_t rend_bounds[4];
  VTKutils::compute2DBounds(zbuf0, in_bounds, rend_bounds);
  
  vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  windowToImageFilter->SetInput(ren);
  windowToImageFilter->SetInputBufferTypeToRGB();
  windowToImageFilter->ShouldRerenderOn();
  windowToImageFilter->Update();
  vtkImageData* img0 = windowToImageFilter->GetOutput();
  
  images.resize(2);

  uint32_t split_size[2] = {VTKutils::IMAGE_X_DIM/x_factor, VTKutils::IMAGE_Y_DIM/y_factor};
  //uint32_t this_size = split_size[0]*split_size[1];

  unsigned char* pPixel = static_cast< unsigned char* >( img0->GetScalarPointer());
  //printf ("IMAGE SIZE %d %d\n", split_size[0],split_size[1]);
  uint32_t this_size = 0;

  for(int i=0; i < images.size(); i++){
    images[i].bounds = new uint32_t[4];
    images[i].rend_bounds = new uint32_t[4];
    
    if(i == 0)
    {
      images[i].bounds[0] = 0;
      images[i].bounds[1] = split_size[0] -1;
    }
    else
    {
      images[i].bounds[0] = split_size[0];
      images[i].bounds[1] = VTKutils::IMAGE_X_DIM -1;
    }

    images[i].bounds[2] = 0;
    images[i].bounds[3] = VTKutils::IMAGE_Y_DIM -1;

    VTKutils::computeIntersection(images[i].bounds, rend_bounds, images[i].rend_bounds);

    this_size = (images[i].rend_bounds[1]-images[i].rend_bounds[0]+1)*(images[i].rend_bounds[3]-images[i].rend_bounds[2]+1);
    images[i].image = new unsigned char[this_size*3];
    images[i].zbuf = new unsigned char[this_size];

    // printf("iso %d\n", id);
    // printf("iso bounds %d %d ^ %d %d\n", images[i].bounds[0],images[i].bounds[1],images[i].bounds[2],images[i].bounds[3]);
    // printf("iso rend_bounds %d %d ^ %d %d\n", images[i].rend_bounds[0],images[i].rend_bounds[1],images[i].rend_bounds[2],images[i].rend_bounds[3]);
  
  }

  for(int i=0; i < images.size(); i++){
    uint32_t* bound = images[i].rend_bounds;
    //uint32_t x_size = bound[1]-bound[0]+1;
    //uint32_t y_size = images[0].bounds[3]-images[0].bounds[2]+1;
    uint32_t c_point = 0;

    for(int y=bound[2]; y < bound[3]+1; y++){
      for(int x=bound[0]; x < bound[1]+1; x++){
         uint32_t idx = (x + y*VTKutils::IMAGE_X_DIM);
         uint32_t imgidx = idx*3;

         uint32_t myidx = c_point;
         uint32_t imgmyidx = myidx*3;

         images[i].zbuf[myidx] = ( unsigned char ) -255+(zbuf0[idx]*255);
         memcpy(images[i].image +imgmyidx, pPixel+imgidx, 3*sizeof(unsigned char));

         c_point++;
       }
    }

  }


  // for(int y=0; y < VTKutils::IMAGE_Y_DIM; y++){
  //   for(int x=0; x < VTKutils::IMAGE_X_DIM; x++){
  //      uint32_t idx = (x + y*VTKutils::IMAGE_X_DIM);
  //      uint32_t imgidx = idx*3;

  //      int sel_img = 0;

  //      if(x >= split_size[0]){
  //         sel_img = 1;
  //      }

  //      uint32_t myidx = (x%(split_size[0])) + ((y%(split_size[1]))*x_size);//((x-images[sel_img].bounds[0]) + (y-images[sel_img].bounds[2])*x_size);
  //      uint32_t imgmyidx = myidx*3;

  //      images[sel_img].zbuf[myidx] = ( unsigned char ) -255+(zbuf0[idx]*255);
  //      memcpy(images[sel_img].image +imgmyidx, pPixel+imgidx, 3*sizeof(unsigned char));

  //    }
  // }
  
  return true;
#endif
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

