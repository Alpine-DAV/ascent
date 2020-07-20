//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory //
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
/// file: ascent_runtime_camera_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_simplex_filters.hpp"
#include "ascent_runtime_camera_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>
#include <ascent_data_object.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/PointRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/rendering/ScalarRenderer.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/Gradient.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/filters/Log.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/Recenter.hpp>
#include <vtkh/filters/Slice.hpp>
#include <vtkh/filters/Statistics.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>
#include <vtkh/filters/Histogram.hpp>
#include <vtkh/filters/HistSampling.hpp>
#include <vtkm/cont/DataSet.h>
#include <vtkm/rendering/raytracing/Camera.h>
#include <vtkm/cont/ArrayCopy.h>


#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <chrono>
#include <stdio.h>

//openCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace conduit;
using namespace std;
using namespace std::chrono;

using namespace flow;

typedef vtkm::rendering::Camera vtkmCamera;

void fibonacciSphere(int i, int samples, double* points)
{
  int rnd = 1;
  //if randomize:
  //    rnd = random.random() * samples

  double offset = 2./samples;
  double increment = M_PI * (3. - sqrt(5.));


  double y = ((i * offset) - 1) + (offset / 2);
  double r = sqrt(1 - pow(y,2));

  double phi = ((i + rnd) % samples) * increment;

  double x = cos(phi) * r;
  double z = sin(phi) * r;
  points[0] = x;
  points[1] = y;
  points[2] = z;
}

#include <cmath> /* using fmod for modulo on doubles */

/* This is a test of overloading it for doubles to use in place of linear interpolation */
void fibonacciSphere(double i, int samples, double* points)
{
  /* This a bit odd but it didn't seem to do well when doing over samples */
  /* It might be bcause of the i + rnd */
  if (i > (samples-1)){
      i = i - (samples - 1);
  }
  /* */

  int rnd = 1;
  //if randomize:
  //    rnd = random.random() * samples

  double offset = 2./samples;
  double increment = M_PI * (3. - sqrt(5.));


  double y = ((i * offset) - 1) + (offset / 2);
  double r = sqrt(1 - pow(y,2));

  double phi = ( fmod((i + rnd), (double)samples) * increment);

  double x = cos(phi) * r;
  double z = sin(phi) * r;
  points[0] = x;
  points[1] = y;
  points[2] = z;
}

Camera
GetCamera2(int frame, int nframes, double radius, double* lookat)
{
//  double t = SineParameterize(frame, nframes, nframes/10);
  double points[3];
  fibonacciSphere(frame, nframes, points);
  Camera c;
  double zoom = 3.0;
//  c.near = zoom/20;
//  c.far = zoom*25;
//  c.angle = M_PI/6;

/*  if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
  {
    if(points[2] >= 0)
      points[2] += radius;
    if(points[2] < 0)
      points[2] -= radius;
  }*/

  c.position[0] = zoom*radius*points[0];
  c.position[1] = zoom*radius*points[1];
  c.position[2] = zoom*radius*points[2];

//cout << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
    
//  c.focus[0] = lookat[0];
//  c.focus[1] = lookat[1];
//  c.focus[2] = lookat[2];
//  c.up[0] = 0;
//  c.up[1] = 1;
//  c.up[2] = 0;
  return c;
}

/* This is a test of overloading it for doubles to use in place of linear interpolation */
Camera
GetCamera2(double frame, int nframes, double radius, double* lookat)
{
//  double t = SineParameterize(frame, nframes, nframes/10);
  double points[3];
  fibonacciSphere(frame, nframes, points);
  Camera c;
  double zoom = 3.0;
//  c.near = zoom/20;
//  c.far = zoom*25;
//  c.angle = M_PI/6;

/*  if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
  {
    if(points[2] >= 0)
      points[2] += radius;
    if(points[2] < 0)
      points[2] -= radius;
  }*/

  c.position[0] = zoom*radius*points[0];
  c.position[1] = zoom*radius*points[1];
  c.position[2] = zoom*radius*points[2];

//cout << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
    
//  c.focus[0] = lookat[0];
//  c.focus[1] = lookat[1];
//  c.focus[2] = lookat[2];
//  c.up[0] = 0;
//  c.up[1] = 1;
//  c.up[2] = 0;
  return c;
}

#include <cmath>
//Use this file stuff later but figured I'd put it next to mine
#include <iostream>
#include <fstream>

Camera
GetCamera3(double x0, double x1, double y0, double y1, double z0, double z1, double radius,
	       	int thetaPos, int numTheta, int phiPos, int numPhi)
{
  Camera c;
  double zoom = 3.0;
  
  double theta = (thetaPos / (numTheta - 1.0)) * M_PI ;
  double phi = (phiPos / (numPhi - 1.0)) * M_PI * 2.0; 
  double xm = (x0 + x1) / 2.0;
  double ym = (y0 + y1) / 2.0;
  double zm = (z0 + z1) / 2.0;

  c.position[0] = ( ( radius * sin(theta) * cos(phi) ) + xm ) * zoom;
  c.position[1] = ( ( radius * sin(theta) * sin(phi) ) + ym ) * zoom;
  c.position[2] = ( ( radius * cos(theta) ) + zm ) * zoom;

  return c;
}

#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/worklet/WorkletMapTopology.h>

class ProcessTriangle : public vtkm::worklet::WorkletVisitCellsWithPoints
{
public:
  // This is to tell the compiler what inputs to expect.
  // For now we'll be providing the CellSet, CooridnateSystem,
  // an input variable, and an output variable.
  using ControlSignature = void(CellSetIn cellset,
                                FieldInPoint points,
                                FieldInPoint variable,
                                FieldOutCell output);

  // After VTK-m does it's magic, you need to tell what information you need
  // from the provided inputs from the ControlSignature.
  // For now :
  // 1. number of points making an individual cell
  // 2. _2 is the 2nd input to ControlSignature : this should give you all points of the triangle.
  // 3. _3 is the 3rd input to ControlSignature : this should give you the variables at those points.
  // 4. _4 is the 4rd input to ControlSignature : this will help you store the output of your calculation.
  using ExecutionSignature = void(PointCount, _2, _3, _4);

  template <typename PointVecType, typename FieldVecType>
  VTKM_EXEC
  void operator()(const vtkm::IdComponent& numPoints,
                  const PointVecType& points,
                  const FieldVecType& variable,
                  Triangle& output) const
  {
    if(numPoints != 3)
      ASCENT_ERROR("We only play with triangles here");
    // Since you only have triangles, numPoints should always be 3
    // PointType is an abstraction of vtkm::Vec3f, which is {x, y, z} of a point
    using PointType = typename PointVecType::ComponentType;
    using FieldType = typename FieldVecType::ComponentType;
    // Following lines will help you extract points for a single triangle
    //PointType vertex0 = points[0]; // {x0, y0, z0}
    //PointType vertex1 = points[1]; // {x1, y1, z1}
    //PointType vertex2 = points[2]; // {x2, y2, z2}
    output.X[0] = points[0][0];
    output.Y[0] = points[0][1];
    output.Z[0] = points[0][2];
    output.X[1] = points[1][0];
    output.Y[1] = points[1][1];
    output.Z[1] = points[1][2];
    output.X[2] = points[2][0];
    output.Y[2] = points[2][1];
    output.Z[2] = points[2][2]; 
  }
};

std::vector<Triangle>
GetTriangles2(vtkh::DataSet &vtkhData, std::string field_name)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<Triangle> tris;

   
     
  //if there is data: loop through domains and grab all triangles.
  if(!vtkhData.IsEmpty())
  {
    for(int i = 0; i < localDomainIds.size(); i++)
    {
      vtkm::cont::DataSet dataset = vtkhData.GetDomain(localDomainIds[i]);
      //Get Data points
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      //Get triangles
      vtkm::cont::DynamicCellSet cellset = dataset.GetCellSet();
      //Get variable
      vtkm::cont::Field field = dataset.GetField(field_name);

      int numTris = cellset.GetNumberOfCells();
      std::vector<Triangle> tmp_tris(numTris);
     
      vtkm::cont::ArrayHandle<Triangle> triangles = vtkm::cont::make_ArrayHandle(tmp_tris);
      vtkm::cont::Invoker invoker;
      invoker(ProcessTriangle{}, cellset, coords, field.GetData().ResetTypes(vtkm::TypeListFieldScalar{}), triangles);

      //combine all domain triangles
      tris.insert(tris.end(), tmp_tris.begin(), tmp_tris.end());

    }
  }
  return tris;
}

std::vector<float>
GetScalarData2(vtkh::DataSet &vtkhData, std::string field_name, int height, int width)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<float> data;

   
     
  //if there is data: loop through domains and grab all triangles.
  if(!vtkhData.IsEmpty())
  {
    for(int i = 0; i < localDomainIds.size(); i++)
    {
      vtkm::cont::DataSet dataset = vtkhData.GetDomain(localDomainIds[i]);
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      vtkm::cont::DynamicCellSet cellset = dataset.GetCellSet();
      //Get variable
      vtkm::cont::Field field = dataset.GetField(field_name);
      
      vtkm::cont::ArrayHandle<float> field_data;
      field.GetData().CopyTo(field_data);
      auto portal = field_data.GetPortalConstControl();

      for(int i = 0; i < height*width; i++)
        data.push_back(portal.Get(i));
      
    }
  }
  return data;
}

Triangle transformTriangle2(Triangle t, Camera c)
{
  bool print = false;
  Matrix camToView, m0, cam, view;
  cam = c.CameraTransform();
  view = c.ViewTransform();
  camToView = Matrix::ComposeMatrices(cam, view);
  m0 = Matrix::ComposeMatrices(camToView, c.DeviceTransform());

  Triangle triangle;
  // Zero XYZ
  double * pointOut = new double[4];
  double * pointIn  = new double[4];
  pointIn[0] = t.X[0];
  pointIn[1] = t.Y[0];
  pointIn[2] = t.Z[0];
  pointIn[3] = 1; //w
  m0.TransformPoint(pointIn, pointOut);
  triangle.X[0] = (pointOut[0]/pointOut[3]); //DIVIDE BY W!!	
  triangle.Y[0] = (pointOut[1]/pointOut[3]);
  triangle.Z[0] = (pointOut[2]/pointOut[3]);

  //One XYZ
  pointIn[0] = t.X[1];
  pointIn[1] = t.Y[1];
  pointIn[2] = t.Z[1];
  pointIn[3] = 1; //w
  m0.TransformPoint(pointIn, pointOut);
  triangle.X[1] = (pointOut[0]/pointOut[3]); //DIVIDE BY W!!	
  triangle.Y[1] = (pointOut[1]/pointOut[3]);
  triangle.Z[1] = (pointOut[2]/pointOut[3]);

  //Two XYZ
  pointIn[0] = t.X[2];
  pointIn[1] = t.Y[2];
  pointIn[2] = t.Z[2];
  pointIn[3] = 1; //w
  m0.TransformPoint(pointIn, pointOut);
  triangle.X[2] = (pointOut[0]/pointOut[3]); //DIVIDE BY W!!	
  triangle.Y[2] = (pointOut[1]/pointOut[3]);
  triangle.Z[2] = (pointOut[2]/pointOut[3]);


  if(print)
  {
    cout << "triangle out: (" << triangle.X[0] << " , " << triangle.Y[0] << " , " << triangle.Z[0] << ") " << endl <<
                         " (" << triangle.X[1] << " , " << triangle.Y[1] << " , " << triangle.Z[1] << ") " << endl <<
                         " (" << triangle.X[2] << " , " << triangle.Y[2] << " , " << triangle.Z[2] << ") " << endl;
  }
  //transfor values

  delete[] pointOut;
  delete[] pointIn;

  return triangle;

}

void CalcSilhouette2(float * data_in, int width, int height, double &length, double &curvature, double &curvatureExtrema, double &entropy)
{
	/*
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<unsigned int> curvatureHistogram(9,0);
  double silhouetteLength = 0; 
  std::vector<double> silhouetteCurvature;

  cv::Mat image_gray;
  cv::Mat image(width, height, CV_32F, data_in); 
  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY );
  cv::blur(image_gray, image_gray, cv::Size(3,3) );
  cv::findContours(image_gray, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

  unsigned int numberOfAngles = 0;
  cout << "CONTOURS SIZE " << contours.size() << endl;
  for( int j = 0; j < contours.size(); j++ )
  {
    silhouetteLength += cv::arcLength( contours.at(j), true );
    unsigned int contourSize = (unsigned int)contours.at(j).size();
    silhouetteCurvature.resize(numberOfAngles + contourSize);
    for( unsigned int k = 0; k < contourSize; k++ )
    {
      cv::Point diff1 = contours.at(j).at(k) - contours.at(j).at((k + 1) % contourSize);
      cv::Point diff2 = contours.at(j).at((k + 1) % contourSize) - contours.at(j).at((k + 2) % contourSize);
      double angle = 0.0;
      if(diff1.x != diff2.x || diff1.y != diff2.y)
      {
        double v1[3];
        double v2[3];
        v1[0] = diff1.x;
        v1[1] = diff1.y;
        v1[2] = 0;
        v2[0] = diff2.x;
        v2[1] = diff2.y;
        v2[2] = 0;
        normalize(v1);
        normalize(v2);
        double dotprod = dotProduct(v1,v2,2);
        double mag1 = magnitude3d(v1);
        double mag2 = magnitude3d(v2);
        double rad = acos(dotprod/(mag1*mag2));
        angle = rad*(double)180/M_PI;
      }
      silhouetteCurvature[numberOfAngles + k] = angle;
    }
    numberOfAngles += contourSize;
  }

  //Calculate Curvature and Entropy Metrics
  entropy = 0;
  curvature = 0;
  curvatureExtrema = 0;
  int num_curves = silhouetteCurvature.size();
  for(int i = 0; i < num_curves; i++)
  {
    double angle = silhouetteCurvature[i];
    curvature += abs(angle)/90.0;
    curvatureExtrema += pow((abs(angle)/90), 2.0);
    int bin = (int) ((angle + 180.0)/45.0);
    curvatureHistogram[bin]++;
  }

  for(int i = 0; i < curvatureHistogram.size(); i++)
  {
    unsigned int value = curvatureHistogram[i];
    if(value != 0)
    {
      double aux = value/(double)num_curves;
      entropy += aux*log2(aux);
    }
  }

  //Final Values
  length           = silhouetteLength;
  curvature        = curvature/(double)num_curves;
  curvatureExtrema = curvatureExtrema/(double)num_curves;
  entropy          = (-1)*entropy;
  */
}


template< typename T >
T calcentropy2( const T* array, long len, int nBins )
{
  T max = std::abs(array[0]);
  T min = std::abs(array[0]);
  for(long i = 0; i < len; i++ )
  {
    max = max > std::abs(array[i]) ? max : std::abs(array[i]);
    min = min < std::abs(array[i]) ? min : std::abs(array[i]);
  }
  T stepSize = (max-min) / (T)nBins;
  if(stepSize == 0)
    return 0.0;

  long* hist = new long[ nBins ];
  for(int i = 0; i < nBins; i++ )
    hist[i] = 0;

  for(long i = 0; i < len; i++ )
  {
    T idx = (std::abs(array[i]) - min) / stepSize;
    if((int)idx == nBins )
      idx -= 1.0;
    hist[(int)idx]++;
  }

  T entropy = 0.0;
  for(int i = 0; i < nBins; i++ )
  {
    T prob = (T)hist[i] / (T)len;
    if(prob != 0.0 )
      entropy += prob * std::log( prob );
  }

  delete[] hist;

  return (entropy * -1.0);
}

float
calculateMetric2(vtkh::DataSet* dataset, std::string metric, std::string field_name, int height, int width)
{
  float score = 0.0;

  if(metric == "data_entropy")
  {
    float entropy = 0.0;
    #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Status status;
      if(rank == 0)
      {
	int size = height*width;
        std::vector<float> field_data = GetScalarData2(*dataset, field_name, height, width);
	for(int i = 0; i < size; i++)
          if(field_data[i] != field_data[i])
	    field_data[i] = -FLT_MAX;
	float field_array[size];
	std::copy(field_data.begin(), field_data.end(), field_array);
	entropy = calcentropy2(field_array, field_data.size(), 100);

      }
      MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      score = entropy;
    #else
      int size = height*width;
      std::vector<float> field_data = GetScalarData2(*dataset, field_name, height, width);
      for(int i = 0; i < size; i++)
        if(field_data[i] != field_data[i])
          field_data[i] = -FLT_MAX;
      float field_array[size];
      std::copy(field_data.begin(), field_data.end(), field_array);
      entropy = calcentropy2(field_array, field_data.size(), 100);
      score = entropy;
    #endif
  }
  else if (metric == "depth_entropy")
  {
    float entropy = 0.0;
    #if ASCENT_MPI_ENABLED 
      // Get the number of processes
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      
      // Get the rank of this process
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Status status;
      if(rank == 0)
      { 
        int size = height*width;
        std::vector<float> depth_data = GetScalarData2(*dataset, "depth", height, width);
        for(int i = 0; i < size; i++)
          if(depth_data[i] != depth_data[i])
            depth_data[i] = -FLT_MAX;
        float depth_array[size];
        std::copy(depth_data.begin(), depth_data.end(), depth_array);
        entropy = calcentropy2(depth_array, depth_data.size(), 100);
      
      }
      MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      score = entropy;
    #else
      int size = height*width;
      std::vector<float> depth_data = GetScalarData2(*dataset, "depth", height, width);
      for(int i = 0; i < size; i++)
        if(depth_data[i] != depth_data[i])
          depth_data[i] = -FLT_MAX;
      float depth_array[size];
      std::copy(depth_data.begin(), depth_data.end(), depth_array);
      entropy = calcentropy2(depth_array, depth_data.size(), 100);
      score = entropy;
    #endif
  }
  else if (metric == "max_depth")
  {
    float depth = -FLT_MAX;
    #if ASCENT_MPI_ENABLED
      // Get the number of processes
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Status status;
      if(rank == 0)
      {
        int size = height*width;
        std::vector<float> depth_data = GetScalarData2(*dataset, "depth", height, width);
        for(int i = 0; i < size; i++)
          if(depth_data[i] == depth_data[i])
            if(depth < depth_data[i])
	      depth = depth_data[i];
        score = depth;
      }
      MPI_Bcast(&score, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    #else
      int size = height*width;
      std::vector<float> depth_data = GetScalarData2(*dataset, "depth", height, width);
      for(int i = 0; i < size; i++)
        if(depth_data[i] == depth_data[i])
	  if(depth < depth_data[i])
            depth = depth_data[i];
      score = depth;
    #endif
  }
  else if (metric == "max_silhouette")
  {
    #if ASCENT_MPI_ENABLED
      // Get the number of processes
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Status status;
      if(rank == 0)
      {
        int size = height*width;
        std::vector<float> depth_data = GetScalarData2(*dataset, "depth", height, width);
        for(int i = 0; i < size; i++)
          if(depth_data[i] == depth_data[i])
	    depth_data[i] = 255.0; //data = white
          else
	    depth_data[i] = 0.0; //background = black

	float data_in[width*height];
	float contour[width*height];
	std::copy(depth_data.begin(), depth_data.end(), data_in);
	double length, curvature, curvatureExtrema, entropy;
	CalcSilhouette2(data_in, width, height, length, curvature, curvatureExtrema, entropy);
	score = (float)length;
        MPI_Bcast(&score, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }
    #else
      int size = height*width;
      std::vector<float> depth_data = GetScalarData2(*dataset, "depth", height, width);
      for(int i = 0; i < size; i++)
        if(depth_data[i] == depth_data[i])
	  depth_data[i] = 255.0;
        else
	  depth_data[i] = 0.0;
      float data_in[size];
      float contour[size];
      std::copy(depth_data.begin(), depth_data.end(), data_in);
      double length, curvature, curvatureExtrema, entropy;
      CalcSilhouette2(data_in, width, height, length, curvature, curvatureExtrema, entropy);
      score = (float)length;
    #endif
    
  }
  else
    ASCENT_ERROR("This metric is not supported. \n");

  return score;
}

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters::detail --
//-----------------------------------------------------------------------------

CameraSimplex::CameraSimplex()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
CameraSimplex::~CameraSimplex()
{
// empty
}

//-----------------------------------------------------------------------------
void
CameraSimplex::declare_interface(Node &i)
{
    i["type_name"]   = "simplex";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
CameraSimplex::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();
    bool res = check_string("field",params, info, true);
    bool metric = check_string("metric",params, info, true);
    bool samples = check_numeric("samples",params, info, true);

    if(!metric)
    {
        info["errors"].append() = "Missing required metric parameter."
                        	  " Currently only supports data_entropy"
				  " for some scalar field"
				  " and depth_entropy.\n";
        res = false;
    }

    if(!samples)
    {
        info["errors"].append() = "Missing required numeric parameter. "
				  "Must specify number of samples.\n";
        res = false;
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("field");
    valid_paths.push_back("metric");
    valid_paths.push_back("samples");
    std::string surprises = surprise_check(valid_paths, params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
    
}

//-----------------------------------------------------------------------------
void
CameraSimplex::execute()
{
    double time = 0.;
    auto time_start = high_resolution_clock::now();
    //cout << "USING SIMPLEX PIPELINE" << endl;
    #if ASCENT_MPI_ENABLED
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif  

    DataObject *data_object = input<DataObject>(0);
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
    std::string field_name = params()["field"].as_string();
    std::string metric     = params()["metric"].as_string();

    if(!collection->has_field(field_name))
    {
      ASCENT_ERROR("Unknown field '"<<field_name<<"'");
    }
    int samples = (int)params()["samples"].as_int64();
    //TODO:Get the height and width of the image from Ascent
    int width  = 1000;
    int height = 1000;
    
    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);
    
    double triangle_time = 0.;
    auto triangle_start = high_resolution_clock::now();
    std::vector<Triangle> triangles;// = GetTriangles2(dataset,field_name);
    auto triangle_stop = high_resolution_clock::now();
    triangle_time += duration_cast<microseconds>(triangle_stop - triangle_start).count();
    /*#if ASCENT_MPI_ENABLED
      cout << "Global bounds: " << dataset.GetGlobalBounds() << endl;
      cout << "rank " << rank << " bounds: " << dataset.GetBounds() << endl;
    #endif*/

    vtkm::Bounds b = dataset.GetGlobalBounds();

    vtkm::Float32 xMin = vtkm::Float32(b.X.Min);
    vtkm::Float32 xMax = vtkm::Float32(b.X.Max);
    vtkm::Float32 yMin = vtkm::Float32(b.Y.Min);
    vtkm::Float32 yMax = vtkm::Float32(b.Y.Max);
    vtkm::Float32 zMin = vtkm::Float32(b.Z.Min);
    vtkm::Float32 zMax = vtkm::Float32(b.Z.Max);

    vtkm::Float32 xb = vtkm::Float32(b.X.Length());
    vtkm::Float32 yb = vtkm::Float32(b.Y.Length());
    vtkm::Float32 zb = vtkm::Float32(b.Z.Length());
    //double bounds[3] = {(double)xb, (double)yb, (double)zb};
    //cout << "x y z bounds " << xb << " " << yb << " " << zb << endl;
    vtkm::Float32 radius = sqrt(xb*xb + yb*yb + zb*zb)/2.0;
    //cout << "radius " << radius << endl;
    //if(radius<1)
      //radius = radius + 1;
    //vtkm::Float32 x_pos = 0., y_pos = 0., z_pos = 0.;
    vtkmCamera *camera = new vtkmCamera;
    camera->ResetToBounds(dataset.GetGlobalBounds());
    vtkm::Vec<vtkm::Float32,3> lookat = camera->GetLookAt();
    double focus[3] = {(double)lookat[0],(double)lookat[1],(double)lookat[2]};

/*
    Screen screen;
    screen.width = width;
    screen.height = height;
    screen.zBufferInitialize();
    screen.triScreenInitialize();
    screen.triCameraInitialize();
    screen.valueInitialize();
*/
    //double winning_scores[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    //int    winning_samples[3] = {-1, -1, -1};
    //loop through number of camera samples.
    double scanline_time = 0.;
    double metric_time   = 0.;


    // Basic winning score while making new camera
    double winning_score = -DBL_MAX;
    int winning_i = -1;
    int winning_j = -1;

    // New theta and phi camera code
    int numTheta = 100;
    int numPhi = 100;

    // File stuff
    ofstream datafile;
    // Open in binary mode and delete if it already existed
    datafile.open("scores.bin", ios::binary | ios::trunc);

    for (int i = 0 ; i < numTheta ; i++) {
      cout << "Step: " << i << endl;
      cout << "  Current Winning Score: " << winning_score << endl;
      for (int j = 0 ; j < numPhi ; j++) {

        Camera cam = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
		       	        radius, i, numTheta, j, numPhi); 

        vtkm::Vec<vtkm::Float32, 3> pos{(float)cam.position[0],
                                  (float)cam.position[1],
                                  (float)cam.position[2]};

        camera->SetPosition(pos);
        vtkh::ScalarRenderer tracer;
        tracer.SetWidth(width);
        tracer.SetHeight(height);
        tracer.SetInput(&dataset); //vtkh dataset by toponame
        tracer.SetCamera(*camera);
        tracer.Update();

        vtkh::DataSet *output = tracer.GetOutput();

        float score = calculateMetric2(output, metric, field_name, height, width);

	//cout << "Camera at: " << cam.position[0] << ", " << cam.position[1] << ", " << cam.position[2] << endl;
        //cout << "Score is: " << score << endl << endl;

        //datafile << score << " ";	
        datafile.write( (char *) &score, sizeof(score));

	if (score > winning_score) {
            winning_score = score;
            winning_i = i;
            winning_j = j;
        }

      }
    }

    cout << "Winning score: " << winning_score << " at (" << winning_i << ", " << winning_j << ")" << endl;

    datafile.close();

    /*
    for(int sample = 0; sample < samples; sample++)
    {
    //================ Scalar Renderer Code ======================//
    //What it does: Quick ray tracing of data (replaces get triangles and scanline).
    //What we need: z buffer, any other important buffers (tri ids, scalar values, etc.)
      
      Camera cam = GetCamera2(sample, samples, radius, focus);
      vtkm::Vec<vtkm::Float32, 3> pos{(float)cam.position[0],
                                (float)cam.position[1],
                                (float)cam.position[2]};

      camera->SetPosition(pos);
      vtkh::ScalarRenderer tracer;
      tracer.SetWidth(width);
      tracer.SetHeight(height);
      tracer.SetInput(&dataset); //vtkh dataset by toponame
      tracer.SetCamera(*camera);
      tracer.Update();

      vtkh::DataSet *output = tracer.GetOutput();

      float score = calculateMetric2(output, metric, field_name, height, width);

      cout << "Sample: " << sample << " Score: " << score << endl;
      //cout << "Camera Positions: " << (float)cam.position[0] << " " << (float)cam.position[1] << " " << (float)cam.position[2] << endl << endl;
     
      delete output;
*/

    /*================ End Scalar Renderer  ======================*/

/*

      screen.width = width;
      screen.height = height;
      screen.visible = 0.0;
      screen.occluded = 0.0;
      screen.zBufferInitialize();
      screen.triScreenInitialize();
      screen.triCameraInitialize();
      screen.valueInitialize();

      Camera c = GetCamera2(sample, samples, radius, focus);
      c.screen = screen;
      int num_tri = triangles.size();
      
      //Scanline timings
      auto scanline_start = high_resolution_clock::now();
      //loop through all triangles
      for(int tri = 0; tri < num_tri; tri++)
      {
	//triangle in world space
        Triangle w_t = triangles[tri];
	
	//triangle in image space
	Triangle i_t = transformTriangle2(w_t, c);
	i_t.vis_counted = false;
	i_t.screen = screen;
	i_t.scanline(tri, c);
	screen = i_t.screen;

      }//end of triangle loop
      auto scanline_stop = high_resolution_clock::now();

      scanline_time += duration_cast<microseconds>(scanline_stop - scanline_start).count();
      //metric timings
      auto metric_start = high_resolution_clock::now();
      double score = calculateMetric2(screen, metric);
      auto metric_stop = high_resolution_clock::now();
      metric_time += duration_cast<microseconds>(metric_stop - metric_start).count();
*/

      /* Commented out original top 3 scores code
      if (score > winning_scores[0]) {
        // Found a better top score, replace all 3 
        winning_scores[2] = winning_scores[1];	      
	winning_samples[2] = winning_samples[1];

        winning_scores[1] = winning_scores[0];	      
	winning_samples[1] = winning_samples[0];

        winning_scores[0] = score;
	winning_samples[0] = sample;
      }
      else if (score > winning_scores[1]) {
        // Found a better second score, replace bottom 2
        winning_scores[2] = winning_scores[1];	      
	winning_samples[2] = winning_samples[1];

        winning_scores[1] = score;
	winning_samples[1] = sample;
      }
      else if (score > winning_scores[2]) {
        // Found a better third score, replace it
        winning_scores[2] = score;	      
	winning_samples[2] = sample;
      }


    } //end of sample loop

    if(winning_samples[0] == -1)
      ASCENT_ERROR("Something went terribly wrong; No camera position was chosen");

    cout << "Top 3 scores are: " << winning_scores[0] << ", " << winning_scores[1] << ", " << winning_scores[2] << endl;
    cout << "With samples: " << winning_samples[0] << ", " << winning_samples[1] << ", " << winning_samples[2] << endl;

    // Now loop around the winning samples to see if we can improve them 
    double winning_samplesNext[3] = {(double)winning_samples[0], (double)winning_samples[1], (double)winning_samples[2]};
    for (int score_idx = 0 ; score_idx < 3 ; score_idx++) {

      cout << "Looking around sample " << winning_samples[score_idx] << endl;
      for (int neg = 1; neg > -1; neg--) {

        for(int i = 1; i < 6; i++) {

	  float sampleNext;
	  if (neg) {
            sampleNext = winning_samples[score_idx] - (.6 - (i * .1) );
          }
          else {
            sampleNext = winning_samples[score_idx] + (i * .1);
          }

          Camera camNext = GetCamera2(sampleNext, samples, radius, focus);
          vtkm::Vec<vtkm::Float32, 3> posNext{(float)camNext.position[0],
                                              (float)camNext.position[1],
                                              (float)camNext.position[2]};

          camera->SetPosition(posNext);
          vtkh::ScalarRenderer tracer;
          tracer.SetWidth(width);
          tracer.SetHeight(height);
          tracer.SetInput(&dataset); //vtkh dataset by toponame
          tracer.SetCamera(*camera);
          tracer.Update();
          vtkh::DataSet *outputNext = tracer.GetOutput();
          float scoreNext = calculateMetric2(outputNext, metric, field_name, height, width);

          cout << "  Sample: " << sampleNext << " Score: " << scoreNext << endl;
          //cout << "Camera Positions: " << (float)camNext.position[0] << " " << (float)camNext.position[1] << " " << (float)camNext.position[2] << endl << endl;
	
          if(scoreNext > winning_scores[score_idx])
          {
            winning_scores[score_idx] = scoreNext;
            winning_samplesNext[score_idx] = sampleNext;
          }

        }
      }
    }

    cout << "Top 3 scores now are: " << winning_scores[0] << ", " << winning_scores[1] << ", " << winning_scores[2] << endl;
    cout << "With samples: " << winning_samplesNext[0] << ", " << winning_samplesNext[1] << ", " << winning_samplesNext[2] << endl;

    double winning_samplesFinal[3] = {winning_samplesNext[0], winning_samplesNext[1], winning_samplesNext[2]};
    for (int score_idx = 0 ; score_idx < 3 ; score_idx++) {

      cout << "Looking around sample " << winning_samplesNext[score_idx] << endl;
      for (int neg = 1; neg > -1; neg--) {

        for(int i = 1; i < 6; i++) {

	  float sampleFinal;
	  if (neg) {
            sampleFinal = winning_samplesNext[score_idx] - (.06 - (i * .01) );
          }
          else {
            sampleFinal = winning_samplesNext[score_idx] + (i * .01);
          }

          Camera camFinal = GetCamera2(sampleFinal, samples, radius, focus);
          vtkm::Vec<vtkm::Float32, 3> posFinal{(float)camFinal.position[0],
                                (float)camFinal.position[1],
                                (float)camFinal.position[2]};

          camera->SetPosition(posFinal);
          vtkh::ScalarRenderer tracer;
          tracer.SetWidth(width);
          tracer.SetHeight(height);
          tracer.SetInput(&dataset); //vtkh dataset by toponame
          tracer.SetCamera(*camera);
          tracer.Update();
          vtkh::DataSet *outputFinal = tracer.GetOutput();
          float scoreFinal = calculateMetric2(outputFinal, metric, field_name, height, width);

          cout << "    Sample: " << sampleFinal << " Score: " << scoreFinal << endl;
          //cout << "Camera Positions: " << (float)camFinal.position[0] << " " << (float)camFinal.position[1] << " " << (float)camFinal.position[2] << endl << endl;
	
          if(scoreFinal > winning_scores[score_idx])
          {
            winning_scores[score_idx] = scoreFinal;
            winning_samplesFinal[score_idx] = sampleFinal;
          }

        } 
      }
    }

    cout << "Top 3 scores now are: " << winning_scores[0] << ", " << winning_scores[1] << ", " << winning_scores[2] << endl;
    cout << "With samples: " << winning_samplesFinal[0] << ", " << winning_samplesFinal[1] << ", " << winning_samplesFinal[2] << endl;
   
    double best_score = winning_scores[0];
    double best_sample = winning_samplesFinal[0];
    if (winning_scores[1] > best_score) {
        best_score = winning_scores[1];
        best_sample = winning_samplesFinal[1];
    }
    if (winning_scores[2] > best_score) {
        best_score = winning_scores[2];
        best_sample = winning_samplesFinal[2];
    }

    cout << "Best score is: " << best_score << endl;
    cout << "Best sample is: " << best_sample << endl;
*/

    //Camera best_c = GetCamera2(best_sample, samples, radius, focus);
    Camera best_c = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
		       	        radius, winning_i, numTheta, winning_j, numPhi);

    vtkm::Vec<vtkm::Float32, 3> pos{(float)best_c.position[0], 
	                            (float)best_c.position[1], 
				    (float)best_c.position[2]}; 
/*
#if ASCENT_MPI_ENABLED
    if(rank == 0)
    {
      cout << "look at: " << endl;
      vtkm::Vec<vtkm::Float32,3> lookat = camera->GetLookAt();
      cout << lookat[0] << " " << lookat[1] << " " << lookat[2] << endl;
      camera->Print();
    }
#endif
*/
    camera->SetPosition(pos);


    if(!graph().workspace().registry().has_entry("camera"))
    {
      //cout << "making camera in registry" << endl;
      graph().workspace().registry().add<vtkm::rendering::Camera>("camera",camera,1);
    }

/*
#if ASCENT_MPI_ENABLED
    if(rank == 0)
      camera->Print();
#endif
*/
    set_output<DataObject>(input<DataObject>(0));
    //set_output<vtkmCamera>(camera);
    auto time_stop = high_resolution_clock::now();
    time += duration_cast<seconds>(time_stop - time_start).count();

    /*#if ASCENT_MPI_ENABLED
      cout << "rank: " << rank << " secs total: " << time << endl;
    #endif*/
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
