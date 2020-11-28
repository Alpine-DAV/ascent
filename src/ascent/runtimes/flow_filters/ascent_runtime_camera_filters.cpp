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
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/worklet/WorkletMapTopology.h>
#include <vtkm/cont/DataSetFieldAdd.h>


#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <chrono>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>

//openCV
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>

using namespace conduit;
using namespace std;
using namespace std::chrono;

using namespace flow;

#if defined(ASCENT_VTKM_ENABLED)
typedef vtkm::rendering::Camera vtkmCamera;
#endif



//make file
void 
MakeFile(std::string filename, double *array, int size)
{
  ofstream myfile(filename, ios::out | ios::app);
  if(myfile.is_open())
  {
    for(int i = 0; i < size; i++)
    {
      myfile << array[i] << "\n";
    }
    myfile << "=========\n";
  } 
}
//Camera Class Functions

Matrix
Camera::CameraTransform(void)
{
  bool print = false;
  double v3[3]; //camera position - focus
  v3[0] = (position[0] - focus[0]);
  v3[1] = (position[1] - focus[1]);
  v3[2] = (position[2] - focus[2]);
  normalize(v3);

  double v1[3]; //UP x (camera position - focus)
  crossProduct(up, v3, v1);
  normalize(v1);

  double v2[3]; // (camera position - focus) x v1
  crossProduct(v3, v1, v2);
  normalize(v2);

  double t[3]; // (0,0,0) - camera position
  t[0] = (0 - position[0]);
  t[1] = (0 - position[1]);
  t[2] = (0 - position[2]);


  if (print)
  {
    cerr << "position " << position[0] << " " << position[1] << " " << position[2] << endl;
    cerr << "focus " << focus[0] << " " << focus[1] << " " << focus[2] << endl;
    cerr << "up " << up[0] << " " << up[1] << " " << up[2] << endl;
    cerr << "v1 " << v1[0] << " " << v1[1] << " " << v1[2] << endl;
    cerr << "v2 " << v2[0] << " " << v2[1] << " " << v2[2] << endl;
    cerr << "v3 " << v3[0] << " " << v3[1] << " " << v3[2] << endl;
    cerr << "t " << t[0] << " " << t[1] << " " << t[2] << endl;
  }

/*
| v1.x v2.x v3.x 0 |
| v1.y v2.y v3.y 0 |
| v1.z v2.z v3.z 0 |
| v1*t v2*t v3*t 1 |
*/
  Matrix camera;

  camera.A[0][0] = v1[0]; //v1.x
  camera.A[0][1] = v2[0]; //v2.x
  camera.A[0][2] = v3[0]; //v3.x
  camera.A[0][3] = 0; //0
  camera.A[1][0] = v1[1]; //v1.y
  camera.A[1][1] = v2[1]; //v2.y
  camera.A[1][2] = v3[1]; //v3.y
  camera.A[1][3] = 0; //0
  camera.A[2][0] = v1[2]; //v1.z
  camera.A[2][1] = v2[2]; //v2.z
  camera.A[2][2] = v3[2]; //v3.z
  camera.A[2][3] = 0; //0
  camera.A[3][0] = dotProduct(v1, t, 3); //v1 dot t
  camera.A[3][1] = dotProduct(v2, t, 3); //v2 dot t
  camera.A[3][2] = dotProduct(v3, t, 3); //v3 dot t
  camera.A[3][3] = 1.0; //1

  if(print)
  {
    cerr << "Camera:" << endl;
    camera.Print(cerr);
  }
  return camera;

};

Matrix
Camera::ViewTransform(void) 
{

/*
| cot(a/2)    0         0            0     |
|    0     cot(a/2)     0            0     |
|    0        0    (f+n)/(f-n)      -1     |
|    0        0         0      (2fn)/(f-n) |
*/
  Matrix view;
  double c = (1.0/(tan(angle/2.0))); //cot(a/2) =    1
				     //		   -----
				     //		  tan(a/2)
  double f = ((far + near)/(far - near));
  double f2 = ((2*far*near)/(far - near));

  view.A[0][0] = c;
  view.A[0][1] = 0;
  view.A[0][2] = 0;
  view.A[0][3] = 0;
  view.A[1][0] = 0;
  view.A[1][1] = c;
  view.A[1][2] = 0;
  view.A[1][3] = 0;
  view.A[2][0] = 0;
  view.A[2][1] = 0;
  view.A[2][2] = f;
  view.A[2][3] = -1.0;
  view.A[3][0] = 0;
  view.A[3][1] = 0;
  view.A[3][2] = f2;
  view.A[3][3] = 0;
  return view;

};


Matrix
Camera::DeviceTransform()
{ //(double x, double y, double z){

/*
| x' 0  0  0 |
| 0  y' 0  0 |
| 0  0  z' 0 |
| 0  0  0  1 |
*/
  Matrix device;
  int width = 1000;
  int height = 1000;
  device.A[0][0] = (width/2);
  device.A[0][1] = 0;
  device.A[0][2] = 0;
  device.A[0][3] = 0;
  device.A[1][0] = 0;
  device.A[1][1] = (height/2);
  device.A[1][2] = 0;
  device.A[1][3] = 0;
  device.A[2][0] = 0;
  device.A[2][1] = 0;
  device.A[2][2] = 1;
  device.A[2][3] = 0;
  device.A[3][0] = (width/2);
  device.A[3][1] = (height/2);
  device.A[3][2] = 0;
  device.A[3][3] = 1;

  return device;
}

Matrix
Camera::DeviceTransform(int width, int height) 
{ //(double x, double y, double z){

/*
| x' 0  0  0 |
| 0  y' 0  0 |
| 0  0  z' 0 |
| 0  0  0  1 |
*/
  Matrix device;

  device.A[0][0] = (width/2);
  device.A[0][1] = 0;
  device.A[0][2] = 0;
  device.A[0][3] = 0;
  device.A[1][0] = 0;
  device.A[1][1] = (height/2);
  device.A[1][2] = 0;
  device.A[1][3] = 0;
  device.A[2][0] = 0;
  device.A[2][1] = 0;
  device.A[2][2] = 1;
  device.A[2][3] = 0;
  device.A[3][0] = (width/2);
  device.A[3][1] = (height/2);
  device.A[3][2] = 0;
  device.A[3][3] = 1;

  return device;
}


//Edge Class Functions


Edge::Edge (double x_1, double y_1, double z_1,  double x_2, double y_2, double z_2, double v_1, double v_2)
{
  x1 = x_1;
  x2 = x_2;
  y1 = y_1;
  y2 = y_2;
  z1 = z_1;
  z2 = z_2;
  value1  = v_1;
  value2  = v_2;

  // find relationship of y1 and y2 for min and max bounds of the line
  if (y1 < y2)
    minY = y1;
  else
    minY = y2;
  if (y1 > y2)
    maxY = y1;
  else
    maxY = y2;

  if (x2 - x1 == 0)
  { //if vertical, return x
    vertical = true;
    slope = x1;
  }
  else
  {
    vertical = false;
    slope = (y2 - y1)/(x2 - x1); //slope is 0 if horizontal, else it has a slope
  }
  b = y1 - slope*x1;
  if (y2 - y1 == 0) //if horizontal disregard
    relevant = false;
  else
    relevant = true;
}

//find x on the line of y1 and y2 and given y with ymin <= y <= ymax.
double
Edge::findX(double y){
  if (vertical == true){
    return slope;
  }
  else{
    if (slope == 0)
      return 0;
    else{
      double x = (y - b)/slope;
      return x;
    }
  }
}

	//A = y1, B = y2, fX = interpolated point, X = desired point, fA = value at A, fB = value at B
	//(x-A)/(B-A) ratio of y between y1 and y2
	//interpolate finds the value (z or rgb or norm vector) at y between y1 and y2.
double
Edge::interpolate(double a, double b, double C, double D, double fa, double fb, double x)
{
  double A, B, fA, fB;
  if(C < D)
  {
    A = a;
    B = b;
    fA = fa;
    fB = fb;
  }
  else
  {
    A = b;
    B = a;
    fA = fb;
    fB = fa;
  }
  double fX = fA + ((x - A)/(B - A))*(fB-fA);
  return fX;
}

double
Edge::findZ(double y)
{
  double z = interpolate(y1, y2, x1, x2, z1, z2, y);
  return z;
}

double
Edge::findValue(double y)
{
  double value = interpolate(y1, y2, x1, x2, value1, value2, y);
  return value;
}


bool
Edge::applicableY(double y)
{
  if (y >= minY && y <= maxY)
    return true;
  else if (nabs(minY - y) < 0.00001 || nabs(maxY - y) < 0.00001)
    return true;
  else
    return false;
}

//Matrix Class Functions
void
Matrix::Print(ostream &o)
{
  for (int i = 0 ; i < 4 ; i++)
  {
      char str[256];
      sprintf(str, "(%.7f %.7f %.7f %.7f)\n", A[i][0], A[i][1], A[i][2], A[i][3]);
      o << str;
  }
}

//multiply two matrices
Matrix
Matrix::ComposeMatrices(const Matrix &M1, const Matrix &M2)
{
  Matrix rv;
  for (int i = 0 ; i < 4 ; i++)
      for (int j = 0 ; j < 4 ; j++)
      {
          rv.A[i][j] = 0;
          for (int k = 0 ; k < 4 ; k++)
              rv.A[i][j] += M1.A[i][k]*M2.A[k][j];
      }

  return rv;
}

//multiply vector by matrix
void
Matrix::TransformPoint(const double *ptIn, double *ptOut)
{
  ptOut[0] = ptIn[0]*A[0][0]
           + ptIn[1]*A[1][0]
           + ptIn[2]*A[2][0]
           + ptIn[3]*A[3][0];
  ptOut[1] = ptIn[0]*A[0][1]
           + ptIn[1]*A[1][1]
           + ptIn[2]*A[2][1]
           + ptIn[3]*A[3][1];
  ptOut[2] = ptIn[0]*A[0][2]
           + ptIn[1]*A[1][2]
           + ptIn[2]*A[2][2]
           + ptIn[3]*A[3][2];
  ptOut[3] = ptIn[0]*A[0][3]
           + ptIn[1]*A[1][3]
           + ptIn[2]*A[2][3]
           + ptIn[3]*A[3][3];
}


//Screen Class Functions
void 
Screen::zBufferInitialize()
{
  zBuff = new double[width*height];
  int i;
  for (i = 0; i < width*height; i++)
    zBuff[i] = -1.0;
}

void 
Screen::triScreenInitialize()
{
  triScreen = new int[width*height];
  int i;
  for (i = 0; i < width*height; i++)
    triScreen[i] = -1;
}

void 
Screen::triCameraInitialize()
{
  triCamera = new double*[width*height];
  int i;
  double* position = new double[3];
  position[0] = 0;
  position[1] = 0;
  position[2] = 0;
  for (i = 0; i < width*height; i++)
    triCamera[i] = position;
}

void 
Screen::valueInitialize()
{
  values = new double[width*height];
  int i;
  for (i = 0; i < width*height; i++)
    values[i] = 0.0;
}

//Triangle Class 

void
Triangle::printTri()
{
  cerr << "X: " << X[0] << " " << X[1] << " " << X[2] << endl;
  cerr << "Y: " << Y[0] << " " << Y[1] << " " << Y[2] << endl;
  cerr << "Z: " << Z[0] << " " << Z[1] << " " << Z[2] << endl;
}

float
Triangle::calculateTriArea(){
  bool print = false;
  float area = 0.0;
  float AC[3];
  float BC[3];

  AC[0] = X[1] - X[0];
  AC[1] = Y[1] - Y[0];
  AC[2] = Z[1] - Z[0];

  BC[0] = X[2] - X[0];
  BC[1] = Y[2] - Y[0];
  BC[2] = Z[2] - Z[0];

  float orthogonal_vec[3];

  orthogonal_vec[0] = AC[1]*BC[2] - BC[1]*AC[2];
  orthogonal_vec[1] = AC[0]*BC[2] - BC[0]*AC[2];
  orthogonal_vec[2] = AC[0]*BC[1] - BC[1]*AC[1];

  area = sqrt(pow(orthogonal_vec[0], 2) +
              pow(orthogonal_vec[1], 2) +
              pow(orthogonal_vec[2], 2))/2.0;

  if(print)
  {
  cerr << "Triangle: (" << X[0] << " , " << Y[0] << " , " << Z[0] << ") " << endl <<
                   " (" << X[1] << " , " << Y[1] << " , " << Z[1] << ") " << endl <<
                   " (" << X[2] << " , " << Y[2] << " , " << Z[2] << ") " << endl <<
           " has surface area: " << area << endl;
  }
  return area;
}

double
Triangle::findMin(double a, double b, double c)
{
  double min = a;
  if (b < min)
    min = b;
  if (c < min)
    min = c;
  return min;
}

double
Triangle::findMax(double a, double b, double c)
{
  double max = a;
  if (b > max)
    max = b;
  if (c > max)
    max = c;
  return max;
}


//Misc Functions
double ceil441(double f)
{
  return ceil(f-0.00001);
}

double floor441(double f)
{
  return floor(f+0.00001);
}

double nabs(double x)
{
  if (x < 0)
    x = (x*(-1));
  return x;
}

double calculateArea(double x0, double y0, double z0, double x1, double y1, double z1, double x2, double y2, double z2)
{
  double area = 0.0;
  double AC[3];
  double BC[3];

  AC[0] = x1 - x0;
  AC[1] = y1 - y0;
  AC[2] = z1 - x0;

  BC[0] = x2 - x0;
  BC[1] = y2 - y0;
  BC[2] = z2 - z0;

  double orthogonal_vec[3];

  orthogonal_vec[0] = AC[1]*BC[2] - BC[1]*AC[2];
  orthogonal_vec[1] = AC[0]*BC[2] - BC[0]*AC[2];
  orthogonal_vec[2] = AC[0]*BC[1] - BC[1]*AC[1];

  area = sqrt(pow(orthogonal_vec[0], 2) +
              pow(orthogonal_vec[1], 2) +
              pow(orthogonal_vec[2], 2))/2.0;
 
  return area;
}

void normalize(double * normal) 
{
  double total = pow(normal[0], 2.0) + pow(normal[1], 2.0) + pow(normal[2], 2.0);
  total = pow(total, 0.5);
  normal[0] = normal[0] / total;
  normal[1] = normal[1] / total;
  normal[2] = normal[2] / total;
}

double dotProduct(double* v1, double* v2, int length)
{
  double dotproduct = 0;	
  for (int i = 0; i < length; i++)
  {
    dotproduct += (v1[i]*v2[i]);
  }
  return dotproduct;
}

void crossProduct(double a[3], double b[3], double output[3])
{
  output[0] = ((a[1]*b[2]) - (a[2]*b[1])); //ay*bz-az*by
  output[1] = ((a[2]*b[0]) - (a[0]*b[2])); //az*bx-ax*bz
  output[2] = ((a[0]*b[1]) - (a[1]*b[0])); //ax*by-ay*bx
}


double SineParameterize(int curFrame, int nFrames, int ramp)
{
  int nNonRamp = nFrames-2*ramp;
  double height = 1./(nNonRamp + 4*ramp/M_PI);
  if (curFrame < ramp)
  {
    double factor = 2*height*ramp/M_PI;
    double eval = cos(M_PI/2*((double)curFrame)/ramp);
    return (1.-eval)*factor;
  }
  else if (curFrame > nFrames-ramp)
  {
    int amount_left = nFrames-curFrame;
    double factor = 2*height*ramp/M_PI;
    double eval =cos(M_PI/2*((double)amount_left/ramp));
    return 1. - (1-eval)*factor;
  }
  double amount_in_quad = ((double)curFrame-ramp);
  double quad_part = amount_in_quad*height;
  double curve_part = height*(2*ramp)/M_PI;
  return quad_part+curve_part;
}

Camera
GetCamera(int frame, int nframes, double radius, double* lookat, float *bounds)
{
//  double t = SineParameterize(frame, nframes, nframes/10);
  double points[3];
  fibonacci_sphere(frame, nframes, points);
  Camera c;
  double zoom = 3.0;
  c.near = zoom/8;
  c.far = zoom*5;
  c.angle = M_PI/6;

/*  if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
  {
    if(points[2] >= 0)
      points[2] += radius;
    if(points[2] < 0)
      points[2] -= radius;
  }*/
  float x = (bounds[0] + bounds[1])/2;
  float y = (bounds[2] + bounds[3])/2;
  float z = (bounds[4] + bounds[5])/2;
  

  c.position[0] = (zoom*radius*points[0]) + lookat[0];
  c.position[1] = (zoom*radius*points[1]) + lookat[1];
  c.position[2] = (zoom*radius*points[2]) + lookat[2];

  //cerr << "radius: " << radius << endl;
  //cerr << "lookat: " << lookat[0] << " " << lookat[1] << " " << lookat[2] << endl;
  //cerr << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
    
  c.focus[0] = lookat[0];
  c.focus[1] = lookat[1];
  c.focus[2] = lookat[2];
  c.up[0] = 0;
  c.up[1] = 1;
  c.up[2] = 0;
  return c;
}


#if defined(ASCENT_VTKM_ENABLED)
class ProcessTriangle : public vtkm::worklet::WorkletVisitCellsWithPoints
{
public:
  // This is to tell the compiler what inputs to expect.
  // For now we'll be providing the CellSet, CooridnateSystem,
  // an input variable, and an output variable.
  using ControlSignature = void(CellSetIn cellset,
                                FieldInPoint points,
                                FieldOutCell output);

  // After VTK-m does it's magic, you need to tell what information you need
  // from the provided inputs from the ControlSignature.
  // For now :
  // 1. number of points making an individual cell
  // 2. _2 is the 2nd input to ControlSignature : this should give you all points of the triangle.
  // 3. _3 is the 3rd input to ControlSignature : this should give you the variables at those points.
  // 4. _4 is the 4rd input to ControlSignature : this will help you store the output of your calculation.
  using ExecutionSignature = void(PointCount, _2, _3);

  template <typename PointVecType>
  VTKM_EXEC
  void operator()(const vtkm::IdComponent& numPoints,
                  const PointVecType& points,
                  Triangle& output) const
  {
    if(numPoints != 3)
      ASCENT_ERROR("We only play with triangles here");
    // Since you only have triangles, numPoints should always be 3
    // PointType is an abstraction of vtkm::Vec3f, which is {x, y, z} of a point
    using PointType = typename PointVecType::ComponentType;
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

class GetTriangleFields : public vtkm::worklet::WorkletVisitCellsWithPoints
{
public:
  // This is to tell the compiler what inputs to expect.
  // For now we'll be providing the CellSet, CooridnateSystem,
  // an input variable, and an output variable.
  using ControlSignature = void(CellSetIn cellset,
                                FieldInPoint points,
                                FieldOutCell x0, FieldOutCell y0, FieldOutCell z0,
		                FieldOutCell x1, FieldOutCell y1, FieldOutCell z1,
		                FieldOutCell x2, FieldOutCell y2, FieldOutCell z2);

  // After VTK-m does it's magic, you need to tell what information you need
  // from the provided inputs from the ControlSignature.
  // For now :
  // 1. number of points making an individual cell
  // 2. _2 is the 2nd input to ControlSignature : this should give you all points of the triangle.
  // 3. _3 is the 3rd input to ControlSignature : this should give you the variables at those points.
  // 4. _4 is the 4rd input to ControlSignature : this will help you store the output of your calculation.
  using ExecutionSignature = void(PointCount, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);

  template <typename PointVecType, typename FieldType>
  VTKM_EXEC
  void operator()(const vtkm::IdComponent& numPoints,
                  const PointVecType& points,
                  FieldType& x0,
                  FieldType& y0,
                  FieldType& z0,
                  FieldType& x1,
                  FieldType& y1,
                  FieldType& z1,
                  FieldType& x2,
                  FieldType& y2,
		  FieldType& z2) const
  {
    if(numPoints != 3)
      ASCENT_ERROR("We only play with triangles here");
    // Since you only have triangles, numPoints should always be 3
    // PointType is an abstraction of vtkm::Vec3f, which is {x, y, z} of a point
    using PointType = typename PointVecType::ComponentType;
    // Following lines will help you extract points for a single triangle
    //PointType vertex0 = points[0]; // {x0, y0, z0}
    //PointType vertex1 = points[1]; // {x1, y1, z1}
    //PointType vertex2 = points[2]; // {x2, y2, z2}
    x0 = points[0][0];
    y0 = points[0][1];
    z0 = points[0][2];
    x1 = points[1][0];
    y1 = points[1][1];
    z1 = points[1][2];
    x2 = points[2][0];
    y2 = points[2][1];
    z2 = points[2][2]; 
  }
};

vtkh::DataSet*
AddTriangleFields(vtkh::DataSet &vtkhData)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  vtkh::DataSet* newDataSet = new vtkh::DataSet;

  //if there is data: loop through domains and grab all triangles.
  if(!vtkhData.IsEmpty())
  {
    vtkm::cont::DataSetFieldAdd dataSetFieldAdd;
    for(int i = 0; i < localDomainIds.size(); i++)
    {
      vtkm::cont::DataSet dataset = vtkhData.GetDomain(i);
      //Get Data points
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      //Get triangles
      vtkm::cont::DynamicCellSet cellset = dataset.GetCellSet();

      int numTris = cellset.GetNumberOfCells();
      //make vectors and array handles for x,y,z triangle points.
      std::vector<double> x0(numTris), y0(numTris), z0(numTris), x1(numTris), y1(numTris), z1(numTris), x2(numTris), y2(numTris), z2(numTris);
      std::vector<double> X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2;
     
      vtkm::cont::ArrayHandle<vtkm::Float64> x_0 = vtkm::cont::make_ArrayHandle(x0);
      vtkm::cont::ArrayHandle<vtkm::Float64> y_0 = vtkm::cont::make_ArrayHandle(y0);
      vtkm::cont::ArrayHandle<vtkm::Float64> z_0 = vtkm::cont::make_ArrayHandle(z0);
      vtkm::cont::ArrayHandle<vtkm::Float64> x_1 = vtkm::cont::make_ArrayHandle(x1);
      vtkm::cont::ArrayHandle<vtkm::Float64> y_1 = vtkm::cont::make_ArrayHandle(y1);
      vtkm::cont::ArrayHandle<vtkm::Float64> z_1 = vtkm::cont::make_ArrayHandle(z1);
      vtkm::cont::ArrayHandle<vtkm::Float64> x_2 = vtkm::cont::make_ArrayHandle(x2);
      vtkm::cont::ArrayHandle<vtkm::Float64> y_2 = vtkm::cont::make_ArrayHandle(y2);
      vtkm::cont::ArrayHandle<vtkm::Float64> z_2 = vtkm::cont::make_ArrayHandle(z2);
      vtkm::cont::Invoker invoker;
      invoker(GetTriangleFields{}, cellset, coords, x_0, y_0, z_0, x_1, y_1, z_1, x_2, y_2, z_2);

      X0.insert(X0.end(), x0.begin(), x0.end());
      Y0.insert(Y0.end(), y0.begin(), y0.end());
      Z0.insert(Z0.end(), z0.begin(), z0.end());
      X1.insert(X1.end(), x1.begin(), x1.end());
      Y1.insert(Y1.end(), y1.begin(), y1.end());
      Z1.insert(Z1.end(), z1.begin(), z1.end());
      X2.insert(X2.end(), x2.begin(), x2.end());
      Y2.insert(Y2.end(), y2.begin(), y2.end());
      Z2.insert(Z2.end(), z2.begin(), z2.end());
      //dataset.AddCellField("X0", X0);
      //dataset.AddCellField("Y0", Y0);
      //dataset.AddCellField("Z0", Z0);
      //dataset.AddCellField("X1", X1);
      //dataset.AddCellField("Y1", Y1);
      //dataset.AddCellField("Z1", Z1);
      //dataset.AddCellField("X2", X2);
      //dataset.AddCellField("Y2", Y2);
      //dataset.AddCellField("Z2", Z2);
      dataSetFieldAdd.AddCellField(dataset, "X0", X0);
      dataSetFieldAdd.AddCellField(dataset, "Y0", Y0);
      dataSetFieldAdd.AddCellField(dataset, "Z0", Z0);
      dataSetFieldAdd.AddCellField(dataset, "X1", X1);
      dataSetFieldAdd.AddCellField(dataset, "Y1", Y1);
      dataSetFieldAdd.AddCellField(dataset, "Z1", Z1);
      dataSetFieldAdd.AddCellField(dataset, "X2", X2);
      dataSetFieldAdd.AddCellField(dataset, "Y2", Y2);
      dataSetFieldAdd.AddCellField(dataset, "Z2", Z2);
      newDataSet->AddDomain(dataset,localDomainIds[i]);
    }
  }
  return newDataSet;
}

std::vector<Triangle>
GetTriangles(vtkh::DataSet &vtkhData)
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
      vtkm::cont::DataSet dataset = vtkhData.GetDomain(i);
      //Get Data points
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      //Get triangles
      vtkm::cont::DynamicCellSet cellset = dataset.GetCellSet();
      //Get variable

      int numTris = cellset.GetNumberOfCells();
      std::vector<Triangle> tmp_tris(numTris);
     
      vtkm::cont::ArrayHandle<Triangle> triangles = vtkm::cont::make_ArrayHandle(tmp_tris);
      vtkm::cont::Invoker invoker;
      invoker(ProcessTriangle{}, cellset, coords, triangles);

      //combine all domain triangles
      tris.insert(tris.end(), tmp_tris.begin(), tmp_tris.end());
    }
  }
  return tris;
}

std::vector<float>
GetScalarData(vtkh::DataSet &vtkhData, std::string field_name, int height, int width)
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
#endif

Triangle transformTriangle(Triangle t, Camera c, int width, int height)
{
  bool print = false;
  Matrix camToView, m0, cam, view;
  cam = c.CameraTransform();
  view = c.ViewTransform();
  camToView = Matrix::ComposeMatrices(cam, view);
  m0 = Matrix::ComposeMatrices(camToView, c.DeviceTransform(width, height));
  /*
  cerr<< "cam" << endl;
  cam.Print(cerr);
  cerr<< "view" << endl;
  view.Print(cerr);
  cerr<< "m0" << endl;
  m0.Print(cerr);
  cerr<< "camToView" << endl;
  camToView.Print(cerr);
  cerr<< "device t" << endl;
  c.DeviceTransform(width, height).Print(cerr);
  */

  Triangle triangle;
  // Zero XYZ
  double pointOut[4];
  double pointIn[4];
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
    cerr << "triangle out: (" << triangle.X[0] << " , " << triangle.Y[0] << " , " << triangle.Z[0] << ") " << endl <<
                         " (" << triangle.X[1] << " , " << triangle.Y[1] << " , " << triangle.Z[1] << ") " << endl <<
                         " (" << triangle.X[2] << " , " << triangle.Y[2] << " , " << triangle.Z[2] << ") " << endl;
  }

  return triangle;

}

double magnitude3d(double* vec)
{
  return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

void CalcSilhouette(float * data_in, int width, int height, double &length, double &curvature, double &curvatureExtrema, double &entropy)
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
  cerr << "CONTOURS SIZE " << contours.size() << endl;
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

void prewittX_kernel(const int rows, const int cols, double * const kernel) 
{
  if(rows != 3 || cols !=3) 
  {
    std::cerr << "Bad Prewitt kernel matrix\n";
    return;
  }
  for(int i=0;i<3;i++) 
  {
    kernel[0 + (i*rows)] = -1.0;
    kernel[1 + (i*rows)] = 0.0;
    kernel[2 + (i*rows)] = 1.0;
  }
}

void prewittY_kernel(const int rows, const int cols, double * const kernel) 
{
  if(rows != 3 || cols !=3) 
  {
    std::cerr << "Bad Prewitt kernel matrix\n";
    return;
  }
  for(int i=0;i<3;i++) 
  {
    kernel[i + (0*rows)] = 1.0;
    kernel[i + (1*rows)] = 0.0;
    kernel[i + (2*rows)] = -1.0;
  }
}

void apply_prewitt(const int rows, const int cols, 
		   float * const in, float * const out) {
  const int dim = 3;
  double kernelY[dim*dim];
  double kernelX[dim*dim];
  prewittY_kernel(3,3,kernelY);
  prewittX_kernel(3,3,kernelX);
  double gY = 0;
  double gX = 0;

  for(int i = 0; i < rows; i++) 
  {
    for(int j = 0; j < cols; j++) 
    {
      const int out_offset = i + (j*rows);
      // For each pixel, do the stencil
      gY = 0;
      gX = 0;
      double intensity = 0;
      for(int x = i - 1, kx = 0; x <= i + 1; x++, kx++) 
      {
        for(int y = j - 1, ky = 0; y <= j + 1; y++, ky++) 
	{
          if(x >= 0 && x < rows && y >= 0 && y < cols) 
	  {
            const int in_offset = x + (y*rows);
            const int k_offset = kx + (ky*dim);
	    gY += in[in_offset]*kernelY[k_offset];
	    gX += in[in_offset]*kernelX[k_offset];
	   }
	 }
      }
      intensity = sqrt(gY*gY + gX*gX);
      out[out_offset] = intensity;
      out[out_offset] = intensity;
      out[out_offset] = intensity;
    }
  }
}


void fibonacci_sphere(int i, int samples, double* points)
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



/*	
  double gr = (sqrt(5.0)+1)/2;
  double ga = (2 - gr)*(2.0*M_PI);
  cerr << "ga: " << ga << endl;
  cerr << "gr: " << gr << endl;
  double lat = asin(-1.0 + 2.0*i/(samples+1.0));
  double lon = ga*i;
  cerr << "lat: " << lat << endl;
  cerr << "lon: " << lon << endl;
  
  double x = cos(lon)*cos(lat);
  double y = sin(lon)*cos(lat);
  double z = sin(lat);
  
  cerr << "x: " << x << endl;
  cerr << "y: " << y << endl;
  cerr << "z: " << z << endl;
*/
/*
  double phi = acos(1 - 2*(i + .5)/samples);
  double gr = (sqrt(5.0)+1)/2;
  double theta = 2*M_PI*(i+.5)/gr;

  double x = cos(theta)*sin(phi);
  double y = sin(theta)*sin(phi);
  double z = cos(phi);

  cerr << "x: " << x << endl;
  cerr << "y: " << y << endl;
  cerr << "z: " << z << endl;
*/
  
  points[0] = x;
  points[1] = y;
  points[2] = z;
}

template< typename T >
T calcentropy( const T* array, long len, int nBins )
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

//calculate (world space) area without camera
float
calcArea(std::vector<float> triangle)
{
  //need to transform triangle to camera viewpoint
  Triangle tri(triangle[0], triangle[1], triangle[2],
               triangle[3], triangle[4], triangle[5],
               triangle[6], triangle[7], triangle[8]);
  return tri.calculateTriArea();

}

//calculate image space area
float
calcArea(std::vector<float> triangle, Camera c, int width, int height)
{
  //need to transform triangle to device space with given camera
  Triangle w_tri(triangle[0], triangle[1], triangle[2], 
	       triangle[3], triangle[4], triangle[5], 
	       triangle[6], triangle[7], triangle[8]);
  Triangle d_tri = transformTriangle(w_tri, c, width, height);
  /*
  cerr << "w_tri: " << endl;
  w_tri.printTri();
  cerr << "d_tri: " << endl;
  d_tri.printTri();
  */
  return d_tri.calculateTriArea();

}

#if defined(ASCENT_VTKM_ENABLED)

float
calculateVisibilityRatio(vtkh::DataSet* dataset, std::vector<Triangle> &local_triangles, int height, int width)
{
  float visibility_ratio = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    int num_local_triangles = local_triangles.size();
    float local_area        = 0.0;
    float global_area       = 0.0;
    
    for(int i = 0; i < num_local_triangles; i++)
    {
      float area = local_triangles[i].calculateTriArea();
      local_area += area;
    }

    MPI_Reduce(&local_area, &global_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      int num_triangles = triangles.size();
      float projected_area = 0.0;
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i]);
        projected_area += area;
      }
      visibility_ratio = projected_area/global_area;
    }
    MPI_Bcast(&visibility_ratio, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //cerr << "visibility_ratio " << visibility_ratio << endl;
    return visibility_ratio;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    int num_triangles = triangles.size();
    int num_local_triangles = local_triangles.size();
    float projected_area = 0.0;
    float total_area     = 0.0;
    
    for(int i = 0; i < num_local_triangles; i++)
    {
      float area = local_triangles[i].calculateTriArea();
      total_area += area;
    }
    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i]);
      projected_area += area;
    }
    visibility_ratio = projected_area/total_area;
    return visibility_ratio;
  #endif
}

float 
calculateViewpointEntropy(vtkh::DataSet* dataset, std::vector<Triangle> &local_triangles, int height, int width, Camera camera)
{
  float viewpoint_entropy = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    int num_local_triangles = local_triangles.size();
    float global_area       = 0.0;
    float local_area        = 0.0;
    for(int i = 0; i < num_local_triangles; i++)
    {
      Triangle t = transformTriangle(local_triangles[i], camera, width, height); 
      float area = t.calculateTriArea();
      local_area += area;
    }
    MPI_Reduce(&local_area, &global_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      int num_triangles     = triangles.size();
      float viewpoint_ratio = 0.0;
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i]);
	if(area != 0.0)
          viewpoint_ratio += ((area/global_area)*std::log(area/global_area));
      }
      viewpoint_entropy = (-1.0)*viewpoint_ratio;
    }
    MPI_Bcast(&viewpoint_entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
//    cerr << "viewpoint_entropy " << viewpoint_entropy << endl;
    return viewpoint_entropy;
  #else
    int size = height*width;

    //Stefan print statement
    //cout << "Size is " << size << endl;
    //End Stefan print statement
    
    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    int num_triangles = triangles.size();

    int num_local_triangles = local_triangles.size();

    float total_area      = 0.0;
    float viewpoint_ratio = 0.0;
    for(int i = 0; i < num_local_triangles; i++)
    {
      Triangle t = transformTriangle(local_triangles[i], camera, width, height);
      float area = t.calculateTriArea();
      total_area += area;
    }

    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i]);

      if(area != 0.0)
        viewpoint_ratio += ((area/total_area)*std::log(area/total_area));
    }

    viewpoint_entropy = (-1.0)*viewpoint_ratio;

    return viewpoint_entropy;
  #endif
}

float
calculateI2(vtkh::DataSet* dataset, std::vector<Triangle> &local_triangles, int height, int width, Camera camera)
{
  float viewpoint_entropy = 0.0;
  float i2 = 0.0;
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
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      int num_triangles     = triangles.size();
      int num_local_triangles = local_triangles.size();
      float total_area      = 0.0;
      float real_total_area = 0.0;
      float viewpoint_ratio = 0.0;
      float hz              = 0.0;
      for(int i = 0; i < num_local_triangles; i++)
      {
        Triangle t = transformTriangle(local_triangles[i], camera, width, height);
        float area = t.calculateTriArea();
        total_area += area;
	real_total_area += local_triangles[i].calculateTriArea();
      }
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i]);
        if(area != 0.0)
          viewpoint_ratio += ((area/total_area)*std::log(area/total_area));
      }
      for(int i = 0; i < num_local_triangles; i++)
      {
        float area = local_triangles[i].calculateTriArea();
	if(area != 0 && real_total_area != 0)
          hz += (area/real_total_area)*log((area/real_total_area));
      }
      viewpoint_entropy = (-1.0)*viewpoint_ratio;

      hz = (-1.0)*hz;
      cerr << "viewpiont ent: " << viewpoint_entropy;
      cerr << "hz: " << hz << endl;

      i2 = hz - viewpoint_entropy;
      i2 = (-1.0)*i2;
    }
    MPI_Bcast(&i2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return i2;
  #else
    int size = height*width;

    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    int num_triangles = triangles.size();

    int num_local_triangles = local_triangles.size();

    float total_area      = 0.0;
    float real_total_area = 0.0;
    float hz              = 0.0;
    float viewpoint_ratio = 0.0;
    for(int i = 0; i < num_local_triangles; i++)
    {
      Triangle t = transformTriangle(local_triangles[i], camera, width, height);
      float area = t.calculateTriArea();
      total_area += area;
    }

    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i]);

      if(area != 0.0)
        viewpoint_ratio += ((area/total_area)*std::log(area/total_area));
    }
    for(int i = 0; i < num_local_triangles; i++)
    {
      float area = local_triangles[i].calculateTriArea();
      if(area != 0 && real_total_area != 0)
        hz += (area/real_total_area)*log((area/real_total_area));
    }
    viewpoint_entropy = (-1.0)*viewpoint_ratio;
    hz = (-1.0)*hz;

    i2 = hz - viewpoint_entropy;
    i2 = (-1.0)*i2;

    return i2;
  #endif
}

float
calculateVKL(vtkh::DataSet* dataset, std::vector<Triangle> &local_triangles, int height, int width, Camera camera)
{
  float vkl = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    //Needs total world area and total image space area
    int num_local_triangles = local_triangles.size();
    float total_area     = 0.0;
    float local_area     = 0.0;
    float w_total_area   = 0.0;
    float w_local_area   = 0.0;
    for(int i = 0; i < num_local_triangles; i++)
    {
      float w_area = local_triangles[i].calculateTriArea();
      Triangle t = transformTriangle(local_triangles[i], camera, width, height);	
      float area = t.calculateTriArea();
      local_area += area;
      w_local_area += w_area;
    }
    MPI_Reduce(&w_local_area, &w_total_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_area, &total_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      int num_triangles = triangles.size();
      float projected_area = 0.0;
      for(int i = 0; i < num_triangles; i++)
      {
	float area = calcArea(triangles[i], camera, width, height);
        projected_area += area;
      }
      for(int i = 0; i < num_triangles; i++)
      {
	float area   = calcArea(triangles[i], camera, width, height);
        float w_area = calcArea(triangles[i]);
	if(area != 0.0 && w_area != 0.0)
	  vkl += (area/projected_area)*std::log((area/projected_area)/(w_area/w_total_area));
      }
    }
    MPI_Bcast(&vkl, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //cerr << "vkl " << vkl << endl;
    return (-1.0)*vkl;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    int num_triangles     = triangles.size();
    int num_local_triangles = local_triangles.size();
    float total_area     = 0.0;
    float w_total_area   = 0.0;
    float projected_area = 0.0;
    for(int i = 0; i < num_local_triangles; i++)
    {
      float w_area = local_triangles[i].calculateTriArea();
      Triangle t = transformTriangle(local_triangles[i], camera, width, height);
      float area = t.calculateTriArea();
      total_area += area;
      w_total_area += w_area;
    }
    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i], camera, width, height);
      projected_area += area;
    }
    for(int i = 0; i < num_triangles; i++)
    {
      float area   = calcArea(triangles[i], camera, width, height);
      float w_area = calcArea(triangles[i]);
      if(area != 0.0 && w_area != 0.0)
        vkl += (area/projected_area)*std::log((area/projected_area)/(w_area/w_total_area));
    }
    return (-1.0)*vkl;
  #endif
}

float
calculateDataEntropy(vtkh::DataSet* dataset, int height, int width,std::string field_name)
{
  float entropy = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now(); 

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> field_data = GetScalarData(*dataset, field_name, height, width);
      std::vector<float> data;
      for(int i = 0; i < size; i++)
        if(field_data[i] == field_data[i])
          data.push_back(field_data[i]);
      float field_array[data.size()];
      std::copy(data.begin(), data.end(), field_array);
      entropy = calcentropy(field_array, data.size(), 100);
    }
    auto time_stop = high_resolution_clock::now();
    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = metric_time;
    MPI_Allgather(&metric_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("metric_times.txt", array, world_size);
    MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #else
    int size = height*width;
    std::vector<float> field_data = GetScalarData(*dataset, field_name, height, width);
    std::vector<float> data;
    for(int i = 0; i < size; i++)
      if(field_data[i] == field_data[i])
        data.push_back(field_data[i]);
    float field_array[data.size()];
    std::copy(data.begin(), data.end(), field_array);
    entropy = calcentropy(field_array, data.size(), 100);
  #endif
  return entropy;
}

float 
calculateDepthEntropy(vtkh::DataSet* dataset, int height, int width)
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
      std::vector<float> depth = GetScalarData(*dataset, "depth", height, width);
      std::vector<float> depth_data;
      for(int i = 0; i < size; i++)
        if(depth[i] == depth[i] && depth[i] <= 1000)
	{
          depth_data.push_back(depth[i]);
	}
          //depth_data[i] = -FLT_MAX;
      float depth_array[depth_data.size()];
      std::copy(depth_data.begin(), depth_data.end(), depth_array);
      entropy = calcentropy(depth_array, depth_data.size(), 100);

    }
    MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #else
    int size = height*width;
    std::vector<float> depth = GetScalarData(*dataset, "depth", height, width);
    std::vector<float> depth_data;
    for(int i = 0; i < size; i++)
      if(depth[i] == depth[i] && depth[i] <= 1000)
      {
        depth_data.push_back(depth[i]);
      }
        //depth_data[i] = -FLT_MAX;
    float depth_array[depth_data.size()];
    std::copy(depth_data.begin(), depth_data.end(), depth_array);
    entropy = calcentropy(depth_array, depth_data.size(), 100);
  #endif
  return entropy;
}

float
calculateVisibleTriangles(vtkh::DataSet *dataset, int height, int width)
{
  float num_triangles = 0.0;
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
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      num_triangles = triangles.size();
    }
    MPI_Bcast(&num_triangles, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    num_triangles = triangles.size();
  #endif
  return num_triangles;
}

float
calculateProjectedArea(vtkh::DataSet* dataset, int height, int width, Camera camera)
{
  float projected_area = 0.0;
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
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      int num_triangles = triangles.size();
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i], camera, width, height);
        projected_area += area;
      }
    }
    MPI_Bcast(&projected_area, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    int num_triangles = triangles.size();
    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i], camera, width, height);
      projected_area += area;
    }
  #endif
  return projected_area;
}

float
calculatePlemenosAndBenayada(vtkh::DataSet *dataset, int num_local_triangles, int height, int width, Camera camera)
{
  float pb_score = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    //Needs total global triangles
    int num_global_triangles = 0;
    MPI_Reduce(&num_local_triangles, &num_global_triangles, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      std::sort(triangles.begin(), triangles.end());
      triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
      int num_triangles = triangles.size();
      float projected_area = 0.0;
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i], camera, width, height);
        projected_area += area;
      }

      float pixel_ratio = projected_area/size;
      float triangle_ratio = (float) num_triangles/(float) num_global_triangles;
      //cerr << "pixel_ratio: " << pixel_ratio << endl;
      //cerr << "triangle_ratio: " << triangle_ratio << endl;
      pb_score = pixel_ratio + triangle_ratio;
    }
    MPI_Bcast(&pb_score, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
    for(int i = 0; i < size; i++)
    {
      if(x0[i] == x0[i]) //!nan
      {
        std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
        triangles.push_back(tri);
       }
    }
    std::sort(triangles.begin(), triangles.end());
    triangles.erase(std::unique(triangles.begin(), triangles.end()), triangles.end());
    int num_triangles = triangles.size();
    float projected_area = 0.0;
    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i], camera, width, height);
      projected_area += area;
    }

    float pixel_ratio = projected_area/size;
    float triangle_ratio = (float) num_triangles/(float) num_local_triangles;
    pb_score = pixel_ratio + triangle_ratio;
  #endif
  return pb_score;

}

float
calculateMaxDepth(vtkh::DataSet *dataset, int height, int width)
{
  float depth = -FLT_MAX;
  #if ASCENT_MPI_ENABLED
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double metric_time = 0.;
    auto time_start = high_resolution_clock::now(); 
    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> depth_data = GetScalarData(*dataset, "depth", height, width);
      for(int i = 0; i < size; i++)
        if(depth_data[i] == depth_data[i])
	{
          if(depth < depth_data[i] && depth_data[i] <= 1000)
	  {
            depth = depth_data[i];
	  }
	}
    }
    auto time_stop = high_resolution_clock::now(); 
    metric_time += duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    time_start = high_resolution_clock::now();
    MPI_Bcast(&depth, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    time_stop = high_resolution_clock::now();
    double coord = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " coordination time: " << coord << " microseconds." << endl;
  #else
    int size = height*width;
    std::vector<float> depth_data = GetScalarData(*dataset, "depth", height, width);
    for(int i = 0; i < size; i++)
      if(depth_data[i] == depth_data[i])
        if(depth < depth_data[i] && depth_data[i] <= 1000)
          depth = depth_data[i];
  #endif
  return depth;
}
/*
float 
calculateMaxSilhouette(vtkh::DataSet *dataset, int height, int width)
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
        std::vector<float> depth_data = GetScalarData(*dataset, "depth", height, width);
        for(int i = 0; i < size; i++)
          if(depth_data[i] == depth_data[i])
            depth_data[i] = 255.0; //data = white
          else
            depth_data[i] = 0.0; //background = black

        float data_in[width*height];
        float contour[width*height];
        std::copy(depth_data.begin(), depth_data.end(), data_in);
        double length, curvature, curvatureExtrema, entropy;
        CalcSilhouette(data_in, width, height, length, curvature, curvatureExtrema, entropy);
        MPI_Bcast(&length, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }
    #else
      int size = height*width;
      std::vector<float> depth_data = GetScalarData(*dataset, "depth", height, width);
      for(int i = 0; i < size; i++)
        if(depth_data[i] == depth_data[i])
          depth_data[i] = 255.0;
        else
          depth_data[i] = 0.0;
      float data_in[size];
      float contour[size];
      std::copy(depth_data.begin(), depth_data.end(), data_in);
      double length, curvature, curvatureExtrema, entropy;
      CalcSilhouette(data_in, width, height, length, curvature, curvatureExtrema, entropy);
    #endif
    return (float)length;
}
*/

float
calculateMetric(vtkh::DataSet* dataset, std::string metric, std::string field_name, std::vector<Triangle> &local_triangles, int height, int width, Camera camera)
{
  float score = 0.0;

  if(metric == "data_entropy")
  {
    score = calculateDataEntropy(dataset, height, width, field_name);
  }
  else if (metric == "visibility_ratio")
  {
    score = calculateVisibilityRatio(dataset, local_triangles,  height, width);
  }
  else if (metric == "viewpoint_entropy")
  {
    score = calculateViewpointEntropy(dataset, local_triangles, height, width, camera);
  }
  else if (metric == "i2")
  {
    score = calculateI2(dataset, local_triangles, height, width, camera);
  }
  else if (metric == "vkl")
  {
    score = calculateVKL(dataset, local_triangles, height, width, camera);
  }
  else if (metric == "visible_triangles")
  {
    score = calculateVisibleTriangles(dataset, height, width);
  }
  else if (metric == "projected_area")
  {
    score = calculateProjectedArea(dataset, height, width, camera);
  }
  else if (metric == "pb")
  {
    int num_local_triangles = local_triangles.size();
    score = calculatePlemenosAndBenayada(dataset, num_local_triangles, height, width, camera); 
  }
  else if (metric == "depth_entropy")
  {
    score = calculateDepthEntropy(dataset, height, width);
  }
  else if (metric == "max_depth")
  {
    score = calculateMaxDepth(dataset, height, width);
  }
  else
    ASCENT_ERROR("This metric is not supported. \n");

  return score;
}

#endif
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

AutoCamera::AutoCamera()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
AutoCamera::~AutoCamera()
{
// empty
}

//-----------------------------------------------------------------------------
void
AutoCamera::declare_interface(Node &i)
{
    i["type_name"]   = "auto_camera";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
AutoCamera::verify_params(const conduit::Node &params,
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
AutoCamera::execute()
{
    double time = 0.;
    auto time_start = high_resolution_clock::now();

    #if defined(ASCENT_VTKM_ENABLED)
      #if ASCENT_MPI_ENABLED
        int rank;
	int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Barrier(MPI_COMM_WORLD);
      
      #endif  
      DataObject *data_object = input<DataObject>(0);
      std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
    //int cycle = params()["state/cycle"].to_int32();
      conduit::Node * meta = graph().workspace().registry().fetch<Node>("metadata");
      int cycle = -1;
      if(meta->has_path("cycle"))
      {
        cycle = (*meta)["cycle"].to_int32();
      }
      cerr << "=====USING CAMERA PIPELINE===== CYCLE: " << cycle << endl;
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
    
      double triangle_time = 0.;
      auto triangle_start = high_resolution_clock::now();
      std::string topo_name = collection->field_topology(field_name);

      vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);
      
      std::vector<Triangle> triangles = GetTriangles(dataset);
      int num_local_triangles = triangles.size();
      vtkh::DataSet* data = AddTriangleFields(dataset);
      int num_domains = data->GetNumberOfDomains();
      for(int i = 0; i < num_domains; i++)
      {
        vtkm::cont::DataSet data_set;
        vtkm::Id domain_id;
        data->GetDomain(i, data_set, domain_id);

      // all the data sets better be the same
        if(data_set.GetCellSet().GetNumberOfCells() == 0)
          data_set.Clear();
      }
      //data->PrintSummary(cerr);

      vtkm::Bounds b = data->GetGlobalBounds();
      vtkm::Float32 xb = vtkm::Float32(b.X.Length());
      vtkm::Float32 yb = vtkm::Float32(b.Y.Length());
      vtkm::Float32 zb = vtkm::Float32(b.Z.Length());
      float bounds[6] = {(float)b.X.Max, (float)b.X.Min, 
	                (float)b.Y.Max, (float)b.Y.Min, 
	                (float)b.Z.Max, (float)b.Z.Min};
      //double bounds[3] = {(double)xb, (double)yb, (double)zb};
      //cerr << "x y z bounds " << xb << " " << yb << " " << zb << endl;

      vtkm::Float32 radius = sqrt(xb*xb + yb*yb + zb*zb)/2.0;
      //cerr << "radius " << radius << endl;
      //if(radius<1)
        //radius = radius + 1;
      //vtkm::Float32 x_pos = 0., y_pos = 0., z_pos = 0.;
      vtkmCamera *camera = new vtkmCamera;
      camera->ResetToBounds(data->GetGlobalBounds());
    //cerr << "vtkm Cam" << endl;
    //camera->Print();
      vtkm::Vec<vtkm::Float32,3> lookat = camera->GetLookAt();
      double focus[3] = {(double)lookat[0],(double)lookat[1],(double)lookat[2]};

      auto triangle_stop = high_resolution_clock::now();
      triangle_time += duration_cast<microseconds>(triangle_stop - triangle_start).count();
      //cerr << "Global bounds: " << dataset.GetGlobalBounds() << endl;
      #if ASCENT_MPI_ENABLED
        cerr << "rank " << rank << " num_local_triangles: " << num_local_triangles << endl;
        //cerr << "Global bounds: " << dataset.GetGlobalBounds() << endl;
        //cerr << "rank " << rank << " bounds: " << dataset.GetBounds() << endl;
//	cerr << "rank " << rank << " data processing time: " << triangle_time << " microseconds. " << endl;
	double array[world_size] = {0};
        array[rank] = triangle_time;
        MPI_Allgather(&triangle_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if(rank == 0)
          MakeFile("processing_times.txt", array, world_size);
      #endif


      double winning_score  = -DBL_MAX;
      int    winning_sample = -1;
      double losing_score   = DBL_MAX;
      int    losing_sample  = -1;
    //loop through number of camera samples.
      double scanline_time = 0.;
      double metric_time   = 0.;

      for(int sample = 0; sample < samples; sample++)
      {
    /*================ Scalar Renderer Code ======================*/
    //What it does: Quick ray tracing of data (replaces get triangles and scanline).
    //What we need: z buffer, any other important buffers (tri ids, scalar values, etc.)
        auto render_start = high_resolution_clock::now();
      
        Camera cam = GetCamera(sample, samples, radius, focus, bounds);
        vtkm::Vec<vtkm::Float32, 3> pos{(float)cam.position[0],
                                (float)cam.position[1],
                                (float)cam.position[2]};

        camera->SetPosition(pos);
        vtkh::ScalarRenderer tracer;
        tracer.SetWidth(width);
        tracer.SetHeight(height);
        tracer.SetInput(data); //vtkh dataset by toponame
        tracer.SetCamera(*camera);
        tracer.Update();

        vtkh::DataSet *output = tracer.GetOutput();
	auto render_stop = high_resolution_clock::now();
	double render_time = duration_cast<microseconds>(render_stop - render_start).count();
	#if ASCENT_MPI_ENABLED
	  double array[world_size] = {0};
          array[rank] = render_time;
          MPI_Allgather(&render_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
          if(rank == 0)
            MakeFile("renderer_times.txt", array, world_size);
//          cerr << "rank: " << rank << " ScalarRenderer time: " << render_time  << " microseconds " << endl;
        #endif
        
        float score = calculateMetric(output, metric, field_name, triangles, height, width, cam);
        std::cerr << "sample " << sample << " score: " << score << std::endl;
        delete output;

    /*================ End Scalar Renderer  ======================*/

        //cerr << "sample " << sample << " score: " << score << endl;
        if(winning_score < score)
        {
          winning_score = score;
	  winning_sample = sample;
        }
        if(losing_score > score)
        {
          losing_score = score;
	  losing_sample = sample;
        }
      } //end of sample loop
      triangles.clear();
      delete data;
      auto setting_camera_start = high_resolution_clock::now();

      if(winning_sample == -1)
        ASCENT_ERROR("Something went terribly wrong; No camera position was chosen");
      cerr << metric << " winning_sample " << winning_sample << " score: " << winning_score << endl;
      cerr << metric << " losing_sample " << losing_sample << " score: " << losing_score << endl;
     // Camera best_c = GetCamera(cycle, 100, radius, focus, bounds);
      //Camera best_c = GetCamera(losing_sample, samples, radius, focus, bounds);
      Camera best_c = GetCamera(winning_sample, samples, radius, focus, bounds);
    
      vtkm::Vec<vtkm::Float32, 3> pos{(float)best_c.position[0], 
	                            (float)best_c.position[1], 
				    (float)best_c.position[2]}; 
/*
#if ASCENT_MPI_ENABLED
    if(rank == 0)
    {
      cerr << "look at: " << endl;
      vtkm::Vec<vtkm::Float32,3> lookat = camera->GetLookAt();
      cerr << lookat[0] << " " << lookat[1] << " " << lookat[2] << endl;
      camera->Print();
    }
#endif
*/
      camera->SetPosition(pos);
      //camera->Print();


      if(!graph().workspace().registry().has_entry("camera"))
      {
      //cerr << "making camera in registry" << endl;
        graph().workspace().registry().add<vtkm::rendering::Camera>("camera",camera,1);
      }
      auto setting_camera_end = high_resolution_clock::now();
      double setting_camera = 0.;
      setting_camera += duration_cast<microseconds>(setting_camera_end - setting_camera_start).count();
      #if ASCENT_MPI_ENABLED
        double array2[world_size] = {0};
        array2[rank] = setting_camera;
        MPI_Allgather(&setting_camera, 1, MPI_DOUBLE, array2, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if(rank == 0)
          MakeFile("setCam_times.txt", array2, world_size);
//        cerr << "rank: " << rank << " Setting Camera time: " << setting_camera  << " microseconds " << endl;
      #endif

    //This breaks everything
    //TODO:Figure out where to delete it, probably after where it's grabbed. 
/*
#if ASCENT_MPI_ENABLED
    if(rank == 0)
      camera->Print();
#endif
*/
      #endif
    set_output<DataObject>(input<DataObject>(0));
    //set_output<vtkmCamera>(camera);
    auto time_stop = high_resolution_clock::now();
    time += duration_cast<microseconds>(time_stop - time_start).count();
    cerr << "========END CAMERA PIPELINE=======" << endl;
    
   #if ASCENT_MPI_ENABLED
     double array3[world_size] = {0};
     array3[rank] = time;
     MPI_Allgather(&time, 1, MPI_DOUBLE, array3, 1, MPI_DOUBLE, MPI_COMM_WORLD);
     if(rank == 0)
       MakeFile("total_times.txt", array3, world_size);
     //cerr << "rank: " << rank << " Total Time: " << time  << " microseconds " << endl;
    #endif
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
