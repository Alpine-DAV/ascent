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
#include <ascent_metadata.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/ScalarRenderer.hpp>
#include <vtkm/cont/DataSet.h>
#include <vtkm/rendering/raytracing/Camera.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/worklet/WorkletMapTopology.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/worklet/FieldHistogram.h>

#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <chrono>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>

#ifdef ASCENT_USE_OPENMP
#include <thread>
#endif

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
MakeFile(std::string filename, float *array, int size)
{
  ofstream myfile(filename, ios::out | ios::app);
  if(myfile.is_open())
  {
    for(int i = 0; i < size; i++)
    {
      myfile << array[i] << "\n";
    }
  } 
}
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
  } 
} 
//Camera Class Functions

Matrix
Camera::CameraTransform(void) const
{
  // bool print = false;
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


  /*
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
  */

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

  /*
  if(print)
  {
    cerr << "Camera:" << endl;
    camera.Print(cerr);
  }
  */
  return camera;

};

Matrix
Camera::ViewTransform(void) const
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
Camera::DeviceTransform() const
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
Camera::DeviceTransform(int width, int height) const
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
  /*
  for (int i = 0 ; i < 4 ; i++)
  {
      char str[256];
      sprintf(str, "(%.7f %.7f %.7f %.7f)\n", A[i][0], A[i][1], A[i][2], A[i][3]);
      o << str;
  }
  */
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
Triangle::printTri() const
{
  cerr << "X: " << X[0] << " " << X[1] << " " << X[2] << endl;
  cerr << "Y: " << Y[0] << " " << Y[1] << " " << Y[2] << endl;
  cerr << "Z: " << Z[0] << " " << Z[1] << " " << Z[2] << endl;
}

float
Triangle::calculateTriArea() const
{
  // bool print = false;
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

  area = vtkm::Sqrt(
    vtkm::Pow(orthogonal_vec[0], 2) +
    vtkm::Pow(orthogonal_vec[1], 2) +
    vtkm::Pow(orthogonal_vec[2], 2)) * 0.5f;

  /*
  if(print)
  {
  cerr << "Triangle: (" << X[0] << " , " << Y[0] << " , " << Z[0] << ") " << endl <<
                   " (" << X[1] << " , " << Y[1] << " , " << Z[1] << ") " << endl <<
                   " (" << X[2] << " , " << Y[2] << " , " << Z[2] << ") " << endl <<
           " has surface area: " << area << endl;
  }
  */
  return area;
}

void
Triangle::cutoff(int width, int height)
{
  if(X[0] < 0) X[0] = 0;
  if(X[0] > width) X[0] = width;
  if(X[1] < 0) X[1] = 0;
  if(X[1] > width) X[1] = width;
  if(X[2] < 0) X[2] = 0;
  if(X[2] > width) X[2] = width;

  if(Y[0] < 0) Y[0] = 0;
  if(Y[0] > height) Y[0] = height;
  if(Y[1] < 0) Y[1] = 0;
  if(Y[1] > height) Y[1] = height;
  if(Y[2] < 0) Y[2] = 0;
  if(Y[2] > height) Y[2] = height;
}

double
Triangle::findMin(double a, double b, double c) const
{
  double min = a;
  if (b < min)
    min = b;
  if (c < min)
    min = c;
  return min;
}

double
Triangle::findMax(double a, double b, double c) const
{
  double max = a;
  if (b > max)
    max = b;
  if (c > max)
    max = c;
  return max;
}

float
findMax(float a, float b, float c)
{
  float max = a;
  if (b > max)
    max = b;
  if (c > max)
    max = c;
  return max;
}

float
findMin(float a, float b, float c)
{
  float min = a;
  if (b < min)
    min = b;
  if (c < min)
    min = c;
  return min;
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

template<typename T>
void normalize(T * normal) 
{
  T total = pow(normal[0], 2.0) + pow(normal[1], 2.0) + pow(normal[2], 2.0);
  total = pow(total, 0.5);
  normal[0] = normal[0] / total;
  normal[1] = normal[1] / total;
  normal[2] = normal[2] / total;
}

template<typename T>
T dotProduct(const T* v1, const T* v2, int length)
{
  T dotproduct = 0;	
  for (int i = 0; i < length; i++)
  {
    dotproduct += (v1[i]*v2[i]);
  }
  return dotproduct;
}

template<typename T>
void crossProduct(const T a[3], const T b[3], T output[3])
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
GetCameraPhiTheta(float* bounds,  double radius, int thetaPos, int numTheta, int phiPos, int numPhi, float *lookat)
{
  Camera c;
  double zoom = 3.0;
  c.near = zoom/20;
  c.far = zoom*25;
  c.angle = M_PI/6;

  cerr << "radius: " << radius << endl;

  double theta = (thetaPos / (numTheta - 1.0)) * M_PI;
  double phi = (phiPos / (numPhi - 1.0)) * M_PI * 2.0;
  
  cerr << "phi: " << phi << " phiPos: " << phiPos << " numPhi: " << numPhi << endl;
  cerr << "theta: " << theta << " thetaPos: " << thetaPos << " numTheta: " << numTheta << endl;
  
  double xm = (bounds[0] + bounds[1])/2;
  double ym = (bounds[2] + bounds[3])/2;
  double zm = (bounds[4] + bounds[5])/2;

  cerr << "sin(theta): " << sin(theta) << " cos(phi): " << cos(phi) << " cos(theta): " << cos(theta) << " sin(phi): " << sin(phi) << endl;
  
  c.position[0] = (  zoom*radius * sin(theta) * cos(phi)  + xm );
  c.position[1] = (  zoom*radius * sin(theta) * sin(phi)  + ym );
  c.position[2] = (  zoom*radius * cos(theta)  + zm );


  //check lookat vs middle
  cerr << "xm ym zm : " << xm <<  " " << ym << " " << zm << endl;
//  cerr << "lookat: " << lookat[0] << " " << lookat[1] << " " << lookat[2] << endl;
  cerr << "position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
  
  c.focus[0] = lookat[0];
  c.focus[1] = lookat[1];
  c.focus[2] = lookat[2];
  c.up[0] = 0;
  c.up[1] = 1;
  c.up[2] = 0;
  return c;
}

Camera
GetCamera(int frame, int nframes, double radius, float* lookat, float *bounds)
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
  /*
  float x = (bounds[0] + bounds[1])/2;
  float y = (bounds[2] + bounds[3])/2;
  float z = (bounds[4] + bounds[5])/2;
  */ 

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
class CalculateArea : public vtkm::worklet::WorkletVisitCellsWithPoints
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

  template <typename PointVecType, typename FieldType>
  VTKM_EXEC
  void operator()(const vtkm::IdComponent& numPoints,
                  const PointVecType& points,
                  FieldType& output) const
  {
    if(numPoints != 3)
    {
      this->RaiseError("We only play with triangles here");
      return;
    }
    // Since you only have triangles, numPoints should always be 3
    // PointType is an abstraction of vtkm::Vec3f, which is {x, y, z} of a point
    using PointType = typename PointVecType::ComponentType;
    // Following lines will help you extract points for a single triangle
    //PointType vertex0 = points[0]; // {x0, y0, z0}
    //PointType vertex1 = points[1]; // {x1, y1, z1}
    //PointType vertex2 = points[2]; // {x2, y2, z2}
    FieldType AC0 = points[1][0] - points[0][0];
    FieldType BC0 = points[2][0] - points[0][0];
    FieldType AC1 = points[1][1] - points[0][1];
    FieldType BC1 = points[2][1] - points[0][1];
    FieldType AC2 = points[1][2] - points[0][2];
    FieldType BC2 = points[2][2] - points[0][2];

    FieldType OV0 = AC1*BC2 - BC1*AC2;
    FieldType OV1 = AC0*BC2 - BC0*AC2;
    FieldType OV2 = AC0*BC1 - BC1*AC1;

    output = sqrt(pow(OV0, 2) +
              pow(OV1, 2) +
              pow(OV2, 2))/2.0;

  }
};


double
GetArea(vtkh::DataSet &vtkhData)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  double total_area = 0.0;
  std::vector<double> local_areas;
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
      std::vector<double> tmp_areas(numTris);
     
      vtkm::cont::ArrayHandle<double> areas = vtkm::cont::make_ArrayHandle(tmp_areas);
      vtkm::cont::Invoker invoker;
      invoker(CalculateArea{}, cellset, coords, areas);

      //combine all domain triangles
      local_areas.insert(local_areas.end(), tmp_areas.begin(), tmp_areas.end());
      for(std::vector<double>::iterator it = tmp_areas.begin(); it != tmp_areas.end(); it++)
        total_area += *it;
    }
  }
  return total_area;
}


class ProcessTriangle : public vtkm::worklet::WorkletVisitCellsWithPoints
{
public:
  // This is to tell the compiler what inputs to expect.
  // For now we'll be providing the CellSet, CooridnateSystem,
  // an input variable, and an output variable.
  using ControlSignature = void(CellSetIn cellset,
                                FieldInPoint points,
                                FieldOutCell triangle_output,
				FieldOutCell area_output);

  // After VTK-m does it's magic, you need to tell what information you need
  // from the provided inputs from the ControlSignature.
  // For now :
  // 1. number of points making an individual cell
  // 2. _2 is the 2nd input to ControlSignature : this should give you all points of the triangle.
  // 3. _3 is the 3rd input to ControlSignature : this should give you the variables at those points.
  // 4. _4 is the 4rd input to ControlSignature : this will help you store the output of your calculation.
  using ExecutionSignature = void(PointCount, _2, _3, _4);

  template <typename PointVecType, typename FieldType>
  VTKM_EXEC
  void operator()(const vtkm::IdComponent& numPoints,
                  const PointVecType& points,
                  Triangle& output,
		  FieldType& local_area) const
  {
    if(numPoints != 3) 
    {
      this->RaiseError("We only play with triangles here");
    }
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

    FieldType AC0 = points[1][0] - points[0][0];
    FieldType BC0 = points[2][0] - points[0][0];
    FieldType AC1 = points[1][1] - points[0][1];
    FieldType BC1 = points[2][1] - points[0][1];
    FieldType AC2 = points[1][2] - points[0][2];
    FieldType BC2 = points[2][2] - points[0][2];

    FieldType OV0 = AC1*BC2 - BC1*AC2;
    FieldType OV1 = AC0*BC2 - BC0*AC2;
    FieldType OV2 = AC0*BC1 - BC1*AC1;

    local_area = sqrt(pow(OV0, 2) +
              pow(OV1, 2) +
              pow(OV2, 2))/2.0;
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
    {
      this->RaiseError("We only play with triangles here");
    }
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

int 
GetBin(float x0, float y0, float z0, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, int xbins, int ybins, int zbins)
{
  bool print = false;
  if(print)
  {
  cerr << "xms: " << xmin << " - " << xmax << "\nyms: " << ymin << " - " << ymax << "\nzms: " << zmin << " - " << zmax << endl;
    cerr << "bins: " << xbins << " " << ybins << " " << zbins << endl;
    cerr << "diffx: " << (std::abs(x0) - xmin) << endl;
    cerr << "diffy: " << (std::abs(y0) - ymin) << endl;
    cerr << "diffz: " << (std::abs(z0) - zmin) << endl;
    cerr << "x: " << x0 << " y: " << y0 << " z0: " << z0 << endl;
  }
  double xStep = (xmax - xmin)/xbins;
  double yStep = (ymax - ymin)/ybins;
  double zStep = (zmax - zmin)/zbins;
  if(print)
    cerr << "xstep: " << xStep << " ystep: " << yStep << " zstep: " << zStep << endl;

  int idx = (int)(std::abs(x0 - xmin)/xStep); 
  if(idx == xbins)
    idx--;
  int idy = (int)(std::abs(y0 - ymin)/yStep); 
  if(idy == ybins)
    idy--;
  int idz = (int)(std::abs(z0 - zmin)/zStep); 
  if(idz == zbins)
    idz--;

  int id = (int)(idx + xbins*(idy + ybins*idz));
  if(print)
  {
    cerr << "idx: " << idx << " idy: " << idy << " idz: " << idz << endl;
    cerr << "id: " << id << endl;
  }
  return id;
}

vtkh::DataSet*
AddTriangleFields(vtkh::DataSet &vtkhData, float &xmin, float &xmax, float &ymin, float &ymax, float &zmin, float &zmax, int xBins, int yBins, int zBins)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  vtkh::DataSet *newDataSet = new vtkh::DataSet;
  //if there is data: loop through domains and grab all triangles.
  if (!vtkhData.IsEmpty())
  {
    for (int i = 0; i < localDomainIds.size(); i++)
    {
      vtkm::cont::DataSet dataset = vtkhData.GetDomain(i);
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      vtkm::cont::DynamicCellSet cellset = dataset.GetCellSet();
    
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> X0;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> Y0;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> Z0;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> X1;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> Y1;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> Z1;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> X2;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> Y2;
      vtkm::cont::ArrayHandle<vtkm::FloatDefault> Z2;
      vtkm::cont::Invoker invoker;
      invoker(GetTriangleFields{}, cellset, coords, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2);

      vtkm::cont::ArrayHandle<double> point_bin;
      if (X0.GetNumberOfValues() > 0)
      {
        int size = X0.GetNumberOfValues();
        point_bin.Allocate(size);
        auto binPortal = point_bin.WritePortal();
        auto x0 = X0.ReadPortal();
        auto y0 = Y0.ReadPortal();
        auto z0 = Z0.ReadPortal();
        for (int i = 0; i < size; i++)
        {
          auto bin =(double)GetBin(x0.Get(i), y0.Get(i), z0.Get(i), xmin, xmax, ymin, ymax, zmin, zmax, xBins, yBins, zBins);
          binPortal.Set(i, bin);
        }
      }
      dataset.AddCellField("X0", X0);
      dataset.AddCellField("Y0", Y0);
      dataset.AddCellField("Z0", Z0);
      dataset.AddCellField("X1", X1);
      dataset.AddCellField("Y1", Y1);
      dataset.AddCellField("Z1", Z1);
      dataset.AddCellField("X2", X2);
      dataset.AddCellField("Y2", Y2);
      dataset.AddCellField("Z2", Z2);
      dataset.AddCellField("Bin", point_bin);
      newDataSet->AddDomain(dataset, localDomainIds[i]);
    }
  }
  return newDataSet;
}

// TODO: Manish
// `tris` and `local_areas` can be converted into ArrayHandles
std::vector<Triangle>
GetTrianglesAndArea(vtkh::DataSet &vtkhData, double &area)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<Triangle> tris;
  std::vector<double> local_areas;
  double total_area = 0.0;
  //if there is data: loop through domains and grab all triangles.
  if (!vtkhData.IsEmpty())
  {
    for (int i = 0; i < localDomainIds.size(); i++)
    {
      vtkm::cont::DataSet& dataset = vtkhData.GetDomain(i);
      vtkm::cont::CoordinateSystem coords = dataset.GetCoordinateSystem();
      vtkm::cont::DynamicCellSet cellset = dataset.GetCellSet();
      vtkm::cont::ArrayHandle<Triangle> triangles;
      vtkm::cont::ArrayHandle<double> areas;
      vtkm::cont::Invoker invoker;
      invoker(ProcessTriangle{}, cellset, coords, triangles, areas);

      //combine all domain triangles
      // copyWithOffset here?
      auto trianglesPortal = triangles.ReadPortal();
      for (auto i = 0; i < trianglesPortal.GetNumberOfValues(); ++i)
      {
        tris.push_back(trianglesPortal.Get(i));
      }

      auto areasPortal = areas.ReadPortal();
      for (auto i = 0; i < areasPortal.GetNumberOfValues(); ++i)
      {
        total_area += areasPortal.Get(i);
      }
    }
    area = total_area;
  }

  return tris;
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
      std::vector<double> tmp_areas(numTris);

      vtkm::cont::ArrayHandle<Triangle> triangles = vtkm::cont::make_ArrayHandle(tmp_tris);
      vtkm::cont::ArrayHandle<double> areas = vtkm::cont::make_ArrayHandle(tmp_areas);

      vtkm::cont::Invoker invoker;
      invoker(ProcessTriangle{}, cellset, coords, triangles, areas);

      //combine all domain triangles
      tris.insert(tris.end(), tmp_tris.begin(), tmp_tris.end());
    }
  }
  return tris;
}

template< typename T >
std::vector<T>
GetScalarData(vtkh::DataSet &vtkhData, std::string field_name, int height, int width)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<T> data;
     
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
      
      long int size = field.GetNumberOfValues();
      
      vtkm::cont::ArrayHandle<T> field_data;
      //using Type = vtkm::cont::ArrayHandle<vtkm::FloatDefault>();
      //if(field.GetData().IsType<Type>())
      //  cerr << "THEY ARE A FLOAT" << endl;
      //else
      //  cerr << "THE DATA IS NOT FLOAT" << endl;      
      field.GetData().CopyTo(field_data);
      auto portal = field_data.GetPortalConstControl();

      for(int i = 0; i < size; i++)
      {
        data.push_back(portal.Get(i));
      }
      
    }
  }
  return data;
}

template <typename T>
std::vector<T>
GetScalarData(vtkh::DataSet &vtkhData, const char *field_name, int height, int width)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  std::vector<T> data;
     
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
      
      long int size = field.GetNumberOfValues();
      
      using data_d = vtkm::cont::ArrayHandle<vtkm::Float64>;
      using data_f = vtkm::cont::ArrayHandle<vtkm::Float32>;
      if(field.GetData().IsType<data_d>())
      {
        vtkm::cont::ArrayHandle<vtkm::Float64> field_data;
        field.GetData().CopyTo(field_data);
        auto portal = field_data.GetPortalConstControl();

        for(int i = 0; i < size; i++)
        {
          data.push_back(portal.Get(i));
        }
      }
      if(field.GetData().IsType<data_f>())
      {
        vtkm::cont::ArrayHandle<vtkm::Float32> field_data;
        field.GetData().CopyTo(field_data);
        auto portal = field_data.GetPortalConstControl();

        for(int i = 0; i < size; i++)
        {
          data.push_back(portal.Get(i));
        }
      }
    }
  }
  //else
    //cerr << "VTKH Data is empty" << endl;
  return data;
}

template <typename FloatType>
class CopyWithOffset : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn src, WholeArrayInOut dest);
  using ExecutionSignature = void(InputIndex, _1, _2);

  VTKM_CONT
  CopyWithOffset(const vtkm::Id offset = 0)
      : Offset(offset)
  {
  }
  template <typename OutArrayType>
  VTKM_EXEC inline void operator()(const vtkm::Id idx, const FloatType &srcValue, OutArrayType &destArray) const
  {
    destArray.Set(idx + this->Offset, srcValue);
  }

private:
  vtkm::Id Offset;
};

template <typename T>
struct MaxValueWithChecks
{
  MaxValueWithChecks(T minValid, T maxValid)
      : MinValid(minValid),
        MaxValid(maxValid)
  {
  }

  VTKM_EXEC_CONT inline T operator()(const T &a, const T &b) const
  {
    if (this->IsValid(a) && this->IsValid(b))
    {
      return (a > b) ? a : b;
    }
    else if (!this->IsValid(a))
    {
      return b;
    }
    else if (!this->IsValid(b))
    {
      return a;
    }
    else
    {
      return this->MinValid;
    }
  }

  VTKM_EXEC_CONT inline bool IsValid(const T &t) const
  {
    return !vtkm::IsNan(t) && t > MinValid && t < MaxValid;
  }

  T MinValid;
  T MaxValid;
};

template <typename SrcType, typename DestType>
void copyArrayWithOffset(const vtkm::cont::ArrayHandle<SrcType> &src, vtkm::cont::ArrayHandle<DestType> &dest, vtkm::Id offset)
{
  vtkm::cont::Invoker invoker;
  invoker(CopyWithOffset<SrcType>(offset), src, dest);
}

enum DataCheckFlags
{
  CheckNan          = 1 << 0,
  CheckZero         = 1 << 1,
  CheckMinExclusive = 1 << 2,
  CheckMaxExclusive = 1 << 3,
};

template<typename T>
struct DataCheckVals
{
  T Min;
  T Max;
};

inline DataCheckFlags operator|(DataCheckFlags lhs, DataCheckFlags rhs)
{
  return static_cast<DataCheckFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

template <typename FloatType>
struct CopyWithChecksMask : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn src, FieldOut dest);
  using ExecutionSignature = void(_1, _2);

  VTKM_CONT
  CopyWithChecksMask(DataCheckFlags checks, DataCheckVals<FloatType> checkVals)
      : Checks(checks),
        CheckVals(checkVals)
  {
  }

  VTKM_EXEC inline void operator()(const FloatType &val, vtkm::IdComponent& mask) const
  {
    bool passed = true;
    if(this->HasCheck(CheckNan))
    {
      passed = passed && !vtkm::IsNan(val);   
    }
    if(this->HasCheck(CheckZero)) 
    {
      passed = passed && (val != FloatType(0));
    }
    if(this->HasCheck(CheckMinExclusive))
    {
      passed = passed && (val > this->CheckVals.Min);
    }
    if(this->HasCheck(CheckMaxExclusive))
    {
      passed = passed && (val < this->CheckVals.Max);
    }

    mask = passed ? 1 : 0;
  }
  
  VTKM_EXEC inline bool HasCheck(DataCheckFlags check) const
  {
    return (Checks & check) == check;
  }

  DataCheckFlags Checks;
  DataCheckVals<FloatType> CheckVals;
};

template<typename SrcType>
vtkm::cont::ArrayHandle<SrcType> copyWithChecks(
  const vtkm::cont::ArrayHandle<SrcType>& src, 
  DataCheckFlags checks, 
  DataCheckVals<SrcType> checkVals = DataCheckVals<SrcType>{})
{
  vtkm::cont::ArrayHandle<vtkm::IdComponent> mask;
  vtkm::cont::Invoker invoker;
  invoker(CopyWithChecksMask<SrcType>(checks, checkVals), src, mask);
  
  vtkm::cont::ArrayHandle<SrcType> dest;
  vtkm::cont::Algorithm::CopyIf(src, mask, dest);
  return dest;
}

template <typename T>
vtkm::cont::ArrayHandle<T>
GetScalarDataAsArrayHandle(vtkh::DataSet &vtkhData, std::string field_name)
{
  //Get domain Ids on this rank
  //will be nonzero even if there is no data
  std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
  vtkm::cont::ArrayHandle<T> totalFieldData;

  if (!vtkhData.IsEmpty())
  {
    // Loop once to get the total number of items and reserve the vector
    vtkm::Id totalNumberOfValues = std::accumulate(
        localDomainIds.begin(),
        localDomainIds.end(),
        0,
        [&](const vtkm::Id &acc, const vtkm::Id domainId)
        {
          const vtkm::cont::DataSet &dataset = vtkhData.GetDomain(domainId);
          const vtkm::cont::Field &field = dataset.GetField(field_name);

          return acc + field.GetData().GetNumberOfValues();
        });

    totalFieldData.Allocate(totalNumberOfValues);
    vtkm::Id offset = 0;
    for (auto &domainId : localDomainIds)
    {
      const vtkm::cont::DataSet &dataset = vtkhData.GetDomain(domainId);
      const vtkm::cont::Field &field = dataset.GetField(field_name);
      const auto fieldData = field.GetData().AsArrayHandle<vtkm::cont::ArrayHandle<T>>();
      copyArrayWithOffset(fieldData, totalFieldData, offset);
      offset += fieldData.GetNumberOfValues();
    }
  }

  return totalFieldData;
}

template <typename FloatType>
struct TriangleCreator : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(
    FieldIn x0,
    FieldIn y0,
    FieldIn z0,
    FieldIn x1,
    FieldIn y1,
    FieldIn z1,
    FieldIn x2,
    FieldIn y2,
    FieldIn z2,
    FieldOut triangle,
    FieldOut stencil);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);

  VTKM_CONT
  TriangleCreator()
  {  }

  VTKM_EXEC void operator()(
    const FloatType& x0,
    const FloatType& y0,
    const FloatType& z0,
    const FloatType& x1,
    const FloatType& y1,
    const FloatType& z1,
    const FloatType& x2,
    const FloatType& y2,
    const FloatType& z2,
    Triangle& triangle,
    vtkm::IdComponent& stencil) const
  {
    stencil = !vtkm::IsNan(x0);
    triangle.X[0] = x0;
    triangle.Y[0] = y0;
    triangle.Z[0] = z0;
    triangle.X[1] = x1;
    triangle.Y[1] = y1;
    triangle.Z[1] = z1;
    triangle.X[2] = x2;
    triangle.Y[2] = y2;
    triangle.Z[2] = z2;
  }
};

struct TriangleSortLess
{
  VTKM_EXEC_CONT bool operator()(const Triangle& t1, const Triangle& t2) const
  {
    for(vtkm::IdComponent i = 0; i < 3; ++i)
    {
      if(t1.X[i] < t2.X[i])
        return true;
      if(t1.X[i] > t2.X[i])
        return false;
      if(t1.Y[i] < t2.Y[i])
        return true;
      if(t1.Y[i] > t2.Y[i])
        return false;
      if(t1.Z[i] < t2.Z[i])
        return true;
      if(t1.Z[i] > t2.Z[i])
        return false;
    }

    return false;
  }
};

struct TriangleComparator
{
  VTKM_EXEC bool operator()(const Triangle& t1, const Triangle& t2) const
  {
    bool result =      this->Compare(t1.X, t2.X);
    result = result && this->Compare(t1.Y, t2.Y);
    result = result && this->Compare(t1.Z, t2.Z);
    return result;
  }

  VTKM_EXEC bool Compare(const float *v1, const float *v2) const
  {
    for(vtkm::IdComponent i = 0; i < 3; ++i) 
    {
      if (v1[i] != v2[i])
      {
        return false;
      }
    }
    return true;
  }
};

vtkm::cont::ArrayHandle<Triangle>
GetUniqueTriangles(vtkh::DataSet *dataset)
{
  auto x0 = GetScalarDataAsArrayHandle<float>(*dataset, "X0");
  auto y0 = GetScalarDataAsArrayHandle<float>(*dataset, "Y0");
  auto z0 = GetScalarDataAsArrayHandle<float>(*dataset, "Z0");
  auto x1 = GetScalarDataAsArrayHandle<float>(*dataset, "X1");
  auto y1 = GetScalarDataAsArrayHandle<float>(*dataset, "Y1");
  auto z1 = GetScalarDataAsArrayHandle<float>(*dataset, "Z1");
  auto x2 = GetScalarDataAsArrayHandle<float>(*dataset, "X2");
  auto y2 = GetScalarDataAsArrayHandle<float>(*dataset, "Y2");
  auto z2 = GetScalarDataAsArrayHandle<float>(*dataset, "Z2");

  vtkm::cont::ArrayHandle<Triangle> allTriangles;
  vtkm::cont::ArrayHandle<vtkm::IdComponent> trianglesStencil;
  vtkm::cont::ArrayHandle<Triangle> triangles;
  if (x0.GetNumberOfValues() > 0) 
  {
    vtkm::cont::Invoker invoker;
    invoker(
      TriangleCreator<float>{}, 
      x0, 
      y0, 
      z0, 
      x1, 
      y1, 
      z1, 
      x2, 
      y2, 
      z2, 
      allTriangles, 
      trianglesStencil);

    vtkm::cont::Algorithm::CopyIf(allTriangles, trianglesStencil, triangles);
    vtkm::cont::Algorithm::Sort(triangles, TriangleSortLess());
    vtkm::cont::Algorithm::Unique(triangles, TriangleComparator());
  }

  return triangles;
}

struct TriangleAreaCalculator : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn triangles, FieldOut areas);
  using ExecutionSignature = void(_1, _2);

  VTKM_CONT
  TriangleAreaCalculator()
    : UseTransform(false)
  {}

  VTKM_CONT
  TriangleAreaCalculator(Camera camera, int width, int height, bool cutoff)
    : UseTransform(true),
      camera(camera),
      Width(width),
      Height(height),
      Cutoff(cutoff)
  {}

  VTKM_EXEC inline void operator()(const Triangle& triangle, float& area) const
  {
    if (this->UseTransform) 
    {
      Triangle transformedTriangle = transformTriangle(triangle, this->camera, this->Width, this->Height);
      if (this->Cutoff) 
      {
        transformedTriangle.cutoff(this->Width, this->Height);
      }
      area = transformedTriangle.calculateTriArea();
    }
    else
    {
      area = triangle.calculateTriArea();
    }
    area = vtkm::IsNan(area) ? 0.0 : area;
  }

  bool UseTransform;
  Camera camera;
  vtkm::Id Width;
  vtkm::Id Height;
  bool Cutoff;
};

vtkm::cont::ArrayHandle<float>
CalculateProjectedTriangleAreas(const vtkm::cont::ArrayHandle<Triangle>& triangles, Camera camera, int width, int height, bool cutoff)
{
  vtkm::cont::ArrayHandle<float> areas;
  vtkm::cont::Invoker invoker;
  invoker(TriangleAreaCalculator{camera, width, height, cutoff}, triangles, areas);
  return areas;
}

vtkm::cont::ArrayHandle<float>
CalculateTriangleAreas(const vtkm::cont::ArrayHandle<Triangle>& triangles)
{
  vtkm::cont::ArrayHandle<float> areas;
  vtkm::cont::Invoker invoker;
  invoker(TriangleAreaCalculator{}, triangles, areas);
  return areas;
}
#endif

EXEC_CONT
Triangle transformTriangle(const Triangle& t, const Camera& c, int width, int height) 
{
  // bool print = true;
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

  /*
  if(print)
  {
    cerr << "triangle out: (" << triangle.X[0] << " , " << triangle.Y[0] << " , " << triangle.Z[0] << ") " << endl <<
                         " (" << triangle.X[1] << " , " << triangle.Y[1] << " , " << triangle.Z[1] << ") " << endl <<
                         " (" << triangle.X[2] << " , " << triangle.Y[2] << " , " << triangle.Z[2] << ") " << endl;
  }
  */

  return triangle;

}

void 
TriangleBounds(std::vector<Triangle> triangles, float &xmin, float &xmax, float &ymin, float &ymax, float &zmin, float &zmax)
{
  float xMin = FLT_MAX, yMin = FLT_MAX, zMin = FLT_MAX;
  float xMax = -FLT_MAX, yMax = -FLT_MAX, zMax = -FLT_MAX;  
 
  int num_tri = triangles.size();
  for(int i = 0; i < num_tri; i++)
  {
    float tmp_xmin = findMin(triangles[i].X[0], triangles[i].X[1], triangles[i].X[2]); 
    float tmp_ymin = findMin(triangles[i].Y[0], triangles[i].Y[1], triangles[i].Y[2]); 
    float tmp_zmin = findMin(triangles[i].Z[0], triangles[i].Z[1], triangles[i].Z[2]); 
    float tmp_xmax = findMax(triangles[i].X[0], triangles[i].X[1], triangles[i].X[2]); 
    float tmp_ymax = findMax(triangles[i].Y[0], triangles[i].Y[1], triangles[i].Y[2]); 
    float tmp_zmax = findMax(triangles[i].Z[0], triangles[i].Z[1], triangles[i].Z[2]); 
    if(xMin > tmp_xmin)
      xMin = tmp_xmin;  
    if(yMin > tmp_ymin)
      yMin = tmp_ymin;  
    if(zMin > tmp_zmin)
      zMin = tmp_zmin;  
    if(xMax < tmp_xmax)
      xMax = tmp_xmax;
    if(yMax < tmp_ymax)
      yMax = tmp_ymax;
    if(zMax < tmp_zmax)
      zMax = tmp_zmax;
  }
  xmax = xMax;
  xmin = xMin;
  ymax = yMax;
  ymin = yMin;
  zmax = zMax;
  zmin = zMin;
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

std::vector<float>
CalculateCentroid(std::vector<float> tri)
{
  std::vector<float> centroid(3,0.0);
  centroid[0] = (tri[0] + tri[3] + tri[6])/3;
  centroid[1] = (tri[1] + tri[4] + tri[7])/3;
  centroid[2] = (tri[2] + tri[5] + tri[8])/3;

  return centroid;
}

float
calcShading(float* viewDirection,float* lightDir, float* normal)
{
  float Ka = 0.3;
  float Kd = 0.7;
  float Ks = 0;
  float alpha = 7.5;
  bool flag;
  flag = false;
  float diffuse = 0, specular = 0, shading = 0;
  diffuse = nabs(dotProduct(lightDir, normal, 3));
  if(diffuse > 1)
    diffuse = 1;
  float Rtemp = 2*(dotProduct(lightDir, normal, 3));
  float R[] = {Rtemp*normal[0] - lightDir[0], Rtemp*normal[1] - lightDir[1], Rtemp*normal[2] - lightDir[2]};
  float dot = (dotProduct(R, viewDirection, 3));
  if (dot < 0)
    flag = true;
  specular = pow(nabs(dotProduct(R, viewDirection, 3)), alpha);
  if (flag)
  {
    specular = -specular;
    flag = false;
  }
  if (specular < 0)
    specular = 0;
  shading = Kd*diffuse + Ka + Ks*specular;
  if(shading != shading)
    shading = 0;
  if(shading > 1)
    shading = 1;

  return shading;
}

void
CalculateNormal(std::vector<float> tri, float normal[3])
{
  bool print = false;
  if(print)
    cerr << "normal before: " << normal[0] << " " << normal[1] << " " << normal[2] << endl;
  //(C-A)x(B-A)
  float ca[3];
  //z[0] - x[0]
  //z[1] - x[1]
  //z[2] - x[2]
 /* ca[0] = tri[2] - tri[0];
  ca[1] = tri[5] - tri[3];
  ca[2] = tri[8] - tri[6];
*/
  ca[0] = tri[6] - tri[0];
  ca[1] = tri[7] - tri[1];
  ca[2] = tri[8] - tri[2];
  float ba[3];
  ba[0] = tri[3] - tri[0];
  ba[1] = tri[4] - tri[1];
  ba[2] = tri[5] - tri[2];
  //y[0] - x[0]
  //y[1] - x[1]
  //y[2] - x[2]
/*  ba[0] = tri[1] - tri[0];
  ba[1] = tri[4] - tri[3];
  ba[2] = tri[7] - tri[6];
*/
  crossProduct(ba, ca, normal);
  if(print)
  {
    cerr << "Triangle: " << tri[0] << " " << tri[1] << " " << tri[2] << endl;
    cerr << tri[3] << " " << tri[4] << " " << tri[5] << endl;
    cerr << tri[6] << " " << tri[7] << " " << tri[8] << endl;
    cerr << "normal after: " << normal[0] << " " << normal[1] << " " << normal[2] << endl;
    cerr << endl;

  }
}

std::vector<float>
CalculateFlatShading(std::vector<std::vector<float>> triangles, Camera cam)
{
  int num_tri = triangles.size();
  std::vector<float> shadings(num_tri,0.0);
  for(int i = 0; i < num_tri; i++)
  {
    //calculate each triangle's normal
    float normal[3];
    CalculateNormal(triangles[i], normal);
    normalize(normal);
    //calculate centroid of each triangle, P
    std::vector<float> centroid = CalculateCentroid(triangles[i]);
    float lightDir[3];
    lightDir[0] = cam.position[0] - centroid[0];
    lightDir[1] = cam.position[1] - centroid[1];
    lightDir[2] = cam.position[2] - centroid[2];
    normalize(lightDir);
    float viewDirection[3];
    viewDirection[0] = cam.position[0] - cam.focus[0];
    viewDirection[1] = cam.position[1] - cam.focus[1];
    viewDirection[2] = cam.position[2] - cam.focus[2];
    normalize(viewDirection);
    float shade = calcShading(viewDirection, lightDir, normal);
    if(shade != shade)
	    cerr << "shade " << shade << endl;
    shadings[i] = shade;
  }
  return shadings;
}

#if defined(ASCENT_VTKM_ENABLED)
struct FlatShadingCalculator : public vtkm::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn triangles, FieldOut shadings);
  using ExecutionSignature = void(_1, _2);
  using Vec3f = vtkm::Vec<vtkm::Float32, 3>;

  VTKM_CONT
  FlatShadingCalculator(Camera camera)
      : CamPos(camera.position[0], camera.position[1], camera.position[2]),
        CamFocus(camera.focus[0], camera.focus[1], camera.focus[2])
  {
  }

  VTKM_EXEC inline void operator()(const Triangle& triangle, float& shading) const
  {
    Vec3f a = Vec3f(triangle.X[0], triangle.Y[0], triangle.Z[0]);
    Vec3f b = Vec3f(triangle.X[1], triangle.Y[1], triangle.Z[1]);
    Vec3f c = Vec3f(triangle.X[2], triangle.Y[2], triangle.Z[2]);
    Vec3f normal = vtkm::Normal(vtkm::TriangleNormal(a, b, c));

    float centroidX = (a[0] + b[0]+ c[0]) / 3.0f;
    float centroidY = (a[1] + b[1]+ c[1]) / 3.0f;
    float centroidZ = (a[2] + b[2]+ c[2]) / 3.0f;
    
    Vec3f lightDir = vtkm::Normal(this->CamPos - Vec3f(centroidX, centroidY, centroidZ));

    Vec3f viewDir = vtkm::Normal(this->CamPos - this->CamFocus);
    shading = this->CalcShading(viewDir, lightDir, normal);
  }

  VTKM_EXEC inline float CalcShading(const Vec3f& viewDir, const Vec3f& lightDir, const Vec3f& normal) const
  {
    float Ka = 0.3;
    float Kd = 0.7;
    float Ks = 0;
    float alpha = 7.5;
    bool flag;
    flag = false;
    float diffuse = 0, specular = 0, shading = 0;
    diffuse = vtkm::Abs(vtkm::Dot(lightDir, normal));
    if (diffuse > 1)
      diffuse = 1;
    float Rtemp = 2 * (vtkm::Dot(lightDir, normal));
    Vec3f R {
      Rtemp * normal[0] - lightDir[0], 
      Rtemp * normal[1] - lightDir[1], 
      Rtemp * normal[2] - lightDir[2]
    };
    float dot = (vtkm::Dot(R, viewDir));
    if (dot < 0)
      flag = true;
    specular = vtkm::Pow(vtkm::Abs(vtkm::Dot(R, viewDir)), alpha);
    if (flag)
    {
      specular = -specular;
      flag = false;
    }
    if (specular < 0)
      specular = 0;
    shading = Kd * diffuse + Ka + Ks * specular;
    if (shading != shading)
      shading = 0;
    if (shading > 1)
      shading = 1;

    return shading;
  }

  Vec3f CamPos;
  Vec3f CamFocus; 
};

vtkm::cont::ArrayHandle<float>
CalculateFlatShading(const vtkm::cont::ArrayHandle<Triangle>& triangles, Camera camera)
{
  vtkm::cont::ArrayHandle<float> shadings;
  vtkm::cont::Invoker invoker;
  invoker(FlatShadingCalculator{camera}, triangles, shadings);
  return shadings;
}
#endif

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
T calcentropyMM( const T* array, long len, int nBins , T field_max, T field_min)
{
  T max = field_max;
  T min = field_min;

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

template< typename T >
T calcentropy( const T* array, long len, int nBins)
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

#if defined(ASCENT_VTKM_ENABLED)
template <typename T>
struct CalculateEntropy
{
  inline VTKM_EXEC_CONT T operator()(const T& numerator, const T& denominator) const
  {
    const T prob = numerator / denominator;
    if (prob == T(0))
    {
      return T(0);
    }
    return prob * vtkm::Log(prob);
  }
};

template <typename T>
struct CalculateVKLEntropy
{
  VTKM_CONT CalculateVKLEntropy(T projected_total_area, T world_total_area)
    : ProjectedTotalArea(projected_total_area),
      WorldTotalArea(world_total_area)
  {}

  inline VTKM_EXEC_CONT T operator()(const T& area, const T& w_area) const
  {
    if(area != 0.0f && w_area != 0.0f)
    {
      T left_term = area / this->ProjectedTotalArea;
      T divisor = w_area / this->WorldTotalArea;
      return left_term * vtkm::Log(left_term / divisor);
    }

    return T(0);
  }

  T ProjectedTotalArea;
  T WorldTotalArea;
};

template <typename T>
T calcentropyMM(const vtkm::cont::ArrayHandle<T>& data, int nBins, T max, T min)
{
  vtkm::worklet::FieldHistogram worklet;
  vtkm::cont::ArrayHandle<vtkm::Id> hist;
  T stepSize;
  worklet.Run(data, nBins, min, max, stepSize, hist);

  auto len = vtkm::cont::make_ArrayHandleConstant(
    static_cast<T>(data.GetNumberOfValues()), 
    hist.GetNumberOfValues());
  vtkm::cont::ArrayHandle<T> subEntropies;
  vtkm::cont::Algorithm::Transform(hist, len, subEntropies, CalculateEntropy<T>{});

  T entropy = vtkm::cont::Algorithm::Reduce(subEntropies, T(0));

  return (entropy * -1.0);
}
#endif

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
  cerr << "d_tri_cutoff: " << endl;
  d_tri.printTri();
*/
  d_tri.cutoff(width, height);

  return d_tri.calculateTriArea();

}

#if defined(ASCENT_VTKM_ENABLED)


float
calculateVisibilityRatio(vtkh::DataSet* dataset, std::vector<Triangle> &local_triangles, float worldspace_local_area, int height, int width)
{
  float visibility_ratio = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_local_triangles = local_triangles.size();
    float global_area       = 0.0;
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();

    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD); 
    if(rank == 0)
      MakeFile("visibilityratio_phase1_metric_times.txt", array, world_size);

    MPI_Reduce(&worldspace_local_area, &global_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    timer.Start();
    
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto triangles = GetUniqueTriangles(dataset);
      if(triangles.GetNumberOfValues() > 0) 
      {
        auto triangle_areas = CalculateTriangleAreas(triangles);
        float triangle_area = vtkm::cont::Algorithm::Reduce(triangle_areas, 0.0f);
        visibility_ratio = triangle_area / global_area;
      }
      #else
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)	      
//      #endif
      if(x0.size())
      {
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
        #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for reduction(+:projected_area)
        #endif
        for(int i = 0; i < num_triangles; i++)
        {
          float area = calcArea(triangles[i]);
          projected_area += area;
        }
        visibility_ratio = projected_area/global_area;
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    total_time = timer.GetElapsedTime();
    MPI_Bcast(&visibility_ratio, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array1[world_size] = {0};
    array1[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array1, 1, MPI_DOUBLE, MPI_COMM_WORLD); 
    if(rank == 0)
      MakeFile("visibilityratio_phase2_metric_times.txt", array1, world_size);
    //cerr << "visibility_ratio " << visibility_ratio << endl;
    
    return visibility_ratio;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//    #ifdef ASCENT_USE_OPENMP
//    #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//    #pragma omp parallel for reduction(merge:triangles)
//    #endif
    if(x0.size())
    {
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
    
      #ifdef ASCENT_USE_OPENMP
      #pragma omp parallel for reduction(+:projected_area)
      #endif
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i]);
        projected_area += area;
      }
      visibility_ratio = projected_area/worldspace_local_area;
    }
    return visibility_ratio;
  #endif
}

int COUNT;

float
calculateShadingEntropy(vtkh::DataSet* dataset, int height, int width, Camera camera)
{
  float shading_entropy = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;

    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto triangles = GetUniqueTriangles(dataset);
      if (triangles.GetNumberOfValues() > 0) 
      {
        auto shadings = CalculateFlatShading(triangles, camera);
        shading_entropy = calcentropyMM(shadings, 100, 1.0f, 0.0f);
      }
      #else
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
      if(x0.size())
      {
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
        //calculate flat shading
        std::vector<float> shadings = CalculateFlatShading(triangles, camera);
        int shadings_size = shadings.size();
        float shadings_arr[shadings_size];
        std::copy(shadings.begin(), shadings.end(), shadings_arr);
        shading_entropy = calcentropyMM(shadings_arr, num_triangles, 100, (float)1, (float)0);
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&shading_entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("shadingentropy_metric_times.txt", array, world_size);
//    cerr << "viewpoint_entropy " << viewpoint_entropy << endl;    
    
    return shading_entropy;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
    cerr << "X0 SIZE: " << x0.size() << endl;
    if(x0.size())
    {
      for(int i = 0; i < size; i++)
      {
        if(x0[i] == x0[i]) //!nan
        {
          std::vector<float> tri{x0[i],y0[i],z0[i],x1[i],y1[i],z1[i],x2[i],y2[i],z2[i]};
          triangles.push_back(tri);
        }
      }
      int num_triangles = triangles.size();
      //calculate flat shading
      std::vector<float> shadings = CalculateFlatShading(triangles, camera);
      long shadings_size = shadings.size();
      //float* shadings_arr = &shadings[0]; //point shadings arr at shadings vec
      float shadings_arr[shadings_size];
      std::copy(shadings.begin(), shadings.end(), shadings_arr);
      shading_entropy = calcentropyMM(shadings.data(), num_triangles, 100, (float)1, (float)0);
      //Makefile
      
      std::string file = "shading_data_dump" + std::to_string(COUNT) + ".txt";
      //MakeFile(file, shadings.data(), shadings.size());
    }

    return shading_entropy;
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    float global_area       = 0.0;
    float local_area        = 0.0;
    #if defined(ASCENT_VTKM_ENABLED)
    auto triangles_AH = vtkm::cont::make_ArrayHandle(local_triangles, vtkm::CopyFlag::Off);
    auto projected_areas = CalculateProjectedTriangleAreas(triangles_AH, camera, width, height, false);
    local_area = vtkm::cont::Algorithm::Reduce(projected_areas, 0.0f);
    #else
    int num_local_triangles = local_triangles.size();
    local_area = 0.0f;
    #ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:local_area)
    #endif
    std::vector<float> areas;
    for(int i = 0; i < num_local_triangles; i++)
    {
      Triangle t = transformTriangle(local_triangles[i], camera, width, height); 
      float area = t.calculateTriArea();
      areas.push_back(area);
      local_area += area;
    }
    #endif
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("viewpointentropy_phase1_metric_times.txt", array, world_size);


    MPI_Reduce(&local_area, &global_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    timer.Start();
    
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto uniqueTriangles = GetUniqueTriangles(dataset);
      if (uniqueTriangles.GetNumberOfValues() > 0) 
      {
        auto projected_areas = CalculateProjectedTriangleAreas(uniqueTriangles, camera, width, height, true);
        vtkm::cont::ArrayHandle<float> sub_entropies;
        vtkm::cont::Algorithm::Transform(
          projected_areas, 
          vtkm::cont::make_ArrayHandleConstant(global_area, projected_areas.GetNumberOfValues()),
          sub_entropies, 
          CalculateEntropy<float>{});
        viewpoint_entropy = -1.0f * vtkm::cont::Algorithm::Reduce(sub_entropies, 0.0f);
      }
      #else
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
      if(x0.size())
      {
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
        #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for reduction(+:viewpoint_ratio)
        #endif
        for(int i = 0; i < num_triangles; i++)
        {
          float area = calcArea(triangles[i], camera, width, height);
	        if(area != 0.0)
            viewpoint_ratio += ((area/global_area)*std::log(area/global_area));
        }
        viewpoint_entropy = (-1.0)*viewpoint_ratio;
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    total_time = timer.GetElapsedTime();
    MPI_Bcast(&viewpoint_entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array1[world_size] = {0};
    array1[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array1, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("viewpointentropy_phase2_metric_times.txt", array1, world_size);
//    cerr << "viewpoint_entropy " << viewpoint_entropy << endl;


    return viewpoint_entropy;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
    if(x0.size())
    {
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
      #ifdef ASCENT_USE_OPENMP
      #pragma omp parallel for reduction(+:total_area)
      #endif
      for(int i = 0; i < num_local_triangles; i++)
      {
        Triangle t = transformTriangle(local_triangles[i], camera, width, height);
        float area = t.calculateTriArea();
        total_area += area;
      }

      #ifdef ASCENT_USE_OPENMP
      #pragma omp parallel for reduction(+:viewpoint_ratio)
      #endif
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i], camera, width, height);

        if(area != 0.0)
          viewpoint_ratio += ((area/total_area)*std::log(area/total_area));
      }

      viewpoint_entropy = (-1.0)*viewpoint_ratio;
    }

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
    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now(); 
    vtkm::cont::Timer timer;
    timer.Start();

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      #ifdef ASCENT_USE_OPENMP
      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
      #pragma omp parallel for reduction(merge:triangles)
      #endif
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

    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      #ifdef ASCENT_USE_OPENMP
      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
      #pragma omp parallel for reduction(merge:triangles)
      #endif
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
calculateVKL(vtkh::DataSet* dataset, std::vector<Triangle> &local_triangles, float worldspace_local_area, int height, int width, Camera camera)
{
  float vkl = FLT_MAX;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

      // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    //Needs total world area and total image space area
    int num_local_triangles = local_triangles.size();
    long double total_area     = 0.0;
    long double local_area     = 0.0;
    long double w_total_area   = 0.0;
    long double w_local_area   = worldspace_local_area;
    #if defined(ASCENT_VTKM_ENABLED)

    auto triangles_AH = vtkm::cont::make_ArrayHandle(local_triangles, vtkm::CopyFlag::Off);
    auto projected_areas = CalculateProjectedTriangleAreas(triangles_AH, camera, width, height, false);
    local_area = vtkm::cont::Algorithm::Reduce(projected_areas, 0.0f);

    #else

    #ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:local_area)
    #endif
    local_area = 0.0;
    for(int i = 0; i < num_local_triangles; i++)
    {
      Triangle t = transformTriangle(local_triangles[i], camera, width, height);	
      float area = t.calculateTriArea();
      local_area += area;
    }
    #endif

    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();

    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("vkl_phase1_metric_times.txt", array, world_size);

    MPI_Reduce(&w_local_area, &w_total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_area, &total_area, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    timer.Start();

    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto triangles = GetUniqueTriangles(dataset);
      if (triangles.GetNumberOfValues() > 0)
      {
        auto areas = CalculateProjectedTriangleAreas(triangles, camera, width, height, true);
        auto w_areas = CalculateTriangleAreas(triangles);
        auto projected_area = vtkm::cont::Algorithm::Reduce(areas, 0.0f);

        vtkm::cont::ArrayHandle<float> sub_entropies;
        vtkm::cont::Algorithm::Transform(
          areas, 
          w_areas,
          sub_entropies, 
          CalculateVKLEntropy<float>{static_cast<float>(projected_area), 
                                     static_cast<float>(w_total_area)});
        vkl = vtkm::cont::Algorithm::Reduce(sub_entropies, 0.0f);
      }
      #else
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
      if(x0.size())
      {
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
        long double projected_area = 0.0;
        #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for reduction(+:projected_area)
        #endif
        for(int i = 0; i < num_triangles; i++)
        {
  	      float area = calcArea(triangles[i], camera, width, height);
          projected_area += area;
        }

	vkl = 0.0;
        #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for reduction(+:vkl)
        #endif
        for(int i = 0; i < num_triangles; i++)
        {
	        float area   = calcArea(triangles[i], camera, width, height);
          float w_area = calcArea(triangles[i]);
	        if(area != 0.0 && w_area != 0.0)
	          vkl += (area/projected_area)*std::log((area/projected_area)/(w_area/w_total_area));
        }
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    total_time = timer.GetElapsedTime();
    MPI_Bcast(&vkl, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array1[world_size] = {0};
    array1[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array1, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("vkl_phase2_metric_times.txt", array1, world_size);
    //cerr << "vkl " << vkl << endl;

    return (-1.0) * vkl;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
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
    float w_total_area   = worldspace_local_area;
    float projected_area = 0.0;
    #ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:total_area)
    #endif
    for(int i = 0; i < num_local_triangles; i++)
    {
      Triangle t = transformTriangle(local_triangles[i], camera, width, height);
      float area = t.calculateTriArea();
      total_area += area;
    }
    #ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:projected_area)
    #endif
    for(int i = 0; i < num_triangles; i++)
    {
      float area = calcArea(triangles[i], camera, width, height);
      projected_area += area;
    }
    vkl = 0.0;
    #ifdef ASCENT_USE_OPENMP
    #pragma omp parallel for reduction(+:vkl)
    #endif
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
calculateDataEntropy(vtkh::DataSet* dataset, int height, int width,std::string field_name, float field_max, float field_min)
{
  float entropy = 0.0;
  #if ASCENT_MPI_ENABLED //pass screens among all ranks
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto field_data = GetScalarDataAsArrayHandle<float>(*dataset, field_name.c_str());
      if (field_data.GetNumberOfValues() > 0) 
      {
        DataCheckFlags checks = CheckNan | CheckZero;
        field_data = copyWithChecks<float>(field_data, checks);
        entropy = calcentropyMM(field_data, 6, field_max, field_min);
      } 
      else
      {
        entropy = 0;
      }
      #else
      int size = height*width;
      std::vector<float> field_data = GetScalarData<float>(*dataset, field_name.c_str(), height, width);
      std::vector<float> data;
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<float> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:data)
//      #endif
      if(field_data.size())
      {
        for(int i = 0; i < size; i++)
          if(field_data[i] == field_data[i] && field_data[i] != 0)
            data.push_back(field_data[i]);
        float field_array[data.size()];
        std::copy(data.begin(), data.end(), field_array);
        entropy = calcentropyMM(field_array, data.size(), 6, field_max, field_min);
      }
      else
        entropy = 0;
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("dataentropy_metric_times.txt", array, world_size);

    return entropy;
  #else
    int size = height*width;
    std::vector<float> field_data = GetScalarData<float>(*dataset, field_name.c_str(), height, width);
    std::vector<float> data;
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<float> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:data)
//      #endif
      
    if(field_data.size())
    {
      for(int i = 0; i < size; i++)
        if(field_data[i] == field_data[i] && field_data[i] != 0)
          data.push_back(field_data[i]);
      float field_array[data.size()];
      std::copy(data.begin(), data.end(), field_array);
      entropy = calcentropyMM(field_array, data.size(), 6, field_max, field_min);
      //MakeFile
      std::string file = "data_data_dump" + std::to_string(COUNT) + ".txt";
//      MakeFile(file, data.data(), data.size());
    }
    else
      entropy = 0;
  #endif
  return entropy;
}

float 
calculateDepthEntropy(vtkh::DataSet* dataset, int height, int width, float diameter)
{

  float entropy = 0.0;
  #if ASCENT_MPI_ENABLED 
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto field_data = GetScalarDataAsArrayHandle<float>(*dataset, "depth");
      if (field_data.GetNumberOfValues() > 0) 
      {
        // TODO: Manish to understand why do we ignore 0 values?
        DataCheckFlags checks = CheckNan | CheckMinExclusive | CheckMaxExclusive;
        DataCheckVals<float> checkVals { .Min = 0, .Max = float(INT_MAX) };
        field_data = copyWithChecks<float>(field_data, checks, checkVals);
        entropy = calcentropyMM(field_data, 1000, diameter, float(0));
      } 
      else
      {
        entropy = 0;
      }
      #else
      int size = height*width;
      std::vector<float> depth = GetScalarData<float>(*dataset, "depth", height, width);
      std::vector<float> depth_data;
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<float> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:depth_data)
//      #endif
      if(depth.size())
      {
        for(int i = 0; i < size; i++)
          if(depth[i] == depth[i] && depth[i] < INT_MAX && depth[i] > 0)
	        {
            depth_data.push_back(depth[i]);
	        }
          //depth_data[i] = -FLT_MAX;
        float depth_array[depth_data.size()];
        std::copy(depth_data.begin(), depth_data.end(), depth_array);
        entropy = calcentropyMM(depth_array, depth_data.size(), 1000, diameter, (float) 0.0);
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("depthentropy_metric_times.txt", array, world_size);

    return entropy;
  #else
    int size = height*width;
    std::vector<float> depth = GetScalarData<float>(*dataset, "depth", height, width);
    std::vector<float> depth_data;
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<float> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:depth_data)
//      #endif
    if(depth.size())
    {
      for(int i = 0; i < size; i++)
        if(depth[i] == depth[i] && depth[i] < INT_MAX && depth[i] > 0)
        {
          depth_data.push_back(depth[i]);
        }
        //depth_data[i] = -FLT_MAX;
      float depth_array[depth_data.size()];
      std::copy(depth_data.begin(), depth_data.end(), depth_array);
      entropy = calcentropyMM(depth_array, depth_data.size(), 1000, diameter, (float) 0.0);
      //MakeFile
      std::string file = "depth_data_dump" + std::to_string(COUNT) + ".txt";
//      MakeFile(file, depth_data.data(), depth_data.size());
    }
  #endif
  return entropy;
}

float
calculateBinEntropy(vtkh::DataSet* dataset, int height, int width, int xBins, int yBins, int zBins)
{

  float entropy = 0.0;
  #if ASCENT_MPI_ENABLED
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      int size = height*width;
      std::vector<float> bin = GetScalarData<float>(*dataset, "Bin", height, width);
      std::vector<float> bin_data;
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<float> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:depth_data)
//      #endif
      if(bin.size())
      {
        for(int i = 0; i < size; i++)
          if(bin[i] == bin[i] && bin[i] <= INT_MAX)
          {
            bin_data.push_back(bin[i]);
          }
          //bin_data[i] = -FLT_MAX;
	int bins = xBins*yBins*zBins;
	int bins_minus1 = bins - 1;
        float bin_array[bin_data.size()];
        std::copy(bin_data.begin(), bin_data.end(), bin_array);
        entropy = calcentropyMM(bin_array, bin_data.size(), bins, (float)bins_minus1 , (float)0);
      }
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&entropy, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("depthentropy_metric_times.txt", array, world_size);

    return entropy;
  #else
    int size = height*width;
    std::vector<float> bin = GetScalarData<float>(*dataset, "Bin", height, width);
    std::vector<float> bin_data;
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<float> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:bin_data)
//      #endif
    if(bin.size())
    {
      for(int i = 0; i < size; i++)
        if(bin[i] == bin[i] && bin[i] < INT_MAX)
        {
          bin_data.push_back(bin[i]);
        }
        //bin_data[i] = -FLT_MAX;
      float bin_array[bin_data.size()];
      std::copy(bin_data.begin(), bin_data.end(), bin_array);
      int bins = xBins*yBins*zBins;
      int bins_minus1 = bins - 1;
    
      entropy = calcentropyMM(bin_array, bin_data.size(), bins, (float)bins_minus1, (float)0);
      //MakeFile
      std::string file = "bin_data_dump" + std::to_string(COUNT) + ".txt";
//      MakeFile(file, bin_data.data(), bin_data.size());
    }
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto triangles = GetUniqueTriangles(dataset);
      if (triangles.GetNumberOfValues() > 0) 
      {
        num_triangles = triangles.GetNumberOfValues();
      }
      #else
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
      if(x0.size())
      {
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
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&num_triangles, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("num_visible_triangles_metric_times.txt", array, world_size);    

    return num_triangles;
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
      #endif
    if(x0.size())
    {
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();    
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto triangles = GetUniqueTriangles(dataset);
      auto projected_areas = CalculateProjectedTriangleAreas(triangles, camera, width, height, true);
      projected_area = vtkm::cont::Algorithm::Reduce(projected_areas, 0.0f);
      #else      
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif

      if(x0.size())
      {
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
        #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for reduction(+:projected_area)
        #endif
        for(int i = 0; i < num_triangles; i++)
        {
          float area = calcArea(triangles[i], camera, width, height);
          projected_area += area;
        }
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&projected_area, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("projectedarea_metric_times.txt", array, world_size);    
    
  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
    if(x0.size())
    {
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
      #ifdef ASCENT_USE_OPENMP
      #pragma omp parallel for reduction(+:projected_area)
      #endif
      for(int i = 0; i < num_triangles; i++)
      {  
        float area = calcArea(triangles[i], camera, width, height);
        projected_area += area;
      }
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    //Needs total global triangles
    int num_global_triangles = 0;
    MPI_Reduce(&num_local_triangles, &num_global_triangles, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
      float projected_area = 0.0;
      int num_triangles = 0;
      #if defined(ASCENT_VTKM_ENABLED)
      auto triangles = GetUniqueTriangles(dataset);
      num_triangles = triangles.GetNumberOfValues();
      auto projected_areas = CalculateProjectedTriangleAreas(triangles, camera, width, height, true);
      projected_area = vtkm::cont::Algorithm::Reduce(projected_areas, 0.0f);
      #else
      int size = height*width;
      std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
      std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
      std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
      std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
      std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
      std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
      std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
      std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
      std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

      std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
      if(x0.size())
      {
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
        projected_area = 0.0;
        #ifdef ASCENT_USE_OPENMP
        #pragma omp parallel for reduction(+:projected_area)
        #endif
        for(int i = 0; i < num_triangles; i++)
        {
          float area = calcArea(triangles[i], camera, width, height);
          projected_area += area;
        }
      }
      #endif // defined(ASCENT_VTKM_ENABLED)
      float pixel_ratio = projected_area / static_cast<float>(height * width);
      float triangle_ratio = static_cast<float>(num_triangles) / static_cast<float>(num_global_triangles);
      pb_score = pixel_ratio + triangle_ratio;
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&pb_score, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("pb_metric_times.txt", array, world_size);

  #else
    int size = height*width;
    std::vector<float> x0 = GetScalarData<float>(*dataset, "X0", height, width);
    std::vector<float> y0 = GetScalarData<float>(*dataset, "Y0", height, width);
    std::vector<float> z0 = GetScalarData<float>(*dataset, "Z0", height, width);
    std::vector<float> x1 = GetScalarData<float>(*dataset, "X1", height, width);
    std::vector<float> y1 = GetScalarData<float>(*dataset, "Y1", height, width);
    std::vector<float> z1 = GetScalarData<float>(*dataset, "Z1", height, width);
    std::vector<float> x2 = GetScalarData<float>(*dataset, "X2", height, width);
    std::vector<float> y2 = GetScalarData<float>(*dataset, "Y2", height, width);
    std::vector<float> z2 = GetScalarData<float>(*dataset, "Z2", height, width);

    std::vector<std::vector<float>> triangles; //<x0,y0,z0,x1,y1,z1,x2,y2,z2>
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp declare reduction (merge : std::vector<std::vector<float>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//      #pragma omp parallel for reduction(merge:triangles)
//      #endif
    if(x0.size())
    {
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
      #ifdef ASCENT_USE_OPENMP
      #pragma omp parallel for reduction(+:projected_area)
      #endif
      for(int i = 0; i < num_triangles; i++)
      {
        float area = calcArea(triangles[i], camera, width, height);
        projected_area += area;
      }

      float pixel_ratio = projected_area/size;
      float triangle_ratio = (float) num_triangles/(float) num_local_triangles;
      pb_score = pixel_ratio + triangle_ratio;
    }
  #endif
  return pb_score;
}

#if defined(ASCENT_VTKM_ENABLED)
template <typename T>
T calculateMaxDepth(const vtkm::cont::ArrayHandle<T> &depthData)
{
  T depth = -1.0 * std::numeric_limits<T>::max();

  if (depthData.GetNumberOfValues() > 0)
  {
    MaxValueWithChecks<T> max{
        -1.0 * std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max()};
    depth = vtkm::cont::Algorithm::Reduce(depthData, depth, max);
  }

  return depth;
}
#endif

float
calculateMaxDepth(vtkh::DataSet *dataset, int height, int width)
{
  float depth = -FLT_MAX;
  #if ASCENT_MPI_ENABLED
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = high_resolution_clock::now();
    vtkm::cont::Timer timer;
    timer.Start();

    // Get the rank of this process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Status status;
    if(rank == 0)
    {
      #if defined(ASCENT_VTKM_ENABLED)
      auto depthData = GetScalarDataAsArrayHandle<float>(*dataset, "depth");
      depth = calculateMaxDepth(depthData);
      #else
      int size = height*width;
      std::vector<float> depth_data = GetScalarData<float>(*dataset, "depth", height, width);
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp parallel for reduction(max:depth)
//      #endif
      if(depth_data.size())
      {
        for(int i = 0; i < size; i++)
          if(depth_data[i] == depth_data[i])
	        {
            if(depth < depth_data[i] && depth_data[i] < INT_MAX)
	          {
              depth = depth_data[i];
	          }
	        }
      }
      #endif
    }
    auto time_stop = high_resolution_clock::now();
    timer.Stop();
    vtkm::Float64 total_time = timer.GetElapsedTime();
    MPI_Bcast(&depth, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double metric_time = duration_cast<microseconds>(time_stop - time_start).count();
//    cerr << "rank " << rank << " metric work time: " << metric_time << " microseconds." << endl;
    double array[world_size] = {0};
    array[rank] = total_time;
    MPI_Allgather(&total_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
      MakeFile("maxdepth_metric_times.txt", array, world_size);

  #else
    int size = height*width;
    std::vector<float> depth_data = GetScalarData<float>(*dataset, "depth", height, width);
//      #ifdef ASCENT_USE_OPENMP
//      #pragma omp parallel for reduction(max:depth)
//      #endif
    if(depth_data.size())
    {
      for(int i = 0; i < size; i++)
        if(depth_data[i] == depth_data[i])
          if(depth < depth_data[i] && depth_data[i] < INT_MAX)
            depth = depth_data[i];
    }
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
      // MPI_Status status;
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
calculateMetricScore(vtkh::DataSet* dataset, std::string metric, std::string field_name, std::vector<Triangle> &local_triangles, double worldspace_local_area, int height, int width, Camera camera, float field_max, float field_min, int xBins, int yBins, int zBins, float diameter)
{
  float score = 0.0;

  if(metric == "data_entropy")
  {
    score = calculateDataEntropy(dataset, height, width, field_name, field_max, field_min);
  }
  else if (metric == "visibility_ratio")
  {
    score = calculateVisibilityRatio(dataset, local_triangles, worldspace_local_area,  height, width);
  }
  else if (metric == "viewpoint_entropy")
  {
    score = calculateViewpointEntropy(dataset, local_triangles, height, width, camera);
  }
  else if (metric == "dds_entropy")
  {
    float shading_score = calculateShadingEntropy(dataset, height, width, camera);
    cerr << "shading score: " << shading_score << endl;
    float data_score = calculateDataEntropy(dataset, height, width, field_name, field_max, field_min);
    cerr << "data_score: " << data_score << endl;
    float depth_score = calculateDepthEntropy(dataset, height, width, diameter);
    score = shading_score+data_score+depth_score;
    cerr << "depth_score: " << depth_score << endl;
    cerr << "dds_score: " << score << endl;
  }
  else if (metric == "shading_entropy")
  {
    score = calculateShadingEntropy(dataset, height, width, camera);
  }
  else if (metric == "i2")
  {
    score = calculateI2(dataset, local_triangles, height, width, camera);
  }
  else if (metric == "vkl")
  {

    score = calculateVKL(dataset, local_triangles, worldspace_local_area, height, width, camera);
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
  else if (metric == "bin_entropy")
  {
    score = calculateBinEntropy(dataset, height, width, xBins, yBins, zBins);
  }
  else if (metric == "depth_entropy")
  {
    score = calculateDepthEntropy(dataset, height, width, diameter);
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
    valid_paths.push_back("sample");
    valid_paths.push_back("phi");
    valid_paths.push_back("theta");
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

    #if defined(ASCENT_VTKM_ENABLED)
      #if ASCENT_MPI_ENABLED
        int rank;
	int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      
      #endif  
      // vtkm::cont::SetStderrLogLevel(vtkm::cont::LogLevel::UserVerboseFirst);
      DataObject *data_object = input<DataObject>(0);
      std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
    //int cycle = params()["state/cycle"].to_int32();
      conduit::Node meta = Metadata::n_metadata;
      int cycle = -1;
      if(meta.has_path("cycle"))
      {
        cycle = meta["cycle"].to_int32();
      }
      cerr << "=====USING CAMERA PIPELINE===== CYCLE: " << cycle << endl;
      std::string field_name = params()["field"].as_string();
      std::string metric     = params()["metric"].as_string();

      if(!collection->has_field(field_name))
      {
        ASCENT_ERROR("Unknown field '"<<field_name<<"'");
      }
      int samples = (int)params()["samples"].as_int64();
      int sample2 = (int)params()["sample"].as_int64();
      int c_phi = (int)params()["phi"].as_int64();
      int c_theta = (int)params()["theta"].as_int64();
    //TODO:Get the height and width of the image from Ascent
      int width  = 1000;
      int height = 1000;

      #if ASCENT_MPI_ENABLED
      MPI_Barrier(MPI_COMM_WORLD);
      double triangle_time = 0.;
      auto triangle_start = high_resolution_clock::now();
      #endif

      std::string topo_name = collection->field_topology(field_name);

      vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);
    
      std::vector<double> field_data = GetScalarData<double>(dataset, field_name.c_str(), height, width);
      
      float datafield_max = 0.;
      float datafield_min = 0.;
      if(field_data.size())
      {
        datafield_max = (float)*max_element(field_data.begin(),field_data.end());
        datafield_min = (float)*min_element(field_data.begin(),field_data.end());
      }
      //TODO: Need global mins and maxes for parallel. MPI send to rank 0.


      double worldspace_local_area = 0;
      std::vector<Triangle> triangles = GetTrianglesAndArea(dataset, worldspace_local_area);
      int num_local_triangles = triangles.size();
      float xmax = 0.0, xmin = 0.0, ymax = 0.0, ymin = 0.0, zmax = 0.0, zmin = 0.0;
      TriangleBounds(triangles,xmin,xmax,ymin,ymax,zmin,zmax);
      //cerr << "Triangle Bounds:\nX: " << xmin << " - " << xmax << "\nY: " << ymin << " - " << ymax << "\nZ: " << zmin << " - " << zmax << endl;
      
      int xBins = 8,yBins = 8,zBins = 8;

      vtkh::DataSet* data = AddTriangleFields(dataset,xmin,xmax,ymin,ymax,zmin,zmax,xBins,yBins,zBins);
//      data->PrintSummary(cerr);

      #if ASCENT_MPI_ENABLED
      auto triangle_stop = high_resolution_clock::now();
      triangle_time += duration_cast<microseconds>(triangle_stop - triangle_start).count();
      double array[world_size] = {0};
      cerr << "world size: " << world_size << endl;
      array[rank] = triangle_time;
      MPI_Allgather(&triangle_time, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);
      if(rank == 0)
        MakeFile("processing_times.txt", array, world_size);
      #endif

      vtkm::Bounds lb = dataset.GetBounds();
      //cerr << " local bounds:\n X: " << lb.X.Min << " - " << lb.X.Max << " \nY: " << lb.Y.Min << " - " << lb.Y.Max << " \nZ: " << lb.Z.Min << " - " << lb.Z.Max << endl; 

      vtkm::Bounds b = dataset.GetGlobalBounds();
      vtkm::Float32 xb = vtkm::Float32(b.X.Length());
      vtkm::Float32 yb = vtkm::Float32(b.Y.Length());
      vtkm::Float32 zb = vtkm::Float32(b.Z.Length());
      float bounds[6] = {(float)b.X.Max, (float)b.X.Min, 
	                (float)b.Y.Max, (float)b.Y.Min, 
	                (float)b.Z.Max, (float)b.Z.Min};
      //cerr << "global bounds: " << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << endl;

      vtkm::Float32 radius = sqrt(xb*xb + yb*yb + zb*zb)/2.0;
      float diameter = sqrt(xb*xb + yb*yb + zb*zb)*6.0;
      vtkmCamera *camera = new vtkmCamera;
      camera->ResetToBounds(dataset.GetGlobalBounds());
      vtkm::Vec<vtkm::Float32,3> lookat = camera->GetLookAt();
      float focus[3] = {(float)lookat[0],(float)lookat[1],(float)lookat[2]};

      //original
      double winning_score  = -DBL_MAX;
      int    winning_sample = -1;
      // int    winning_phi = -1;
      // int    winning_theta = -1;
      double losing_score   = DBL_MAX;
      int    losing_sample  = -1;
      // int    losing_phi  = -1;
      // int    losing_theta  = -1;

      // int phi = 100;
      // int theta = 100;
      int count = 0;


      //loop through number of camera samples.
      for(int sample = 0; sample < samples; sample++)
      {
        COUNT = sample;
        cerr<< "Sample: " << count << endl;
    /*================ Scalar Renderer Code ======================*/
    //What it does: Quick ray tracing of data (replaces get triangles and scanline).
    //What we need: z buffer, any other important buffers (tri ids, scalar values, etc.)
      
//        Camera cam = GetCamera(sample, samples, radius, focus, bounds);
        Camera cam = GetCamera(sample, samples, radius, focus, bounds);
        vtkm::Vec<vtkm::Float32, 3> pos{(float)cam.position[0],
                                (float)cam.position[1],
                                (float)cam.position[2]};
        #if ASCENT_MPI_ENABLED
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
	auto render_start = high_resolution_clock::now();

        camera->SetPosition(pos);
        vtkh::ScalarRenderer tracer;
        tracer.SetWidth(width);
        tracer.SetHeight(height);
        tracer.SetInput(data); //vtkh dataset by toponame
        tracer.SetCamera(*camera);
        tracer.Update();
        //camera->GetViewUp().Print();
//	cerr << "Camera for " << count << endl;
//        camera->Print();
	
	

        vtkh::DataSet *output = tracer.GetOutput();
//	output->PrintSummary(std::cerr);

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


        cerr << "Starting metric" << endl;

        //original 
        //float score = calculateMetricScore(output, metric, field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);
        
	//new
        float data_entropy_score = calculateMetricScore(output, "data_entropy", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float depth_entropy_score = calculateMetricScore(output, "depth_entropy", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float shading_entropy_score = calculateMetricScore(output, "shading_entropy", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float max_depth_score = calculateMetricScore(output, "max_depth", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float projected_area_score = calculateMetricScore(output, "projected_area", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float pb_score = calculateMetricScore(output, "pb", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float visible_triangles_score = calculateMetricScore(output, "visible_triangles", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float visibility_ratio_score = calculateMetricScore(output, "visibility_ratio", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float viewpoint_entropy_score = calculateMetricScore(output, "viewpoint_entropy", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float vkl_score = calculateMetricScore(output, "vkl", field_name, triangles, worldspace_local_area, height, width, cam, datafield_max, datafield_min, xBins, yBins, zBins, diameter);

        float score = data_entropy_score;

	std::cerr << "sample " << sample << " " << metric << " score: " << score << std::endl;
	cerr << endl;
        delete output;

    /*================ End Scalar Renderer  ======================*/

	//original
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
        count++;
      } //end of sample loop
      triangles.clear();
//    delete data;
//
      #if ASCENT_MPI_ENABLED
      MPI_Barrier(MPI_COMM_WORLD);
      #endif
      auto setting_camera_start = high_resolution_clock::now();

      if(winning_sample == -1)
        ASCENT_ERROR("Something went terribly wrong; No camera position was chosen");
      cerr << metric << " winning_sample " << winning_sample << " score: " << winning_score << endl;
      cerr << metric << " losing_sample " << losing_sample << " score: " << losing_score << endl;
     // Camera best_c = GetCamera(cycle, 100, radius, focus, bounds);
      //Camera best_c = GetCamera(losing_sample, samples, radius, focus, bounds);
      Camera best_c = GetCamera(winning_sample, samples, radius, focus, bounds);
//      cerr << "Writing out camera: " << sample2 << endl; 
//      Camera best_c = GetCamera(sample2, samples, radius, focus, bounds);
    
      vtkm::Vec<vtkm::Float32, 3> pos{(float)best_c.position[0], 
	                            (float)best_c.position[1], 
				    (float)best_c.position[2]}; 
      camera->SetPosition(pos);
      //camera->GetViewUp().Print();
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

    #endif //vtkm enabled
    set_output<DataObject>(input<DataObject>(0));
    cerr << "========END CAMERA PIPELINE=======" << endl;
    return;
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
