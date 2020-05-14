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


#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <chrono>
#include <stdio.h>

using namespace conduit;
using namespace std;
using namespace std::chrono;

using namespace flow;

typedef vtkm::rendering::Camera vtkmCamera;

//Camera Class Functions

Matrix
Camera::CameraTransform(void)
{
  bool print = false;
  double* v3 = new double[3]; //camera position - focus
  v3[0] = (position[0] - focus[0]);
  v3[1] = (position[1] - focus[1]);
  v3[2] = (position[2] - focus[2]);
  normalize(v3);

  double* v1 = new double[3]; //UP x (camera position - focus)
  v1 = crossProduct(up, v3);
  normalize(v1);

  double* v2 = new double[3]; // (camera position - focus) x v1
  v2 = crossProduct(v3, v1);
  normalize(v2);

  double* t = new double[3]; // (0,0,0) - camera position
  t[0] = (0 - position[0]);
  t[1] = (0 - position[1]);
  t[2] = (0 - position[2]);


  if (print)
  {
    cout << "position " << position[0] << " " << position[1] << " " << position[2] << endl;
    cout << "focus " << focus[0] << " " << focus[1] << " " << focus[2] << endl;
    cout << "up " << up[0] << " " << up[1] << " " << up[2] << endl;
    cout << "v1 " << v1[0] << " " << v1[1] << " " << v1[2] << endl;
    cout << "v2 " << v2[0] << " " << v2[1] << " " << v2[2] << endl;
    cout << "v3 " << v3[0] << " " << v3[1] << " " << v3[2] << endl;
    cout << "t " << t[0] << " " << t[1] << " " << t[2] << endl;
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
    cout << "Camera:" << endl;
    camera.Print(cout);
  }
  delete[] v1;
  delete[] v2;
  delete[] v3;
  delete[] t;
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

  device.A[0][0] = (screen.width/2);
  device.A[0][1] = 0;
  device.A[0][2] = 0;
  device.A[0][3] = 0;
  device.A[1][0] = 0;
  device.A[1][1] = (screen.height/2);
  device.A[1][2] = 0;
  device.A[1][3] = 0;
  device.A[2][0] = 0;
  device.A[2][1] = 0;
  device.A[2][2] = 1;
  device.A[2][3] = 0;
  device.A[3][0] = (screen.width/2);
  device.A[3][1] = (screen.height/2);
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
  cout << "X: " << X[0] << " " << X[1] << " " << X[2] << endl;
  cout << "Y: " << Y[0] << " " << Y[1] << " " << Y[2] << endl;
  cout << "Z: " << Z[0] << " " << Z[1] << " " << Z[2] << endl;
}

void
Triangle::findDepth()
{
  minDepth = std::min({Z[0], Z[1], Z[3]});
  maxDepth = std::max({Z[0], Z[1], Z[3]});
}

void
Triangle::calculateTriArea(){
  bool print = false;
  area = 0.0;
  double AC[3];
  double BC[3];

  AC[0] = X[1] - X[0];
  AC[1] = Y[1] - Y[0];
  AC[2] = Z[1] - Z[0];

  BC[0] = X[2] - X[0];
  BC[1] = Y[2] - Y[0];
  BC[2] = Z[2] - Z[0];

  double orthogonal_vec[3];

  orthogonal_vec[0] = AC[1]*BC[2] - BC[1]*AC[2];
  orthogonal_vec[1] = AC[0]*BC[2] - BC[0]*AC[2];
  orthogonal_vec[2] = AC[0]*BC[1] - BC[1]*AC[1];

  area = sqrt(pow(orthogonal_vec[0], 2) +
              pow(orthogonal_vec[1], 2) +
              pow(orthogonal_vec[2], 2))/2.0;

  if(print)
  {
  cout << "Triangle: (" << X[0] << " , " << Y[0] << " , " << Z[0] << ") " << endl <<
                   " (" << X[1] << " , " << Y[1] << " , " << Z[1] << ") " << endl <<
                   " (" << X[2] << " , " << Y[2] << " , " << Z[2] << ") " << endl <<
           " has surface area: " << area << endl;
  }
  return;
}

void
Triangle::calculateCentroid(){
  bool print = false;
  radius = 0;
  centroid[0] = (X[0]+X[1]+X[2])/3;
  centroid[1] = (Y[0]+Y[1]+Y[2])/3;
  centroid[2] = (Z[0]+Z[1]+Z[2])/3;


  double median[3];
  median[0] = (X[0]+X[1])/2;
  median[1] = (Y[0]+Y[1])/2;
  median[2] = (Z[0]+Z[1])/2;

  radius = sqrt(pow(median[0] - centroid[0], 2) +
                pow(median[1] - centroid[1], 2));

  if(print)
  {
    cout << endl << "centroid: (" << centroid[0] << " , " << centroid[1] << " , " << centroid[2] << ")" << endl;
    cout << "radius: " << radius << endl;
  }
}

void
Triangle::scanline(int i, Camera c)
{
  double minX;
  double maxX;

  double minY = findMin(Y[0], Y[1], Y[2]);
  double maxY = findMax(Y[0], Y[1], Y[2]);
  minY = ceil441(minY);
  maxY = floor441(maxY);
  if (minY < 0)
    minY = 0;
  if (maxY > screen.height-1)
    maxY = screen.height-1;

  Edge e1 = Edge(X[0], Y[0], Z[0], X[1], Y[1], Z[1], value[0], value[1]);
  Edge e2 = Edge(X[1], Y[1], Z[1], X[2], Y[2], Z[2], value[1], value[2]);
  Edge e3 = Edge(X[2], Y[2], Z[2], X[0], Y[0], Z[0], value[2], value[0]);

  double t, rightEnd, leftEnd;
  Edge leftLine, rightLine;
  //loop through Y and find X values, then color the pixels given min and max X found
  for(int y = minY; y <= maxY; y++){
    int row = screen.width*3*y;
    leftEnd = 1000*1000;
    rightEnd = -1000*1000;

    if (e1.relevant)
    { //not horizontal
      if(e1.applicableY(y))
      { // y is within y1 and y2 for e1
        t = e1.findX(y);
        if(t < leftEnd)
	{ //find applicable left edge of triangle for given y
          leftEnd = t;
          leftLine = e1;
        }
        if(t > rightEnd)
	{ //find applicable right edge of triangle for given y
          rightEnd = t;
          rightLine = e1;
	}
      }
    }
    if(e2.relevant)
    { //not horizontal
      if(e2.applicableY(y))
      { //y is on e2s line
        t = e2.findX(y);
        if(t < leftEnd)
	{
          leftEnd = t;
          leftLine = e2;
        }
        if(t > rightEnd)
	{
          rightEnd = t;
          rightLine = e2;
        }
      }
    }
    if(e3.relevant)
    { //not horizontal
      if(e3.applicableY(y))
      { //line has given y
        t = e3.findX(y);
        if(t < leftEnd)
	{
          leftEnd = t;
          leftLine = e3;
        }
        if(t > rightEnd)
	{
          rightEnd = t;
          rightLine = e3;
        }
      }
    }

    minX = leftEnd;
    minX = ceil441(minX);
    maxX = rightEnd;
    maxX = floor441(maxX);
    if (minX < 0)
      minX = 0;
    if (maxX > screen.width-1)
      maxX = screen.width;

    //use the y value to interpolate and find the value at the end points of each x-line.-->[Xmin, Xmax]
    double leftZ, rightZ, leftV, rightV;
    leftZ  = leftLine.findZ((double)y); //leftmost z value of x
    rightZ = rightLine.findZ((double)y); //rightmost z value of x
    leftV  = leftLine.findValue((double)y);
    rightV = rightLine.findValue((double)y);
    
    //loop through all the pixels that have the bottom left in the triangle.
    double ratio, z, value = 0.0;
    for (int x = minX; x <= maxX; x++)
    {
      if (leftEnd == rightEnd)//don't divide by 0 & do not use rounded x min/max values
        ratio = 1.0;
      else
        ratio = ((double)x - leftEnd)/(rightEnd - leftEnd);//ratio between unrounded x values on the current row (y)

      //double distance = sqrt(pow(centroid[0] - x, 2) + pow(centroid[1] - y,2));
      z = leftZ + ratio*(rightZ - leftZ);
      value = leftV + ratio*(rightV - leftV);
      if(z > screen.zBuff[y*screen.width + x])
      {
        screen.triScreen[y*screen.width + x] = i;
        screen.triCamera[y*screen.width + x] = c.position;
        screen.values[y*screen.width + x]    = value;
        screen.zBuff[y*screen.width + x] = z;
        /*if(distance <= radius)//inside radius to the centroid
        {
          screen.triScreen[y*screen.width + x] = i;
        }*/
      }
    }
  }
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

double* crossProduct(double * a, double * b)
{
  double* cross = new double[3]; 
  cross[0] = ((a[1]*b[2]) - (a[2]*b[1])); //ay*bz-az*by
  cross[1] = ((a[2]*b[0]) - (a[0]*b[2])); //az*bx-ax*bz
  cross[2] = ((a[0]*b[1]) - (a[1]*b[0])); //ax*by-ay*bx

  return cross;
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
GetCamera(int frame, int nframes, double radius, double* lookat)
{
  double t = SineParameterize(frame, nframes, nframes/10);
  double points[3];
  fibonacci_sphere(frame, nframes, points);
  Camera c;
  double zoom = 1.0;
  c.near = zoom/20;
  c.far = zoom*25;
  c.angle = M_PI/6;

/*  if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
  {
    if(points[2] >= 0)
      points[2] += radius;
    if(points[2] < 0)
      points[2] -= radius;
  }*/

  c.position[0] = radius*points[0];
  c.position[1] = radius*points[1];
  c.position[2] = radius*points[2];

//cout << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
    
  c.focus[0] = lookat[0];
  c.focus[1] = lookat[1];
  c.focus[2] = lookat[2];
  c.up[0] = 0;
  c.up[1] = 1;
  c.up[2] = 0;
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
    output.value[0] = variable[0];
    output.value[1] = variable[1];
    output.value[2] = variable[2];
  }
};


std::vector<Triangle>
GetTriangles(vtkh::DataSet &vtkhData, std::string field_name)
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
      //dataset.PrintSummary(std::cout);
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


Triangle transformTriangle(Triangle t, Camera c)
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
  int i;
  for (i = 0; i < 3; i++)
  {
    triangle.value[i] = t.value[i];
  }
  //component ID -- currently unused
  triangle.compID = t.compID;

  delete[] pointOut;
  delete[] pointIn;

  return triangle;

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

double
calculateMetric(Screen screen, std::string metric)
{
  if(metric == "data_entropy")
  {
    #if ASCENT_MPI_ENABLED //pass screens among all ranks
      // Get the number of processes
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      // Get the rank of this process
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Status status;
      if(rank != 0)
      { 
	//send values to rank 0
        MPI_Send(screen.values, screen.width*screen.height, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	//send z-buffer to rank 0
        MPI_Send(screen.zBuff, screen.width*screen.height, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        //Get final values back from rank 0
	MPI_Recv(screen.values, screen.width*screen.height, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
      }
      if(rank == 0)
      {
	double *zBuff  = (double *) malloc(sizeof(double)*screen.width*screen.height);
	double *values = (double *)  malloc(sizeof(double)*screen.width*screen.height);
        for(int i = 1; i < world_size; i++)
        {
          MPI_Recv(values, screen.width*screen.height, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
          MPI_Recv(zBuff, screen.width*screen.height, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
	  for(int pixel = 0; pixel < screen.height*screen.width; pixel++)
          {
            //new buffer wins, replace with new value
            if(zBuff[pixel] > screen.zBuff[pixel])
	      screen.values[pixel] = values[pixel]; 
          }
	}
	free(zBuff);
	free(values);

	for(int i = 1; i < world_size; i++)
        {
	  //send values back
          MPI_Send(screen.values, screen.width*screen.height, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }
      }
    #endif
    return calcentropy(screen.values, screen.width*screen.height, 100);
  }
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
                        	  " Currently only supports data_entropy.\n";
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
    //cout << "USING CAMERA PIPELINE" << endl;
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
    std::vector<Triangle> triangles = GetTriangles(dataset,field_name);
    auto triangle_stop = high_resolution_clock::now();
    triangle_time += duration_cast<microseconds>(triangle_stop - triangle_start).count();
    #if ASCENT_MPI_ENABLED
      cout << "rank: " << rank << " has " << triangles.size() << " triangles. " << endl;
      cout << "rank: " << rank << " GetTri " << triangle_time << endl;
    //cout << "dataset bounds: " << dataset.GetGlobalBounds() << endl;
      
    #endif

    vtkm::Bounds b = dataset.GetGlobalBounds();
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


    Screen screen;
    screen.width = width;
    screen.height = height;
    screen.zBufferInitialize();
    screen.triScreenInitialize();
    screen.triCameraInitialize();
    screen.valueInitialize();

    double winning_score = -DBL_MAX;
    int    winning_sample = -1;
    //loop through number of camera samples.
    double scanline_time = 0.;
    double metric_time   = 0.;
    for(int sample = 0; sample < samples; sample++)
    {
    /*================ Scalar Renderer Code ======================*/
    //What it does: Quick ray tracing of data (replaces get triangles and scanline).
    //What we need: z buffer, any other important buffers (tri ids, scalar values, etc.)
      
      Camera cam = GetCamera(sample, samples, radius, focus);
      vtkm::Vec<vtkm::Float32, 3> pos{(float)cam.position[0],
                                (float)cam.position[1],
                                (float)cam.position[2]};

      camera->SetPosition(pos);
      vtkh::ScalarRenderer tracer;
      tracer.SetWidth(width);
      tracer.SetHeight(height);
      tracer.SetInput(&dataset); //vtkh dataset by toponame
      tracer.SetCamera(*camera);
      //if(!dataset.IsEmpty()) //not sure if necessary? 
      //all ranks get stuck whether they are empty or not.
      tracer.Update();
      //Getting stuck
      cout << "here " << endl;

      vtkh::DataSet *output = tracer.GetOutput();
    #if ASCENT_MPI_ENABLED
      if(output != NULL)
	      cout << "output is not null on rank: " << rank << endl;
      else
	      cout << "output is NULL on rank: " << rank << endl;
    #endif
      //VTKHCollection *new_coll = new VTKHCollection();
      //new_coll->add(*output, topo_name);
      //DataObject *res = new DataObject(new_coll);
      delete output;
      //set_output<DataObject>(res); //don't actually want this

    /*================ End Scalar Renderer  ======================*/

      screen.width = width;
      screen.height = height;
      screen.visible = 0.0;
      screen.occluded = 0.0;
      screen.zBufferInitialize();
      screen.triScreenInitialize();
      screen.triCameraInitialize();
      screen.valueInitialize();

      Camera c = GetCamera(sample, samples, radius, focus);
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
	Triangle i_t = transformTriangle(w_t, c);
	i_t.vis_counted = false;
	i_t.screen = screen;
	i_t.scanline(tri, c);
	screen = i_t.screen;

      }//end of triangle loop
      auto scanline_stop = high_resolution_clock::now();

      scanline_time += duration_cast<microseconds>(scanline_stop - scanline_start).count();

      //metric timings
      auto metric_start = high_resolution_clock::now();
      double score = calculateMetric(screen, metric);
      auto metric_stop = high_resolution_clock::now();
      metric_time += duration_cast<microseconds>(metric_stop - metric_start).count();
      //cout << "sample " << sample << " score: " << score << endl;
      if(winning_score < score)
      {
        winning_score = score;
	winning_sample = sample;
      }
    } //end of sample loop

    #if ASCENT_MPI_ENABLED
      cout << "rank: " << rank << " scanline: " << scanline_time/(double)samples << endl;
      cout << "rank: " << rank << " metric: " << metric_time/(double)samples << endl;
    #endif

    if(winning_sample == -1)
      ASCENT_ERROR("Something went terribly wrong; No camera position was chosen");
    //cout << "winning_sample " << winning_sample << " score: " << winning_score << endl;
    Camera best_c = GetCamera(winning_sample, samples, radius, focus);

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

    #if ASCENT_MPI_ENABLED
      cout << "rank: " << rank << " secs total: " << time << endl;
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
