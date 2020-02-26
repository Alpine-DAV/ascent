//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
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
/// file: ascent_runtime_vtkh_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_CAMERA_FILTERS
#define ASCENT_RUNTIME_CAMERA_FILTERS

#include <ascent.hpp>

#include <flow_filter.hpp>




//-----------------------------------------------------------------------------
//Misc Functions
//-----------------------------------------------------------------------------
double ceil441(double f);
double floor441(double f);
double nabs(double x);
double calculateArea(double x0, double y0, double z0, double x1, double y1, double z1, double x2, double y2, double z2);
void normalize(double * normal);
double* normalize2(double * normal);
double dotProduct(double* v1, double* v2, int length);
double magnitude2d(double* vec);
double magnitude3d(double* vec);
double* crossProduct(double * a, double * b);
double SineParameterize(int curFrame, int nFrames, int ramp);
void fibonacci_sphere(int i, int samples, double* points);

//-----------------------------------------------------------------------------
//Matrix Class
//-----------------------------------------------------------------------------

class Matrix
{
  public:
    double          A[4][4];

    void            TransformPoint(const double *ptIn, double *ptOut);
    static Matrix   ComposeMatrices(const Matrix &, const Matrix &);
    void            Print(std::ostream &o);

};



//-----------------------------------------------------------------------------
//Screen Class
//-----------------------------------------------------------------------------
class Screen
{
  public:
      unsigned char *buffer;
      int           width, height;
      double*       zBuff;
      double        visible;
      double        occluded;
      int*          triScreen;
      double**      triCamera;

      void zBufferInitialize();
      void triScreenInitialize();
      void triCameraInitialize();
};

//-----------------------------------------------------------------------------
//Camera Class
//-----------------------------------------------------------------------------

class Camera
{
  public:
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];
    Screen          screen;

    Matrix          CameraTransform(void);
    Matrix          ViewTransform(void);
    Matrix          DeviceTransform(void);
};

Camera GetCamera(int frame, int nframes);
//-----------------------------------------------------------------------------
//Edge Class
//-----------------------------------------------------------------------------

class Edge{
  public:
	double x1, x2, y1, y2, z1, z2, r1, r2, g1, g2, b1, b2, slope, b, minY, maxY, shade1, shade2; //as in y = mx + b
	double *normal1, *normal2;
	bool   vertical, relevant; //find the vertical line, horizontal line and hypotenuse. relevant = hypotenuse; not relevant = horizontal
	Edge(){}
	Edge (double x_1, double y_1, double z_1, double r_1, double g_1, double b_1, double* norm1, double s1, double x_2, double y_2, double z_2, double r_2, double g_2, double b_2, double* norm2, double s2);
	double findX(double y);
	double interpolate(double a, double b, double C, double D, double fa, double fb, double x);
	double findZ(double y);
	double findRed(double y);
	double findGreen(double y);
	double findBlue(double y);
	double normalZ(double y);
	double normalX(double y);
	double normalY(double y);
	double findShade(double y);
	bool   applicableY(double y);
};





//-----------------------------------------------------------------------------
//Triangle Class
//-----------------------------------------------------------------------------
#include <cfloat>

class Triangle
{
  public:
      double         X[3];
      double         Y[3];
      double         Z[3];
      double         colors[3][3];
      double         normals[3][3];
      Screen         screen;
      double         view[3];
      double         shading[3];
      double         centroid[3];
      double         radius;
      bool           vis_counted = false;
      int            compID;
      bool           occ_counted = false;
      double         area = 0.0;
      double         minDepth = DBL_MAX;
      double         maxDepth = -DBL_MAX;


      void printTri();
      void findDepth();
      void calculateTriArea();

      void calculateCentroid();
      void calculatePhongShading(Camera c);
      void scanline(int i, Camera c);
      double findMin(double a, double b, double c);
      double findMax(double a, double b, double c);
};


Triangle transformTriangle(Triangle t, Camera c);
std::vector<Triangle>
GetTriangles(const char *, int type);
double CalculateNormalCameraDot(double* cameraPositions, Triangle tri);

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
///
/// Camera Filters
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ASCENT_API AutoCamera : public ::flow::Filter
{
public:
    AutoCamera();
    virtual ~AutoCamera();

    virtual void   declare_interface(conduit::Node &i);
    virtual bool   verify_params(const conduit::Node &params,
                                 conduit::Node &info);
    virtual void   execute();
};

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




#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
