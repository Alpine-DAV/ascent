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
#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/DataSet.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#define EXEC_CONT VTKM_EXEC_CONT
#else
#define EXEC_CONT 
#endif

//-----------------------------------------------------------------------------
//Misc Functions
//-----------------------------------------------------------------------------
float nabs(float x);
float calculateArea(float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2);
template<typename T> EXEC_CONT void normalize(T * normal);
template<typename T> EXEC_CONT T dotProduct(const T* v1, const T* v2, int length);
template<typename T> EXEC_CONT void crossProduct(const T a[3], const T b[3], T c[3]);
float SineParameterize(int curFrame, int nFrames, int ramp);
void fibonacci_sphere(int i, int samples, float* points);

//-----------------------------------------------------------------------------
//Matrix Class
//-----------------------------------------------------------------------------

class Matrix
{
  public:
    double          A[4][4];

    EXEC_CONT void   TransformPoint(const double *ptIn, double *ptOut);
    EXEC_CONT static Matrix ComposeMatrices(const Matrix &m1, const Matrix &m2);
    EXEC_CONT void   Print(std::ostream &o);

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

    EXEC_CONT Matrix CameraTransform(void) const;
    EXEC_CONT Matrix ViewTransform(void) const;
    EXEC_CONT Matrix DeviceTransform(int width, int height) const;
    EXEC_CONT Matrix DeviceTransform() const;
};

Camera GetCamera(int frame, int nframes, float radius, float *lookat, float *bounds);


//-----------------------------------------------------------------------------
//Triangle Class
//-----------------------------------------------------------------------------
#include <cfloat>

class Triangle
{
  public:
      float          X[3];
      float          Y[3];
      float          Z[3];

      EXEC_CONT 
      Triangle(){};

      EXEC_CONT 
      Triangle(float x0, float y0, float z0,
	       float x1, float y1, float z1,
	       float x2, float y2, float z2)
      {
        X[0] = x0; Y[0] = y0; Z[0] = z0;
        X[1] = x1; Y[1] = y1; Z[1] = z1;
        X[2] = x2; Y[2] = y2; Z[2] = z2;
      }	      
      
      void printTri() const;
      EXEC_CONT float calculateTriArea() const;
      EXEC_CONT float findMin(float a, float b, float c) const;
      EXEC_CONT float findMax(float a, float b, float c) const;
      EXEC_CONT void cutoff(int w, int h);
};

#if defined(ASCENT_VTKM_ENABLED)
EXEC_CONT Triangle transformTriangle(const Triangle& t, const Camera& c, int width, int height);
std::vector<Triangle>
GetTriangles(vtkh::DataSet &vtkhData, std::string field_name );
double CalculateNormalCameraDot(double* cameraPositions, Triangle tri);

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
