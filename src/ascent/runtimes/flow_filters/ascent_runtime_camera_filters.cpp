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

#include <stdio.h>

using namespace conduit;
using namespace std;

using namespace flow;

typedef vtkm::rendering::Camera vtkmCamera;

//Camera Class Functions

Matrix
Camera::CameraTransform(void) {
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


	if (print){
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

	if(print){
		cout << "camera" << endl;
		camera.Print(cout);
	}
	delete[] v1;
	delete[] v2;
	delete[] v3;
	delete[] t;
	return camera;

};

Matrix
Camera::ViewTransform(void) {

        bool print = false;

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
	if (print){
		cout << "view" << endl;
		view.Print(cout);
	}
	return view;

    };


Matrix
Camera::DeviceTransform() { //(double x, double y, double z){

        bool print = false;
	/*if (x > screen.width)
		x = screen.width;
	if (x < 0)
		x = 0;

	if (y > screen.height)
		y = screen.height;
	if (y < 0)
		y = 0;
	double x_, y_, z_;
	x_ = (((screen.width)*(x+1))/2); //n*(x+1)/2
	y_ = (((screen.height)*(y+1))/2); //m*(y+1)/2
	z_ = z; //z
	if (print){
		cout << "x " << x << " x_ " << x_ << endl;
		cout << " screen width " << screen.width << endl;
	}*/

/*
| x' 0  0  0 |
| 0  y' 0  0 |
| 0  0  z' 0 |
| 0  0  0  1 |
*/
	Matrix device;
/*
	device.A[0][0] = x_;
	device.A[0][1] = 0;
	device.A[0][2] = 0;
	device.A[0][3] = 0;
	device.A[1][0] = 0;
	device.A[1][1] = y_;
	device.A[1][2] = 0;
	device.A[1][3] = 0;
	device.A[2][0] = 0;
	device.A[2][1] = 0;
	device.A[2][2] = z_;
	device.A[2][3] = 0;
	device.A[3][0] = 0;
	device.A[3][1] = 0;
	device.A[3][2] = 0;
	device.A[3][3] = 1;
*/

/*This is the matrix Andy posted on piazza*/

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


	if(print){
		cout << "device" << endl;
		device.Print(cout);
	}
	return device;
}


//Edge Class Functions


Edge::Edge (double x_1, double y_1, double z_1, double r_1, double g_1, double b_1, double* norm1, double s1, double x_2, double y_2, double z_2, double r_2, double g_2, double b_2, double* norm2, double s2){
		x1 = x_1;
		x2 = x_2;
		y1 = y_1;
		y2 = y_2;
		z1 = z_1;
		z2 = z_2;
		r1 = r_1;
		r2 = r_2;
		g1 = g_1;
		g2 = g_2;
		b1 = b_1;
		b2 = b_2;
		normal1 = norm1;
		normal2 = norm2;
		shade1  = s1;
		shade2  = s2;

		// find relationship of y1 and y2 for min and max bounds of the line
		if (y1 < y2)
			minY = y1;
		else
			minY = y2;
		if (y1 > y2)
			maxY = y1;
		else
			maxY = y2;

		if (x2 - x1 == 0){ //if vertical, return x
			vertical = true;
			slope = x1;
		}
		else{
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
Edge::interpolate(double a, double b, double C, double D, double fa, double fb, double x){
		double A, B, fA, fB;
		if(C < D){
			A = a;
			B = b;
			fA = fa;
			fB = fb;
		}
		else{
			A = b;
			B = a;
			fA = fb;
			fB = fa;
		}
		double fX = fA + ((x - A)/(B - A))*(fB-fA);
		return fX;
	}

double
Edge::findZ(double y){
		double z = interpolate(y1, y2, x1, x2, z1, z2, y);
		return z;
	}

double
Edge::findRed(double y){
		double red = interpolate(y1, y2, x1, x2, r1, r2, y);
		return red;
	}

double
Edge::findGreen(double y){
		double green = interpolate(y1, y2, x1, x2, g1, g2, y);
		return green;
	}


double
Edge::findBlue(double y){
		double blue = interpolate(y1, y2, x1, x2, b1, b2, y);
		return blue;
	}

double
Edge::normalZ(double y){
		double normZ = interpolate(y1, y2, x1, x2, normal1[2], normal2[2], y);
		return normZ;
	}

double
Edge::normalX(double y){
		double normX = interpolate(y1, y2, x1, x2, normal1[0], normal2[0], y);
		return normX;
	}

double
Edge::normalY(double y){
		double normY = interpolate(y1, y2, x1, x2, normal1[1], normal2[1], y);
		return normY;
	}

double
Edge::findShade(double y){
		double normY = interpolate(y1, y2, x1, x2, shade1, shade2, y);
		return normY;
	}


bool
Edge::applicableY(double y){
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

//Triangle Class 

void
Triangle::printTri(){
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

  // would some methods for transforming the triangle in place be helpful?
void
Triangle::scanline(int i, Camera c){
        bool print = false;
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

	Edge e1 = Edge(X[0], Y[0], Z[0], colors[0][0], colors[0][1], colors[0][2], normals[0], shading[0], X[1], Y[1], Z[1], colors[1][0], colors[1][1], colors[1][2], normals[1], shading[1]);
	Edge e2 = Edge(X[1], Y[1], Z[1], colors[1][0], colors[1][1], colors[1][2], normals[1], shading[1], X[2], Y[2], Z[2], colors[2][0], colors[2][1], colors[2][2], normals[2], shading[2]);
	Edge e3 = Edge(X[2], Y[2], Z[2], colors[2][0], colors[2][1], colors[2][2], normals[2], shading[2], X[0], Y[0], Z[0], colors[0][0], colors[0][1], colors[0][2], normals[0], shading[0]);

	double t, rightEnd, leftEnd;
	Edge leftLine, rightLine;
	//loop through Y and find X values, then color the pixels given min and max X found
	for(int y = minY; y <= maxY; y++){
		int row = screen.width*3*y;

		leftEnd = 1000*1000;
		rightEnd = -1000*1000;

		if (e1.relevant){ //not horizontal
			if( e1.applicableY(y)){ // y is within y1 and y2 for e1
				t = e1.findX(y);
				if (t < leftEnd){ //find applicable left edge of triangle for given y
					leftEnd = t;
					leftLine = e1;
				}
				if (t > rightEnd){ //find applicable right edge of triangle for given y
					rightEnd = t;
					rightLine = e1;
				}
			}
		}
		if (e2.relevant){ //not horizontal
			if ( e2.applicableY(y)){ //y is on e2s line
				t = e2.findX(y);
				if (t < leftEnd){
					leftEnd = t;
					leftLine = e2;
				}
				if (t > rightEnd){
					rightEnd = t;
					rightLine = e2;
				}
			}
		}
		if (e3.relevant){ //not horizontal
			if ( e3.applicableY(y)){ //line has given y
				t = e3.findX(y);
				if (t < leftEnd){
					leftEnd = t;
					leftLine = e3;
				}
				if (t > rightEnd){
					rightEnd = t;
					rightLine = e3;
				}
			}
		}

		if(print){
			cout << " leftend " << leftEnd << " rightend " << rightEnd << endl;
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
		double leftZ, rightZ, leftRed, rightRed, leftBlue, rightBlue, leftGreen, rightGreen, leftShading, rightShading;
                double leftNormal[3], rightNormal[3];
		leftZ          = leftLine.findZ((double)y); //leftmost z value of x
		rightZ         = rightLine.findZ((double)y); //rightmost z value of x
		leftRed        = leftLine.findRed((double)y);
		rightRed       = rightLine.findRed((double)y);
		leftBlue       = leftLine.findBlue((double)y);
		rightBlue      = rightLine.findBlue((double)y);
		leftGreen      = leftLine.findGreen((double)y);
		rightGreen     = rightLine.findGreen((double)y);
		leftShading    = leftLine.findShade((double)y);
		rightShading   = rightLine.findShade((double)y);
                leftNormal[0]  = leftLine.normalX((double)y);
                leftNormal[1]  = leftLine.normalY((double)y);
                leftNormal[2]  = leftLine.normalZ((double)y);
                rightNormal[0] = rightLine.normalX((double)y);
                rightNormal[1] = rightLine.normalY((double)y);
                rightNormal[2] = rightLine.normalZ((double)y);
		//loop through all the pixels that have the bottom left in the triangle.

                //cout << "left norms: " << leftNormal[0] << " " << leftNormal[1] << " " << leftNormal[2] << endl;
                //cout << "right norms: " << rightNormal[0] << " " << rightNormal[1] << " " << rightNormal[2] << endl;

		double ratio, z;

		for (int x = minX; x <= maxX; x++){

			if (leftEnd == rightEnd)//don't divide by 0 & do not use rounded x min/max values
				ratio = 1.0;
			else
				ratio = ((double)x - leftEnd)/(rightEnd - leftEnd);//ratio between unrounded x values on the current row (y)

                        double distance = sqrt(pow(centroid[0] - x, 2) + pow(centroid[1] - y,2));
			z = leftZ + ratio*(rightZ - leftZ);
			if(z > screen.zBuff[y*screen.width + x])
                        {
                          screen.triScreen[y*screen.width + x] = i;
                          screen.triCamera[y*screen.width + x] = c.position;
                          /*if(distance <= radius)//inside radius to the centroid
                          {
                            screen.triScreen[y*screen.width + x] = i;
                          }*/

			  double shading = leftShading + ratio*(rightShading - leftShading);
			  if (print)
		            cout << "shading for x y " << x <<  " " << y << " "  << shading << endl;
			  double red, green, blue;
			  red = (leftRed + ratio*(rightRed - leftRed))*shading;
			  if (red > 1.0)
			    red = 1.0;
			  if (red < 0)
			    red = 0;
			  green = (leftGreen + ratio*(rightGreen - leftGreen))*shading;
			  if (green > 1.0)
			    green = 1.0;
			  if (green < 0)
			    green = 0;
			  blue = (leftBlue + ratio*(rightBlue - leftBlue))*shading;
			  if (blue > 1.0)
			    blue = 1.0;
			  if (blue < 0)
			    blue = 0;

			  screen.zBuff[y*screen.width + x] = z;
			  screen.buffer[row + 3*x]     = (unsigned char) ceil441(255.0*red);
		          screen.buffer[row + 3*x + 1] = (unsigned char) ceil441(255.0*green);
			  screen.buffer[row + 3*x + 2] = (unsigned char) ceil441(255.0*blue);
			}
		}
	}
}

double
Triangle::findMin(double a, double b, double c){
		double min = a;
		if (b < min)
			min = b;
		if (c < min)
			min = c;
		return min;
	}

double
Triangle::findMax(double a, double b, double c){
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

double nabs(double x){
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

void normalize(double * normal) {
	double total = pow(normal[0], 2.0) + pow(normal[1], 2.0) + pow(normal[2], 2.0);
	//if (nabs(total) < 0.0001){	
	//	normal[0] = 0;
	//	normal[1] = 0;
	//	normal[2] = 0;
	//}
	//else{
		total = pow(total, 0.5);
		normal[0] = normal[0] / total;
		normal[1] = normal[1] / total;
		normal[2] = normal[2] / total;
	//}
}

double* normalize2(double * normal) {
	double total = pow(normal[0], 2.0) + pow(normal[1], 2.0) + pow(normal[2], 2.0);
	if (nabs(total) < 0.0001){	
		normal[0] = 0;
		normal[1] = 0;
		normal[2] = 0;
	}
	else{
		total = pow(total, 0.5);
		normal[0] = normal[0] / total;
		normal[1] = normal[1] / total;
		normal[2] = normal[2] / total;
	}

return normal;
}

double dotProduct(double* v1, double* v2, int length){
	double dotproduct = 0;	
	for (int i = 0; i < length; i++){
		dotproduct += (v1[i]*v2[i]);
	}
	return dotproduct;
}

double magnitude2d(double* vec)
{
  return sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
}


double magnitude3d(double* vec)
{
  return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}


double* crossProduct(double * a, double * b){
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
GetCamera(int frame, int nframes, double* bounds)
{
    double t = SineParameterize(frame, nframes, nframes/10);
    double points[3];
    fibonacci_sphere(frame, nframes, points);
    Camera c;
    double zoom = 6.0;
    c.near = zoom/8;
    c.far = zoom*5;
    c.angle = M_PI/6;
    //MINE 
    c.position[0] = zoom*points[0] + (bounds[0]+bounds[1])/2;
    c.position[1] = zoom*points[1] + (bounds[2] + bounds[3])/2;
    c.position[2] = zoom*points[2]  + (bounds[4]+bounds[5])/2;
    //Hanks
    //c.position[0] = zoom*sin(2*M_PI*t) + (bounds[0]+bounds[1])/2;
    //c.position[1] = zoom*cos(2*M_PI*t) + (bounds[2] + bounds[3])/2;
    //c.position[2] = zoom;// + (bounds[4]+bounds[5])/2;

//cout << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;

    c.focus[0] = (bounds[0]+bounds[1])/2;
    c.focus[1] = (bounds[2]+bounds[3])/2;
    c.focus[2] = (bounds[4]+bounds[5])/2;
    c.up[0] = 0;
    c.up[1] = 1;
    c.up[2] = 0;
    return c;
}

Camera
GetCamera(int frame, int nframes)
{   
    double t = SineParameterize(frame, nframes, nframes/10);
    double points[3];
    fibonacci_sphere(frame, nframes, points);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    //MINE 
//    c.position[0] = 40*points[0];
//    c.position[1] = 40*points[1];
//    c.position[2] = 40*points[2];
    //Hanks
    double zoom = 5.0;
    c.position[0] = zoom*sin(2*M_PI*t);
    c.position[1] = zoom*cos(2*M_PI*t);
    c.position[2] = zoom;
//cout << "camera position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
    c.focus[0] = 0;
    c.focus[1] = 0;
    c.focus[2] = 0;
    c.up[0] = 0;
    c.up[1] = 1;
    c.up[2] = 0;
    return c;
}

void GetRange(double* range, int size, double* values)
{
  vector<double> vector;
  for(int i = 0; i < size; i++)
    vector.push_back(values[i]);

  sort(vector.begin(), vector.end());
  range[0] = 0;
  range[1] = 0;
  //cout << "vector front and back " << vector.front() << " " << vector.back() << endl;
  range[0] = vector.front();
  range[1] = vector.back();
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
    FieldType v = variable[1];
    cout << "variable " << v << endl;
    output.X[0] = points[0][0];
    output.Y[0] = points[0][1];
    output.Z[0] = points[0][2];
    output.X[1] = points[1][0];
    output.Y[1] = points[1][1];
    output.Z[1] = points[1][2];
    output.X[2] = points[2][0];
    output.Y[2] = points[2][1];
    output.Z[2] = points[2][2]; 
    //output.value[0] = v;
    //output.value[1] = variable[1];
    //output.value[2] = variable[2];
  
    output.printTri();

  }
};


std::vector<Triangle>
GetTriangles(vtkh::DataSet &vtkhData, std::string field_name)
{
    
    //vtkm::cont::Field field = vtkhData->GetField(field_name);
    //Get domain Ids on this rank
    std::vector<vtkm::Id> localDomainIds = vtkhData.GetDomainIds();
    std::vector<Triangle> tris;
    cout << "number of domains: " << localDomainIds.size() << endl;
    //loop through domains and grab all triangles.
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
      cout << "number of cells " << cellset.GetNumberOfCells() << endl;
      std::vector<Triangle> tmp_tris(numTris);
     
      vtkm::cont::ArrayHandle<Triangle> triangles = vtkm::cont::make_ArrayHandle(tmp_tris);
      vtkm::cont::Invoker invoker;
      invoker(ProcessTriangle{}, cellset, coords,field.GetData(), triangles);

      //combine all domain triangles
      tris.insert(tris.end(), tmp_tris.begin(), tmp_tris.end());

    }
    return tris;
}


Triangle transformTriangle(Triangle t, Camera c){
        bool print = false;

	Matrix camToView, m0, cam, view;
	cam = c.CameraTransform();
	view = c.ViewTransform();
	camToView = Matrix::ComposeMatrices(cam, view);
	m0 = Matrix::ComposeMatrices(camToView, c.DeviceTransform());
	if (print){
		cout<< "m0" << endl;
		m0.Print(cout);
	}
  if(print)
  {
    cout << "triangle in: (" << t.X[0] << " , " << t.Y[0] << " , " << t.Z[0] << ") " << endl <<
                        " (" << t.X[1] << " , " << t.Y[1] << " , " << t.Z[1] << ") " << endl <<
                        " (" << t.X[2] << " , " << t.Y[2] << " , " << t.Z[2] << ") " << endl;
  }

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
	//transfor colors and normals
	int i, j;
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			triangle.colors[i][j]  = t.colors[i][j];
			triangle.normals[i][j] = t.normals[i][j];
		}
	}
	for (i = 0; i < 3; i++){
		triangle.shading[i] = t.shading[i];
	}
        triangle.compID = t.compID;

	delete[] pointOut;
	delete[] pointIn;

	return triangle;

}

double CalculateNormalCameraDot(double* cameraPositions, Triangle tri)
{
  double interpolatedNormals[3];
  interpolatedNormals[0] = (tri.normals[0][0] + tri.normals[1][0] + tri.normals[2][0])/3;
  interpolatedNormals[1] = (tri.normals[0][1] + tri.normals[1][1] + tri.normals[2][1])/3;
  interpolatedNormals[2] = (tri.normals[0][2] + tri.normals[1][2] + tri.normals[2][2])/3;

  normalize(interpolatedNormals);
  normalize(cameraPositions);
  return dotProduct(cameraPositions, interpolatedNormals, 3);  
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
    cout << "USING CAMERA PIPELINE" << endl;

    DataObject *data_object = input<DataObject>(0);
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      ASCENT_ERROR("Unknown field '"<<field_name<<"'");
    }
    
    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);
    GetTriangles(dataset,field_name);
//    cout << "dataset bounds: " << dataset.GetGlobalBounds() << endl;
  
    vtkmCamera *camera = new vtkmCamera;
    vtkm::Bounds bounds = dataset.GetGlobalBounds();
    vtkm::Float32 xb = vtkm::Float32(bounds.X.Length());
    vtkm::Float32 yb = vtkm::Float32(bounds.Y.Length());
    vtkm::Float32 zb = vtkm::Float32(bounds.Z.Length());
    //cout << "x y z " << xb << " " << yb << " " << zb << endl;
    vtkm::Float32 radius = sqrt(xb*xb + yb*yb + zb*zb)/2.0;
    //cout << "radius " << radius << endl;
    if(radius<1)
      radius = radius + 1;
    camera->ResetToBounds(dataset.GetGlobalBounds());
    camera->Print();
    double sphere_points[3];
    //TODO: loop through number of samples THEN add parallelism + z buffer.
    //fibonacci_sphere(sample, num_samples, sphere_points);
    fibonacci_sphere(1, 100, sphere_points);
    
    vtkm::Float32 x_pos = sphere_points[0]*radius;// + xb/2.0;
    vtkm::Float32 y_pos = sphere_points[1]*radius;// + yb/2.0;
    vtkm::Float32 z_pos = sphere_points[2]*radius;// + zb/2.0;

    if(abs(x_pos) < radius && abs(y_pos) < radius && abs(z_pos) < radius)
      if(z_pos >= 0)
        z_pos += radius;
      if(z_pos < 0)
        z_pos -= radius;
    vtkm::Vec<vtkm::Float32, 3> pos{x_pos, y_pos, z_pos}; 
    camera->SetPosition(pos);


    if(!graph().workspace().registry().has_entry("camera"))
    {
      cout << "making camera in registry" << endl;
      graph().workspace().registry().add<vtkm::rendering::Camera>("camera",camera,1);
    }


    camera->Print();
    set_output<DataObject>(input<DataObject>(0));
    //set_output<vtkmCamera>(camera);
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
