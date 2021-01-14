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
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/worklet/WorkletMapTopology.h>


#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#endif

#include <chrono>
#include <stdio.h>
#include <math.h>

////openCV
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
/* This is stuff for old camera, leaving it in here for now
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

#include <cmath> // using fmod for modulo on doubles 

void fibonacciSphere(double i, int samples, double* points)
{
  if (i > (samples-1)){
      i = i - (samples - 1);
  }

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

  //if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
 // {
   // if(points[2] >= 0)
   //   points[2] += radius;
   // if(points[2] < 0)
   //   points[2] -= radius;
  //}

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

  //if(abs(points[0]) < radius && abs(points[1]) < radius && abs(points[2]) < radius)
  //{
  //  if(points[2] >= 0)
  //    points[2] += radius;
  //  if(points[2] < 0)
  //    points[2] -= radius;
  //}

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
*/

#include <cmath>
//Use this file stuff later but figured I'd put it next to mine
#include <iostream>
#include <fstream>

// Stuff for random numbers
#include<time.h>

Camera
GetCamera3(double x0, double x1, double y0, double y1, double z0, double z1, double radius,
	       	int thetaPos, int numTheta, int phiPos, int numPhi, double *lookat)
{
  Camera c;
  double zoom = 1.0;
  c.near = zoom/20;
  c.far = zoom*25;
  c.angle = M_PI/6;

  /* New version that didn't work
  double theta = 2 * M_PI * (thetaPos / (numTheta - 1.0));
  double phi = acos(1 - 2 * (phiPos / (numPhi - 1)));

  double xmid = (x0 + x1) / 2.0;
  double ymid = (y0 + y1) / 2.0;
  double zmid = (z0 + z1) / 2.0;

  double x = sin(phi) * cos(theta);
  double y = sin(phi) * sin(theta);
  double z = cos(phi);

  c.position[0] = (zoom * 3 * radius * x) + xmid;
  c.position[1] = (zoom * 3 * radius * y) + ymid;
  c.position[2] = (zoom * 3 * radius * z) + zmid;
  */

  ///* This is our old version, saved just in case 
  double theta = (thetaPos / (numTheta - 1.0)) * M_PI ;
  double phi = (phiPos / (numPhi - 1.0)) * M_PI * 2.0; 
  double xm = (x0 + x1) / 2.0;
  double ym = (y0 + y1) / 2.0;
  double zm = (z0 + z1) / 2.0;

  c.position[0] = (  zoom*3*radius * sin(theta) * cos(phi)  + xm );
  c.position[1] = (  zoom*3*radius * sin(theta) * sin(phi)  + ym );
  c.position[2] = (  zoom*3*radius * cos(theta)  + zm );
  //*/

  //check lookat vs middle
  //cerr << "xm ym zm : " << xm <<  " " << ym << " " << zm << endl;
  //cerr << "lookat: " << lookat[0] << " " << lookat[1] << " " << lookat[2] << endl;
  //cerr << "position: " << c.position[0] << " " << c.position[1] << " " << c.position[2] << endl;
  c.focus[0] = lookat[0];
  c.focus[1] = lookat[1];
  c.focus[2] = lookat[2];
  c.up[0] = 0;
  c.up[1] = 1;
  c.up[2] = 0;
  return c;
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
    valid_paths.push_back("sample");
    valid_paths.push_back("i");
    valid_paths.push_back("j");
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
    double time_var = 0.;
    auto time_start = high_resolution_clock::now();
    //cout << "USING SIMPLEX PIPELINE" << endl;
    #if ASCENT_MPI_ENABLED
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif  

    #if defined(ASCENT_VTKM_ENABLED)
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
    //std::vector<Triangle> triangles;// = GetTriangles2(dataset,field_name);
    std::vector<Triangle> triangles = GetTriangles(dataset);
    float total_triangles = (float) triangles.size();
    vtkh::DataSet* data = AddTriangleFields(dataset);
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
    float bounds[3] = {(float)xb, (float)yb, (float)zb};
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

    double losing_score = DBL_MAX;
    int losing_i = -1;
    int losing_j = -1;

    // New theta and phi camera code
    //int numTheta = 100;
    //int numPhi = 100;

/* 
    // Code for 20 images from spiral, must change sample from 0->20 in the yaml file 
    int sample = (int)params()["sample"].as_int64();
    
    cout << "Getting picture for sample: " << sample << endl;

    Camera cam = GetCamera(sample, samples, radius, focus, bounds);  

    vtkm::Vec<vtkm::Float32, 3> postest{(float)cam.position[0],
                                      (float)cam.position[1],
                                      (float)cam.position[2]};

    camera->SetPosition(postest);
    vtkh::ScalarRenderer tracer;
    tracer.SetWidth(width);
    tracer.SetHeight(height);
    tracer.SetInput(data); //vtkh dataset by toponame
    tracer.SetCamera(*camera);
    tracer.Update();

    vtkh::DataSet *output = tracer.GetOutput();

    float score = calculateMetric(output, metric, field_name,
                 		     triangles, height, width, cam);

*/

///*
    // Code for getting all scores from the 20 image spirals
    int sample = (int)params()["sample"].as_int64();
    
    int metric_num = 0;
    string metrics[] = {"data_entropy", "depth_entropy", "max_depth",
	                  "pb", "projected_area", "viewpoint_entropy", 
			  "visibility_ratio", "visible_triangles", "vkl"};
    
    for (metric_num ; metric_num < 9 ; metric_num++) {
      metric = metrics[metric_num];
      string filename = metrics[metric_num];
      filename += "_scores.txt"; 

      ofstream myfile;
      myfile.open(filename);
       
      double known_min = DBL_MAX;
      double known_max = -DBL_MAX;
      int number = 0;

      cout << endl << "Gathering max and min data for: " << metric << endl;

      // First loop, find min and max
      for (int i = 0 ; i < 20 ; ++i) {

          Camera cam = GetCamera(i, samples, radius, focus, bounds);  

          vtkm::Vec<vtkm::Float32, 3> postest{(float)cam.position[0],
                                    (float)cam.position[1],
                                    (float)cam.position[2]};
  
          camera->SetPosition(postest);
          vtkh::ScalarRenderer tracer;
          tracer.SetWidth(width);
          tracer.SetHeight(height);
          tracer.SetInput(data); //vtkh dataset by toponame
          tracer.SetCamera(*camera);
          tracer.Update();

          vtkh::DataSet *output = tracer.GetOutput();

          float score = calculateMetric(output, metric, field_name,
  		          triangles, height, width, cam);

   	  if (score < known_min) {
            known_min = score;
	  }

	  if (score > known_max) {
            known_max = score;
	  }

          cout << "Natural score for sample " << i << " is " << score << endl;
      }

      cout << endl << "Writing score file for: " << metric << endl;

      // Second loop, put relative scores in file
      for (int i = 0 ; i < 20 ; ++i) {

          Camera cam = GetCamera(i, samples, radius, focus, bounds);  

          vtkm::Vec<vtkm::Float32, 3> postest{(float)cam.position[0],
                                    (float)cam.position[1],
                                    (float)cam.position[2]};

          camera->SetPosition(postest);
          vtkh::ScalarRenderer tracer;
          tracer.SetWidth(width);
          tracer.SetHeight(height);
          tracer.SetInput(data); //vtkh dataset by toponame
          tracer.SetCamera(*camera);
          tracer.Update();

          vtkh::DataSet *output = tracer.GetOutput();

          float score = calculateMetric(output, metric, field_name,
	  	          triangles, height, width, cam);

          float relative = (score - known_min) / (known_max - known_min);
	  float result = relative * 10;

          myfile << result << endl;
          
          cout << "Relative score for sample " << i << " is " << result << endl;

          number += 377;
      }

      myfile.close();

    }

//*/ 

/*
    // Main block for theta and phi
    cout << "Gathering data for metric: " << metric.c_str() << endl;

    // Check for i and j before main loop
    int yaml_i = (int)params()["i"].as_int64();
    int yaml_j = (int)params()["j"].as_int64();
    cout << "yaml i: " << yaml_i << " , yaml j: " << yaml_j << endl;

    if ((yaml_i >= 0) && (yaml_j >= 0)) {
      // Skip main loop
      cout << "i and j positive so skipping loop" << endl;

      winning_i = yaml_i;
      winning_j = yaml_j;

      Camera cam = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
          	        radius, winning_i, numTheta, winning_j, numPhi, focus); 

      vtkm::Vec<vtkm::Float32, 3> postest{(float)cam.position[0],
                                    (float)cam.position[1],
                                    (float)cam.position[2]};

      camera->SetPosition(postest);
      vtkh::ScalarRenderer tracer;
      tracer.SetWidth(width);
      tracer.SetHeight(height);
      tracer.SetInput(data); //vtkh dataset by toponame
      tracer.SetCamera(*camera);
      tracer.Update();

      vtkh::DataSet *output = tracer.GetOutput();

      float score = calculateMetric(output, metric, field_name,
                 		     triangles, height, width, cam);

      cout << "Score at (" << winning_i << ", " << winning_j << ") is " << score << endl << endl;
    }

    else {
      // Main lopp
      cout << "i or j negative so running loop" << endl;


      // File stuff
      FILE *datafile;
      float buffer[numTheta][numPhi];

      // Get nice filename
      char dataFileName[metric.length() + 5];
      strcpy(dataFileName, metric.c_str());
      dataFileName[metric.length()] = '.';
      dataFileName[metric.length() + 1] = 'b';
      dataFileName[metric.length() + 2] = 'i';
      dataFileName[metric.length() + 3] = 'n';
      dataFileName[metric.length() + 4] = '\0';
  
      datafile = fopen(dataFileName, "wb");

      for (int i = 0 ; i < numTheta ; i++) {
        cout << "Step: " << i << endl;
        cout << "  Current Winning Score: " << winning_score << endl;
        cout << "  Current Losing Score: " << losing_score << endl;
        for (int j = 0 ; j < numPhi ; j++) {

          Camera cam = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
		       	        radius, i, numTheta, j, numPhi, focus); 

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

          float score = calculateMetric(output, metric, field_name,
		       triangles, height, width, cam);

          buffer[i][j] = score;

  	 delete output;

   	 //cout << "Camera at: " << cam.position[0] << ", " << cam.position[1] << ", " << cam.position[2] << endl;
         //cout << "Score is: " << score << endl << endl;
	 if (score > winning_score) {
              winning_score = score;
              winning_i = i;
              winning_j = j;
          }

	 if (score < losing_score) {
              losing_score = score;
              losing_i = i;
              losing_j = j;
         }

        }
      }

      cout << "Winning score: " << winning_score << " at (" << winning_i << ", " << winning_j << ")" << endl;
      cout << "Losing score: " << losing_score << " at (" << losing_i << ", " << losing_j << ")" << endl;

      for (int k = 0 ; k < numTheta ; k++) {
        fwrite(buffer[k], sizeof(float), numPhi, datafile);
      }

      fclose(datafile);
  }

*/ 

    /*================ End Scalar Renderer  ======================*/

/*
    // Code for scores for quizzes with theta and phi
  
    	
    int metric_num = 0;
    string metrics[] = {"data_entropy", "depth_entropy", "max_depth",
	                  "pb", "projected_area", "viewpoint_entropy",
			  "visibility_ratio", "visible_triangles", "vkl"};
    
    for (metric_num ; metric_num < 9 ; metric_num++) {
      metric = metrics[metric_num];
      string filename = metrics[metric_num];
      filename += "_scores.txt"; 

      ofstream myfile;
      myfile.open(filename);
       
      double known_min = DBL_MAX;
      double known_max = -DBL_MAX;
      int number = 0;

      cout << endl << "Gathering max and min data for: " << metric << endl;

      // First loop, find min and max
      while (number < 10000) {
          winning_i = number / 100;
          winning_j = number % 100;

          Camera cam = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
          	        radius, winning_i, numTheta, winning_j, numPhi, focus); 

          vtkm::Vec<vtkm::Float32, 3> postest{(float)cam.position[0],
                                    (float)cam.position[1],
                                    (float)cam.position[2]};
  
          camera->SetPosition(postest);
          vtkh::ScalarRenderer tracer;
          tracer.SetWidth(width);
          tracer.SetHeight(height);
          tracer.SetInput(data); //vtkh dataset by toponame
          tracer.SetCamera(*camera);
          tracer.Update();

          vtkh::DataSet *output = tracer.GetOutput();

          float score = calculateMetric(output, metric, field_name,
  		          triangles, height, width, cam);

   	  if (score < known_min) {
            known_min = score;
	  }

	  if (score > known_max) {
            known_max = score;
	  }

          cout << "Natural score at (" << winning_i << ", " << winning_j << ") is " << score << endl;

          number += 377;
      }

      number = 0;

      cout << endl << "Writing score file for: " << metric << endl;

      // Second loop, put relative scores in file
      while (number < 10000) {
          winning_i = number / 100;
          winning_j = number % 100;

          Camera cam = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
          	        radius, winning_i, numTheta, winning_j, numPhi, focus); 

          vtkm::Vec<vtkm::Float32, 3> postest{(float)cam.position[0],
                                    (float)cam.position[1],
                                    (float)cam.position[2]};

          camera->SetPosition(postest);
          vtkh::ScalarRenderer tracer;
          tracer.SetWidth(width);
          tracer.SetHeight(height);
          tracer.SetInput(data); //vtkh dataset by toponame
          tracer.SetCamera(*camera);
          tracer.Update();

          vtkh::DataSet *output = tracer.GetOutput();

          float score = calculateMetric(output, metric, field_name,
	  	          triangles, height, width, cam);

          float relative = (score - known_min) / (known_max - known_min);
	  float result = relative * 10;

          myfile << result << endl;
          
          cout << "Relative score at (" << winning_i << ", " << winning_j << ") is " << result << endl;

          number += 377;
      }

      myfile.close();

    } 

*/


    //Camera best_c = GetCamera3(xMin, xMax, yMin, yMax, zMin, zMax,
    //		       	        radius, winning_i, numTheta, winning_j, numPhi, focus);
    
    Camera best_c = GetCamera(sample, samples, radius, focus, bounds);  

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
    #endif
    set_output<DataObject>(input<DataObject>(0));
    //set_output<vtkmCamera>(camera);
    auto time_stop = high_resolution_clock::now();
    time_var += duration_cast<seconds>(time_stop - time_start).count();

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
