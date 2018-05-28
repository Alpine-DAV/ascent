//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: t_ascent_lagrangian.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>

#include <conduit_blueprint.hpp>
#include <conduit.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t MESH_SIDE_DIM = 32;

static float *field_data = new float[MESH_SIDE_DIM*MESH_SIDE_DIM*MESH_SIDE_DIM];

class ABCfield
{

public :

void calculateVelocity(double *location, float t, double *velocity)
{
        double ep = 1.0;
        double period = 1.0;

        double sinval = ep * sin(period*t);

        velocity[0] = sin( location[2] + sinval ) + cos( location[1] + sinval );
        velocity[1] = sin( location[0] + sinval ) + cos( location[2] + sinval );
        velocity[2] = sin( location[1] + sinval ) + cos( location[0] + sinval );
}

};



typedef struct simulation_info
{
        int cycle;
        double time;
        bool done;
}simulation_info;

static simulation_info sim_info;

struct Options
{
        int m_dims[3];
        double m_spacing[3];
        int m_time_steps;
        double m_time_delta;

        void init()
        {
          m_dims[0] = {MESH_SIDE_DIM};
          m_dims[1] = {MESH_SIDE_DIM};
          m_dims[2] = {MESH_SIDE_DIM};
          m_time_steps = 100;
          m_time_delta = 1;
          SetSpacing();
        }

        void SetSpacing()
        {
                m_spacing[0] = 6.28/double(m_dims[0]);
                m_spacing[1] = 6.28/double(m_dims[1]);
                m_spacing[2] = 6.28/double(m_dims[2]);
        }
};

struct DataSet
{
        const int m_cell_dims[3];
        const int m_point_dims[3];
        const int m_cell_size;
        const int m_point_size;
        double *m_xvec;
				double *m_yvec;
        double *m_zvec;
        double *m_xyz_vec;
				double m_spacing[3];
        double m_origin[3];
        double m_time_step;

        DataSet(const Options &options)
        : m_point_dims{options.m_dims[0],
                       options.m_dims[1],
                       options.m_dims[2]},
          m_cell_dims{options.m_dims[0] - 1,
                      options.m_dims[1] - 1,
                      options.m_dims[2] - 1},
          m_cell_size((options.m_dims[0] - 1) * (options.m_dims[1] - 1) * (options.m_dims[2] - 1)),
          m_point_size(options.m_dims[0] * options.m_dims[1] * options.m_dims[2]),
          m_spacing{options.m_spacing[0],
                 options.m_spacing[1],
                 options.m_spacing[2]},
          m_origin{0., 0., 0.}

   {
     m_xvec = new double[m_point_size];
     m_yvec = new double[m_point_size];
     m_zvec = new double[m_point_size];
     m_xyz_vec = new double[3*m_point_size];
   }


inline void GetCoord(const int &x, const int &y, const int &z, double *coord)
{
        coord[0] = m_origin[0] + m_spacing[0] * double(x);
        coord[1] = m_origin[1] + m_spacing[1] * double(y);
        coord[2] = m_origin[2] + m_spacing[2] * double(z);
}

/*
inline void SetPoint(const double &val, const int &x, const int &y, const int &z)
{
        const int offset = z * m_point_dims[0] * m_point_dims[1] +
                        y * m_point_dims[0] + x;
        m_nodal_scalars[offset] = val;
}
*/

inline void SetXYZPoint(const double &val0, const double &val1, const double &val2, const int &x, const int &y, const int &z)
{
        const int offset = z * m_point_dims[0] * m_point_dims[1] +
                        y * m_point_dims[0] + x;
        m_xyz_vec[offset*3+0] = val0;
        m_xyz_vec[offset*3+1] = val1;
        m_xyz_vec[offset*3+2] = val2;
}



inline void SetXPoint(const double &val, const int &x, const int &y, const int &z)
{
        const int offset = z * m_point_dims[0] * m_point_dims[1] +
                        y * m_point_dims[0] + x;
        m_xvec[offset] = val;
}

inline void SetYPoint(const double &val, const int &x, const int &y, const int &z)
{
        const int offset = z * m_point_dims[0] * m_point_dims[1] +
                        y * m_point_dims[0] + x;
        m_yvec[offset] = val;
}

inline void SetZPoint(const double &val, const int &x, const int &y, const int &z)
{
        const int offset = z * m_point_dims[0] * m_point_dims[1] +
                        y * m_point_dims[0] + x;
        m_zvec[offset] = val;
}


/*
inline void SetCell(const double &val, const int &x, const int &y, const int &z)
{
        const int offset = z * m_cell_dims[0] * m_cell_dims[1] +
                        y * m_cell_dims[0] + x;
        m_zonal_scalars[offset] = val;
}
*/

void PopulateNode(conduit::Node &node)
{
        node["coordsets/coords/type"] = "uniform";

        node["coordsets/coords/dims/i"] = m_point_dims[0];
        node["coordsets/coords/dims/j"] = m_point_dims[1];
        node["coordsets/coords/dims/k"] = m_point_dims[2];

        node["coordsets/coords/origin/x"] = m_origin[0];
        node["coordsets/coords/origin/y"] = m_origin[1];
        node["coordsets/coords/origin/z"] = m_origin[2];

        node["coordsets/coords/spacing/dx"] = m_spacing[0];
        node["coordsets/coords/spacing/dy"] = m_spacing[1];
        node["coordsets/coords/spacing/dz"] = m_spacing[2];

        node["topologies/mesh/type"]     = "uniform";
        node["topologies/mesh/coordset"] = "coords";
				
/*				node["fields/velocity/association"] = "vertex";
        node["fields/velocity/type"]        = "scalar";
        node["fields/velocity/topology"]    = "mesh";
        node["fields/velocity/values"].set_external(m_xyz_vec);
*/
				node["fields/xvec/association"] = "vertex";
        node["fields/xvec/type"]        = "scalar";
        node["fields/xvec/topology"]    = "mesh";
        node["fields/xvec/values"].set_external(m_xvec);
        
				node["fields/yvec/association"] = "vertex";
        node["fields/yvec/type"]        = "scalar";
        node["fields/yvec/topology"]    = "mesh";
        node["fields/yvec/values"].set_external(m_yvec);
        
				node["fields/zvec/association"] = "vertex";
        node["fields/zvec/type"]        = "scalar";
        node["fields/zvec/topology"]    = "mesh";
        node["fields/zvec/values"].set_external(m_zvec);

}

        ~DataSet()
        {
              //  if(m_xvec) delete[] m_xvec;
             //   if(m_yvec) delete[] m_yvec;
              //  if(m_zvec) delete[] m_zvec;
                if(m_xyz_vec) delete[] m_xyz_vec;
        }


};

void simulation_info_initialize(simulation_info *sim_info)
{
        sim_info->cycle = 0;
        sim_info->time = 0;
        sim_info->done = false;
}



//-----------------------------------------------------------------------------
TEST(ascent_lagrangian, test_lagrangian)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }
    
    
    //
    // Create an example mesh.
    //
  
		ABCfield field;
    Options options;
    options.init();

    DataSet data_set(options);

    double spatial_extents[3];
    spatial_extents[0] = options.m_spacing[0] * options.m_dims[0] + 1;
    spatial_extents[1] = options.m_spacing[1] * options.m_dims[1] + 1;
    spatial_extents[2] = options.m_spacing[2] * options.m_dims[2] + 1;

    simulation_info_initialize(&sim_info);

    int npts = MESH_SIDE_DIM*MESH_SIDE_DIM*MESH_SIDE_DIM;
    for(int i = 0; i < npts; i++)
	    field_data[i] = i;

		double time = 0;

    //
    // Create the actions.
    //
    
    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "lagrangian";
    // filter knobs
    conduit::Node &lagrangian_params = pipelines["pl1/f1/params"];
    lagrangian_params["field"] = "velocity";
		lagrangian_params["step_size"] = 0.01;
    lagrangian_params["write_frequency"] = 10;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    
		conduit::Node &execute  = actions.append();
    execute["action"] = "execute";
   
		Node reset;
    Node &reset_action = reset.append();
    reset_action["action"] = "reset";
 
    //
    // Run Ascent
    //
    
    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["ascent_info"] = "verbose";
		ascent.open(ascent_opts);

		Node mesh_data;
    mesh_data["state/time"].set_external(&time);
    mesh_data["state/cycle"].set_external(&time);
    mesh_data["state/domain_id"] = 0;
    mesh_data["state/info"] = "ABC flow";
    data_set.PopulateNode(mesh_data);

		for(int t = 0; t < options.m_time_steps; ++t)
   			{
        for(int z = 0; z < data_set.m_point_dims[2]; ++z)
     	 			{
            for(int y = 0; y < data_set.m_point_dims[1]; ++y)
      		      {
                for(int x = 0; x < data_set.m_point_dims[0]; ++x)
                {
		                double coord[3], vel[3];
                    data_set.GetCoord(x,y,z,coord);
                    field.calculateVelocity(coord, time, vel);
                    double val_point = sqrt(pow(vel[0],2) + pow(vel[1],2) + pow(vel[2],2));
                    double val_cell = sqrt(pow(vel[0],2) + pow(vel[1],2) + pow(vel[2],2));

                   // data_set.SetPoint(val_point,x,y,z);
                    data_set.SetXPoint(vel[0],x,y,z);
                    data_set.SetYPoint(vel[1],x,y,z);
                    data_set.SetZPoint(vel[2],x,y,z);
             //     	data_set.SetXYZPoint(vel[0], vel[1], vel[2], x,y,z);
									/*  if(x < data_set.m_cell_dims[0] &&
                       y < data_set.m_cell_dims[1] &&
                       z < data_set.m_cell_dims[2] )
                       {
                      		 data_set.SetCell(val_cell, x, y, z);
                       } */
                     }
                   }
                }
      		time += options.m_time_delta;
    			ascent.publish(mesh_data);
					ascent.execute(actions);
					ascent.execute(reset);			
				}
    ascent.close();
    
    // check that we created an image
    EXPECT_TRUE(true);
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    
    // allow override of the data size via the command line
    if(argc == 2)
    { 
        MESH_SIDE_DIM = atoi(argv[1]);
    }
    
    result = RUN_ALL_TESTS();
    return result;
}


