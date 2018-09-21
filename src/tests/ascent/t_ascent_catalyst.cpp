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
/// file: t_ascent_flow_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <ascent_main_runtime.hpp>

#include "ascent_vtk_data_adapter.hpp"
#include "ascent_runtime_vtkh_filters.hpp"

#include "vtkIndent.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkNew.h"
#include "vtkDataObject.h"
#include "vtkXMLMultiBlockDataWriter.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkh/rendering/MeshRenderer.hpp>
#include <vtkh/rendering/VolumeRenderer.hpp>
#include <vtkh/filters/Clip.hpp>
#include <vtkh/filters/ClipField.hpp>
#include <vtkh/filters/IsoVolume.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/filters/NoOp.hpp>
#include <vtkh/filters/Slice.hpp>
#include <vtkh/filters/Threshold.hpp>
#include <vtkm/cont/DataSet.h>

#include <ascent_vtkh_data_adapter.hpp>


#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
class CatalystExtract: public ::flow::Filter
{
public:
  static bool s_was_called;

  static void reset_was_called(){ s_was_called = false;}
  static bool was_called(){ return s_was_called;}

  CatalystExtract():Filter()
  {}
  ~CatalystExtract()
  {}

  void declare_interface(Node &i)
  {
    i["type_name"]   = "my_noop_extract";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
  }

  void execute()
  {
    s_was_called = true;
    //ascent::about(input(0));
    if(input(0).check_type<Node>())
    {
      std::cerr
        << "\n\n===***%%%%%%\n"
        << "We have a blueprint?\n";

      // convert from blueprint to vtk
      const Node* nd = input<Node>(0);
      std::cerr << "\nNode " << nd << "\n";
      conduit::Node info;
      bool success = conduit::blueprint::verify("mesh",*nd,info);

      if(!success)
      {
        info.print();
        ASCENT_ERROR("conduit::Node input to EnsureBlueprint is non-conforming")
      }

      auto vtk_data = VTKDataAdapter::BlueprintToVTKMultiBlock(
        *nd, /*zero_copy*/ true, /*topo_name*/ "");
      if (vtk_data)
      {
        vtkIndent indent;
        vtk_data->PrintSelf(std::cout, indent);
        vtkNew<vtkXMLMultiBlockDataWriter> wri;
        wri->SetInputDataObject(vtk_data);
        wri->SetFileName("/tmp/foob.vtm");
        wri->Write();
      }
      else
      {
        std::cout << "No VTK conversion\n";
      }

      auto nit = nd->children();
      while (nit.has_next())
      {
        const Node& nn = nit.next();
        NodeConstIterator itr = nn["topologies"].children();
        itr.next();
        std::string topo_name = itr.name();

        const Node &n_topo   = nn["topologies"][topo_name];
        string mesh_type     = n_topo["type"].as_string();

        string coords_name   = n_topo["coordset"].as_string();
        const Node &n_coords = nn["coordsets"][coords_name];

        std::cout
          << "***\nExtract exec.. topo " << topo_name
          << " mesh_type " << mesh_type
          << " coordset name " << coords_name
          << "\n***\n";
      }
    }
    else if (input(0).check_type<vtkDataObject>())
    {
      std::cout << "Got VTK data!\n";
    }

    set_output(input(0));
  }
};

bool CatalystExtract::s_was_called = false;

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_register_extract)
{
    AscentRuntime::register_filter_type<CatalystExtract>("extracts","my_extract");

    conduit::Node extracts;
    conduit::Node actions;
    conduit::Node pipelines;

    // pipeline 1
    pipelines["pl1/f1/type"] = "ensure_vtk";

    extracts["e1/type"]  = "my_extract";
    // extracts["e1/pipeline"]  = "pl1";


    auto& convert = actions.append();
    convert["action"] = "add_filter";
    convert["type_name"] = "ensure_vtk";
    convert["name"] = "vtk_data";

    auto& add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    auto& connect = actions.append();
    connect["action"] = "connect";
    connect["src"]  = "vtk_data";
    connect["dest"] = extracts;
    /*
    */

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    Node data, info;
    conduit::blueprint::mesh::examples::braid("quads",
                                               5,
                                               5,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,info));

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    CatalystExtract::reset_was_called();
    EXPECT_FALSE(CatalystExtract::was_called());
    ascent.execute(actions);
    EXPECT_TRUE(CatalystExtract::was_called());
    // ascent.info(info);
    // info.print();
    ascent.close();
}
