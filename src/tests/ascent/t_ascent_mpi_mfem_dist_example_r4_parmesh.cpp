//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_mfem_dist_example.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>
#include "mfem.hpp"
#include <conduit_blueprint.hpp>
#include <conduit_relay_mpi.hpp>
#include <conduit_blueprint_mpi.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;


//
void mpi_distribute(const Node &src_mesh,
                    const Node &opts,
                    Node &dest_mesh,
                    MPI_Comm comm)
{
    // get par_rank and par_size
    int par_rank = conduit::relay::mpi::rank(comm);
    int par_size = conduit::relay::mpi::size(comm);
    
    // gen domain to rank map
    // note: this only handles non domain overloaded inputs cases
    Node d2r_map;
    conduit::blueprint::mpi::mesh::generate_domain_to_rank_map(src_mesh,
                                                               d2r_map,
                                                               comm);

    if(par_rank ==0)
    {
        d2r_map.print();
    }

    index_t_accessor d2r_vals = d2r_map.value();

    // local domain pointers
    std::vector<const Node *> local_domains = ::conduit::blueprint::mesh::domains(src_mesh);
    // map from domain id to local domain index
    std::map<index_t,index_t> local_domain_ids;

    // populate domain id to local domain index map
    for(index_t local_domain_idx = 0;
        local_domain_idx < (index_t)local_domains.size();
        local_domain_idx++)
    {
        index_t domain_id = par_rank;
        const conduit::Node *domain = local_domains[local_domain_idx];
        // see if we have read domain  id
        if(domain->has_child("state") && domain->fetch("state").has_child("domain_id"))
        {
            domain_id = domain->fetch("state/domain_id").to_index_t();
        }

        local_domain_ids[domain_id] = local_domain_idx;
    }

    // clear output mesh
    dest_mesh.reset();

    const Node &domain_map = opts["domain_map"];
    // domain map is an o2m
    // full walk the map to queue sends and recvs
    blueprint::o2mrelation::O2MIterator o2m_iter(domain_map);
    // O2MMap o2m_rel(domain_map);
    index_t_accessor dmap_values = domain_map["values"].value();

    conduit::relay::mpi::communicate_using_schema isr(comm);
    // isr.set_logging(true);
    int tag = 422000; // unique tag start for dist

    // full walk
    while(o2m_iter.has_next(conduit::blueprint::o2mrelation::DATA))
    // for( int i = 0; i < o2m_rel.size(); i++)
    {
        int i = o2m_iter.next(conduit::blueprint::o2mrelation::ONE);
        if(par_rank ==0)
        std::cout << "PROC DOMAIN " << i << std::endl;
        // i is domain id
        // check if we have domain w/ domain id == i
        // if so we will send
        bool have_domain = local_domain_ids.find(i) != local_domain_ids.end();
        // loop over all dests for domain i
        o2m_iter.to_front(conduit::blueprint::o2mrelation::MANY);
        while(o2m_iter.has_next(conduit::blueprint::o2mrelation::MANY))
        // for (int j = 0; j < o2m_rel.size(i); j++)
        {
            o2m_iter.next(conduit::blueprint::o2mrelation::MANY);
            // index_t o2m_idx = o2m_rel.map(i,j);
            index_t o2m_idx  = o2m_iter.index(conduit::blueprint::o2mrelation::DATA);
            index_t des_rank = dmap_values[o2m_idx];
            if(par_rank ==0)
            std::cout << "des_rank = " << des_rank << std::endl;
            if(have_domain)
            {
                const Node &send_dom = *local_domains[local_domain_ids[i]];
                if(par_rank == des_rank)
                {
                    std::cout << "rank " << par_rank << " self send domain " << i <<std::endl;
                    // self send ... simply copy out
                    dest_mesh.append().set(send_dom);
                }
                else
                {
                    std::cout << "rank " << par_rank << " qsend domain " << i
                              << " to " << des_rank
                              << " tag = " << tag << std::endl;
                    // queue send of domain
                    isr.add_isend(send_dom,des_rank,tag);
                }
            }
            else if(par_rank == des_rank)
            {
                // look up who is sending
                index_t send_rank = d2r_vals[i];
                // queue recv of domain
                std::cout << "rank " << par_rank << " qrecv domain " << i << " from " << send_rank
                          << " tag = " << tag << std::endl;
                Node &res_domain = dest_mesh.append();
                isr.add_irecv(res_domain,send_rank,tag);
            }
            //this count allows each pair to have a unique tag
            tag++;
        }
    }
    isr.execute();
}


mfem::ParMesh *load_parmesh(MPI_Comm comm)
{
   
    // mfem::DataCollection *dc = new mfem::VisItDataCollection("/Users/harrison37/Work/alpine/ascent/ex01-fichera");
 //    dc->SetPadDigitsCycle(-1);
 //    dc->SetPadDigitsRank(-1);
 //    dc->Load(0);
 //
 //     if (dc->Error() != mfem::DataCollection::NO_ERROR)
 //     {
 //
 //      }
 //      else
 //      {
 //          std::cout << "OK!" << std::endl;
 //      }
    
    mfem::Mesh m = mfem::Mesh::LoadFromFile("/Users/harrison37/Work/alpine/ascent/ex01-fichera.mesh");
    
    mfem::ParMesh *par_mesh = new  mfem::ParMesh(comm,m);
    
    return par_mesh;
}


//-----------------------------------------------------------------------------
TEST(ascent_mpi_mfem_dist_example, test_4_ranks_parmesh)
{
    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    MPI_Barrier(comm);

    
    // load a mfem dataset
    mfem::ParMesh *pmesh = load_parmesh(comm);
    // create a parmesh from it 
    conduit::Node data;

    // wrap the par mesh into a conduit node


    mfem::ConduitDataCollection::MeshToBlueprintMesh(pmesh,data);
    
    conduit::relay::mpi::io::blueprint::save_mesh(data,"tout_mpi_dist_input","hdf5",comm);


    Node opts,res;
    opts["domain_map/values"] = {0,1,   //  domain 0 on ranks 0,1
                                 1,2,   //  domain 1 on ranks 1,2
                                 2,3,   //  domain 2 on ranks 2,3
                                 3,0};  //  domain 3 on ranks 3,0
    opts["domain_map/sizes"]   = {2,2,2,2};
    opts["domain_map/offsets"] = {0,2,4,6};
    mpi_distribute(data,opts,res,comm);

    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_result","hdf5",comm);

    // if(par_rank == 0)
    // {
    //     EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    //     EXPECT_EQ(res[1]["state/domain_id"].to_index_t(),3);
    // }
    // else if(par_rank == 1)
    // {
    //     EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    //     EXPECT_EQ(res[1]["state/domain_id"].to_index_t(),1);
    // }
    // else if(par_rank == 2)
    // {
    //     EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),1);
    // }
    // else if(par_rank == 2)
    // {
    //     EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    // }

}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}


