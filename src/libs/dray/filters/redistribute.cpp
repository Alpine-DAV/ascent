#include <dray/filters/redistribute.hpp>
#include <dray/dray_node_to_dataset.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/utils/data_logger.hpp>

#include <conduit.hpp>
#include <algorithm>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>

#define DRAY_CHECK_MPI_ERROR( check_mpi_err_code )                  \
{                                                                   \
    if( static_cast<int>(check_mpi_err_code) != MPI_SUCCESS)        \
    {                                                               \
        char check_mpi_err_str_buff[MPI_MAX_ERROR_STRING];          \
        int  check_mpi_err_str_len=0;                               \
        MPI_Error_string( check_mpi_err_code ,                      \
                         check_mpi_err_str_buff,                    \
                         &check_mpi_err_str_len);                   \
                                                                    \
        DRAY_ERROR("MPI call failed: \n"                            \
                      << " error code = "                           \
                      <<  check_mpi_err_code  << "\n"               \
                      << " error message = "                        \
                      <<  check_mpi_err_str_buff << "\n");          \
    }                                                               \
}

#endif

namespace dray
{

namespace detail
{

void pack_grid_function(conduit::Node &n_gf,
                        std::vector<std::pair<size_t,unsigned char*>> &gf_ptrs)
{
  size_t values_bytes = n_gf["values"].total_bytes_compact();
  unsigned char * values_ptr = (unsigned char*)n_gf["values"].data_ptr();

  std::pair<size_t, unsigned char*> value_pair;
  value_pair.first = values_bytes;
  value_pair.second = values_ptr;

  gf_ptrs.push_back(value_pair);

  size_t conn_bytes = n_gf["conn"].total_bytes_compact();
  unsigned char * conn_ptr = (unsigned char*)n_gf["conn"].data_ptr();

  std::pair<size_t, unsigned char*> conn_pair;
  conn_pair.first = conn_bytes;
  conn_pair.second = conn_ptr;

  gf_ptrs.push_back(conn_pair);
}

void pack_dataset(conduit::Node &n_dataset,
                  std::vector<std::pair<size_t,unsigned char*>> &gf_ptrs)
{
  const int32 num_meshes = n_dataset["meshes"].number_of_children();
  for(int32 i = 0; i < num_meshes; ++i)
  {
    conduit::Node &n_mesh = n_dataset["meshes"].child(i);
    pack_grid_function(n_mesh["grid_function"], gf_ptrs);
  }

  const int32 num_fields = n_dataset["fields"].number_of_children();
  for(int32 i = 0; i < num_fields; ++i)
  {
    conduit::Node &field = n_dataset["fields"].child(i);
    pack_grid_function(field["grid_function"], gf_ptrs);
  }
}

void strip_helper(conduit::Node &node)
{
  const int32 num_children = node.number_of_children();

  if(num_children == 0)
  {
    if(node.name() == "conn" || node.name() == "values")
    {
      node.reset();
    }
  }

  for(int32 i = 0; i < num_children; ++i)
  {
    strip_helper(node.child(i));
  }
}

void strip_arrays(const conduit::Node &input, conduit::Node &output)
{
  output.set_external(input);
  strip_helper(output);
}


}//namespace detail



Redistribute::Redistribute()
{
}


Collection
Redistribute::execute(Collection &collection,
                      const std::vector<int32> &src_list,
                      const std::vector<int32> &dest_list)
{
#ifdef DRAY_MPI_ENABLED
  DRAY_LOG_OPEN("redistribute");
  Collection res;
  build_schedule(collection, res, src_list, dest_list);
  send_recv_metadata(collection);
  send_recv(collection, res);
  DRAY_LOG_CLOSE();
  return res;
#else
  // if we are not parallel, nothing to do
  return collection;
#endif
}


void
Redistribute::build_schedule(Collection &collection,
                             Collection &output,
                             const std::vector<int32> &src_list,
                             const std::vector<int32> &dest_list)
{
  const int32 total_domains = collection.size();
  if(src_list.size() != total_domains)
  {
    DRAY_ERROR("src list needs to be of size total domains");
  }
#ifdef DRAY_MPI_ENABLED

  int32 local_size = collection.local_size();
  int32 rank, procs;

  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &procs);

  std::vector<int32> dom_counts;
  dom_counts.resize(procs);

  MPI_Allgather(&local_size, 1, MPI_INT, &dom_counts[0], 1, MPI_INT, comm);

  std::vector<int32> dom_offsets;
  dom_offsets.resize(procs);
  dom_offsets[0] = 0;

  for(int i = 1; i < procs; ++i)
  {
    dom_offsets[i] = dom_offsets[i-1] + dom_counts[i-1];
  }

  m_comm_info.clear();

  const int32 list_size = src_list.size();

  // figure out sends
  const int32 rank_offset = dom_offsets[rank];
  int32 dom_count = dom_counts[rank];
  for(int32 i = 0; i < dom_count; ++i)
  {
    const int32 index = i + rank_offset;
    if(dest_list[index] != rank)
    {
      CommInfo send;
      send.m_src_idx = i;
      send.m_src_rank = rank;
      send.m_dest_rank = dest_list[index];
      send.m_domain_id = index;
      m_comm_info.push_back(send);
    }
    else if(src_list[index] == rank && dest_list[index] == rank)
    {
      // pass through the domain to the output
      output.add_domain(collection.domain(i));
    }
  }

  // figure out recvs
  for(int32 i = 0; i < list_size; ++i)
  {
    if(dest_list[i] == rank && src_list[i] != rank)
    {
      CommInfo recv;
      recv.m_src_rank = src_list[i];
      recv.m_dest_rank = rank;
      recv.m_domain_id = i;
      m_comm_info.push_back(recv);
    }
  }

  // in order to not-deadlock, we will use domain_id
  // as a global ordering. Interleave sends and recvs based
  // on this ordering

  struct CompareCommInfo
  {
    bool operator()(const CommInfo &a, const CommInfo &b) const
    {
      return a.m_domain_id < b.m_domain_id;
    }
  };

  std::sort(m_comm_info.begin(), m_comm_info.end(), CompareCommInfo());
#endif
}


void Redistribute::send_recv_metadata(Collection &collection)
{
  // we will send and recv everything but the actual arrays
  // so we can allocated space and recv the data via Isend/Irecv

#ifdef DRAY_MPI_ENABLED
  const int32 total_comm = m_comm_info.size();
  Timer timer;

  int32 rank = dray::mpi_rank();
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());

  for(int32 i = 0; i < total_comm; ++i)
  {
    CommInfo info = m_comm_info[i];
    bool send = info.m_src_rank == rank;

    if(send)
    {
      conduit::Node n_domain;
      DataSet domain = collection.domain(info.m_src_idx);
      domain.to_node(n_domain);
      conduit::Node meta;
      detail::strip_arrays(n_domain,meta);
      conduit::relay::mpi::send_using_schema(meta,
                                             info.m_dest_rank,
                                             info.m_domain_id,
                                             comm);
      DRAY_INFO("Send domain "<<info.m_domain_id<<" to rank "<<info.m_dest_rank);
    }
    else
    {

      conduit::Node n_domain_meta;
      conduit::relay::mpi::recv_using_schema(n_domain_meta,
                                             info.m_src_rank,
                                             info.m_domain_id,
                                             comm);
      // allocate a data set to for recvs
      m_recv_q[info.m_domain_id] = to_dataset(n_domain_meta);
      DRAY_INFO("Recv domain "<<info.m_domain_id<<" from rank "<<info.m_src_rank);
    }

  }
  DRAY_LOG_ENTRY("send_recv_meta", timer.elapsed());
#endif
}

void Redistribute::send_recv(Collection &collection,
                             Collection &output)
{

#ifdef DRAY_MPI_ENABLED
  Timer timer;
  const int32 total_comm = m_comm_info.size();

  int32 rank = dray::mpi_rank();
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());

  // get all the pointers together

  // tag/vector< size, ptr>
  std::map<int32,std::vector<std::pair<size_t,unsigned char*>>> send_buffers;
  std::map<int32, int32> send_dests;
  std::map<int32,std::vector<std::pair<size_t,unsigned char*>>> recv_buffers;
  std::map<int32, int32> recv_srcs;
  for(int32 i = 0; i < total_comm; ++i)
  {
    CommInfo info = m_comm_info[i];
    bool send = info.m_src_rank == rank;
    const int32 base_tag = info.m_domain_id * 1000;
    // we don't need to keep the conduit nodes around
    // since they point directly to dray memory
    if(send)
    {
      conduit::Node n_domain;
      DataSet domain = collection.domain(info.m_src_idx);
      domain.to_node(n_domain);
      detail::pack_dataset(n_domain, send_buffers[base_tag]);
      send_dests[base_tag] = info.m_dest_rank;
    }
    else
    {
      DataSet &domain = m_recv_q[info.m_domain_id];
      conduit::Node n_domain;
      domain.to_node(n_domain);
      detail::pack_dataset(n_domain, recv_buffers[base_tag]);
      recv_srcs[base_tag] = info.m_src_rank;
    }
  }

  std::vector<MPI_Request> requests;

  // send it
  for(auto &domain : send_buffers)
  {
    const int32 base_tag = domain.first;
    int32 tag_counter = 0;
    // TODO: check for max int size
    for(auto &buffer : domain.second)
    {
      MPI_Request request;
      int32 mpi_error = MPI_Isend(buffer.second,
                                  static_cast<int>(buffer.first),
                                  MPI_BYTE,
                                  send_dests[base_tag],
                                  base_tag + tag_counter,
                                  comm,
                                  &request);
      DRAY_CHECK_MPI_ERROR(mpi_error);
      requests.push_back(request);
      tag_counter++;
    }
  }

  // recv it
  for(auto &domain : recv_buffers)
  {
    const int32 base_tag = domain.first;
    int32 tag_counter = 0;
    // TODO: check for max int size
    for(auto &buffer : domain.second)
    {
      MPI_Request request;
      int32 mpi_error = MPI_Irecv(buffer.second,
                                  static_cast<int>(buffer.first),
                                  MPI_BYTE,
                                  recv_srcs[base_tag],
                                  base_tag + tag_counter,
                                  comm,
                                  &request);
      DRAY_CHECK_MPI_ERROR(mpi_error);
      requests.push_back(request);
      tag_counter++;
    }
  }
  std::vector<MPI_Status> status;
  status.resize(requests.size());
  int32 mpi_error = MPI_Waitall(requests.size(), &requests[0], &status[0]);
  DRAY_CHECK_MPI_ERROR(mpi_error);
  for(auto &recv : m_recv_q)
  {
    output.add_domain(recv.second);
  }
  m_recv_q.clear();
  DRAY_LOG_ENTRY("send_recv", timer.elapsed());
#endif
}

}//namespace dray
