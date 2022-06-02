#ifndef APCOMP_MPI_COLLECT_HPP
#define APCOMP_MPI_COLLECT_HPP

#include <apcomp/image.hpp>
#include <diy/mpi.hpp>
#include <sstream>

namespace apcomp
{

static void MPICollect(Image &image, MPI_Comm comm)
{

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int bounds[4];
  bounds[0] = image.m_bounds.m_min_x;
  bounds[1] = image.m_bounds.m_min_y;
  bounds[2] = image.m_bounds.m_max_x;
  bounds[3] = image.m_bounds.m_max_y;

  int xmin = image.m_bounds.m_min_x;
  int ymin = image.m_bounds.m_min_y;
  int xmax = image.m_bounds.m_max_x;
  int ymax = image.m_bounds.m_max_y;
  int pixels = (xmax - xmin + 1) *(ymax - ymin + 1);

  int *pixel_sizes = nullptr;
  int *pixel_bounds = nullptr;
  if(rank == 0)
  {
    pixel_sizes = new int[size];
    pixel_bounds = new int[size*4];
  }

  MPI_Gather(&bounds, 4, MPI_INT, pixel_bounds, 4, MPI_INT, 0, comm);

  MPI_Barrier(comm);
  // create the final image
  Bounds final_bounds;
  final_bounds.m_min_x = 1;
  final_bounds.m_min_y = 1;
  final_bounds.m_max_x = 1;
  final_bounds.m_max_y = 1;

  if(rank == 0)
  {
    final_bounds = image.m_orig_bounds;
  }

  Image final_image(final_bounds);
  if(rank == 0)
  {
    image.SubsetTo(final_image);
  }

  if(rank != 0)
  {
    MPI_Send(&image.m_pixels[0], pixels * 4, MPI_UNSIGNED_CHAR, 0, 0, comm);
    MPI_Send(&image.m_depths[0], pixels, MPI_FLOAT, 0, 0, comm);
  }
  else
  {
    for(int i = 1; i < size; ++i)
    {
      Bounds inbound;
      inbound.m_min_x = pixel_bounds[i*4 + 0];
      inbound.m_min_y = pixel_bounds[i*4 + 1];
      inbound.m_max_x = pixel_bounds[i*4 + 2];
      inbound.m_max_y = pixel_bounds[i*4 + 3];
      Image incoming(inbound);

      int rec_size = (inbound.m_max_x - inbound.m_min_x + 1) *
                     (inbound.m_max_y - inbound.m_min_y + 1);

      MPI_Status status;
      MPI_Recv(&(incoming.m_pixels[0]), rec_size * 4, MPI_UNSIGNED_CHAR, i, 0, comm, &status);
      MPI_Recv(&(incoming.m_depths[0]), rec_size, MPI_FLOAT, i, 0, comm, &status);
      incoming.SubsetTo(final_image);
    }
  }
  if(rank == 0)
  {
    delete[] pixel_sizes;
    delete[] pixel_bounds;
    image.Swap(final_image);
  }
  //MPI_Barrier(comm);
}

}// namespace apcomp
#endif
