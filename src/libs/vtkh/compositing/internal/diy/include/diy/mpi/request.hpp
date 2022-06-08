namespace vtkhdiy
{
namespace mpi
{
  struct request
  {
    status              wait()              { status s; MPI_Wait(&r, &s.s); return s; }
    inline
    optional<status>    test();
    void                cancel()            { MPI_Cancel(&r); }

    MPI_Request         r;
  };
}
}

vtkhdiy::mpi::optional<vtkhdiy::mpi::status>
vtkhdiy::mpi::request::test()
{
  status s;
  int flag;
  MPI_Test(&r, &flag, &s.s);
  if (flag)
    return s;
  return optional<status>();
}
