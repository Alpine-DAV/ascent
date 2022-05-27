// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <iostream>
#include <mfem.hpp>
#include <mfem/fem/conduitdatacollection.hpp>

using namespace std;

// Returns pointer to new mesh and grid function.
// Caller is responsible to delete mesh_ptr and sol.
void construct_example_data (const int num_el,
                             mfem::Mesh *&mesh_ptr,
                             mfem::GridFunction *&sol,
                             int order = 2,
                             std::string fname = "beam-hex.mesh");


int main (int argc, char *argv[])
{
  std::string config_file = "";

  if (argc < 2)
  {
    std::cout << "missing mesh file\n";
    exit (1);
  }

  if (argc < 3)
  {
    std::cout << "missing output name\n";
    exit (1);
  }

  if (argc != 4)
  {
    std::cout << "missing polynomial order\n";
    exit (1);
  }

  int order = atoi(argv[3]);
  std::string mesh_file(argv[1]);
  std::string output_name(argv[2]);

  mfem::Mesh *mesh_ptr;
  mfem::GridFunction *field_ptr;
  construct_example_data (4, mesh_ptr, field_ptr, order, mesh_file);

  mfem::ConduitDataCollection col (output_name);
  col.SetMesh (mesh_ptr);
  col.RegisterField ("diffusion", field_ptr);
  col.SetProtocol ("hdf5");
  col.Save ();
}

void construct_example_data (const int in_max_els,
                             mfem::Mesh *&out_mesh_ptr,
                             mfem::GridFunction *&out_sol_ptr,
                             int order,
                             std::string fname)
{
  using namespace mfem;

  std::string file_name = fname;
  std::cout << "File name " << file_name << "\n";

  Mesh *mesh = new Mesh (file_name.c_str (), 1, 1);
  int dim = mesh->Dimension ();
  bool static_cond = false;
  int sdim = mesh->SpaceDimension ();
  std::cout << "Dim : " << dim << "\n"; //  Dims in referene space
  std::cout << "Space Dim : " << sdim << "\n";

  const float max_els = in_max_els;
  // 3. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 50,000
  //    elements.
  {
    int ref_levels = (int)floor (log (max_els / mesh->GetNE ()) / log (2.) / dim);
    for (int l = 0; l < ref_levels; l++)
    {
      mesh->UniformRefinement ();
    }
  }

  mesh->ReorientTetMesh ();

  // 4. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  if (order > 0)
  {
    fec = new H1_FECollection (order, dim);
  }
  else if (mesh->GetNodes ())
  {
    fec = mesh->GetNodes ()->OwnFEC ();
    cout << "Using isoparametric FEs: " << fec->Name () << endl;
  }
  else
  {
    fec = new H1_FECollection (order = 1, dim);
  }
  FiniteElementSpace *fespace = new FiniteElementSpace (mesh, fec);
  cout << "Number of finite element unknowns: " << fespace->GetTrueVSize () << endl;

  // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size ())
  {
    Array<int> ess_bdr (mesh->bdr_attributes.Max ());
    ess_bdr = 1;
    fespace->GetEssentialTrueDofs (ess_bdr, ess_tdof_list);
  }

  // 6. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  LinearForm *b = new LinearForm (fespace);
  ConstantCoefficient one (1.0);
  b->AddDomainIntegrator (new DomainLFIntegrator (one));
  b->Assemble ();

  // 7. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  GridFunction *_x = new GridFunction (fespace);
  GridFunction &x = *_x;
  x = 0.0;

  // 8. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    domain integrator.
  BilinearForm *a = new BilinearForm (fespace);
  a->AddDomainIntegrator (new DiffusionIntegrator (one));

  // 9. Assemble the bilinear form and the corresponding linear system,
  //    applying any necessary transformations such as: eliminating boundary
  //    conditions, applying conforming constraints for non-conforming AMR,
  //    static condensation, etc.
  if (static_cond)
  {
    a->EnableStaticCondensation ();
  }
  a->Assemble ();

  SparseMatrix A;
  Vector B, X;
  a->FormLinearSystem (ess_tdof_list, x, *b, A, X, B);

  cout << "Size of linear system: " << A.Height () << endl;

  GSSmoother M (A);
  PCG (A, M, B, X, 1, 200, 1e-12, 0.0);

  // 11. Recover the solution as a finite element grid function.
  a->RecoverFEMSolution (X, *b, x);

  // Output to arguments.
  out_mesh_ptr = mesh;
  out_sol_ptr = _x;

  // TODO didn't there used to be some "delete" statements?
}
