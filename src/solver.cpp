#include <Epetra_ConfigDefs.h>
#ifdef EPETRA_MPI
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_Map.h>
#include <Epetra_Import.h>

// EpetraExt
#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_RowMatrixOut.h>
#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_VectorIn.h>
#include <EpetraExt_MultiVectorOut.h>
#include <EpetraExt_MultiVectorIn.h>
#include <EpetraExt_BlockMapIn.h>
#include <EpetraExt_BlockMapOut.h>
#include <Xpetra_EpetraUtils.hpp>
#include <Xpetra_EpetraMultiVector.hpp>

#include <AztecOO.h>
#include <MueLu_CreateEpetraPreconditioner.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <string>

using Teuchos::RCP;

std::string mtxFileName = "A.mtx";
std::string bFileName = "b.mtx";

int main(int argc, char *argv[]) {
#ifdef EPETRA_MPI
  MPI_Init(&argc, &argv);
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif

  Kokkos::initialize(argc, argv);

  int myRank = comm.MyPID();

  // Read matrix A from Matrix Market file
  Epetra_CrsMatrix* A_raw;
  EpetraExt::MatrixMarketFileToCrsMatrix(mtxFileName.c_str(), comm, A_raw);
  RCP<Epetra_CrsMatrix> A = Teuchos::rcp(A_raw);

  // Ensure matrix is filled and ready to use
  A->FillComplete();


  // Create a non-const Epetra_BlockMap from the RowMap
  const Epetra_Map& constRowMap = A->RowMap();
  Epetra_BlockMap rowMap(constRowMap);

  // Create vector B from Matrix Market file
  Epetra_Vector* B_raw;
  EpetraExt::MatrixMarketFileToVector(bFileName.c_str(), rowMap, B_raw);
  RCP<Epetra_Vector> B = Teuchos::rcp(B_raw);

  // Create solution vector X
  Epetra_Vector Xraw(A->RowMap(), true); // Initialize with zero
  RCP<Epetra_Vector> X = Teuchos::rcp(&Xraw);

  // Print the matrix and vectors to be sure we have
  // the data initialized correctly
  /*std::cout << "Printing A matrix:" << std::endl;
  A->Print(std::cout);
  std::cout << "Printing B vector:" << std::endl;
  B->Print(std::cout);
  std::cout << "Printing V vector:" << std::endl;
  X->Print(std::cout);*/

  // Create MueLu preconditioner
  Teuchos::ParameterList paramList;
  paramList.set("verbosity", "high");
  paramList.set("max levels", 3);
  paramList.set("coarse: max size", 5000);
  paramList.set("multigrid algorithm", "sa");
  paramList.set("sa: damping factor", 1.33);
  paramList.set("reuse: type", "full");
  paramList.set("smoother: type", "RELAXATION");
  paramList.sublist("smoother: params").set("relaxation: type", "Gauss-Seidel");
  paramList.sublist("smoother: params").set("relaxation: sweeps", 3);
  paramList.sublist("smoother: params").set("relaxation: damping factor", 1.0);
  paramList.sublist("smoother: params").set("relaxation: zero starting solution", true);
  paramList.sublist("level 1").set("sa: damping factor", 0.0);
  paramList.set("aggregation: type", "uncoupled");
  paramList.set("aggregation: min agg size", 4);
  paramList.set("aggregation: max agg size", 36);
  paramList.set("aggregation: drop tol", 0.04);
  paramList.set("repartition: enable", false);
  paramList.set("repartition: partitioner", "zoltan2");
  paramList.set("repartition: start level", 1);
  paramList.set("repartition: min rows per proc", 20000000);
  paramList.set("repartition: target rows per proc", 0);
  paramList.set("repartition: max imbalance", 1.1);
  paramList.set("repartition: remap parts", true);
  paramList.set("repartition: rebalance P and R", false);
  paramList.sublist("repartition: params").set("algorithm", "multijagged");

  RCP<MueLu::EpetraOperator> preconditioner = MueLu::CreateEpetraPreconditioner(A, paramList);

  Epetra_LinearProblem epetraProblem(A.get(), X.get(), B.get());

  // Set up AztecOO solver
  AztecOO solver(epetraProblem);
  solver.SetAztecOption(AZ_solver, AZ_gmres);
  solver.SetAztecOption(AZ_kspace, 40);
  solver.SetAztecOption(AZ_output, 1);
  solver.SetAztecOption(AZ_max_iter, 50);
  solver.SetPrecOperator(preconditioner.get());

  int maxIterations = 50;
  double convgTol = 1e-8;

  // Solve the linear system
  solver.Iterate(maxIterations, convgTol);

  // Calculate and print the final residual norm
  Epetra_Vector R(B->Map());
  A->Multiply(false, *X, R);
  R.Update(1.0, *B, -1.0);

  double finalNorm;
  R.Norm2(&finalNorm);
  std::cout << "Final residual norm: " << finalNorm << std::endl;

#ifdef EPETRA_MPI
  int finalize_retcode = MPI_Finalize();
  if(0 == myRank) fprintf(stderr, "Process, return_code\n");
  fprintf(stderr, "%i, %i\n", myRank, finalize_retcode);
#endif

  Kokkos::finalize();

  return 0;
}

