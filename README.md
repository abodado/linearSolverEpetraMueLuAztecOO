# linearSolverEpetraMueLuAztecOO
Solve linear system using Trilinos Epetra, MueLu, and AztecOO.

The initial version of this code uses Trilinos libraries and functions to read a system matrix A.mtx and RHS b.mtx (in MatrixMarket format) supplied by the user and then solve the system using AztecOO with MueLu as the preconditioner. Eventually, I would like to add functionality to read in a xml file of preconditioner settings to investigate the optimal settings for my linear system. I intend to have a separate repository that uses Tpetra, MueLu, and Belos such that one can compare the performance of the solvers. 
