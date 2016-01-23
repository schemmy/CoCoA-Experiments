#include <iostream>
#include <iostream>
#include <math.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <iomanip>      // std::setprecision
#include <set>
using namespace std;

#include <gsl/gsl_blas.h> // for BLAS
#include <gsl/gsl_linalg.h> // for LU, QR, ...

#ifndef QRSOLVER_H_
#define QRSOLVER_H_

void geneRandIdx(std::vector<unsigned int> oneToN, std::vector<unsigned int> &randSet,
                 unsigned int max, unsigned int size) {

	if (size > floor(max * 0.4)) {
		for (int i = max - 1; i > 0; --i)
		{
			//get swap index
			int j = rand() % i;
			//swap
			int temp = oneToN[i];
			oneToN[i] = oneToN[j];
			oneToN[j] = temp;
		}

		for (int i = 0; i < size; ++i)
			randSet[i] = oneToN[i];
	}
	else {
		int tempNum;			// temp variable to hold random number
		bool found;			// used for value in array

		unsigned int index = 0;
		while (index < size)
		{
			found = false;
			tempNum = floor(rand() / (0.0 + RAND_MAX) * max);

			for (int index2 = 0; index2 < index; index2++) {
				if (tempNum == randSet[index2])
				{
					found = true;
					break;
				}
			}
			if (found == false) {
				randSet[index] = tempNum;
				index++;
			}
		}
	}

	//for (int i = 0; i < size; ++i) cout<<randSet[i]<<"  ";
}


void gaussEliminationSolver(std::vector<double> &A, int n,
                            std::vector<double> &b, std::vector<double> &x) {

	std::vector<double> A_fix(n * n, 0);
	std::vector<double> b_fix(n, 0);
	cblas_dcopy(n * n, &A[0], 1, &A_fix[0], 1);
	cblas_dcopy(n, &b[0], 1, &b_fix[0], 1);

	for (int col_idx = 0; col_idx < n - 1; col_idx++) {
		for (int row_idx = col_idx + 1; row_idx < n; row_idx++) {
			int idx1 = row_idx * n + col_idx;
			int idx2 = col_idx * n + col_idx;
			double multiplier = 0.0;
			multiplier = 1.0 * A_fix[idx1] / A_fix[idx2];
			//http://www.mathkeisan.com/usersguide/man/daxpy.html
			int idx_start1 = row_idx * n + col_idx;
			int idx_start2 = col_idx * n + col_idx;
			cblas_daxpy(n - col_idx, -multiplier, &A_fix[idx_start2], 1, &A_fix[idx_start1], 1);
			b_fix[row_idx] -= 1.0 * multiplier * b_fix[col_idx];
		}
	}

	for (int i = n - 1; i >= 0; i--) {
		int idx = i * n + i + 1;

		double dot_product = cblas_ddot(n - i - 1, &A_fix[idx], 1, &x[i + 1], 1);
		x[i] = (b_fix[i] - dot_product) / A_fix[idx - 1];
	}

}



void QRGramSchmidtSolver(std::vector<double> &A, int n,
                         std::vector<double> &b, std::vector<double> &x) {

	// Generating Q^T
	std::vector<double> QT(n * n, 0);
	// Copy A^T to Q^T
	for (int i = 0; i < n; i++)
		cblas_dcopy(n, &A[i], n, &QT[i * n], 1);

	// Compute Q
	for  (int i = 1; i < n; i++) {
		double denom = cblas_dnrm2(n, &QT[(i - 1) * n], 1);
		for (int j = i; j < n; j++) {
			double nom = cblas_ddot(n, &QT[(i - 1) * n], 1, &A[j], n);
			double factor = 1.0 * nom / denom / denom;
			// u_j = u_j - proj_{u_i}a_j
			cblas_daxpy(n, -factor, &QT[(i - 1)*n], 1, &QT[j * n], 1);
		}
	}
	// normalize Q^T
	for (int i = 0; i < n; i++) {
		double oneOverNorm = 1.0 / cblas_dnrm2(n, &QT[i * n], 1);
		cblas_dscal(n, oneOverNorm, &QT[i * n], 1);
	}

	std::vector<double> b_new(n, 0);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, &QT[0], n, &b[0], 1, 0.0, &b_new[0], 1);

	// generating R
	std::vector<double> R(n * n, 0);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			if (i > j)
				R[i * n + j] = 0;
			else
				R[i * n + j] = cblas_ddot(n, &QT[i * n], 1, &A[j], n);
		}
	}

	// Back substitution
	for (int i = n - 1; i >= 0; i--) {
		int idx = i * n + i + 1;
		double dot_product = cblas_ddot(n - i - 1, &R[idx], 1, &x[i + 1], 1);
		//cout<<idx<<"   "<<A[idx-1]<<"   "<<dot_product<<endl;
		x[i] = (b_new[i] - dot_product) / R[idx - 1];
	}
}



void CGSolver(std::vector<double> &A, int n,
              std::vector<double> &b, std::vector<double> &x) {

	std::vector<double> r(n);
	std::vector<double> p(n);
	std::vector<double> Ap(n);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, &A[0],
	            n, &x[0], 1, 1.0, &r[0], 1);
	cblas_daxpy(n, -1.0, &b[0], 1, &r[0], 1);
	cblas_dcopy(n, &r[0], 1, &p[0], 1);
	cblas_dscal(n, -1.0, &p[0], 1);

	double tol = 1e-16;
	int iter = 0;

	while (1) {
		cblas_set_to_zero(Ap);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, &A[0],
		            n, &p[0], 1, 1.0, &Ap[0], 1);

		double nom = cblas_ddot(n, &r[0], 1, &r[0], 1);
		double denom = cblas_ddot(n, &p[0], 1, &Ap[0], 1);
		double alpha = nom / denom;

		cblas_daxpy(n, alpha, &p[0], 1, &x[0], 1);
		cblas_daxpy(n, alpha, &Ap[0], 1, &r[0], 1);

		double nomNew = cblas_ddot(n, &r[0], 1, &r[0], 1);
		double beta = nomNew / nom;

		cblas_dscal(n, beta, &p[0], 1);
		cblas_daxpy(n, -1.0, &r[0], 1, &p[0], 1);
		iter += 1;
		//cout<<sqrt(nomNew)<<endl;
		if (sqrt(nomNew) <= tol || iter > 1000) {
			//cout<< iter <<endl;
			break;
		}

	}

}


void SGDSolver(ProblemData<unsigned int, double> &instance,
               unsigned int &n, std::vector<double> &b, std::vector<double> &x, double &diag) {

	double eta = 1.0/100;
	double xTs = 0.0;
	std::vector<double> gradIdx(n * instance.n);
	std::vector<double> gradAvg(n);
	double nomNew = 1.0;

	while (nomNew > 1e-10) {

		for (unsigned int iter = 0; iter < instance.n; iter++) {

			xTs = 0.0;
			unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
			for (unsigned int i = 0; i < n; i++) {
				gradAvg[i] -= gradIdx[idx * n + i];
				gradIdx[idx * n + i] = 0.0;
			}
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				xTs += instance.A_csr_values[i] * x[instance.A_csr_col_idx[i]];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				gradIdx[idx * n + instance.A_csr_col_idx[i]] = instance.A_csr_values[i] * xTs;
			for (unsigned int i = 0; i < n; i++) {
				gradIdx[idx * n + i] = gradIdx[idx * n + i] 
												- b[i]  + diag * x[i];
				gradAvg[i] += gradIdx[idx * n + i];
				x[i] -= eta / instance.n * gradAvg[i];
			}
		}

		std::vector<double> grad(n);
		for (unsigned int j = 0; j < instance.n; j++) {
			unsigned int idx = j;
			xTs = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				xTs += instance.A_csr_values[i] * x[instance.A_csr_col_idx[i]];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				grad[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * xTs / instance.n;
		}
		for (unsigned int i = 0; i < n; i++) {
			grad[i] = grad[i] - b[i] + diag * x[i];
		}
		nomNew = cblas_ddot(n, &grad[0], 1, &grad[0], 1);
		//cout << nomNew << endl;

	}

}




// void geneWoodburyH(ProblemData<unsigned int, double> &instance,
//                    unsigned int &p, std::vector<double> &woodburyH, double & diag) {

// 	for (unsigned int idx1 = 0; idx1 < p; idx1++) {
// 		for (unsigned int idx2 = 0; idx2 < p; idx2++) {
// 			unsigned int i = instance.A_csr_row_ptr[idx1];
// 			unsigned int j = instance.A_csr_row_ptr[idx2];
// 			while (i < instance.A_csr_row_ptr[idx1 + 1] && j < instance.A_csr_row_ptr[idx2 + 1]) {
// 				if (instance.A_csr_col_idx[i] == instance.A_csr_col_idx[j]) {
// 					woodburyH[idx1 * p + idx2] += instance.A_csr_values[i] * instance.A_csr_values[j]
// 					                              * instance.b[idx1] * instance.b[idx2] / diag / p;
// 					i++;
// 					j++;
// 				}
// 				else if (instance.A_csr_col_idx[i] < instance.A_csr_col_idx[j])
// 					i++;
// 				else
// 					j++;
// 			}
// 		}
// 	}
// 	for (unsigned int idx = 0; idx < p; idx++)
// 		woodburyH[idx * p + idx] += 1.0;

// }


// void geneWoodburyHLogistic(ProblemData<unsigned int, double> &instance,
//                            unsigned int &p, std::vector<double> &woodburyH, std::vector<double> &wTx, double & diag) {

// 	double temp, scalar;
// 	cblas_set_to_zero(woodburyH);

// 	for (unsigned int idx1 = 0; idx1 < p; idx1++) {

// 		temp = exp(-wTx[idx1]);
// 		scalar = temp / (temp + 1) / (temp + 1);

// 		for (unsigned int idx2 = 0; idx2 < p; idx2++) {

// 			unsigned int i = instance.A_csr_row_ptr[idx1];
// 			unsigned int j = instance.A_csr_row_ptr[idx2];
// 			while (i < instance.A_csr_row_ptr[idx1 + 1] && j < instance.A_csr_row_ptr[idx2 + 1]) {
// 				if (instance.A_csr_col_idx[i] == instance.A_csr_col_idx[j]) {
// 					woodburyH[idx1 * p + idx2] += instance.A_csr_values[i]  * instance.A_csr_values[j]
// 					                              * instance.b[idx1] * instance.b[idx2] / diag / p * scalar;
// 					i++;
// 					j++;
// 				}
// 				else if (instance.A_csr_col_idx[i] < instance.A_csr_col_idx[j])
// 					i++;
// 				else
// 					j++;
// 			}
// 		}
// 	}
// 	for (unsigned int idx = 0; idx < p; idx++)
// 		woodburyH[idx * p + idx] += 1.0;

// }



// void WoodburySolverForDiscoLogistic(ProblemData<unsigned int, double> &instance,
//                                     unsigned int &n, unsigned int &p, std::vector<double> &woodburyH,
//                                     std::vector<double> &b, std::vector<double> &x, std::vector<double> &wTx, double & diag) {

// 	std::vector<double> woodburyVTy(p);
// 	std::vector<double> woodburyHVTy(p);
// 	std::vector<double> woodburyZHVTy(n);

// 	double temp, scalar;

// 	for (unsigned int idx = 0; idx < p; idx++) {

// 		temp = exp(-wTx[idx]);
// 		scalar = temp / (temp + 1) / (temp + 1);
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyVTy[idx] += instance.A_csr_values[i] * instance.b[idx] * b[instance.A_csr_col_idx[i]]
// 			                    / diag / p  * scalar;
// 		}
// 	}

// 	//CGSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);
// 	//gaussEliminationSolver(woodburyH, p, woodburyVTy, woodburyHVTy);
// 	QRGramSchmidtSolver(woodburyH, p, woodburyVTy, woodburyHVTy);

// 	for (unsigned int idx = 0; idx < p; idx++) {

// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx] / diag
// 			        * woodburyHVTy[idx];
// 		}
// 	}

// 	for (unsigned int i = 0; i < n; i++) {
// 		x[i] = b[i] / diag - woodburyZHVTy[i];
// 	}

// }
// void WoodburySolverForOcsidLogistic(ProblemData<unsigned int, double> &preConData, ProblemData<unsigned int, double> &instance,
//                                     unsigned int &n, unsigned int &p, std::vector<double> &woodburyH, std::vector<double> &b, std::vector<double> &x,
//                                     std::vector<double> &wTx, double & diag, boost::mpi::communicator & world) {

// 	std::vector<double> woodburyVTy(p);
// 	std::vector<double> woodburyVTy_World(p);
// 	std::vector<double> woodburyHVTy(p);
// 	std::vector<double> woodburyZHVTy(n);

// 	double temp, scalar;

// 	for (unsigned int idx = 0; idx < p; idx++) {

// 		temp = exp(-wTx[idx]);
// 		scalar = temp / (temp + 1) / (temp + 1);
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyVTy[idx] += instance.A_csr_values[i] * instance.b[idx] * b[instance.A_csr_col_idx[i]] / diag / p * scalar;
// 		}
// 	}
// 	vall_reduce(world, woodburyVTy, woodburyVTy_World);


// 	//CGSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);
// 	//gaussEliminationSolver(woodburyH, p, woodburyVTy, woodburyHVTy);
// 	QRGramSchmidtSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);

// 	for (unsigned int idx = 0; idx < p; idx++) {
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
// 			        / diag * woodburyHVTy[idx];
// 		}
// 	}

// 	for (unsigned int i = 0; i < n; i++) {
// 		x[i] = b[i] / diag - woodburyZHVTy[i];
// 	}
// }


// void WoodburySolverForOcsid(ProblemData<unsigned int, double> &preConData, ProblemData<unsigned int, double> &instance,
//                             unsigned int &n, unsigned int &p, std::vector<double> &woodburyH, std::vector<double> &b, std::vector<double> &x,
//                             double & diag, boost::mpi::communicator & world) {

// 	std::vector<double> woodburyVTy(p);
// 	std::vector<double> woodburyVTy_World(p);
// 	std::vector<double> woodburyHVTy(p);
// 	std::vector<double> woodburyZHVTy(n);


// 	for (unsigned int idx = 0; idx < p; idx++) {
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyVTy[idx] += instance.A_csr_values[i] * instance.b[idx] * b[instance.A_csr_col_idx[i]] / diag / p;
// 		}
// 	}
// 	vall_reduce(world, woodburyVTy, woodburyVTy_World);

// 	//CGSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);
// 	QRGramSchmidtSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);

// 	for (unsigned int idx = 0; idx < p; idx++) {
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
// 			        / diag * woodburyHVTy[idx];
// 		}
// 	}

// 	for (unsigned int i = 0; i < n; i++) {
// 		x[i] = b[i] / diag - woodburyZHVTy[i];
// 	}
// }

// void WoodburySolverForDisco(ProblemData<unsigned int, double> &instance, unsigned int &n,
//                             std::vector<unsigned int> &randIdx, unsigned int &p, std::vector<double> &woodburyH,
//                             std::vector<double> &b, std::vector<double> &x, double & diag) {

// 	std::vector<double> woodburyVTy(p);
// 	std::vector<double> woodburyHVTy(p);
// 	std::vector<double> woodburyZHVTy(n);


// 	for (unsigned int idx = 0; idx < p; idx++) {
// 		//unsigned int idx = randIdx[j];
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyVTy[idx] += instance.A_csr_values[i] * instance.b[idx] * b[instance.A_csr_col_idx[i]] / diag / p;
// 		}
// 	}

// 	//CGSolver(woodburyH, p, woodburyVTy, woodburyHVTy);
// 	QRGramSchmidtSolver(woodburyH, p, woodburyVTy, woodburyHVTy);

// 	for (unsigned int idx = 0; idx < p; idx++) {
// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
// 			woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
// 			        / diag * woodburyHVTy[idx];
// 		}
// 	}

// 	for (unsigned int i = 0; i < n; i++) {
// 		x[i] = b[i] / diag - woodburyZHVTy[i];
// 	}

// }



void ifNoPreconditioning(int n,
                         std::vector<double> &b, std::vector<double> &x) {

	for (int i = 0; i < n; i++) {
		x[i] = 1.0 * b[i];
	}

}


#endif
