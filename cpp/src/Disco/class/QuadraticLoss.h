/*
 * QuadraticLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */
#ifndef QUADRATICLOSS_H_
#define QUADRATICLOSS_H_

#include "LossFunction.h"
#include "QR_solver.h"

template<typename L, typename D>
class QuadraticLoss: public LossFunction<L, D>  {
public:

	QuadraticLoss() {}

	virtual ~QuadraticLoss() {}

	virtual int getName() {
		return 1;
	}

	virtual void init(ProblemData<L, D> & instance) {


	}


	virtual void computeVectorTimesData(std::vector<double> &vec, ProblemData<unsigned int, double> &instance,
	                                    std::vector<double> &result, boost::mpi::communicator &world, int &mode) {

		double temp;
		std::vector<double> res(result.size());
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			temp = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				temp += vec[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

			res[idx] = temp * instance.b[idx];
		}

		if (mode == 1)
			cblas_dcopy(result.size(), &res[0], 1, &result[0], 1);
		else if (mode == 2)
			vall_reduce(world, res, result);
	}

	virtual void computeObjective(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                              std::vector<double> &xTw, double & obj, boost::mpi::communicator & world, int &mode) {

		obj = 0.0;

		for (unsigned int idx = 0; idx < instance.n; idx++)
			obj += 0.5 * (xTw[idx]  - 1.0) * (xTw[idx] - 1.0);

		double wNorm = cblas_l2_norm(w.size(), &w[0], 1);

		if (mode == 1) {
			double obj_local;
			obj_local = 1.0 / instance.total_n * obj + 0.5 * instance.lambda * wNorm * wNorm / world.size();
			vall_reduce(world, &obj_local, &obj, 1);
		}
		else if (mode == 2) {
			double obj_local;
			obj_local = 1.0 / instance.total_n * obj / world.size() + 0.5 * instance.lambda * wNorm * wNorm;
			vall_reduce(world, &obj_local, &obj, 1);
		}


	}


	virtual void computePrimalAndDualObjective(ProblemData<unsigned int, double> &instance,
	        std::vector<double> &alpha, std::vector<double> &w, double & rho, double & finalDualError,
	        double & finalPrimalError) {

		double localError = 0.0;
		for (unsigned int i = 0; i < instance.n; i++) {
			double tmp = alpha[i] * alpha[i] * 0.5 -  alpha[i];
			localError += tmp;
		}

		double localQuadLoss = 0.0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			double dotProduct = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx];
			        i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
			}
			double single = 1.0  -  dotProduct * instance.b[idx];
			double tmp = 0.5 * single * single;

			localQuadLoss += tmp;
		}

		finalPrimalError = 0;
		finalDualError = 0;

		double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
		finalDualError = 1.0 / instance.n * localError
		                 + 0.5 * instance.lambda * tmp2 * tmp2;
		finalPrimalError =  1.0 / instance.n * localQuadLoss
		                    + 0.5 * instance.lambda * tmp2 * tmp2;

	}


	virtual void computeLocalGradient(std::vector<double> &w, std::vector<double> &grad, std::vector<double> &xTw,
	                                  ProblemData<unsigned int, double> &instance, boost::mpi::communicator & world, int &mode) {

		cblas_set_to_zero(grad);

		if (mode == 1) {
			for (unsigned int idx = 0; idx < instance.n; idx++)
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					grad[instance.A_csr_col_idx[i]] += (xTw[idx] - 1.0) * instance.A_csr_values[i] * instance.b[idx]
					                                   / instance.n;

			cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);
		}

	}

	virtual void computeGradient(std::vector<double> &w, std::vector<double> &grad, std::vector<double> &xTw,
	                             ProblemData<unsigned int, double> &instance, boost::mpi::communicator & world, int &mode) {

		cblas_set_to_zero(grad);

		for (unsigned int idx = 0; idx < instance.n; idx++)
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				grad[instance.A_csr_col_idx[i]] += (xTw[idx] - 1.0) * instance.A_csr_values[i] * instance.b[idx]
				                                   / instance.total_n;

		if (mode == 1) {
			cblas_daxpy(instance.m, instance.lambda / world.size(), &w[0], 1, &grad[0], 1);
			std::vector<double> grad_world(instance.m);
			vall_reduce(world, grad, grad_world);
			cblas_dcopy(instance.m, &grad_world[0], 1, &grad[0], 1);
		}
		else if (mode == 2)
			cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);

	}


	virtual void computeHessianTimesAU(std::vector<double> &u, std::vector<double> &Hu, std::vector<double> &xTw,
	                                   std::vector<double> &xTu, ProblemData<unsigned int, double> &instance,
	                                   unsigned int &batchSizeH, std::vector<unsigned int> &randIdx,
	                                   boost::mpi::communicator & world, int &mode) {

		cblas_set_to_zero(Hu);

		for (unsigned int j = 0; j < batchSizeH; j++) {
			unsigned int idx = randIdx[j];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				Hu[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx] * xTu[idx] / batchSizeH;
			}
		}

		if (mode == 1) {
			std::vector<double> Hu_local(instance.m);
			for (unsigned int i = 0; i < instance.m; i++)
				Hu_local[i] = Hu[i] / world.size() + instance.lambda / world.size() * u[i];
			vall_reduce(world, Hu_local, Hu);
		}
		else if (mode == 2) {
			for (unsigned int i = 0; i < instance.m; i++)
				Hu[i] += instance.lambda * u[i];

		}

	}



	virtual void getWoodburyH(ProblemData<unsigned int, double> &instance,
	                          unsigned int &p, std::vector<double> &woodburyH, std::vector<double> &wTx, double & diag) {

		cblas_set_to_zero(woodburyH);
		for (unsigned int idx1 = 0; idx1 < p; idx1++) {
			for (unsigned int idx2 = 0; idx2 < p; idx2++) {
				unsigned int i = instance.A_csr_row_ptr[idx1];
				unsigned int j = instance.A_csr_row_ptr[idx2];
				while (i < instance.A_csr_row_ptr[idx1 + 1] && j < instance.A_csr_row_ptr[idx2 + 1]) {
					if (instance.A_csr_col_idx[i] == instance.A_csr_col_idx[j]) {
						woodburyH[idx1 * p + idx2] += instance.A_csr_values[i] * instance.A_csr_values[j]
						                              * instance.b[idx1] * instance.b[idx2] / diag / p;
						i++;
						j++;
					}
					else if (instance.A_csr_col_idx[i] < instance.A_csr_col_idx[j])
						i++;
					else
						j++;
				}
			}
		}
		for (unsigned int idx = 0; idx < p; idx++)
			woodburyH[idx * p + idx] += 1.0;

	}


	virtual void WoodburySolver(ProblemData<unsigned int, double> &instance, unsigned int &n,
	                            std::vector<unsigned int> &randIdx, unsigned int &p, std::vector<double> &woodburyH,
	                            std::vector<double> &b, std::vector<double> &x, std::vector<double> &wTx, double & diag,
	                            boost::mpi::communicator & world, int &mode) {

		std::vector<double> woodburyVTy(p);
		std::vector<double> woodburyVTy_World(p);
		std::vector<double> woodburyHVTy(p);
		std::vector<double> woodburyZHVTy(n);


		for (unsigned int idx = 0; idx < p; idx++) {
			//unsigned int idx = randIdx[j];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				woodburyVTy[idx] += instance.A_csr_values[i] * instance.b[idx] * b[instance.A_csr_col_idx[i]] / diag / p;
			}
		}

		if (mode == 1) {
			//QRGramSchmidtSolver(woodburyH, p, woodburyVTy, woodburyHVTy);
			CGSolver(woodburyH, p, woodburyVTy, woodburyHVTy);
		}
		if (mode == 2) {
			vall_reduce(world, woodburyVTy, woodburyVTy_World);
			QRGramSchmidtSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);
			//CGSolver(woodburyH, p, woodburyVTy_World, woodburyHVTy);
		}

		for (unsigned int idx = 0; idx < p; idx++) {
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
				        / diag * woodburyHVTy[idx];
			}
		}

		for (unsigned int i = 0; i < n; i++)
			x[i] = b[i] / diag - woodburyZHVTy[i];

	}



	virtual void SAGSolver(ProblemData<unsigned int, double> &instance,
	                       unsigned int &n, std::vector<double> &xTw, std::vector<double> &b, std::vector<double> &x, int nEpoch, double &diag) {

		double eta = 0.01;
		double kappa = 1.0;
		int em = 0;
		//nEpoch = 100;
		std::vector<double> gradAvg(instance.m);
		std::vector<double> y(instance.n);
		std::vector<double> C(instance.n);
		std::vector<int> V(instance.m, 1);
		std::vector<double> z(instance.m);
		cblas_dcopy(instance.m, &x[0], 1, &z[0], 1);
		int iter = 0;
		std::vector<double> S(nEpoch * instance.n);
		std::vector<double> Sb(nEpoch * instance.n);
		double xTs = 0.0;
		double nomNew = 1.0;
		double nom0 = 1.0;
		unsigned int k;

//	for (int iter = 1; iter < maxIter; iter++) {

		for (k = 1; k < instance.n * nEpoch; k++) {
			unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
			if (C[idx] == 0) {
				em++;
				C[idx] = 1;
			}

			// Just-in-time calculation of needed values of z
			xTs = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				z[instance.A_csr_col_idx[i]] = z[instance.A_csr_col_idx[i]]
				                               - (S[k - 1] - S[V[instance.A_csr_col_idx[i]] - 1]) * gradAvg[instance.A_csr_col_idx[i]]
				                               + (Sb[k - 1] - Sb[V[instance.A_csr_col_idx[i]] - 1]) * b[instance.A_csr_col_idx[i]];
				V[instance.A_csr_col_idx[i]] = k;
				xTs += instance.A_csr_values[i] * z[instance.A_csr_col_idx[i]] * instance.b[idx];
			}
			//Update the memory y and the direction
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				gradAvg[instance.A_csr_col_idx[i]] -= y[idx] * instance.A_csr_values[i] * instance.b[idx];
			}
			y[idx] = xTs * kappa;
			//y[idx] = -1.0 * exp(-xTs * instance.b[idx]) / (1.0 + exp(-xTs * instance.b[idx])) * kappa;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				gradAvg[instance.A_csr_col_idx[i]] += y[idx] * instance.A_csr_values[i] * instance.b[idx];
			}
			//Update kappa and the sum needed for z updates.
			kappa = kappa * (1.0 - eta * diag);
			S[k] = S[k - 1] + eta / kappa / em;
			Sb[k] = Sb[k - 1] + eta / kappa;
		}

		for (unsigned int i = 0; i < instance.m; i++) {
			x[i] = kappa * (z[i] - (S[k - 1] - S[V[i] - 1]) * gradAvg[i]
			                + (Sb[k - 1] - Sb[V[i] - 1]) * b[i]);
		}



		// 	std::vector<double> grad(n);
		// 	for (unsigned int j = 0; j < instance.n; j++) {
		// 		unsigned int idx = j;
		// 		xTs = 0.0;
		// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
		// 			xTs += instance.A_csr_values[i] * x[instance.A_csr_col_idx[i]];
		// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
		// 			grad[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * xTs / instance.n;
		// 	}
		// 	for (unsigned int i = 0; i < n; i++) {
		// 		grad[i] = grad[i] - b[i] + diag * x[i];
		// 	}
		// 	nomNew = cblas_ddot(n, &grad[0], 1, &grad[0], 1);
		// 	//cout << nomNew << endl;
		// 	if (iter == 1)
		// 		nom0 = nomNew;
		// 	if (nomNew < 1e-5 * nom0)
		// 		break;

		// }

// Naive implementation
		//std::vector<double> gradIdx(instance.m * instance.n);
		// double xTs = 0.0;
		// double nomNew = 1.0;
		// double nom0 = 1.0;
		//  while (nomNew > 1e-5 * nom0) {
		// for (unsigned int ii = 0; ii < instance.n * 10; ii++) {
		// 	xTs = 0.0;
		// 	unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
		// 	for (unsigned int i = 0; i < n; i++) {
		// 		gradAvg[i] -= gradIdx[idx * n + i];
		// 		gradIdx[idx * n + i] = 0.0;
		// 	}
		// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
		// 		xTs += instance.A_csr_values[i] * x[instance.A_csr_col_idx[i]];
		// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++){
		// 		gradIdx[idx * n + instance.A_csr_col_idx[i]] = instance.A_csr_values[i] * xTs;
		// 		gradAvg[instance.A_csr_col_idx[i]] += gradIdx[idx * n + instance.A_csr_col_idx[i]];
		// 	}
		// 	for (unsigned int i = 0; i < n; i++) {
		// 		x[i] -= eta * (1.0/ instance.n * gradAvg[i] - b[i] + diag * x[i]);
		// 	}
		// }





	}

	virtual void oneCocoaSDCAUpdate(ProblemData<unsigned int, double> &instance, std::vector<double> &w,
	                                std::vector<double> &deltaAlpha, std::vector<double> &deltaW) {


		L idx = rand() / (0.0 + RAND_MAX) * instance.n;
		// compute "delta alpha" = argmin
		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (w[instance.A_csr_col_idx[i]]
			               + 1.0 * instance.penalty * deltaW[instance.A_csr_col_idx[i]])
			              * instance.A_csr_values[i];
		}
		D alphaI = instance.x[idx] + deltaAlpha[idx];
		D deltaAl = 0; // FINISH
		deltaAl = (1.0  - alphaI - dotProduct * instance.b[idx]) * instance.Li[idx];
		deltaAlpha[idx] += deltaAl;

		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl
			                                     * instance.A_csr_values[i] * instance.b[idx];

	}



	virtual void computeStoGrad_SVRG(int iter, int freq, unsigned int batchGrad,
	                                 std::vector<double> &w, std::vector<double> &wRec,
	                                 ProblemData<unsigned int, double> instance,
	                                 std::vector<double> &xTw, std::vector<double> &xTwRec,
	                                 std::vector<double> &gradientFull, std::vector<double> &gradientRec,
	                                 std::vector<double> &gradient, std::vector<unsigned int> &randIdxGrad) {


		if (iter % freq == 1) {
			// computeVectorTimesData(w, instance, xTw, world, mode);
			// computeGradient(w, gradientFull, xTw, instance, world, mode);
			cblas_set_to_zero(gradientFull);
			for (unsigned int idx = 0; idx < instance.n; idx++) {
				xTw[idx] = 0;

				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					xTw[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

				xTw[idx] = xTw[idx] * instance.b[idx];
				xTwRec[idx] = xTw[idx];
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					gradientFull[instance.A_csr_col_idx[i]] += (xTw[idx] - 1.0) *
					        instance.A_csr_values[i] * instance.b[idx] / instance.n;
			}
			//cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &gradientFull[0], 1);
			cblas_dcopy(instance.m, &w[0], 1, &wRec[0], 1);
		}

		cblas_set_to_zero(gradient);
		cblas_set_to_zero(gradientRec);
		for (unsigned int j = 0; j < batchGrad; j++) {
			unsigned int idx = randIdxGrad[j];
			xTw[idx] = 0;

			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				xTw[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

			xTw[idx] = xTw[idx] * instance.b[idx];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				gradient[instance.A_csr_col_idx[i]] += (xTw[idx] - 1.0) * instance.A_csr_values[i] * instance.b[idx]
				                                       / batchGrad;
				gradientRec[instance.A_csr_col_idx[i]] += (xTwRec[idx] - 1.0) * instance.A_csr_values[i] * instance.b[idx]
				        / batchGrad;
			}
		}

		// cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &gradient[0], 1);
		// cblas_daxpy(instance.m, instance.lambda, &wRec[0], 1, &gradientRec[0], 1);

		for (unsigned int i = 0; i < instance.m; i++) {
			gradient[i] = gradient[i] + instance.lambda * w[i] - gradientRec[i] + gradientFull[i];

		}

	}

	virtual void StoWoodburyHGet(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                             unsigned int &batchHessian, std::vector<double> &woodburyH,
	                             std::vector<double> &wTx, std::vector<unsigned int> &randIdx, double & diag) {
		cblas_set_to_zero(woodburyH);
		for (unsigned int ii = 0; ii < batchHessian; ii++) {
			unsigned int idx1 = randIdx[ii];
			for (unsigned int jj = 0; jj < batchHessian; jj++) {
				unsigned int idx2 = randIdx[jj];
				unsigned int i = instance.A_csr_row_ptr[idx1];
				unsigned int j = instance.A_csr_row_ptr[idx2];
				while (i < instance.A_csr_row_ptr[idx1 + 1] && j < instance.A_csr_row_ptr[idx2 + 1]) {
					if (instance.A_csr_col_idx[i] == instance.A_csr_col_idx[j]) {
						woodburyH[ii * batchHessian + jj] += instance.A_csr_values[i] * instance.A_csr_values[j]
						                                     * instance.b[idx1] * instance.b[idx2] / diag / batchHessian;
						i++;
						j++;
					}
					else if (instance.A_csr_col_idx[i] < instance.A_csr_col_idx[j])
						i++;
					else
						j++;
				}
			}
		}
		for (unsigned int idx = 0; idx < batchHessian; idx++)
			woodburyH[idx * batchHessian + idx] += 1.0;

	}
	virtual void StoWoodburySolve(unsigned int batchHessian, std::vector<double> &w, ProblemData<unsigned int, double> instance,
	                              std::vector<double> &woodburyH,
	                              std::vector<double> &gradient, std::vector<double> &woodburyZHVTy,
	                              std::vector<double> &woodburyVTy, std::vector<double> &woodburyHVTy,
	                              std::vector<double> &vk, std::vector<unsigned int> &randIdx) {


		cblas_set_to_zero(woodburyVTy);

		for (unsigned int j = 0; j < batchHessian; j++) {
			unsigned int idx = randIdx[j];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				woodburyVTy[j] += instance.A_csr_values[i] * instance.b[idx] *
				                  gradient[instance.A_csr_col_idx[i]] / instance.lambda / batchHessian;
			}
		}

		CGSolver(woodburyH, batchHessian, woodburyVTy, woodburyHVTy);

		for (unsigned int j = 0; j < batchHessian; j++) {
			unsigned int idx = randIdx[j];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
				        / instance.lambda * woodburyHVTy[j];
			}
		}

		for (unsigned int i = 0; i < instance.m; i++) {
			vk[i] =  (gradient[i] / instance.lambda  - woodburyZHVTy[i]);
			woodburyZHVTy[i] = 0;
		}

	}


	// virtual void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	//                              ProblemData<unsigned int, double> &preConData, double &mu,
	//                              std::vector<double> &vk, double &deltak, unsigned int &batchSizeP, unsigned int &batchSizeH,
	//                              boost::mpi::communicator &world, std::ofstream &logFile, int &mode) {


	// 	std::vector<int> flag(2);

	// 	std::vector<double> constantLocal(8);
	// 	std::vector<double> constantSum(8);

	// 	double start = 0;
	// 	double finish = 0;
	// 	double elapsedTime = 0;
	// 	double grad_norm;

	// 	double epsilon;
	// 	double alpha = 0.0;
	// 	double beta = 0.0;

	// 	std::vector<double> objective(2);
	// 	std::vector<double> v(instance.m);
	// 	std::vector<double> s(instance.m);
	// 	std::vector<double> r(instance.m);
	// 	std::vector<double> u(instance.m);
	// 	std::vector<double> xTu(instance.n);
	// 	std::vector<double> xTw(instance.n);
	// 	std::vector<double> Hv(instance.m);
	// 	std::vector<double> Hu(instance.m);
	// 	std::vector<double> gradient(instance.m);
	// 	std::vector<unsigned int> randIdx(batchSizeH);
	// 	std::vector<unsigned int> oneToN(instance.n);
	// 	for (unsigned int idx = 0; idx < instance.n; idx++)
	// 		oneToN[idx] = idx;
	// 	std::vector<double> woodburyH(batchSizeP * batchSizeP);
	// 	double diag = instance.lambda + mu;

	// 	computeVectorTimesData(w, instance, xTw, world, mode);
	// 	computeObjective(w, instance, xTw, objective[0], world, mode);
	// 	computeGradient(w, gradient, xTw, instance, world, mode);
	// 	if (mode == 1) {
	// 		geneWoodburyH(instance, batchSizeP, woodburyH, diag);
	// 		if (world.rank() == 0) {
	// 			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
	// 			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
	// 			       0, 0, grad_norm, objective[0]);
	// 			logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << objective[0] << endl;
	// 		}
	// 	}
	// 	else if (mode == 2) {
	// 		geneWoodburyH(preConData, batchSizeP, woodburyH, diag);
	// 		grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
	// 		constantLocal[6] = grad_norm * grad_norm;
	// 		vall_reduce(world, constantLocal, constantSum);
	// 		constantSum[6] = sqrt(constantSum[6]);
	// 		if (world.rank() == 0) {
	// 			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
	// 			       0, 0, constantSum[6], objective[0]);
	// 			logFile << 0 << "," << 0 << "," << 0 << "," << constantSum[6]  << "," << objective[0] << endl;
	// 		}
	// 	}
	// 	for (int iter = 1; iter <= 100; iter++) {

	// 		geneRandIdx(oneToN, randIdx, instance.n, batchSizeH);
	// 		//for (unsigned int idx = 0; idx < batchSizeH; idx++) cout<<randIdx[idx]<<"   ";

	// 		start = gettime_();
	// 		if (mode == 1) {
	// 			flag[0] = 1;
	// 			flag[1] = 1;
	// 			vbroadcast(world, w, 0);

	// 		}
	// 		else if (mode == 2) {
	// 			flag[0] = 1;
	// 			constantLocal[5] = flag[0];
	// 			constantSum[5] = flag[0] * world.size();
	// 		}

	// 		cblas_set_to_zero(v);
	// 		cblas_set_to_zero(Hv);
	// 		computeVectorTimesData(w, instance, xTw, world, mode);
	// 		computeGradient(w, gradient, xTw, instance, world, mode);


	// 		if (mode == 1) {
	// 			if (world.rank() == 0) {
	// 				grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
	// 				epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
	// 				if (grad_norm < 1e-10) {
	// 					flag[1] = 0;
	// 				}
	// 				cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);
	// 				// s= p^-1 r
	// 				if (batchSizeP == 0)
	// 					ifNoPreconditioning(instance.m, r, s);
	// 				//SGDSolver(instance, instance.m, r, s, diag);
	// 				else
	// 					WoodburySolverForDisco(instance, instance.m, randIdx, batchSizeP, woodburyH, r, s, diag);

	// 				cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
	// 			}
	// 		}
	// 		else if (mode == 2) {
	// 			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
	// 			constantLocal[6] = grad_norm * grad_norm;
	// 			epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
	// 			cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);
	// 			// s= p^-1 r
	// 			if (batchSizeP == 0)
	// 				ifNoPreconditioning(instance.m, r, s);
	// 			else
	// 				WoodburySolverForOcsid(preConData, instance, instance.m, batchSizeP, woodburyH, r, s, diag, world);

	// 			cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
	// 		}

	// 		int inner_iter = 0;
	// 		while (1) { //		while (flag != 0)
	// 			if (mode == 1) {
	// 				vbroadcast(world, u, 0);
	// 				if (flag[0] == 0)
	// 					break;
	// 			}
	// 			else if (mode == 2) {
	// 				vall_reduce(world, constantLocal, constantSum);
	// 				if (constantSum[5] == 0)
	// 					break; // stop if all the inner flag = 0.
	// 			}

	// 			computeVectorTimesData(u, instance, xTu, world, mode);
	// 			computeHessianTimesAU(u, Hu, xTu, instance, batchSizeH, randIdx, world, mode);

	// 			if (mode == 1) {
	// 				if (world.rank() == 0) {
	// 					//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
	// 					double nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
	// 					double denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
	// 					alpha = nom / denom;

	// 					for (unsigned int i = 0; i < instance.m ; i++) {
	// 						v[i] += alpha * u[i];
	// 						Hv[i] += alpha * Hu[i];
	// 						r[i] -= alpha * Hu[i];
	// 					}
	// 					// solve linear system to get new s
	// 					if (batchSizeP == 0)
	// 						ifNoPreconditioning(instance.m, r, s);
	// 					//SGDSolver(instance, instance.m, r, s, diag);
	// 					else
	// 						WoodburySolverForDisco(instance, instance.m, randIdx, batchSizeP, woodburyH, r, s, diag);

	// 					double nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
	// 					beta = nom_new / nom;

	// 					for (unsigned int i = 0; i < instance.m ; i++)
	// 						u[i] = beta * u[i] + s[i];

	// 					double r_norm = cblas_l2_norm(instance.m, &r[0], 1);

	// 					if (r_norm <= epsilon || inner_iter > 500) {
	// 						cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
	// 						double vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
	// 						double vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
	// 						deltak = sqrt(vHv + alpha * vHu);
	// 						flag[0] = 0;
	// 					}
	// 					inner_iter++;
	// 				}
	// 				vbroadcast(world, flag, 0);
	// 			}
	// 			else if (mode == 2) {
	// 				double rsLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
	// 				double uHuLocal = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
	// 				constantLocal[0] = rsLocal;
	// 				constantLocal[1] = uHuLocal;
	// 				vall_reduce(world, constantLocal, constantSum);

	// 				alpha = constantSum[0] / constantSum[1];
	// 				for (unsigned int i = 0; i < instance.m ; i++) {
	// 					v[i] += alpha * u[i];
	// 					Hv[i] += alpha * Hu[i];
	// 					r[i] -= alpha * Hu[i];
	// 				}

	// 				//CGSolver(P, instance.m, r, s);
	// 				if (batchSizeP == 0)
	// 					ifNoPreconditioning(instance.m, r, s);
	// 				else
	// 					WoodburySolverForOcsid(preConData, instance, instance.m, batchSizeP, woodburyH, r, s, diag, world);

	// 				double rsNextLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
	// 				constantLocal[2] = rsNextLocal;
	// 				vall_reduce(world, constantLocal, constantSum);
	// 				beta = constantSum[2] / constantSum[0];

	// 				for (unsigned int i = 0; i < instance.m ; i++) {
	// 					u[i] = beta * u[i] + s[i];
	// 				}
	// 				double r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);

	// 				if ( r_normLocal <= epsilon || inner_iter > 500) {			//	if (r_norm <= epsilon || inner_iter > 100)
	// 					cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
	// 					double vHvLocal = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
	// 					double vHuLocal = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
	// 					constantLocal[3] = vHvLocal;
	// 					constantLocal[4] = vHuLocal;
	// 					flag[0] = 0;
	// 					constantLocal[5] = flag[0];
	// 				}
	// 				inner_iter++;
	// 			}

	// 		}
	// 		if (mode == 1) {
	// 			if (world.rank() == 0)
	// 				cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
	// 			vbroadcast(world, w, 0);
	// 		}
	// 		else if (mode == 2) {
	// 			vall_reduce(world, constantLocal, constantSum);
	// 			deltak = sqrt(constantSum[3] + alpha * constantSum[4]);
	// 			cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
	// 			constantSum[6] = sqrt(constantSum[6]);
	// 		}

	// 		finish = gettime_();
	// 		elapsedTime += finish - start;

	// 		computeVectorTimesData(w, instance, xTw, world, mode);
	// 		computeObjective(w, instance, xTw, objective[0], world, mode);

	// 		output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);
	// 		if (mode == 1) {
	// 			if (flag[1] == 0)
	// 				break;
	// 		}
	// 		else if (mode == 2) {
	// 			if (constantSum[6] < 1e-10) {
	// 				break;
	// 			}
	// 		}


	// 	}
	// }


	// virtual void output(ProblemData<unsigned int, double> &instance, int &iter, int &inner_iter, double & elapsedTime,
	//                     std::vector<double> &constantSum, std::vector<double> &objective, double & grad_norm,
	//                     std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

	// 	if (mode == 1) {
	// 		if (world.rank() == 0) {
	// 			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
	// 			       iter, inner_iter, grad_norm, objective[0]);
	// 			logFile << iter << "," << inner_iter << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
	// 		}
	// 	}
	// 	else if (mode == 2) {
	// 		if (world.rank() == 0) {
	// 			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
	// 			       iter, inner_iter, constantSum[6], objective[0]);
	// 			logFile << iter << "," << inner_iter << "," << elapsedTime << "," << constantSum[6] << "," << objective[0] << endl;
	// 		}

	// 	}

	// }

	virtual void computeInitialW(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double & rho, int rank) {
		std::vector<double> deltaW(instance.m);
		std::vector<double> deltaAlpha(instance.n);
		std::vector<double> alpha(instance.n);
		std::vector<double> Li(instance.n);
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			double norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
			                            &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			Li[idx] = 1.0 / (norm * norm / rho / instance.n + 1.0);
		}

		for (unsigned int jj = 0; jj < 10; jj++) {
			cblas_set_to_zero(deltaW);
			cblas_set_to_zero(deltaAlpha);

			for (unsigned int it = 0; it < floor(instance.n / 10); it++) {
				unsigned int idx = rand() / (0.0 + RAND_MAX) * instance.n;

				double dotProduct = 0;
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
					dotProduct += (w[instance.A_csr_col_idx[i]]
					               + 1.0 * deltaW[instance.A_csr_col_idx[i]])
					              * instance.A_csr_values[i];
				}
				double alphaI = alpha[idx] + deltaAlpha[idx];
				double deltaAl = 0;
				deltaAl = (1.0 - alphaI - dotProduct * instance.b[idx]) * Li[idx];
				deltaAlpha[idx] += deltaAl;
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					deltaW[instance.A_csr_col_idx[i]] += 1.0 / instance.n / rho * deltaAl
					                                     * instance.A_csr_values[i] * instance.b[idx];

			}
			cblas_daxpy(instance.m, 1.0, &deltaW[0], 1, &w[0], 1);
			cblas_daxpy(instance.n, 1.0, &deltaAlpha[0], 1, &alpha[0], 1);
		}
		double primalError;
		double dualError;

		computePrimalAndDualObjective(instance, alpha, w, rho, dualError, primalError);
		printf("No. %i node now has the duality gap %E \n", rank, primalError + dualError);

	}

};

// virtual void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
//                              std::vector<double> &vk, double &deltak, unsigned int &batchSize,
//                              boost::mpi::communicator &world, std::ofstream &logFile) {

// 	int mode = 1;

// 	std::vector<int> flag(2);
// 	mpi::request reqs[1];

// 	double start = 0;
// 	double finish = 0;
// 	double elapsedTime = 0;
// 	double grad_norm;

// 	double epsilon;
// 	double alpha = 0.0;
// 	double beta = 0.0;
// 	std::vector<double> v(instance.m);
// 	std::vector<double> s(instance.m);
// 	std::vector<double> r(instance.m);
// 	std::vector<double> u(instance.m);
// 	std::vector<double> xTu(instance.n);
// 	std::vector<double> xTw(instance.n);
// 	std::vector<double> Hu(instance.m);
// 	std::vector<double> Hv(instance.m);
// 	std::vector<double> gradient(instance.m);
// 	std::vector<unsigned int> randPick(batchSize);
// 	std::vector<double> woodburyH(batchSize * batchSize);
// 	std::vector<double> objective(2);
// 	double diag = instance.lambda + mu;

// 	computeVectorTimesData(w, instance, xTw, world, mode);
// 	computeObjective(w, instance, xTw, objective[0], world, mode);
// 	computeGradient(w, gradient, xTw, instance, world, mode);

// 	geneWoodburyH(instance, batchSize, woodburyH, diag);

// 	if (world.rank() == 0) {
// 		grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
// 		printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
// 		       0, 0, grad_norm, objective[0]);
// 		logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << objective[0] << endl;
// 	}

// 	for (unsigned int iter = 1; iter <= 100; iter++) {

// 		start = gettime_();

// 		flag[0] = 1;
// 		flag[1] = 1;

// 		cblas_set_to_zero(v);
// 		cblas_set_to_zero(Hv);
// 		vbroadcast(world, w, 0);
// 		computeVectorTimesData(w, instance, xTw, world, mode);
// 		computeGradient(w, gradient, xTw, instance, world, mode);

// 		if (world.rank() == 0) {
// 			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
// 			epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
// 			if (grad_norm < 1e-8) {
// 				flag[1] = 0;
// 			}

// 			cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

// 			// s= p^-1 r
// 			if (batchSize == 0)
// 				ifNoPreconditioning(instance.m, r, s);
// 			else
// 				WoodburySolverForDisco(instance, instance.m, batchSize, woodburyH, r, s, diag);

// 			cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);

// 		}

// 		int inner_iter = 0;
// 		while (flag[0] != 0) {
// 			vbroadcast(world, u, 0);
// 			computeVectorTimesData(u, instance, xTu, world, mode);
// 			computeHessianTimesAU(u, Hu, xTu, instance, world, mode);

// 			if (world.rank() == 0) {
// 				//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
// 				double nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
// 				double denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
// 				alpha = nom / denom;

// 				for (unsigned int i = 0; i < instance.m ; i++) {
// 					v[i] += alpha * u[i];
// 					Hv[i] += alpha * Hu[i];
// 					r[i] -= alpha * Hu[i];
// 				}
// 				// solve linear system to get new s
// 				if (batchSize == 0)
// 					ifNoPreconditioning(instance.m, r, s);
// 				else
// 					WoodburySolverForDisco(instance, instance.m, batchSize, woodburyH, r, s, diag);

// 				double nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
// 				beta = nom_new / nom;

// 				for (unsigned int i = 0; i < instance.m ; i++) {
// 					u[i] = beta * u[i] + s[i];
// 				}

// 				double r_norm = cblas_l2_norm(instance.m, &r[0], 1);

// 				if (r_norm <= epsilon || inner_iter > 500) {
// 					cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
// 					double vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
// 					double vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
// 					deltak = sqrt(vHv + alpha * vHu);
// 					flag[0] = 0;
// 				}
// 				inner_iter++;
// 			}

// 			vbroadcast(world, flag, 0);

// 		}

// 		if (world.rank() == 0) {
// 			cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
// 		}

// 		finish = gettime_();
// 		elapsedTime += finish - start;
// 		vbroadcast(world, w, 0);

// 		computeVectorTimesData(w, instance, xTw, world, mode);
// 		computeObjective(w, instance, xTw, objective[0], world, mode);
// 		if (world.rank() == 0) {
// 			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
// 			       iter, inner_iter, grad_norm, objective[0]);
// 			logFile << iter << "," << 2 * inner_iter + 2 << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
// 		}

// 		if (flag[1] == 0)
// 			break;


// 	}


// }

#endif /* QUADRATICLOSS_H_ */
