/*
 * LogisticLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */
#ifndef LOGISTICLOSS_H_
#define LOGISTICLOSS_H_

#include "LossFunction.h"
#include "QR_solver.h"

template<typename L, typename D>
class LogisticLoss : public LossFunction<L, D> {
public:

	LogisticLoss() {}

	virtual ~LogisticLoss() {}

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
			obj += log(1.0 + exp(-xTw[idx]));

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
	        std::vector<double> &alpha, std::vector<double> &w, double &rho, double &finalDualError,
	        double &finalPrimalError) {

		double localError = 0;
		for (unsigned int i = 0; i < instance.n; i++) {
			double tmp = 0;
			if (instance.b[i] == -1.0) {
				if (alpha[i] < 0) {
					tmp += -alpha[i] * log(-alpha[i]) ;
				}
				if (alpha[i] > -1) {
					tmp += (1.0 + alpha[i]) * log(1.0 + alpha[i]);
				}

			}
			if (instance.b[i] == 1.0) {
				if (alpha[i] > 0) {
					tmp += alpha[i] * log(alpha[i]) ;
				}
				if (alpha[i] < 1) {
					tmp += (1.0 - alpha[i]) * log(1.0 - alpha[i]);
				}
			}
			localError += tmp;
		}

		double localLogisticLoss = 0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			double dotProduct = 0;
			for (unsigned int i = instance.A_csr_row_ptr[idx];
			        i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += (w[instance.A_csr_col_idx[i]])
				              * instance.A_csr_values[i];
			}

			double tmp = -1.0 * instance.b[idx] * instance.b[idx] * dotProduct;
			localLogisticLoss += log(1.0 + exp(tmp));

		}
		finalPrimalError = 0;
		finalDualError = 0;

		double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
		finalDualError = 1.0 / instance.n * localError
		                 + 0.5 * rho * tmp2 * tmp2;
		finalPrimalError =  1.0 / instance.n * localLogisticLoss
		                    + 0.5 * rho * tmp2 * tmp2;

	}



	virtual void computeGradient(std::vector<double> &w, std::vector<double> &grad, std::vector<double> &xTw,
	                             ProblemData<unsigned int, double> &instance, boost::mpi::communicator & world, int &mode) {

		cblas_set_to_zero(grad);

		double temp;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			temp = exp(-1.0 * xTw[idx]);
			temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.total_n;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i];
		}

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

		double temp, scalar;
		cblas_set_to_zero(Hu);

		for (unsigned int i = 0; i < batchSizeH ; i++) {
			unsigned int idx = randIdx[i];
			temp = exp(-xTw[idx]);
			scalar = temp / (temp + 1) / (temp + 1);
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				Hu[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx] * scalar
				                                 * xTu[idx] / batchSizeH;
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





	virtual void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                             ProblemData<unsigned int, double> &preConData, double &mu,
	                             std::vector<double> &vk, double &deltak, unsigned int &batchSizeP, unsigned int &batchSizeH,
	                             boost::mpi::communicator &world, std::ofstream &logFile, int &mode) {


		std::vector<int> flag(2);

		std::vector<double> constantLocal(8);
		std::vector<double> constantSum(8);

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		double grad_norm;

		double epsilon;
		double alpha = 0.0;
		double beta = 0.0;

		std::vector<double> objective(2);
		std::vector<double> v(instance.m);
		std::vector<double> s(instance.m);
		std::vector<double> r(instance.m);
		std::vector<double> u(instance.m);
		std::vector<double> xTu(instance.n);
		std::vector<double> xTw(instance.n);
		std::vector<double> Hv(instance.m);
		std::vector<double> Hu(instance.m);
		std::vector<double> gradient(instance.m);
		std::vector<unsigned int> randIdx(batchSizeH);
		std::vector<unsigned int> oneToN(instance.n);
		for (unsigned int idx = 0; idx < instance.n; idx++)
			oneToN[idx] = idx;
		std::vector<double> woodburyH(batchSizeP * batchSizeP);
		double diag = instance.lambda + mu;

		computeVectorTimesData(w, instance, xTw, world, mode);
		computeObjective(w, instance, xTw, objective[0], world, mode);
		computeGradient(w, gradient, xTw, instance, world, mode);

		if (mode == 1) {
			if (world.rank() == 0) {
				grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
				       0, 0, grad_norm, objective[0]);
				logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << objective[0] << endl;
			}
		}
		else if (mode == 2) {
			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			constantLocal[6] = grad_norm * grad_norm;
			vall_reduce(world, constantLocal, constantSum);
			if (world.rank() == 0) {
				printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
				       0, 0, constantSum[6], objective[0]);
				logFile << 0 << "," << 0 << "," << 0 << "," << constantSum[6]  << "," << objective[0] << endl;
			}
		}
		for (int iter = 1; iter <= 100; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchSizeH);

			start = gettime_();

			if (mode == 1) {
				flag[0] = 1;
				flag[1] = 1;
				vbroadcast(world, w, 0);

			}
			else if (mode == 2) {
				flag[0] = 1;
				constantLocal[5] = flag[0];
				constantSum[5] = flag[0] * world.size();
			}

			cblas_set_to_zero(v);
			cblas_set_to_zero(Hv);
			computeVectorTimesData(w, instance, xTw, world, mode);
			computeGradient(w, gradient, xTw, instance, world, mode);


			if (mode == 1) {
				if (world.rank() == 0) {
					grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
					epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
					if (grad_norm < 1e-10) {
						flag[1] = 0;
					}
					cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

					computeVectorTimesData(w, instance, xTw, world, mode);
					geneWoodburyHLogistic(instance, batchSizeP, woodburyH, xTw, diag);
					// s= p^-1 r
					if (batchSizeP == 0)
						ifNoPreconditioning(instance.m, r, s);
					else
						WoodburySolverForDiscoLogistic(instance, instance.m, batchSizeP, woodburyH, r, s, xTw, diag);

					cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
				}
			}
			else if (mode == 2) {
				grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				constantLocal[6] = grad_norm * grad_norm;
				epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
				cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);
				// s= p^-1 r
				computeVectorTimesData(w, instance, xTw, world, mode);
				geneWoodburyHLogistic(instance, batchSizeP, woodburyH, xTw, diag);
				if (batchSizeP == 0)
					ifNoPreconditioning(instance.m, r, s);
				else
					WoodburySolverForOcsidLogistic(preConData, instance, instance.m, batchSizeP, woodburyH, r, s, xTw, diag, world);

				cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
			}

			int inner_iter = 0;
			while (1) { //		while (flag != 0)
				if (mode == 1) {
					vbroadcast(world, u, 0);
					if (flag[0] == 0)
						break;
				}
				else if (mode == 2) {
					if (constantSum[5] == 0)
						break; // stop if all the inner flag = 0.
				}

				computeVectorTimesData(u, instance, xTu, world, mode);
				computeHessianTimesAU(u, Hu, xTw, xTu, instance, batchSizeH, randIdx, world, mode);

				if (mode == 1) {
					if (world.rank() == 0) {
						//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
						double nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
						double denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
						alpha = nom / denom;

						for (unsigned int i = 0; i < instance.m ; i++) {
							v[i] += alpha * u[i];
							Hv[i] += alpha * Hu[i];
							r[i] -= alpha * Hu[i];
						}
						// solve linear system to get new s
						if (batchSizeP == 0)
							ifNoPreconditioning(instance.m, r, s);
						else
							WoodburySolverForDiscoLogistic(instance, instance.m, batchSizeP, woodburyH, r, s, xTw, diag);

						double nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
						beta = nom_new / nom;

						for (unsigned int i = 0; i < instance.m ; i++)
							u[i] = beta * u[i] + s[i];

						double r_norm = cblas_l2_norm(instance.m, &r[0], 1);

						if (r_norm <= epsilon || inner_iter > 500) {
							cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
							double vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
							double vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
							deltak = sqrt(vHv + alpha * vHu);
							flag[0] = 0;
						}
						inner_iter++;
					}
					vbroadcast(world, flag, 0);
				}
				else if (mode == 2) {
					double rsLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					double uHuLocal = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
					constantLocal[0] = rsLocal;
					constantLocal[1] = uHuLocal;
					vall_reduce(world, constantLocal, constantSum);

					alpha = constantSum[0] / constantSum[1];
					for (unsigned int i = 0; i < instance.m ; i++) {
						v[i] += alpha * u[i];
						Hv[i] += alpha * Hu[i];
						r[i] -= alpha * Hu[i];
					}

					//CGSolver(P, instance.m, r, s);
					if (batchSizeP == 0)
						ifNoPreconditioning(instance.m, r, s);
					else
						WoodburySolverForOcsidLogistic(preConData, instance, instance.m, batchSizeP, woodburyH, r, s, xTw, diag, world);

					double rsNextLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					constantLocal[2] = rsNextLocal;
					vall_reduce(world, constantLocal, constantSum);
					beta = constantSum[2] / constantSum[0];

					for (unsigned int i = 0; i < instance.m ; i++) {
						u[i] = beta * u[i] + s[i];
					}
					double r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);

					if ( r_normLocal <= epsilon || inner_iter > 500) {			//	if (r_norm <= epsilon || inner_iter > 100)
						cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
						double vHvLocal = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
						double vHuLocal = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
						constantLocal[3] = vHvLocal;
						constantLocal[4] = vHuLocal;
						flag[0] = 0;
						constantLocal[5] = flag[0];
					}
					inner_iter++;
				}

			}
			if (mode == 1) {
				if (world.rank() == 0)
					cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
				vbroadcast(world, w, 0);
			}
			else if (mode == 2) {
				vall_reduce(world, constantLocal, constantSum);
				deltak = sqrt(constantSum[3] + alpha * constantSum[4]);
				cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
			}

			finish = gettime_();
			elapsedTime += finish - start;

			computeVectorTimesData(w, instance, xTw, world, mode);
			computeObjective(w, instance, xTw, objective[0], world, mode);

			output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);
			if (mode == 1) {
				if (flag[1] == 0)
					break;
			}
			else if (mode == 2) {
				if (constantSum[6] < 1e-10) {
					break;
				}
			}
		}
	}


	virtual void output(ProblemData<unsigned int, double> &instance, int &iter, int &inner_iter, double & elapsedTime,
	                    std::vector<double> &constantSum, std::vector<double> &objective, double & grad_norm,
	                    std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (mode == 1) {
			if (world.rank() == 0) {
				printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
				       iter, inner_iter, grad_norm, objective[0]);
				logFile << iter << "," << inner_iter << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
			}
		}
		else if (mode == 2) {
			if (world.rank() == 0) {
				printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
				       iter, inner_iter, constantSum[6], objective[0]);
				logFile << iter << "," << inner_iter << "," << elapsedTime << "," << constantSum[6] << "," << objective[0] << endl;
			}

		}

	}


	virtual void computeInitialW(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double & rho, int rank) {
		std::vector<double> deltaW(instance.m);
		std::vector<double> deltaAlpha(instance.n);
		std::vector<double> alpha(instance.n);

		for (unsigned int jj = 0; jj < 10; jj++) {
			cblas_set_to_zero(deltaW);
			cblas_set_to_zero(deltaAlpha);

			for (unsigned int it = 0; it < floor(instance.n / 10); it++) {
				unsigned int idx = rand() / (0.0 + RAND_MAX) * instance.n;

				double dotProduct = 0;
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
					dotProduct += (w[instance.A_csr_col_idx[i]] + 1.0 * deltaW[instance.A_csr_col_idx[i]])
					              * instance.A_csr_values[i];
				}
				double alphaI = alpha[idx] + deltaAlpha[idx];

				double norm = cblas_l2_norm(
				                  instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
				                  &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

				double deltaAl = 0.0;
				double epsilon = 1e-5;
				if (alphaI == 0) {deltaAl = 0.1 * instance.b[idx];}
				double FirstDerivative = 1.0 * deltaAl / instance.n / rho  * norm * norm
				                         + dotProduct * instance.b[idx] - log(1.0 - (alphaI + deltaAl) / instance.b[idx]) / instance.b[idx]
				                         + log((alphaI + deltaAl) / instance.b[idx]) / instance.b[idx];

				while (FirstDerivative > epsilon || FirstDerivative < - 1.0 * epsilon)
				{
					double SecondDerivative = 1.0 * norm * norm / instance.n / rho
					                          + 1.0 / (1.0 - (alphaI + deltaAl) / instance.b[idx]) + 1.0 / (alphaI + deltaAl) / instance.b[idx];
					deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
					if (instance.b[idx] == 1.0)
						deltaAl = (deltaAl > 1 - alphaI) ? 1 - alphaI - 1e-15 : (deltaAl < -alphaI ? -alphaI + 1e-15 : deltaAl);
					else if (instance.b[idx] == -1.0)
						deltaAl = (deltaAl > -alphaI) ? -alphaI - 1e-15 : (deltaAl < -1.0 - alphaI ? -1.0 - alphaI + 1e-15 : deltaAl);
					FirstDerivative = 1.0 * deltaAl / instance.n / rho * norm * norm
					                  + dotProduct * instance.b[idx] - log(1.0 - (alphaI + deltaAl) / instance.b[idx]) / instance.b[idx]
					                  + log((alphaI + deltaAl) / instance.b[idx]) / instance.b[idx];
				}
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

#endif /* LOGISTICLOSS_H_ */
