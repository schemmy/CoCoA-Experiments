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
	QuadraticLoss() {

	}

	virtual ~QuadraticLoss() {}



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
			obj += 0.5 * (xTw[idx]  - instance.b[idx]) * (xTw[idx] - instance.b[idx]);

		double wNorm = cblas_l2_norm(w.size(), &w[0], 1);

		if (mode == 1) {
			std::vector<double> obj_local(2);
			std::vector<double> obj_world(2);
			obj_local[0] = 1.0 / instance.total_n * obj + 0.5 * instance.lambda * wNorm * wNorm / world.size();
			vall_reduce(world, obj_local, obj_world);
			obj = obj_world[0];
		}
		else if (mode == 2)
			obj = 1.0 / instance.total_n * obj + 0.5 * instance.lambda * wNorm * wNorm;


	}


	virtual void computePrimalAndDualObjective(ProblemData<unsigned int, double> &instance,
	        std::vector<double> &alpha, std::vector<double> &w, double & rho, double & finalDualError,
	        double & finalPrimalError) {

		double localError = 0.0;
		for (unsigned int i = 0; i < instance.n; i++) {
			double tmp = alpha[i] * alpha[i] * 0.5 - instance.b[i] * alpha[i];
			localError += tmp;
		}

		double localQuadLoss = 0.0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			double dotProduct = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx];
			        i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
			}
			double single = 1.0 * instance.b[idx] -  dotProduct * instance.b[idx];
			double tmp = 0.5 * single * single;

			localQuadLoss += tmp;
		}

		finalPrimalError = 0;
		finalDualError = 0;

		double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
		finalDualError = 1.0 / instance.n * localError
		                 + 0.5 * rho * tmp2 * tmp2;
		finalPrimalError =  1.0 / instance.n * localQuadLoss
		                    + 0.5 * rho * tmp2 * tmp2;

	}

	virtual void computeGradient(std::vector<double> &w, std::vector<double> &grad, std::vector<double> &xTw,
	                             ProblemData<unsigned int, double> &instance, boost::mpi::communicator & world, int &mode) {

		cblas_set_to_zero(grad);

		for (unsigned int idx = 0; idx < instance.n; idx++)
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				grad[instance.A_csr_col_idx[i]] += (xTw[idx] - instance.b[idx]) * instance.A_csr_values[i] * instance.b[idx]
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


	virtual void computeHessianTimesAU(std::vector<double> &u, std::vector<double> &Hu,
	                                   std::vector<double> &xTu, ProblemData<unsigned int, double> &instance,
	                                   boost::mpi::communicator & world, int &mode) {

		cblas_set_to_zero(Hu);

		for (unsigned int idx = 0; idx < instance.n ; idx++) {
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				Hu[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx] * xTu[idx] / instance.total_n;
			}
		}


		for (unsigned int i = 0; i < instance.m; i++)
			Hu[i] += instance.lambda / world.size() * u[i];

		if (mode == 1) {
			std::vector<double> Hu_world(instance.m);
			vall_reduce(world, Hu, Hu_world);
			cblas_dcopy(instance.m, &Hu_world[0], 1, &Hu[0], 1);
		}

	}

	virtual void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
	                             std::vector<double> &vk, double &deltak, unsigned int &batchSize,
	                             boost::mpi::communicator &world, std::ofstream &logFile) {

		int mode = 1;

		std::vector<int> flag(2);
		mpi::request reqs[1];

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		double grad_norm;

		double epsilon;
		double alpha = 0.0;
		double beta = 0.0;

		std::vector<double> v(instance.m);
		std::vector<double> s(instance.m);
		std::vector<double> r(instance.m);
		std::vector<double> u(instance.m);
		std::vector<double> xTu(instance.n);
		std::vector<double> xTw(instance.n);
		std::vector<double> Hu_local(instance.m);
		std::vector<double> Hu(instance.m);
		std::vector<double> Hv(instance.m);
		std::vector<double> gradient(instance.m);
		std::vector<double> local_gradient(instance.m);
		std::vector<unsigned int> randPick(batchSize);
		std::vector<double> woodburyH(batchSize * batchSize);
		std::vector<double> objective(2);
		double diag = instance.lambda + mu;

		computeVectorTimesData(w, instance, xTw, world, mode);
		computeObjective(w, instance, xTw, objective[0], world, mode);
		computeGradient(w, gradient, xTw, instance, world, mode);

		geneWoodburyH(instance, batchSize, woodburyH, diag);

		if (world.rank() == 0) {
			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
			       0, 0, grad_norm, objective[0]);
			logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << objective[0] << endl;
		}

		for (unsigned int iter = 1; iter <= 100; iter++) {

			start = gettime_();

			flag[0] = 1;
			flag[1] = 1;

			cblas_set_to_zero(v);
			cblas_set_to_zero(Hv);
			vbroadcast(world, w, 0);
			computeVectorTimesData(w, instance, xTw, world, mode);
			computeGradient(w, gradient, xTw, instance, world, mode);

			if (world.rank() == 0) {
				grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
				if (grad_norm < 1e-8) {
					flag[1] = 0;
				}

				cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

				// s= p^-1 r
				if (batchSize == 0)
					ifNoPreconditioning(instance.m, r, s);
				else
					WoodburySolverForDisco(instance, instance.m, batchSize, woodburyH, r, s, diag);
				cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);

			}

			int inner_iter = 0;
			while (flag[0] != 0) {
				vbroadcast(world, u, 0);
				computeVectorTimesData(u, instance, xTu, world, mode);
				computeHessianTimesAU(u, Hu, xTu, instance, world, mode);

				if (world.rank() == 0) {
					//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
					double nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					double denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
					alpha = nom / denom;

					cblas_daxpy(instance.m, alpha, &u[0], 1, &v[0], 1);
					cblas_daxpy(instance.m, alpha, &Hu[0], 1, &Hv[0], 1);
					cblas_daxpy(instance.m, -alpha, &Hu[0], 1, &r[0], 1);

					// solve linear system to get new s
					if (batchSize == 0)
						ifNoPreconditioning(instance.m, r, s);
					else
						WoodburySolverForDisco(instance, instance.m, batchSize, woodburyH, r, s, diag);

					double nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					beta = nom_new / nom;
					cblas_dscal(instance.m, beta, &u[0], 1);
					cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

					double r_norm = cblas_l2_norm(instance.m, &r[0], 1);
					inner_iter++;

					if (r_norm <= epsilon || inner_iter > 500) {
						cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
						double vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
						double vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
						deltak = sqrt(vHv + alpha * vHu);
						flag[0] = 0;
					}
				}

				vbroadcast(world, flag, 0);

			}

			if (world.rank() == 0) {
				cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
			}

			finish = gettime_();
			elapsedTime += finish - start;
			vbroadcast(world, w, 0);

			computeVectorTimesData(w, instance, xTw, world, mode);
			computeObjective(w, instance, xTw, objective[0], world, mode);
			if (world.rank() == 0) {
				printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
				       iter, inner_iter, grad_norm, objective[0]);
				logFile << iter << "," << 2 * inner_iter + 2 << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
			}

			if (flag[1] == 0)
				break;


		}


	}



	virtual void computeInitialW(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &rho, int rank) {
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
				deltaAl = (1.0 * instance.b[idx] - alphaI - dotProduct * instance.b[idx]) * Li[idx];
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

#endif /* QUADRATICLOSS_H_ */
