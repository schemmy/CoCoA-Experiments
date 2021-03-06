/*
 * QuadraticLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef QUADRATICLOSSCD_H_
#define QUADRATICLOSSCD_H_

#include "QuadraticLoss.h"

template<typename L, typename D>
class QuadraticLossCD: public QuadraticLoss<L, D> {
public:
	QuadraticLossCD() {

	}
	virtual ~QuadraticLossCD() {
	}

	virtual void init(ProblemData<L, D> & instance) {

		instance.Li.resize(instance.n);
		instance.vi.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
			                       &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1.0 / (norm * norm * instance.penalty * instance.oneOverLambdaN + 1.0);

			instance.vi[idx] = norm * norm;
		}

	}


	virtual void subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, std::vector<D> &w,
	                                    std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                    mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {
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
					deltaAl = (1.0 * instance.b[idx] - alphaI - dotProduct * instance.b[idx]) * instance.Li[idx];
					deltaAlpha[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl
						                                     * instance.A_csr_values[i] * instance.b[idx];

				}
				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}
	virtual void subproblem_solver_accelerated_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW,
	        DistributedSettings & distributedSettings, mpi::communicator &world, D gamma, Context &ctx,
	        std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double theta = 1.0 * distributedSettings.iterationsPerThread / instance.n;
		std::vector<double> zk(instance.n);
		std::vector<double> uk(instance.n);
		std::vector<double> Ayk(instance.m);
		std::vector<double> yk(instance.n);
		std::vector<double> deltayk(instance.n);
		std::vector<double> deltaAyk(instance.m);
		cblas_set_to_zero(uk);
		cblas_set_to_zero(yk);
		cblas_set_to_zero(Ayk);
		std::vector<double> AykBuffer(instance.m);

		cblas_dcopy(instance.n, &instance.x[0], 1, &zk[0], 1);

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(deltayk);
				cblas_set_to_zero(deltaAyk);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++)
					this->accelerated_SDCA_oneIteration(instance, deltaAlpha, w, deltaW, zk, uk, yk, deltayk, Ayk,
					                                    deltaAyk, theta, distributedSettings);

				double thetasq = theta * theta;
				theta = 0.5 * sqrt(thetasq * thetasq + 4 * thetasq) - 0.5 * thetasq;
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
				cblas_sum_of_vectors(yk, deltayk, gamma);

				//				vectormatrix_b(instance.x, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
				//						instance.b, instance.oneOverLambdaN, instance.n, deltaW);

				//				vectormatrix_b(yk, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
				//						instance.b, instance.oneOverLambdaN, instance.n, deltaAyk);

				//				cblas_set_to_zero(w);
				//				cblas_set_to_zero(Ayk);

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				vall_reduce(world, deltaAyk, AykBuffer);
				cblas_sum_of_vectors(Ayk, AykBuffer, gamma);

			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;
			}

		}
	}

	virtual void accelerated_SDCA_oneIteration(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &deltaW, std::vector<D> &zk, std::vector<D> &uk, std::vector<D> &yk,
	        std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk, D &theta,
	        DistributedSettings & distributedSettings) {

		D thetasquare = theta * theta;
		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct +=
			    (1.0 * Ayk[instance.A_csr_col_idx[i]] + instance.penalty * deltaAyk[instance.A_csr_col_idx[i]])
			    * instance.A_csr_values[i];
		}
		//matrixvector(instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr ,
		//		yk, instance.m, Apotent);

		D tk = (1.0 * instance.b[idx] - zk[idx] - dotProduct * instance.b[idx]) //- thetasquare * uk[idx])
		       / (instance.vi[idx] * instance.n / distributedSettings.iterationsPerThread * theta * instance.penalty
		          * instance.oneOverLambdaN + 1.);
		zk[idx] += tk;
		uk[idx] -= (1.0 - theta * instance.n / distributedSettings.iterationsPerThread) / (thetasquare) * tk;

		D deltaAl = thetasquare * uk[idx] + zk[idx] - instance.x[idx] - deltaAlpha[idx];
		deltaAlpha[idx] += deltaAl;
		//cout<<idx<<"     "<<deltaAlpha[idx]<<endl;
		//instance.x[idx] = theta * theta * uk[idx] + zk[idx];
		//cout <<idx<<"          "<< theta << "    "<<uk[idx] << "    " <<zk[idx] <<endl;

		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i] * deltaAl
			                                     * instance.b[idx];
		}

		D thetanext = theta;
		thetanext = 0.5 * sqrt(thetasquare * thetasquare + 4 * thetasquare) - 0.5 * thetasquare;
		D dyk = thetanext * thetanext * uk[idx] + zk[idx] - yk[idx] - deltayk[idx];
		deltayk[idx] += dyk;
		//yk[idx] = thetanext * thetanext * uk[idx] + zk[idx];

		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			deltaAyk[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i] * dyk
			                                       * instance.b[idx];
		}

	}

	virtual void subproblem_solver_steepestdescent(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW,
	        DistributedSettings & distributedSettings, mpi::communicator &world, D gamma, Context &ctx,
	        std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double dualobj;
		std::vector < D > gradient(instance.n);
		double a;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				a = instance.lambda * 500.0 * instance.n;
				dualobj = 0;
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {

					this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);
					this->backtrack_linesearch(instance, deltaAlpha, gradient, w, dualobj, a);
					//cout<<a<<endl;
					//D gradNorm = cblas_l2_norm(instance.n, &gradient[0], 1);
					//cout<<gradNorm<<endl;
					if (a <= 1e-12)
						break;
				}

				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i]
						                                     * deltaAlpha[idx] * instance.b[idx];
				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				//cout << deltaW[0]<<"   "<<w[0]<<"   "<<wBuffer[0]<<endl;
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);

			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}
	}


	virtual void subproblem_solver_LBFGS(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, std::vector<D> &w,
	                                     std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                     mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double dualobj = 0;
		int limit_BFGS =  distributedSettings.iterationsPerThread;
		std::vector<double> old_grad(instance.n);
		std::vector<double> old_deltaAlpha(instance.n);
		std::vector<double> sk(instance.n * limit_BFGS);
		std::vector<double> rk(instance.n * limit_BFGS);

		std::vector < D > gradient(instance.n);
		std::vector < D > Hf(instance.n);
		std::vector < D > Af(instance.m);
		std::vector < D > search_direction(instance.n);
		int flag_BFGS = 0;
		std::vector < D > oneoversy(limit_BFGS);
		D stepsize;
		//distributedSettings.iters_communicate_count = 3;
		std::vector<D> gradient_temp1(instance.n);
		std::vector<D> gradient_temp2(instance.m);
		std::vector<D> gradient_temp3(instance.n);

		std::vector<D> AdeltaAlpha(instance.m);
		std::vector<D> Ad(instance.m);
		std::vector<D> AAdeltaAlpha(instance.n);
		std::vector<D> AAd(instance.n);
		D temp = 1.0 / instance.total_n / instance.lambda;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(gradient_temp3);
				this->compute_subproproblem_obj(instance, deltaAlpha, w, dualobj);
				stepsize = 0.001 * instance.n;
				matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, w, instance.n, gradient_temp1);
				// vectormatrix_b(deltaAlpha, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
				//             instance.b, 1.0, instance.n, gradient_temp2);
				// matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
				//          	gradient_temp2, instance.n, gradient_temp3);

				for (L iter_counter = 0; iter_counter < distributedSettings.iterationsPerThread; iter_counter++) {

					//cout<<gradient_temp3[0]-gradient_temp4[0]<<endl;
					for (L i = 0; i < instance.n; i++) {
						gradient[i] = 1.0 / instance.total_n * ( gradient_temp1[i] * instance.b[i] + instance.x[i] - instance.b[i]
						              + 1.0 * instance.penalty * temp * gradient_temp3[i] * instance.b[i] + deltaAlpha[i] );
					}
					//this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);

					this->LBFGS_update(instance, search_direction, old_grad, sk, rk, gradient, oneoversy, iter_counter,
					                   limit_BFGS, flag_BFGS);
					cblas_dcopy(instance.n, &deltaAlpha[0], 1, &old_deltaAlpha[0], 1);
					cblas_dcopy(instance.n, &gradient[0], 1, &old_grad[0], 1);

					vectormatrix_b(deltaAlpha, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
					               instance.b, 1.0, instance.n, AdeltaAlpha);
					vectormatrix_b(search_direction, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
					               instance.b, 1.0, instance.n, Ad);
					matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, AdeltaAlpha, instance.n, AAdeltaAlpha);
					matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, Ad, instance.n, AAd);

					this->wolfe_linesearch(instance, deltaAlpha, search_direction, gradient_temp1,
					                       AdeltaAlpha, Ad, AAdeltaAlpha, AAd, w, dualobj, stepsize);
					//stepsize =  (old_deltaAlpha[0] - deltaAlpha[0])/search_direction[0];
//cout<<stepsize<<endl;
					for (L idx = 0; idx < instance.n; idx++) {
						sk[instance.n * flag_BFGS + idx] = deltaAlpha[idx] - old_deltaAlpha[idx];
						gradient_temp3[idx] = AAdeltaAlpha[idx] - stepsize * AAd[idx];
					}

					flag_BFGS++;
					if (flag_BFGS == limit_BFGS)
						flag_BFGS = 0;

				}
				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i]
						                                     * deltaAlpha[idx] * instance.b[idx];
				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}

			finish = gettime_();
			elapsedTime += finish - start;

			double primalError;
			double dualError;
			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}

	virtual void subproblem_solver_CG(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, std::vector<D> &w,
	                                  std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                  mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {
		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		std::vector<double> cg_b(instance.n);
		std::vector<double> cg_r(instance.n);
		std::vector<double> cg_p(instance.n);
		std::vector<double> b_part(instance.n);

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				cblas_set_to_zero(cg_b);
				cblas_set_to_zero(cg_r);
				cblas_set_to_zero(cg_p);
				cblas_set_to_zero(b_part);

				D cg_a = 0.0;
				D cg_beta = 0.0;

				this->compute_subproproblem_gradient(instance, cg_r, deltaAlpha, w);
				cblas_dscal(instance.n, instance.total_n, &cg_r[0], 1);
				for (unsigned int idx = 0; idx < instance.n; idx++)
					cg_p[idx] = -cg_r[idx];

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) { //control it upper bound

					D denom = 0.0;
					D nomer = 0.0;
					std::vector<double> cg_Ap(instance.m);
					std::vector<double> cg_AAp(instance.n);

					vectormatrix_b(cg_p, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
					               instance.b, 1.0, instance.n, cg_Ap); // Ap
					matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, cg_Ap,
					             instance.n, cg_AAp); //A'Ap

					D con = instance.penalty / instance.lambda / instance.total_n;
					denom = cblas_ddot(instance.m, &cg_Ap[0], 1, &cg_Ap[0], 1);
					denom = denom * con;
					denom += cblas_ddot(instance.n, &cg_p[0], 1, &cg_p[0], 1);

					nomer = cblas_ddot(instance.n, &cg_r[0], 1, &cg_r[0], 1);

					if (denom < 1e-50)
						cg_a = 1.0;
					else
						cg_a = nomer / denom;

					D nomer_next = 0.0;
					cblas_daxpy(instance.n, cg_a, &cg_p[0], 1, &deltaAlpha[0], 1);
					for (unsigned int idx = 0; idx < instance.n; idx++) {
						cg_r[idx] += cg_a * cg_AAp[idx] * instance.b[idx] * con
						             + cg_a * cg_p[idx];
					}
					nomer_next = cblas_ddot(instance.n, &cg_r[0], 1, &cg_r[0], 1);
					cg_beta = nomer_next / nomer;

					for (unsigned int idx = 0; idx < instance.n; idx++)
						cg_p[idx] = -cg_r[idx] + cg_beta * cg_p[idx];

					D r_norm = cblas_l2_norm(instance.n, &cg_r[0], 1);
					//cout<<distributedSettings.iterationsPerThread<<endl;
					if (r_norm < 1e-12) {
						//cout<<it<<endl;
						break;
					}
				}
				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i]
						                                     * deltaAlpha[idx] * instance.b[idx];
				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}

	virtual void subproblem_solver_SDCA_without_duality(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW,
	        DistributedSettings & distributedSettings, mpi::communicator &world, D gamma, Context &ctx,
	        std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		std::vector < D > deltaAlphabuf(instance.m, 0);

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlphabuf);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {
					L idx = rand() / (0.0 + RAND_MAX) * instance.n;
					// compute "delta alpha" = argmin
					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						instance.penalty = 0; // accuracy solution
						dotProduct += (w[instance.A_csr_col_idx[i]]
						               + 1.0 * instance.penalty * deltaW[instance.A_csr_col_idx[i]])
						              * instance.A_csr_values[i];
					}
					std::vector < D > unbiasedEstimator(instance.m, 0);
					D nablabuff = dotProduct - instance.b[idx];
					for (L ii = instance.A_csr_row_ptr[idx]; ii < instance.A_csr_row_ptr[idx + 1]; ii++) {
						int ibuf = instance.A_csr_col_idx[ii];
						unbiasedEstimator[ibuf] = nablabuff * instance.A_csr_values[ibuf];
					}
					cblas_sum_of_vectors(unbiasedEstimator, deltaAlphabuf);
					D beta = 0.0001;
					cblas_sum_of_vectors(deltaAlphabuf, unbiasedEstimator, -beta);
					cblas_sum_of_vectors(deltaW, unbiasedEstimator, -beta / (instance.lambda * instance.n));
				}
				vall_reduce(world, deltaW, wBuffer);
				std::vector < D > deltaAlphaBuffer(instance.m, 0);
				vall_reduce(world, deltaAlphabuf, deltaAlphaBuffer);
				gamma = 1; // set np = 1
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlphaBuffer, gamma);
			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}

	virtual void subproblem_solver_BB(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, std::vector<D> &w,
	                                  std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                  mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {
		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		double dualobj = 0;

		std::vector < D > gradient(instance.n);
		std::vector < D > gradient_old(instance.n);
		std::vector < D > yk(instance.n);
		std::vector < D > sk(instance.n);
		std::vector<D> gradient_temp1(instance.n);
		std::vector<D> gradient_temp2(instance.m);
		std::vector<D> gradient_temp3(instance.n);

		std::vector<D> AdeltaAlpha(instance.m);
		std::vector<D> Ad(instance.m);
		std::vector<D> AAdeltaAlpha(instance.n);
		std::vector<D> AAd(instance.n);


		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, w, instance.n, gradient_temp1);
				double a = 1.0;
				this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);
				//this->backtrack_linesearch(instance, deltaAlpha, gradient, w, dualobj, a);
				vectormatrix_b(gradient, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
				               instance.b, 1.0, instance.n, Ad);
				matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, Ad, instance.n, AAd);
				this->wolfe_linesearch(instance, deltaAlpha, gradient, gradient_temp1,
				                       AdeltaAlpha, Ad, AAdeltaAlpha, AAd, w, dualobj, a);

				cblas_dcopy(instance.n, &deltaAlpha[0], 1, &sk[0], 1);
				cblas_dcopy(instance.n, &gradient[0], 1, &gradient_old[0], 1);

				for (int iter = 0; iter < distributedSettings.iterationsPerThread; iter++) {

					this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);

					for (unsigned int idx = 0; idx < instance.n; idx++) {
						yk[idx] = gradient[idx] - gradient_old[idx];
					}
					double denom = cblas_ddot(instance.n, &sk[0], 1, &yk[0], 1);
					double nom = cblas_ddot(instance.n, &sk[0], 1, &sk[0], 1);
					double stepsize;
					if (denom < 1e-50)
						stepsize = 1e-40;
					else
						stepsize = 1.0 * nom / denom;
					//cout<<stepsize<< "  "<<denom<<endl;

					cblas_daxpy(instance.n, -1.0 * stepsize, &gradient[0], 1, &deltaAlpha[0], 1);
					for (unsigned int idx = 0; idx < instance.n; idx++) {
						sk[idx] = -gradient[idx] * stepsize;
					}
					cblas_dcopy(instance.n, &gradient[0], 1, &gradient_old[0], 1);
				}

				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i]
						                                     * deltaAlpha[idx] * instance.b[idx];
				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}

	virtual void subproblem_solver_FISTA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, std::vector<D> &w,
	                                     std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                     mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {
		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		double t0 = 1.0;
		double t1 = 1.0;
		double Lip = 10.0 / instance.n ; // initial Lip constant estimate
		//double Lip = 0.001 / instance.n ; // initial Lip constant estimate
		double eta = 1.5;

		std::vector<double> y(instance.n);
		std::vector < D > gradient(instance.n);
		std::vector < D > potential(instance.n);

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			t0 = 1.0;
			t1 = 1.0; // set initial values for FISTA step-size parameters

			cblas_set_to_zero(deltaW);
			cblas_set_to_zero(deltaAlpha);
			cblas_set_to_zero(y);
			for (unsigned int kk = 0; kk < distributedSettings.iterationsPerThread ; kk++) {

				this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);

				t1 = 0.5 * (1.0 + sqrt(1.0 + t0 * t0 * 4.0));
				double tmpFrac = (t0 - 1) / t1;
				Lip = 1.0 / instance.lambda / 500.0 / instance.n;
				int iter = 0;
				while (1) {
					Lip = Lip * eta;
					for (unsigned int idx = 0; idx < instance.n; idx++) {
						double temp = deltaAlpha[idx];
						potential[idx] = deltaAlpha[idx] - gradient[idx] / Lip;
						y[idx] = potential[idx] + tmpFrac * (potential[idx] - temp);
					}
					double obj = 0.0;
					this->compute_subproproblem_obj(instance, potential, w, obj);

					double obj_appro = 0.0;
					double obj_appro_part = 0.0;
					this->compute_subproproblem_obj(instance, deltaAlpha, w, obj_appro_part);
					this->compute_subproproblem_gradient(instance, gradient, y, w);
					for (unsigned int idx = 0; idx < instance.n; idx++) {
						obj_appro += (potential[idx] - y[idx]) * gradient[idx]
						             + 0.5 * Lip * (potential[idx] - y[idx]) * (potential[idx] - y[idx]);
					}
					obj_appro += obj_appro_part;

					if (obj < obj_appro || iter > 20) {
						cblas_dcopy(instance.n, &potential[0], 1, &deltaAlpha[0], 1);
						//cout << obj << "   " << obj_appro << "   Lip constant estimate " << Lip << "   " << iter << endl;
						break;
					}
					iter++;
				}

				t0 = t1;
			}
			for (unsigned int idx = 0; idx < instance.n; idx++)
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i] * deltaAlpha[idx] * instance.b[idx];

			//cout<<y[0]<<endl;

			vall_reduce(world, deltaW, wBuffer);
			cblas_sum_of_vectors(w, wBuffer, gamma);
			cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);


			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}

// Qihang paper, better for rcv1. For a1a, make mu smaller to get faster convergence
	virtual void Acce_subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, std::vector<D> &w,
	        std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	        mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		std::vector<double> u(instance.n);
		std::vector<double> v(instance.n);
		std::vector<double> p(instance.m);
		std::vector<double> q(instance.m);
		std::vector<double> deltap(instance.m);
		std::vector<double> deltaq(instance.m);
		std::vector<double> pBuffer(instance.m);
		std::vector<double> qBuffer(instance.m);
		std::vector<double> delta(instance.n);

		double gma = 0.1;
		//double mu = gma / instance.oneOverLambdaN / (gma / instance.oneOverLambdaN);
		double mu = gma / instance.oneOverLambdaN / (0.1 + gma / instance.oneOverLambdaN);
		double rho = (1.0 - sqrt(mu) / world.size()) / (1.0 + sqrt(mu) / world.size());
		double rhoMul = rho;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltap);
				cblas_set_to_zero(deltaq);
				cblas_set_to_zero(delta);
				double c1 = -(1.0 - sqrt(mu)) / 2 / rhoMul;
				double c2 = (1.0 + sqrt(mu)) / 2;

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {
					L idx = rand() / (0.0 + RAND_MAX) * instance.n;
					// compute "delta alpha" = argmin
					D dotProduct1 = 0;
					D dotProduct2 = 0;
					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						dotProduct1 += (p[instance.A_csr_col_idx[i]]
						                + 1.0 * instance.penalty * deltap[instance.A_csr_col_idx[i]])
						               * instance.A_csr_values[i];
						dotProduct2 += (q[instance.A_csr_col_idx[i]]
						                + 1.0 * instance.penalty * deltaq[instance.A_csr_col_idx[i]])
						               * instance.A_csr_values[i];
					}
					dotProduct = rhoMul * dotProduct1 + dotProduct2;
					D alphaI = -1.0 * rhoMul * u[idx] + v[idx] + delta[idx];

					D norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
					                       &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
					instance.Li[idx] = 1.0 / (sqrt(mu) * instance.oneOverLambdaN * (norm * norm * instance.penalty +
					                          gma / instance.oneOverLambdaN) + 1.0 - gma);

					D deltaAl = 0; // FINISH
					deltaAl = (1.0 * instance.b[idx] - alphaI - dotProduct * instance.b[idx]
					           - 2.0 * gma * rhoMul * u[idx]) * instance.Li[idx];
					delta[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						deltap[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * c1
						                                     * instance.A_csr_values[i] * instance.b[idx];
						deltaq[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * c2
						                                     * instance.A_csr_values[i] * instance.b[idx];
					}

				}
				vall_reduce(world, deltap, pBuffer);
				vall_reduce(world, deltaq, qBuffer);
				cblas_sum_of_vectors(p, pBuffer, gamma);
				cblas_sum_of_vectors(q, qBuffer, gamma);
				cblas_sum_of_vectors(u, delta, gamma * c1);
				cblas_sum_of_vectors(v, delta, gamma * c2);

				rhoMul *= rho;
			}

			finish = gettime_();
			elapsedTime += finish - start;
			for (unsigned int idx = 0; idx < instance.n; idx++)
				instance.x[idx] = rhoMul / rho * u[idx] + v[idx];
			for (unsigned int i = 0; i < instance.m; i++)
				w[i] = rhoMul / rho * p[i] + q[i];
			double primalError;
			double dualError;


			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				     << dualError << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;

			}
		}

	}

//Peter APProx, better for a1a
	virtual void Acce_subproblem_solver_SDCAss(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	        mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;
		std::vector<double> u(instance.n);
		std::vector<double> z(instance.n);
		std::vector<double> y(instance.n);
		std::vector<double> zA(instance.m);
		std::vector<double> uA(instance.m);
		std::vector<double> deltaZA(instance.m);
		std::vector<double> deltaUA(instance.m);
		std::vector<double> ZABuffer(instance.m);
		std::vector<double> UABuffer(instance.m);
		std::vector<double> delta(instance.n);
		double muPsi = 100000.0;
		double theta = 0.5 * sqrt(muPsi * muPsi + 4 * muPsi) - 0.5 * muPsi;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaZA);
				cblas_set_to_zero(deltaUA);
				cblas_set_to_zero(delta);
				double c1 = 1.0;
				double c2 = - (1.0 - world.size() * theta) / theta / theta;
				double c3 = world.size() * theta;

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {

					L idx = rand() / (0.0 + RAND_MAX) * instance.n;

					D dotProduct1 = 0;
					D dotProduct2 = 0;
					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						dotProduct1 += (zA[instance.A_csr_col_idx[i]] + 1.0 * instance.penalty * deltaZA[instance.A_csr_col_idx[i]])
						               * instance.A_csr_values[i];
						dotProduct2 += (uA[instance.A_csr_col_idx[i]] + 1.0 * instance.penalty * deltaUA[instance.A_csr_col_idx[i]])
						               * instance.A_csr_values[i];
					}
					dotProduct = dotProduct1 + theta * theta * dotProduct2;

					D norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
					                       &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
					instance.Li[idx] = 1.0 / (norm * norm * instance.penalty *
					                          instance.oneOverLambdaN * theta * world.size() + 1.0);

					D alphaI = z[idx] + delta[idx];
					D deltaAl = 0;
					deltaAl = (1.0 * instance.b[idx] - alphaI - dotProduct * instance.b[idx]) * instance.Li[idx];
					delta[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						deltaZA[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * c1
						                                      * instance.A_csr_values[i] * instance.b[idx];
						deltaUA[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * c2
						                                      * instance.A_csr_values[i] * instance.b[idx];

					}

				}
				vall_reduce(world, deltaZA, ZABuffer);
				vall_reduce(world, deltaUA, UABuffer);
				cblas_sum_of_vectors(zA, ZABuffer, gamma);
				cblas_sum_of_vectors(uA, UABuffer, gamma);
				cblas_sum_of_vectors(z, delta, gamma * c1);
				cblas_sum_of_vectors(u, delta, gamma * c2);
				//theta = 0.5 * sqrt(gamma*gamma*thetasquare*thetasquare + 4*thetasquare) - 0.5 * gamma*thetasquare;
			}

			finish = gettime_();
			elapsedTime += finish - start;

			for (unsigned int idx = 0; idx < instance.n; idx++) {
				instance.x[idx] = theta * theta * u[idx] + z[idx];
			}
			for (unsigned int i = 0; i < instance.m; i++) {
				w[i] = theta * theta * uA[i] + zA[i];
			}

			double primalError;
			double dualError;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError << endl;

			}
		}

	}

	
};

#endif /* QUADRATICLOSSCD_H_ */

