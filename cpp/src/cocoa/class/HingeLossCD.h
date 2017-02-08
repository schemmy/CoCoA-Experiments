/*
 * HingeLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef HINGELOSSCD_H_
#define HINGELOSSCD_H_

#include "HingeLoss.h"

template<typename L, typename D>

class HingeLossCD: public HingeLoss<L, D> {
public:
	HingeLossCD() {

	}

	virtual ~HingeLossCD() {}

	virtual void init(ProblemData<L, D> & instance) {

		instance.Li.resize(instance.n);
		instance.vi.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(
			             instance.A_csr_row_ptr[idx + 1]
			             - instance.A_csr_row_ptr[idx],
			             &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1 / (norm * norm * instance.penalty * instance.oneOverLambdaN);

			instance.vi[idx] = norm * norm;
		}

	}




	virtual void subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	                                    std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                    mpi::communicator & world, D gamma, Context & ctx, std::ofstream & logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		std::vector<double> trainLabel(instance.n);
		unsigned int trainError;
		unsigned int totalTrainError;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
				        it++) {

					L idx = rand() / (0.0 + RAND_MAX) * instance.n;

					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						dotProduct += (w[instance.A_csr_col_idx[i]] + instance.penalty * deltaW[instance.A_csr_col_idx[i]])
						              * instance.A_csr_values[i];
					}

					D alphaI = instance.x[idx] + deltaAlpha[idx];

					D deltaAl = 0;

					D part = (1.0 - instance.b[idx] * dotProduct) * instance.Li[idx];

					deltaAl = (part > 1 - alphaI) ? 1 - alphaI : (part < -alphaI ? -alphaI : part);
					deltaAlpha[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						                                     * instance.A_csr_values[i] * deltaAl * instance.b[idx];
					}

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

			// trainError = 0;
			// cblas_set_to_zero(trainLabel);
			// for (unsigned int idx = 0; idx < instance.n; idx++) {
			// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			// 		trainLabel[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
			// 	}
			// 	if (trainLabel[idx] * instance.b[idx] <= 0 )
			// 		trainError += 1;
			// }
			// vall_reduce(world, &trainError, &totalTrainError, 1);
			//cout << 1.0 * totalTrainError / instance.total_n << endl;

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError
				     //<< "    " << 1.0 * totalTrainError / instance.total_n
				     << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError 
				        //<< "," << 1.0 * totalTrainError / instance.total_n 
				        << endl;

			}

		}

	}


	virtual void subproblem_solver_accelerated_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	        mpi::communicator & world, D gamma, Context & ctx, std::ofstream & logFile) {

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

		for (L i = 0; i < instance.n; i++)
			zk[i] = instance.x[i];

		for (L t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(deltayk);
				cblas_set_to_zero(deltaAyk);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++)
					this->accelerated_SDCA_oneIteration(instance, deltaAlpha, w, deltaW,
					                                    zk, uk, yk, deltayk, Ayk, deltaAyk, theta, distributedSettings);

				double thetasq = theta * theta;
				theta = 0.5 * sqrt(thetasq * thetasq + 4 * thetasq) - 0.5 * thetasq;
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);

//				for (L i = 0; i < instance.n; i++)
//					instance.x[i] = (instance.x[i] > 1 ) ? 1.0 : (instance.x[i] < 0 ? 0.1 : instance.x[i]);

				cblas_sum_of_vectors(yk, deltayk, gamma);

				vectormatrix_b(instance.x, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
				               instance.b, instance.oneOverLambdaN, instance.n, deltaW);

				vectormatrix_b(yk, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
				               instance.b, instance.oneOverLambdaN, instance.n, deltaAyk);

				cblas_set_to_zero(w);
				cblas_set_to_zero(Ayk);

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
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError << endl;
			}


		}
	}


	virtual void accelerated_SDCA_oneIteration(ProblemData<L, D> &instance,
	        std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW, std::vector<D> &zk, std::vector<D> &uk,
	        std::vector<D> &yk, std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk,
	        D & theta, DistributedSettings & distributedSettings) {

		D thetasquare = theta * theta;
		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (1.0 * Ayk[instance.A_csr_col_idx[i]] + instance.penalty * deltaAyk[instance.A_csr_col_idx[i]])
			              * instance.A_csr_values[i];
		}

		D tk = (1.0 - instance.b[idx] * dotProduct)
		       / (instance.vi[idx] * instance.n / distributedSettings.iterationsPerThread
		          * theta * instance.penalty * instance.oneOverLambdaN);
		zk[idx] += tk;
		uk[idx] -= (1.0 - theta * instance.n / distributedSettings.iterationsPerThread) / (thetasquare) * tk;

		D deltaAl = thetasquare * uk[idx] + zk[idx] - instance.x[idx] - deltaAlpha[idx];
		deltaAlpha[idx] += deltaAl;

		D thetanext = theta;
		thetanext = 0.5 * sqrt(thetasquare * thetasquare + 4 * thetasquare) - 0.5 * thetasquare;
		D dyk = thetanext * thetanext * uk[idx] + zk[idx] - yk[idx] - deltayk[idx];
		deltayk[idx] += dyk;

	}


	virtual void subproblem_solver_steepestdescent(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	        std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	        mpi::communicator & world, D gamma, Context & ctx, std::ofstream & logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double dualobj = 0;
		std::vector<D> gradient(instance.n);
		D rho = 0.8;
		D c1ls = 0.1;
		D a = 20.00;
		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);


				for (L line_search_iter = 0; line_search_iter < distributedSettings.iterationsPerThread; line_search_iter++) {

					this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);
					this->backtrack_linesearch(instance, deltaAlpha, gradient, w, dualobj, rho, c1ls, a, distributedSettings);

				}
				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						                                     * instance.A_csr_values[i] * deltaAlpha[idx] * instance.b[idx];
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
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError << endl;


			}
		}
	}


	virtual void subproblem_solver_LBFGS(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	                                     std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                     mpi::communicator & world, D gamma, Context & ctx, std::ofstream & logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double dualobj = 0;
		int limit_BFGS = 10;
		std::vector<double> old_grad(instance.n);
		std::vector<double> sk(instance.n * limit_BFGS);
		std::vector<double> rk(instance.n * limit_BFGS);
		cblas_set_to_zero(sk);
		cblas_set_to_zero(rk);

		std::vector<D> old_deltaAlpha(instance.n);
		std::vector<D> gradient(instance.n);
		std::vector<D> search_direction(instance.n);
		int flag_BFGS = 0;
		std::vector<D> oneoversy(limit_BFGS);

		D rho = 0.8;
		D c1ls = 0.1;
		D a = 20.0;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			for (int jj = 0; jj < 1; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(old_deltaAlpha);

				for (L iter_counter = 0; iter_counter < 10; iter_counter++) {

					this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);
					this->LBFGS_update(instance, search_direction, old_grad,
					                   sk, rk, gradient, oneoversy, iter_counter, limit_BFGS, flag_BFGS);
					this->backtrack_linesearch(instance, deltaAlpha, search_direction, w, dualobj, rho, c1ls, a, distributedSettings);

					for (L idx = 0; idx < instance.n; idx++) {
						sk[instance.n * flag_BFGS + idx] = deltaAlpha[idx] - old_deltaAlpha[idx];
						old_deltaAlpha[idx] = deltaAlpha[idx];////////////////////// ?????????????????????????????
					}

					flag_BFGS += 1;
					if (flag_BFGS == limit_BFGS)
						flag_BFGS = 0;

				}
				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						                                     * instance.A_csr_values[i] * deltaAlpha[idx] * instance.b[idx];
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
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError << endl;

			}
		}

	}


	virtual void subproblem_solver_CG(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	                                  std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                  mpi::communicator & world, D gamma, Context & ctx, std::ofstream & logFile) {
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

				for (unsigned int idx = 0; idx < instance.n; idx++)
					cg_p[idx] = -cg_r[idx];

				for (unsigned int it = 0; it < 1000; it++) {

					D denom = 0.0;
					std::vector<double> cg_Ap(instance.m);
					std::vector<double> cg_AAp(instance.n);

					vectormatrix_b(cg_p, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
					               instance.b, 1.0, instance.n, cg_Ap); // Ap
					matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
					             cg_Ap, instance.n, cg_AAp); //A'Ap

					for (unsigned int i = 0; i < instance.m; i++)
						denom += cg_Ap[i] * cg_Ap[i];
//					if (it==0)	cout<<denom<<endl;
					denom = denom * instance.penalty * instance.oneOverLambdaN;

					D nomer = 0.0;
					for (unsigned int idx = 0; idx < instance.n; idx++) {
						nomer += cg_r[idx] * cg_r[idx];
						denom += cg_p[idx] * cg_p[idx];
					}
					cg_a = nomer / denom;

					D nomer_next = 0.0;
					for (unsigned int idx = 0; idx < instance.n; idx++) {
						deltaAlpha[idx] += cg_a * cg_p[idx];////////////////////// ?????????????????????????????
						deltaAlpha[idx] = (deltaAlpha[idx] > 1 - instance.x[idx]) ? 1 - instance.x[idx]
						                  : (deltaAlpha[idx] < -instance.x[idx] ? -instance.x[idx] : deltaAlpha[idx]);
						cg_r[idx] += cg_a * cg_AAp[idx] * instance.b[idx] * instance.penalty * instance.oneOverLambdaN
						             + cg_a * cg_p[idx];
						nomer_next += cg_r[idx] * cg_r[idx];
					}

					cg_beta = nomer_next / nomer;

					for (unsigned int idx = 0; idx < instance.n; idx++)
						cg_p[idx] = -cg_r[idx] + cg_beta * cg_p[idx];

					D r_norm = cblas_l2_norm(instance.n, &cg_r[0], 1);
					if (r_norm < 1e-6) {
						//cout<<it<<endl;
						break;
					}
				}
				for (unsigned int idx = 0; idx < instance.n; idx++) {
					for (unsigned int i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						                                     * instance.A_csr_values[i] * deltaAlpha[idx] * instance.b[idx];
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
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError << endl;

			}
		}

	}

	virtual void Acce_subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
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
		double theta = 1.0;
		double thetaOld = theta;
		double thetasquare;
		std::vector<double> trainLabel(instance.n);
		unsigned int trainError;
		unsigned int totalTrainError;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaZA);
				cblas_set_to_zero(deltaUA);
				cblas_set_to_zero(delta);
				double c1 = 1.0;
				double c2 = -(1.0 - theta) / theta / theta;
				double c3 = theta;

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
					                          instance.oneOverLambdaN * theta * world.size());

					D alphaI = z[idx] + delta[idx];
					D deltaAl = 0;
					D part = (1.0 - instance.b[idx] * dotProduct) * instance.Li[idx];
					deltaAl = (part > 1 - alphaI) ? 1 - alphaI : (part < -alphaI ? -alphaI : part);
					delta[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						deltaZA[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * c1
						                                      * instance.A_csr_values[i] * instance.b[idx];
						deltaUA[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * c2
						                                      * instance.A_csr_values[i] * instance.b[idx];

					}

				}
				// for (unsigned int i = 0; i < instance.m; i++) {
				// 	deltaW[i] = thetaOld * thetaOld * deltaUA[i] + deltaZA[i];
				// }
				// vall_reduce(world, deltaW, wBuffer);
				//cblas_sum_of_vectors(w, wBuffer, gamma);
				vall_reduce(world, deltaZA, ZABuffer);
				vall_reduce(world, deltaUA, UABuffer);
				cblas_sum_of_vectors(zA, ZABuffer, gamma);
				cblas_sum_of_vectors(uA, UABuffer, gamma);
				cblas_sum_of_vectors(z, delta, gamma * c1);
				cblas_sum_of_vectors(u, delta, gamma * c2);
				thetaOld = theta;
				thetasquare = theta * theta;
				theta = 0.5 * sqrt(gamma*gamma*thetasquare*thetasquare+4*thetasquare) - 0.5*gamma*thetasquare;
			}

			finish = gettime_();
			elapsedTime += finish - start;

			for (unsigned int idx = 0; idx < instance.n; idx++) {
				instance.x[idx] = thetaOld * thetaOld * u[idx] + z[idx];
			}
			for (unsigned int i = 0; i < instance.m; i++) {
				w[i] = thetaOld * thetaOld * uA[i] + zA[i];
			}

			double primalError;
			double dualError;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			// trainError = 0;
			// cblas_set_to_zero(trainLabel);
			// for (unsigned int idx = 0; idx < instance.n; idx++) {
			// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			// 		trainLabel[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
			// 	}
			// 	if (trainLabel[idx] * instance.b[idx] <= 0 )
			// 		trainError += 1;
			// }
			// vall_reduce(world, &trainError, &totalTrainError, 1);
			//cout << 1.0 * totalTrainError / instance.total_n << endl;

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "  error " << primalError << "    " << dualError
				     << "    " << primalError + dualError
				     //<< "    " << 1.0 * totalTrainError / instance.total_n
				     << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
				        << dualError << "," << primalError + dualError 
				        // << ","<< 1.0 * totalTrainError / instance.total_n
				        << endl;

			}



		}

	}




	virtual void L1CoCoA_subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
	                                    std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
	                                    mpi::communicator & world, D gamma, Context & ctx, std::ofstream & logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;


		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
				        it++) {

					L idx = rand() / (0.0 + RAND_MAX) * instance.n;

					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						dotProduct += (w[instance.A_csr_col_idx[i]] + instance.penalty * deltaW[instance.A_csr_col_idx[i]]
										 -instance.b[instance.A_csr_col_idx[i]] ) * instance.A_csr_values[i];
					}

					D alphaI = instance.x[idx] + deltaAlpha[idx];

					D deltaAl = 0;

					D partA = (-instance.lambda - dotProduct) * instance.Li[idx];
					D partB = (instance.lambda - dotProduct) * instance.Li[idx];

					if (partA > -alphaI)
						deltaAl = partA;
					else if (partB < -alphaI)
						deltaAl = partB;
					else
						deltaAl = -alphaI;
					
					// deltaAl = (deltaAl > 0.5 - alphaI) ? 0.5 - alphaI : (deltaAl < -0.5-alphaI ? -0.5-alphaI : deltaAl);

					deltaAlpha[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						deltaW[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * deltaAl;
					}

				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);

			}
		
			double dualError;
			double sparse;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValueL1CoCoA(instance, world, w, dualError, sparse);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "    " <<setprecision(15)<<setw(15)<< dualError
				     << "    " <<setprecision(6)<<setw(6)<< sparse
				     << endl;

				logFile << t << "," << elapsedTime << ","
				     << "," <<setprecision(15)<<setw(15)<< dualError
				     << "," <<setprecision(6)<<setw(6)<< sparse
				     << endl;

			}

		}
	}


	virtual void L1CoCoA_Acce_subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
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
		double theta = 1.0;
		double thetaOld = theta;
		double thetasquare;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaZA);
				cblas_set_to_zero(deltaUA);
				cblas_set_to_zero(delta);
				double c1 = 1.0;
				double c2 = -(1.0 - theta) / theta / theta;
				double c3 = theta;

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {

					L idx = rand() / (0.0 + RAND_MAX) * instance.n;

					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						dotProduct += (zA[instance.A_csr_col_idx[i]] + 1.0 * instance.penalty * deltaZA[instance.A_csr_col_idx[i]] 
							+ theta * theta * (uA[instance.A_csr_col_idx[i]] + 1.0 * instance.penalty * deltaUA[instance.A_csr_col_idx[i]])
						 			- instance.b[instance.A_csr_col_idx[i]] ) * instance.A_csr_values[i];
					}

					// D norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
					//                        &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
					// instance.Li[idx] = 1.0 / (norm * norm * instance.penalty *
					//                           instance.oneOverLambdaN * theta * world.size());

					D alphaI = z[idx] + delta[idx];
					D deltaAl = 0;

					D partA = (-instance.lambda - dotProduct) * instance.Li[idx] / theta/ world.size();
					D partB = (instance.lambda - dotProduct) * instance.Li[idx] / theta/ world.size() ;

					if (partA > -alphaI)
						deltaAl = partA;
					else if (partB < -alphaI)
						deltaAl = partB;
					else
						deltaAl = -alphaI;
					
					// deltaAl = (deltaAl > 0.5 - alphaI) ? 0.5 - alphaI : (deltaAl < -0.5-alphaI ? -0.5-alphaI : deltaAl);

					delta[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
						deltaZA[instance.A_csr_col_idx[i]] += deltaAl * c1 * instance.A_csr_values[i];
						deltaUA[instance.A_csr_col_idx[i]] += deltaAl * c2 * instance.A_csr_values[i];

					}

				}
				// for (unsigned int i = 0; i < instance.m; i++) {
				// 	deltaW[i] = thetaOld * thetaOld * deltaUA[i] + deltaZA[i];
				// }
				// vall_reduce(world, deltaW, wBuffer);
				//cblas_sum_of_vectors(w, wBuffer, gamma);
				vall_reduce(world, deltaZA, ZABuffer);
				vall_reduce(world, deltaUA, UABuffer);
				cblas_sum_of_vectors(zA, ZABuffer, gamma);
				cblas_sum_of_vectors(uA, UABuffer, gamma);
				cblas_sum_of_vectors(z, delta, gamma * c1);
				cblas_sum_of_vectors(u, delta, gamma * c2);
				thetaOld = theta;
				thetasquare = theta * theta;
				theta = 0.5 * sqrt(gamma*gamma*thetasquare*thetasquare+4*thetasquare) - 0.5*gamma*thetasquare;
			}

			finish = gettime_();
			elapsedTime += finish - start;

			for (unsigned int idx = 0; idx < instance.n; idx++) {
				instance.x[idx] = thetaOld * thetaOld * u[idx] + z[idx];
			}
			for (unsigned int i = 0; i < instance.m; i++) {
				w[i] = thetaOld * thetaOld * uA[i] + zA[i];
			}

			double sparse;
			double dualError;

			this->computeObjectiveValueL1CoCoA(instance, world, w, dualError, sparse);


			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
				     << "    " <<setprecision(15)<<setw(15)<< dualError
				     << "    " <<setprecision(6)<<setw(6)<< sparse
				     << endl;

				logFile << t << "," << elapsedTime << ","
				     << "," <<setprecision(15)<<setw(15)<< dualError
				     << "," <<setprecision(6)<<setw(6)<< sparse
				     << endl;

			}


		}

	}


};

#endif /* HINGELOSSCD_H_ */
