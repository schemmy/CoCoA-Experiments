#ifndef DANE_H
#define DANE_H

#include "QuadraticLoss.h"
#include "LogisticLoss.h"
#include "QR_solver.h"

template<typename I, typename D>
class Dane {
public:

	LossFunction<I, D>* lossFunction;
	D mu;
	D diag;
	int maxIter;
	D tol;
	D objective;
	D start;
	D finish;
	D elapsedTime;
	std::vector<D> gradientWorld;
	std::vector<D> gradient;
	std::vector<D> xTw;
	std::vector<D> v;

	std::vector<D> gradIdx;
	std::vector<D> gradAvg;

	Dane() {
	}

	~Dane() {
	}

	Dane(ProblemData<I, D> & instance,
	     D mu_, LossFunction<I, D>* lossFunction_) {

		lossFunction = lossFunction_;
		mu = mu_;
		tol = 1e-10;
		maxIter = 100;
		start = 0;
		finish = 0;
		elapsedTime = 0;
		diag =  mu;

		v.resize(instance.m);
		gradient.resize(instance.m);
		gradientWorld.resize(instance.m);
		xTw.resize(instance.n);
		gradIdx.resize(instance.m * instance.n);
		gradAvg.resize(instance.m);
	}

	void solver(std::vector<D> &w, ProblemData<I, D> & instance,
	            boost::mpi::communicator &world, std::ofstream &logFile) {

		int mode = 1;
		for (int iter = 0; iter < maxIter; iter++) {
			start = gettime_();
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeLocalGradient(w, gradient, xTw, instance, world, mode);
			vall_reduce(world, gradient, gradientWorld);
			cblas_dscal(instance.m, 1.0 / world.size(), &gradientWorld[0], 1);

			solveDaneSubproblem(instance, w);

			vall_reduce(world, v, w);
			cblas_dscal(instance.m, 1.0 / world.size(), &w[0], 1);


			D grad_norm = cblas_l2_norm(instance.m, &gradientWorld[0], 1);
			finish = gettime_();
			elapsedTime += finish - start;
			output(instance, iter, elapsedTime, objective, grad_norm, logFile, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective, world, mode);
			if (grad_norm < tol)
				break;
		}

	}

	void solveDaneSubproblem(ProblemData<I, D> & instance, std::vector<D> &w) {

		double eta = 0.005;
		double xTs = 0.0;
		double nomNew = 1.0;
		double nom0 = 1.0;
		int iter = 0;
		cblas_set_to_zero(gradIdx);
		cblas_set_to_zero(gradAvg);
		cblas_set_to_zero(v);
		std::vector<double> b(instance.m);
		int n = instance.m;

		for (unsigned int i = 0; i < instance.m; i++)
			b[i] = gradient[i] - gradientWorld[i];

		//while (nomNew > 1e-10 * nom0) {

			for (unsigned int ii = 0; ii < instance.n *10 ; ii++) {

				xTs = 0.0;
				unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
				for (unsigned int i = 0; i < n; i++) {
					gradAvg[i] -= gradIdx[idx * n + i];
					gradIdx[idx * n + i] = 0.0;
				}
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					xTs += instance.A_csr_values[i] * v[instance.A_csr_col_idx[i]];
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++){
					gradIdx[idx * n + instance.A_csr_col_idx[i]] = instance.A_csr_values[i] * xTs;
					gradAvg[instance.A_csr_col_idx[i]] += gradIdx[idx * n + instance.A_csr_col_idx[i]];
				}
				for (unsigned int i = 0; i < n; i++) {
					v[i] = v[i] - eta * (1.0/ instance.n * gradAvg[i] - (gradient[i] - gradientWorld[i])
							 + diag * v[i] - mu * w[i]);
				}
			}

			// 		unsigned int idx = j;
		// 		xTs = 0.0;
		// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
		// 			xTs += instance.A_csr_values[i] * v[instance.A_csr_col_idx[i]];
		// 		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
		// 			grad[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * xTs / instance.n;
		// 	}
		// 	for (unsigned int i = 0; i < n; i++) {
		// 		grad[i] = grad[i] - (gradient[i] - gradientWorld[i]) + diag * v[i] - mu * w[i];
		// 	}
		// 	nomNew = cblas_ddot(n, &grad[0], 1, &grad[0], 1);
		// 	//cout << nomNew << endl;
		// 	iter++;
		// 	if (iter == 1)
		// 		nom0 = nomNew;
		// }


	}



	void output(ProblemData<unsigned int, double> &instance, int &iter, double & elapsedTime,
	            D &objective, double & grad_norm,
	            std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (world.rank() == 0) {
			printf("%ith: time %f, norm of gradient %E, objective %E\n",
			       iter, elapsedTime, grad_norm, objective);
			logFile << iter << "," << elapsedTime << "," << grad_norm << "," << objective << endl;
		}
	}


};

#endif // DANE_H
