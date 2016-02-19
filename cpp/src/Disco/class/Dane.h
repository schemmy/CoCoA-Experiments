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
	std::vector<D> wWolrd;
	std::vector<D> v;
	I nEpoch;
	D eta;
	std::vector<D> gradIdx;
	std::vector<D> gradAvg;

	Dane() {
	}

	~Dane() {
	}

	Dane(ProblemData<I, D> & instance,
	     D mu_, D nEpoch_, LossFunction<I, D>* lossFunction_) {

		lossFunction = lossFunction_;
		mu = mu_;
		nEpoch = nEpoch_;
		tol = 1e-10;
		maxIter = 200;
		start = 0;
		finish = 0;
		elapsedTime = 0;
		diag = instance.lambda + mu;
		eta = 1.0;
		v.resize(instance.m);
		gradient.resize(instance.m);
		gradientWorld.resize(instance.m);
		xTw.resize(instance.n);
		//gradIdx.resize(instance.m * instance.n);
		gradAvg.resize(instance.m);
		wWolrd.resize(instance.m);
	}

	void solver(std::vector<D> &w, ProblemData<I, D> & instance,
	            boost::mpi::communicator &world, std::ofstream &logFile) {

		int mode = 1;
		for (int iter = 0; iter < maxIter; iter++) {
			start = gettime_();
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
			vall_reduce(world, gradient, gradientWorld);
			cblas_dscal(instance.m, 1.0 / world.size(), &gradientWorld[0], 1);

			for (int jj = 0; jj < 12; jj++)
				lossFunction->SAGSolver(instance, instance.m, xTw, gradientWorld, v, nEpoch, diag);
			//solveDaneSubproblem(instance, w);
			cblas_daxpy(instance.m, -eta, &v[0], 1, &w[0], 1);

			vall_reduce(world, w, wWolrd);
			for (unsigned int i = 0; i < instance.m; i++)
				w[i] = wWolrd[i] / world.size();

			D grad_norm = cblas_l2_norm(instance.m, &gradientWorld[0], 1);
			finish = gettime_();
			elapsedTime += finish - start;
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective, world, mode);
			output(instance, iter, 2, elapsedTime, objective, grad_norm, logFile, world, mode);
			if (grad_norm < tol)
				break;
		}

	}

	void solveDaneSubproblem(ProblemData<I, D> & instance, std::vector<D> &w) {

		double eta = 0.05;
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

		for (unsigned int ii = 0; ii < instance.n * nEpoch ; ii++) {

			xTs = 0.0;
			unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
			for (unsigned int i = 0; i < n; i++) {
				gradAvg[i] -= gradIdx[idx * n + i];
				gradIdx[idx * n + i] = 0.0;
			}
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				xTs += instance.A_csr_values[i] * v[instance.A_csr_col_idx[i]];
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				gradIdx[idx * n + instance.A_csr_col_idx[i]] = instance.A_csr_values[i] * xTs;
				gradAvg[instance.A_csr_col_idx[i]] += gradIdx[idx * n + instance.A_csr_col_idx[i]];
			}
			for (unsigned int i = 0; i < n; i++) {
				v[i] = v[i] - eta * (1.0 / instance.n * gradAvg[i] - (gradient[i] - gradientWorld[i])
				                     + diag * v[i] - mu * w[i]);
			}
		}

		// 	std::vector<double> grad(n);
		// 	for (unsigned int j = 0; j < instance.n; j++) {
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



	void output(ProblemData<unsigned int, double> &instance, int &iter, int inner_iter, double & elapsedTime,
	            D &objective, double & grad_norm,
	            std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (world.rank() == 0) {
			printf("%ith: %i comm, time %f, norm of gradient %E, objective %E\n",
			       iter, inner_iter, elapsedTime/10.0, grad_norm, objective);
			logFile << iter << "," << inner_iter << "," << elapsedTime/4.0 << "," << grad_norm << "," << objective << endl;
		}
	}


};

#endif // DANE_H
