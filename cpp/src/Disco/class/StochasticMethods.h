#ifndef STOCHASTICMETHODS_H
#define STOCHASTICMETHODS_H


#include "QuadraticLoss.h"
#include "LogisticLoss.h"


template<typename I, typename D>
class StochasticMethods {
public:


	LossFunction<I, D>* lossFunction;

	std::vector<D> objective;
	std::vector<D> gradient;
	std::vector<D> gradientFull;
	std::vector<D> gradientRec;
	std::vector<D> wRec;
	std::vector<D> xTwRec;
	std::vector<D> xTw;
	std::vector<D> woodburyZHVTy;
	std::vector<D> woodburyVTy;
	std::vector<D> woodburyVTy_World;
	std::vector<D> woodburyHVTy;
	std::vector<D> woodburyH;
	std::vector<D> vk;
	std::vector<I> randIdx;
	std::vector<I> randIdxGrad;
	std::vector<I> oneToN;
	I batchHessian;
	I batchGrad;
	// std::vector<D> w_try;
	// std::vector<D> xTw_try;
	D diag;
	int maxIter;
	int SVRGFreq;
	D start;
	D finish;
	D elapsedTime;

	StochasticMethods() {
	}

	~StochasticMethods() {
	}

	StochasticMethods(ProblemData<I, D> & instance, I &batchGrad_, I &batchHessian_,
	                  LossFunction<I, D>* lossFunction_) {

		lossFunction = lossFunction_;
		batchGrad = batchGrad_;
		batchHessian = batchHessian_;
		diag = instance.lambda;

		objective.resize(2);
		gradient.resize(instance.m);
		gradientFull.resize(instance.m);
		gradientRec.resize(instance.m);
		wRec.resize(instance.m);
		xTw.resize(instance.n);
		xTwRec.resize(instance.n);
		woodburyZHVTy.resize(instance.m);
		woodburyVTy.resize(batchHessian);
		woodburyVTy_World.resize(batchHessian);
		woodburyHVTy.resize(batchHessian);
		woodburyH.resize(batchHessian * batchHessian);
		vk.resize(instance.m);
		randIdx.resize(batchHessian);
		randIdxGrad.resize(batchGrad);
		oneToN.resize(instance.n);
		for (I idx = 0; idx < instance.n; idx++)
			oneToN[idx] = idx;
		
		maxIter = 10000;
		SVRGFreq = 10;

	}


	void SH_SVRG(std::vector<D> &w, ProblemData<I, D> & instance, boost::mpi::communicator & world, std::ofstream &logFile) {

		int mode = 1;
		objective[0] = 1.0;

		for (int iter = 1; iter <= maxIter; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchHessian);
			geneRandIdx(oneToN, randIdxGrad, instance.n, batchGrad);

			start = gettime_();

			lossFunction->computeStoGrad_SVRG(iter, SVRGFreq, batchGrad, w, wRec, instance, xTw, xTwRec,
			                                  gradientFull, gradientRec, gradient, randIdxGrad);


			lossFunction->StoWoodburyHGet(w, instance, batchHessian, woodburyH, xTw, randIdx, diag);
			// cblas_set_to_zero(woodburyH);
			// for (unsigned int ii = 0; ii < batchHessian; ii++) {
			// 	unsigned int idx1 = randIdx[ii];
			// 	for (unsigned int jj = 0; jj < batchHessian; jj++) {
			// 		unsigned int idx2 = randIdx[jj];
			// 		unsigned int i = instance.A_csr_row_ptr[idx1];
			// 		unsigned int j = instance.A_csr_row_ptr[idx2];
			// 		while (i < instance.A_csr_row_ptr[idx1 + 1] && j < instance.A_csr_row_ptr[idx2 + 1]) {
			// 			if (instance.A_csr_col_idx[i] == instance.A_csr_col_idx[j]) {
			// 				woodburyH[ii * batchHessian + jj] += instance.A_csr_values[i] * instance.A_csr_values[j]
			// 				                                   * instance.b[idx1] * instance.b[idx2] / diag / batchHessian;
			// 				i++;
			// 				j++;
			// 			}
			// 			else if (instance.A_csr_col_idx[i] < instance.A_csr_col_idx[j])
			// 				i++;
			// 			else
			// 				j++;
			// 		}
			// 	}
			// }
			// for (unsigned int idx = 0; idx < batchHessian; idx++)
			// 	woodburyH[idx * batchHessian + idx] += 1.0;


			lossFunction->StoWoodburySolve(batchHessian, w, instance, woodburyH, gradient, woodburyZHVTy,
			                          woodburyVTy, woodburyHVTy, vk, randIdx);

			// cblas_set_to_zero(woodburyZHVTy);
			// cblas_set_to_zero(woodburyVTy);

			// for (unsigned int j = 0; j < batchHessian; j++) {
			// 	unsigned int idx = randIdx[j];
			// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			// 		woodburyVTy[j] += instance.A_csr_values[i] * instance.b[idx] *
			// 		                  gradient[instance.A_csr_col_idx[i]] / diag / batchHessian;
			// 	}
			// }

			// CGSolver(woodburyH, batchHessian, woodburyVTy, woodburyHVTy);

			// for (unsigned int j = 0; j < batchHessian; j++) {
			// 	unsigned int idx = randIdx[j];
			// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			// 		woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
			// 		        / diag * woodburyHVTy[j];
			// 	}
			// }

			// for (unsigned int i = 0; i < instance.m; i++)
			// 	vk[i] =  (gradient[i] / diag - woodburyZHVTy[i]);


			// line search
			// double vk_norm = cblas_l2_norm(instance.m, &vk[0], 1);

			// double stepsize = 1.0;
			// double obj_try = 0.0;
			// while (stepsize > 1e-5) {

			// 	for (unsigned int i = 0; i < instance.m; i++)
			// 		w_try[i] =  w[i] - stepsize * vk[i];

			// 	lossFunction->computeVectorTimesData(w_try, instance, xTw_try, world, mode);
			// 	lossFunction->computeObjective(w_try, instance, xTw_try, obj_try, world, mode);

			// 	if (obj_try < objective[0] - stepsize * 0.001 * vk_norm * vk_norm) {
			// 		cblas_dcopy(instance.m, &w_try[0], 1, &w[0], 1);
			// 		break;
			// 	}
			// 	stepsize *= 0.8;

			// }

			for (unsigned int i = 0; i < instance.m; i++)
				w[i] =  w[i] - 0.1 * vk[i];

			finish = gettime_();
			elapsedTime += finish - start;
			if ( iter % 2 == 1) {
				lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
				lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
				lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
				double grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				int inner_iter = 0;
				output(instance, iter, inner_iter, elapsedTime, objective, grad_norm, logFile, world, mode);
			}


		}

	}




	void output(ProblemData<unsigned int, double> &instance, int &iter, int &inner_iter, double & elapsedTime,
	            std::vector<double> &objective, double & grad_norm,
	            std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (mode == 1) {
			if (world.rank() == 0) {
				printf("%ith: %i CG iters, time %f, norm of gradient %E, objective %E\n",
				       iter, 2 * inner_iter, elapsedTime, grad_norm, objective[0]);
				logFile << iter << "," << 2 * inner_iter << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
			}
		}


	}

};






#endif // STOCHASTICMETHODS_H
