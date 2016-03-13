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
		SVRGFreq = 1000;

	}


	void SH_SVRG(std::vector<D> &w, ProblemData<I, D> & instance, boost::mpi::communicator & world, std::ofstream &logFile) {

		int mode = 1;
		objective[0] = 1.0;
		std::vector<double> trueH(instance.m * instance.m);

		for (int iter = 1; iter <= maxIter; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchHessian);
			geneRandIdx(oneToN, randIdxGrad, instance.n, batchGrad);

			start = gettime_();

			lossFunction->computeStoGrad_SVRG(iter, SVRGFreq, batchGrad, w, wRec, instance, xTw, xTwRec,
			                                  gradientFull, gradientRec, gradient, randIdxGrad);

			lossFunction->StoWoodburyHGet(w, instance, batchHessian, woodburyH, xTw, randIdx, diag);

			lossFunction->StoWoodburySolve(batchHessian, w, instance, woodburyH, gradient, woodburyZHVTy,
			                               woodburyVTy, woodburyHVTy, vk, randIdx);


			for (unsigned int i = 0; i < instance.m; i++)
				w[i] =  w[i] - 0.001 * vk[i];

			finish = gettime_();
			elapsedTime += finish - start;
			if ( iter % 1000 == 1) {
				lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
				lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
				lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
				double grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				int inner_iter = 0;
				output(instance, iter, inner_iter, elapsedTime, objective, grad_norm, logFile, world, mode);
			}




			// cblas_set_to_zero(woodburyH);
			// cblas_set_to_zero(trueH);
			// woodburyH.resize(instance.m * instance.m);
			// lossFunction -> computeVectorTimesData(w, instance, xTw, world, mode);
			// lossFunction -> computeGradient(w, gradient, xTw, instance, world, mode);
			// for (unsigned int j = 0; j < batchHessian; j++) {
			// 	unsigned int idx = randIdx[j];
			// 	double temp = exp(-1.0 * xTw[idx]);
			// 	double scalar = temp / (temp + 1) / (temp + 1);
			// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			// 		for (unsigned int j = instance.A_csr_row_ptr[idx]; j < instance.A_csr_row_ptr[idx + 1]; j++)
			// 			woodburyH[instance.A_csr_col_idx[i] * instance.m + instance.A_csr_col_idx[j]] +=
			// 			    instance.A_csr_values[i] * instance.A_csr_values[j] * scalar / batchHessian;

			// }
			// for (unsigned int idx = 0; idx < instance.n; idx++) {
			// 	double temp = exp(-1.0 * xTw[idx]);
			// 	double scalar = temp / (temp + 1) / (temp + 1);
			// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			// 		for (unsigned int j = instance.A_csr_row_ptr[idx]; j < instance.A_csr_row_ptr[idx + 1]; j++)
			// 			trueH[instance.A_csr_col_idx[i] * instance.m + instance.A_csr_col_idx[j]] +=
			// 			    instance.A_csr_values[i] * instance.A_csr_values[j] * scalar / instance.n;

			// }
			// for (unsigned int idx = 0; idx < instance.m; idx++){
			// 	trueH[idx * instance.m + idx] += instance.lambda;
			// 	woodburyH[idx * instance.m + idx] += instance.lambda;
			// }

			// CGSolver(trueH, instance.m, gradient, vk);
			// cblas_daxpy(instance.m * instance.m, -1.0, &woodburyH[0], 1, &trueH[0], 1);
			// double err = cblas_l2_norm(instance.m * instance.m, &trueH[0], 1);
			// cout<<err<<endl;


		}

	}

	// useless
	void SH_SAG(std::vector<D> &w, ProblemData<I, D> & instance, boost::mpi::communicator & world, std::ofstream &logFile) {

		int mode = 1;
		objective[0] = 1.0;
		std::vector<double> vkList(instance.m * 100);
		std::vector<double> vkAvg(instance.m);

		for (int iter = 1; iter <= maxIter; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchHessian);
			geneRandIdx(oneToN, randIdxGrad, instance.n, batchGrad);

			start = gettime_();

			lossFunction->computeStoGrad_SVRG(iter, SVRGFreq, batchGrad, w, wRec, instance, xTw, xTwRec,
			                                  gradientFull, gradientRec, gradient, randIdxGrad);

			lossFunction->StoWoodburyHGet(w, instance, batchHessian, woodburyH, xTw, randIdx, diag);

			lossFunction->StoWoodburySolve(batchHessian, w, instance, woodburyH, gradient, woodburyZHVTy,
			                               woodburyVTy, woodburyHVTy, vk, randIdx);

			int yu = (iter - 1) % 100;


			for (unsigned int i = 0; i < instance.m; i++) {
				vkAvg[i] -= vkList[yu * instance.m + i];
				vkAvg[i] += vk[i];
				w[i] =  w[i] - 0.000005 * vkAvg[i];
			}
			cblas_dcopy(instance.m, &vk[0], 1, &vkList[yu * instance.m], 1);


			finish = gettime_();
			elapsedTime += finish - start;
			if ( iter % 1000 == 1) {
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
