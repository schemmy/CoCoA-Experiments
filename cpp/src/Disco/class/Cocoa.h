#ifndef COCOA_H
#define COCOA_H

#include "QuadraticLoss.h"
#include "LogisticLoss.h"

template<typename I, typename D>
class Cocoa {
public:


	LossFunction<I, D>* lossFunction;



	std::vector<D> wBuffer;
	std::vector<D> deltaW;
	std::vector<D> deltaAlpha;
	std::vector<D> gradient;
	std::vector<D> xTw;

	D start;
	D finish;
	D elapsedTime;
	D grad_norm;
	D epsilon;
	D objective;
	D alpha;
	D beta;
	I batchSizeH;
	I batchSizeP;

	D diag;
	D tol;
	int maxIter;

	Cocoa() {
	}

	~Cocoa() {
	}

	Cocoa(ProblemData<I, D> & instance, LossFunction<I, D>* lossFunction_) {

		lossFunction = lossFunction_;

		instance.x.resize(instance.n);
		wBuffer.resize(instance.m);
		deltaW.resize(instance.m);
		deltaAlpha.resize(instance.n);
		gradient.resize(instance.m);
		xTw.resize(instance.n);
		instance.x.resize(instance.n);

		instance.oneOverLambdaN = 1.0 / (instance.total_n * instance.lambda);

		start = 0;
		finish = 0;
		elapsedTime = 0;
		instance.Li.resize(instance.n);
		instance.vi.resize(instance.n);

		for (I idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
			                       &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1.0 / (norm * norm * instance.penalty * instance.oneOverLambdaN + 1.0);

			instance.vi[idx] = norm * norm;
		}


		tol = 1e-10;
		maxIter = 200;
	}

	void solverSDCA(std::vector<D> &w, ProblemData<I, D> &instance,
	                DistributedSettings & distributedSettings,
	                mpi::communicator &world, D gamma, std::ofstream &logFile) {

		int mode = 1;
		int inIter = distributedSettings.iters_bulkIterations_count;
		for (int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();
			for (unsigned int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {

					lossFunction->oneCocoaSDCAUpdate(instance, w, deltaAlpha, deltaW);

				}
				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}
			finish = gettime_();
			elapsedTime += finish - start;

			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
			D grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			lossFunction->computeObjective(w, instance, xTw, objective, world, mode);
			output(instance, t, inIter, 
							elapsedTime, objective, grad_norm, logFile, world, mode);

		}

	}

	void output(ProblemData<I, D> &instance, int &iter, int &inner_iter, double & elapsedTime,
	            D &objective, D & grad_norm,
	            std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (mode == 1) {
			if (world.rank() == 0) {
				printf("%ith: %i CG iters, time %f, norm of gradient %E, objective %E\n",
				       iter, inner_iter, elapsedTime, grad_norm, objective);
				logFile << iter << "," << inner_iter << "," << elapsedTime << "," << grad_norm << "," << objective << endl;
			}
		}


	}



};

#endif // COCOA_H
