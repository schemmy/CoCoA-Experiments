#ifndef CG_H
#define CG_H

#include "QuadraticLoss.h"
#include "LogisticLoss.h"
#include "QR_solver.h"

template<typename I, typename D>
class CG {
public:


	LossFunction<I, D>* lossFunction;


	std::vector<int> flag;

	std::vector<D> constantLocal;
	std::vector<D> constantSum;
	D start;
	D finish;
	D elapsedTime;
	D grad_norm;
	D epsilon;
	D alpha;
	D beta;
	I batchSizeH;
	I batchSizeP;


	std::vector<D> objective;
	std::vector<D> v;
	std::vector<D> s;
	std::vector<D> r;
	std::vector<D> u;
	std::vector<D> xTu;
	std::vector<D> xTw;
	std::vector<D> Hv;
	std::vector<D> Hu;
	std::vector<D> gradient;
	std::vector<I> randIdx;
	std::vector<I> oneToN;
	std::vector<D> woodburyH;
	std::vector<D> vk;
	D deltak;

	D diag;
	D tol;
	int maxIter;

	CG() {
	}

	~CG() {
	}

	CG(ProblemData<I, D> & instance, I &batchSizeP_, I &batchSizeH_,
	   D mu, LossFunction<I, D>* lossFunction_) {

		lossFunction = lossFunction_;
		flag.resize(2);
		constantLocal.resize(8);
		constantSum.resize(8);
		objective.resize(2);
		v.resize(instance.m);
		s.resize(instance.m);
		r.resize(instance.m);
		u.resize(instance.m);
		xTu.resize(instance.n);
		xTw.resize(instance.n);
		Hv.resize(instance.m);
		Hu.resize(instance.m);
		gradient.resize(instance.m);
		oneToN.resize(instance.n);
		randIdx.resize(batchSizeH_);
		woodburyH.resize(batchSizeP_ * batchSizeP_);
		vk.resize(instance.m);
		deltak = 0.0;
		start = 0;
		finish = 0;
		elapsedTime = 0;
		alpha = 0.0;
		beta = 0.0;

		for (I idx = 0; idx < instance.n; idx++)
			oneToN[idx] = idx;

		diag = instance.lambda + mu;

		batchSizeP = batchSizeP_;
		batchSizeH = batchSizeH_;
		tol = 1e-10;
		maxIter = 200;
	}

	void CGDistributedBySamples(std::vector<D> &w,
	                            ProblemData<I, D> & instance,  ProblemData<I, D> &preConData,
	                            boost::mpi::communicator &world, std::ofstream &logFile) {

		int mode = 1;
		lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
		lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
		lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);

		if (lossFunction->getName() == 1)
			lossFunction->getWoodburyH(instance, batchSizeP, woodburyH, xTw, diag);
		if (world.rank() == 0) {
			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
			       0, 0, grad_norm, objective[0]);
			logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << objective[0] << endl;
		}

		for (int iter = 1; iter <= 100; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchSizeH);

			start = gettime_();
			flag[0] = 1;
			flag[1] = 1;
			vbroadcast(world, w, 0);

			cblas_set_to_zero(v);
			cblas_set_to_zero(Hv);
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);

			if (world.rank() == 0) {
				grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
				if (grad_norm < tol) {
					flag[1] = 0;
				}
				cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

				if (lossFunction->getName() == 2) {
					lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
					lossFunction->getWoodburyH(instance, batchSizeP, woodburyH, xTw, diag);
				}
				// s= p^-1 r
				if (batchSizeP == 0)
					ifNoPreconditioning(instance.m, r, s);
				//SGDSolver(instance, instance.m, r, s, diag);
				else
					lossFunction->WoodburySolver(instance, instance.m, randIdx, batchSizeP, woodburyH, r, s, xTw, diag, world, mode);

				cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
			}
			int inner_iter = 0;
			while (1) { //		while (flag != 0)
				vbroadcast(world, u, 0);
				if (flag[0] == 0)
					break;

				lossFunction->computeVectorTimesData(u, instance, xTu, world, mode);
				lossFunction->computeHessianTimesAU(u, Hu, xTw, xTu, instance, batchSizeH, randIdx, world, mode);

				if (world.rank() == 0) {
					//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
					D nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					D denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
					alpha = nom / denom;

					for (I i = 0; i < instance.m ; i++) {
						v[i] += alpha * u[i];
						Hv[i] += alpha * Hu[i];
						r[i] -= alpha * Hu[i];
					}
					// solve linear system to get new s
					if (batchSizeP == 0)
						ifNoPreconditioning(instance.m, r, s);
					//SGDSolver(instance, instance.m, r, s, diag);
					else
						lossFunction->WoodburySolver(instance, instance.m, randIdx, batchSizeP, woodburyH, r, s, xTw, diag, world, mode);

					D nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					beta = nom_new / nom;
					for (I i = 0; i < instance.m ; i++)
						u[i] = beta * u[i] + s[i];

					D r_norm = cblas_l2_norm(instance.m, &r[0], 1);

					if (r_norm <= epsilon || inner_iter > maxIter) {
						cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
						D vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
						D vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
						deltak = sqrt(vHv + alpha * vHu);
						flag[0] = 0;
					}
					inner_iter++;
				}
				vbroadcast(world, flag, 0);

			}


			if (world.rank() == 0)
				cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);

			vbroadcast(world, w, 0);

			finish = gettime_();
			elapsedTime += finish - start;

			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
			output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);
			if (flag[1] == 0)
				break;


		}
	}


	void CGDistributedByFeatures(std::vector<D> &w,
	                             ProblemData<I, D> & instance,  ProblemData<I, D> &preConData,
	                             boost::mpi::communicator &world, std::ofstream &logFile) {

		int mode = 2;
		lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
		lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
		lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);

		if (lossFunction->getName() == 1)
			lossFunction->getWoodburyH(preConData, batchSizeP, woodburyH, xTw, diag);
		grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
		constantLocal[6] = grad_norm * grad_norm;
		vall_reduce(world, constantLocal, constantSum);
		constantSum[6] = sqrt(constantSum[6]);
		if (world.rank() == 0) {
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
			       0, 0, constantSum[6], objective[0]);
			logFile << 0 << "," << 0 << "," << 0 << "," << constantSum[6]  << "," << objective[0] << endl;
		}

		for (int iter = 1; iter <= 100; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchSizeH);

			start = gettime_();
			flag[0] = 1;
			constantLocal[5] = flag[0];
			constantSum[5] = flag[0] * world.size();
			cblas_set_to_zero(v);
			cblas_set_to_zero(Hv);
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);

			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			constantLocal[6] = grad_norm * grad_norm;
			epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
			cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);
			// s= p^-1 r
			if (lossFunction->getName() == 2) {
				lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
				lossFunction->getWoodburyH(preConData, batchSizeP, woodburyH, xTw, diag);
			}
			if (batchSizeP == 0)
				ifNoPreconditioning(instance.m, r, s);
			else
				lossFunction->WoodburySolver(instance, instance.m, randIdx, batchSizeP, woodburyH, r, s, xTw, diag, world, mode);

			cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);

			int inner_iter = 0;
			while (1) { //		while (flag != 0)
				vall_reduce(world, constantLocal, constantSum);
				if (constantSum[5] == 0)
					break; // stop if all the inner flag = 0.

				lossFunction->computeVectorTimesData(u, instance, xTu, world, mode);
				lossFunction->computeHessianTimesAU(u, Hu, xTw, xTu, instance, batchSizeH, randIdx, world, mode);

				D rsLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
				D uHuLocal = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
				constantLocal[0] = rsLocal;
				constantLocal[1] = uHuLocal;
				vall_reduce(world, constantLocal, constantSum);

				alpha = constantSum[0] / constantSum[1];
				for (I i = 0; i < instance.m ; i++) {
					v[i] += alpha * u[i];
					Hv[i] += alpha * Hu[i];
					r[i] -= alpha * Hu[i];
				}
				//CGSolver(P, instance.m, r, s);
				if (batchSizeP == 0)
					ifNoPreconditioning(instance.m, r, s);
				else
					lossFunction->WoodburySolver(instance, instance.m, randIdx, batchSizeP, woodburyH, r, s, xTw, diag, world, mode);


				D rsNextLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
				constantLocal[2] = rsNextLocal;
				vall_reduce(world, constantLocal, constantSum);
				beta = constantSum[2] / constantSum[0];

				for (I i = 0; i < instance.m ; i++) {
					u[i] = beta * u[i] + s[i];
				}
				D r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);

				if ( r_normLocal <= epsilon || inner_iter > maxIter) {			//	if (r_norm <= epsilon || inner_iter > 100)
					cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
					D vHvLocal = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
					D vHuLocal = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
					constantLocal[3] = vHvLocal;
					constantLocal[4] = vHuLocal;
					flag[0] = 0;
					constantLocal[5] = flag[0];
				}
				inner_iter++;

			}
			vall_reduce(world, constantLocal, constantSum);
			deltak = sqrt(constantSum[3] + alpha * constantSum[4]);
			cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
			constantSum[6] = sqrt(constantSum[6]);

			finish = gettime_();
			elapsedTime += finish - start;

			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);

			output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);

			if (constantSum[6] < tol) {
				break;
			}

		}

	}


	void CG_SAG(std::vector<D> &w,
	            ProblemData<I, D> & instance,  ProblemData<I, D> &preConData,
	            boost::mpi::communicator &world, std::ofstream &logFile) {

		int mode = 1;
		lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
		lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
		lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);

		if (world.rank() == 0) {
			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective is %E\n",
			       0, 0, grad_norm, objective[0]);
			logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << objective[0] << endl;
		}

		for (int iter = 1; iter <= 100; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchSizeH);

			start = gettime_();
			flag[0] = 1;
			flag[1] = 1;
			vbroadcast(world, w, 0);

			cblas_set_to_zero(v);
			cblas_set_to_zero(Hv);
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);

			if (world.rank() == 0) {
				grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
				if (grad_norm < tol) {
					flag[1] = 0;
				}
				cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

				// s= p^-1 r
				lossFunction->SAGSolver(instance, instance.m, xTw, r, s, batchSizeP, diag);

				cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
			}
			int inner_iter = 0;
			while (1) { //		while (flag != 0)
				vbroadcast(world, u, 0);
				if (flag[0] == 0)
					break;

				lossFunction->computeVectorTimesData(u, instance, xTu, world, mode);
				lossFunction->computeHessianTimesAU(u, Hu, xTw, xTu, instance, batchSizeH, randIdx, world, mode);

				if (world.rank() == 0) {
					//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
					D nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					D denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
					alpha = nom / denom;

					for (I i = 0; i < instance.m ; i++) {
						v[i] += alpha * u[i];
						Hv[i] += alpha * Hu[i];
						r[i] -= alpha * Hu[i];
					}
					// solve linear system to get new s

					lossFunction->SAGSolver(instance, instance.m, xTw, r, s, batchSizeP, diag);

					D nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
					beta = nom_new / nom;
					for (I i = 0; i < instance.m ; i++)
						u[i] = beta * u[i] + s[i];

					D r_norm = cblas_l2_norm(instance.m, &r[0], 1);

					if (r_norm <= epsilon || inner_iter > maxIter) {
						cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
						D vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
						D vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
						deltak = sqrt(vHv + alpha * vHu);
						flag[0] = 0;
					}
					inner_iter++;
				}
				vbroadcast(world, flag, 0);

			}


			if (world.rank() == 0)
				cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);

			vbroadcast(world, w, 0);

			finish = gettime_();
			elapsedTime += finish - start;

			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);

			output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);
			if (flag[1] == 0)
				break;


		}
	}

	void output(ProblemData<unsigned int, double> &instance, int &iter, int &inner_iter, double & elapsedTime,
	            std::vector<double> &constantSum, std::vector<double> &objective, double & grad_norm,
	            std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (mode == 1) {
			if (world.rank() == 0) {
				printf("%ith: %i CG iters, time %f, norm of gradient %E, objective %E\n",
				       iter, 2*inner_iter, elapsedTime, grad_norm, objective[0]);
				logFile << iter << "," << 2*inner_iter << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
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



};

#endif // CG_H
