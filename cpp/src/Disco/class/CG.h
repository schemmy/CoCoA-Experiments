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


// delete this, this is rubish
	void SH(std::vector<D> &w, ProblemData<I, D> & instance, boost::mpi::communicator & world, std::ofstream &logFile) {

		int mode = 1;
		diag = instance.lambda;

		std::vector<double> w_try(instance.m);
		std::vector<double> xTw_try(instance.n);
		std::vector<double> woodburyZHVTy(instance.m);
		//batchSizeH = min(instance.n, batchSizeH + 100);
		std::vector<double> woodburyVTy(batchSizeH);
		std::vector<double> woodburyVTy_World(batchSizeH);
		std::vector<double> woodburyHVTy(batchSizeH);
		woodburyH.resize(batchSizeH * batchSizeH);
		randIdx.resize(batchSizeH);

		unsigned int batchGrad = floor(instance.n/1);
		std::vector<unsigned int> randIdxGrad(batchGrad);
		objective[0] = 1.0;

		for (int iter = 1; iter <= 1000; iter++) {

			geneRandIdx(oneToN, randIdx, instance.n, batchSizeH);
			geneRandIdx(oneToN, randIdxGrad, instance.n, batchGrad);

			start = gettime_();
			cblas_set_to_zero(xTw);
			for (unsigned int j = 0; j < batchGrad; j++) {
				unsigned int idx = randIdxGrad[j];
				double temp = 0.0;
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					temp += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

				xTw[idx] = temp * instance.b[idx];
			}

			//			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
			cblas_set_to_zero(gradient);
			for (unsigned int j = 0; j < batchGrad; j++) {
				unsigned int idx = randIdxGrad[j];
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					gradient[instance.A_csr_col_idx[i]] += (xTw[idx] - instance.b[idx]) * instance.A_csr_values[i] * instance.b[idx]
					                                       / batchGrad;
			}
			cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &gradient[0], 1);

			// lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			// lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);


			//			lossFunction->getWoodburyH(instance, batchSizeH, woodburyH, xTw, diag);
			cblas_set_to_zero(woodburyH);
			for (unsigned int ii = 0; ii < batchSizeH; ii++) {
				unsigned int idx1 = randIdx[ii];
				for (unsigned int jj = 0; jj < batchSizeH; jj++) {
					unsigned int idx2 = randIdx[jj];
					unsigned int i = instance.A_csr_row_ptr[idx1];
					unsigned int j = instance.A_csr_row_ptr[idx2];
					while (i < instance.A_csr_row_ptr[idx1 + 1] && j < instance.A_csr_row_ptr[idx2 + 1]) {
						if (instance.A_csr_col_idx[i] == instance.A_csr_col_idx[j]) {
							woodburyH[ii * batchSizeH + jj] += instance.A_csr_values[i] * instance.A_csr_values[j]
							                                   * instance.b[idx1] * instance.b[idx2] / diag / batchSizeH;
							i++;
							j++;
						}
						else if (instance.A_csr_col_idx[i] < instance.A_csr_col_idx[j])
							i++;
						else
							j++;
					}
				}
			}
			for (unsigned int idx = 0; idx < batchSizeH; idx++)
				woodburyH[idx * batchSizeH + idx] += 1.0;

			cblas_set_to_zero(woodburyZHVTy);
			cblas_set_to_zero(woodburyVTy);

			for (unsigned int j = 0; j < batchSizeH; j++) {
				unsigned int idx = randIdx[j];
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
					woodburyVTy[j] += instance.A_csr_values[i] * instance.b[idx] *
					                  gradient[instance.A_csr_col_idx[i]] / diag / batchSizeH;
				}
			}

			CGSolver(woodburyH, batchSizeH, woodburyVTy, woodburyHVTy);

			for (unsigned int j = 0; j < batchSizeH; j++) {
				unsigned int idx = randIdx[j];
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
					woodburyZHVTy[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx]
					        / diag * woodburyHVTy[j];
				}
			}

			for (unsigned int i = 0; i < instance.m; i++)
				vk[i] =  (gradient[i] / diag - woodburyZHVTy[i]);


			//line search
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
				w[i] =  w[i] - 0.0005 * vk[i];

			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
			double grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			int inner_iter = 0;
			finish = gettime_();
			elapsedTime += finish - start;
			output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);



		}

	}



	void SGD(std::vector<D> &w, ProblemData<I, D> & instance, boost::mpi::communicator & world, std::ofstream &logFile) {

		int mode = 1;
		diag = instance.lambda;

		std::vector<double> w_try(instance.m);
		std::vector<double> xTw_try(instance.n);
		std::vector<double> woodburyZHVTy(instance.m);
		//batchSizeH = min(instance.n, batchSizeH + 100);
		std::vector<double> woodburyVTy(batchSizeH);
		std::vector<double> woodburyVTy_World(batchSizeH);
		std::vector<double> woodburyHVTy(batchSizeH);
		woodburyH.resize(batchSizeH * batchSizeH);
		randIdx.resize(instance.n);
		objective[0] = 1.0;

		for (int iter = 1; iter <= 100; iter++) {

			start = gettime_();


			for (int ii = 1; ii <= instance.n * 4; ii++) {

				//			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
				cblas_set_to_zero(xTw);
				unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
				double temp = 0.0;
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					temp += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

				xTw[idx] = temp * instance.b[idx];
				cblas_set_to_zero(gradient);

				//			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
				for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
					gradient[instance.A_csr_col_idx[i]] += (xTw[idx] - instance.b[idx])
					                                       * instance.A_csr_values[i] * instance.b[idx];
				cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &gradient[0], 1);


				for (unsigned int i = 0; i < instance.m; i++)
					w[i] =  w[i] - 0.0001 * gradient[i];
			}
			lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
			lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
			lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
			double grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			int inner_iter = 0;
			finish = gettime_();
			elapsedTime += finish - start;
			output(instance, iter, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);



		}

	}

	void SAG(std::vector<D> &w, ProblemData<I, D> & instance, boost::mpi::communicator & world, std::ofstream &logFile) {

		int mode = 1;
		diag = instance.lambda;

		objective[0] = 1.0;

		double eta = 0.01;
		double kappa = 1.0;
		int em = 0;
		int nEpoch = 100;
		double xTs = 0.0;
		double nomNew = 1.0;
		double nom0 = 1.0;
		unsigned int k = 1;
		std::vector<double> gradIdx(instance.m * instance.n);



		// for (unsigned int ii = 0; ii < instance.n * 1; ii++) {
		// 	xTs = 0.0;
		// 	unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
		// 	for (unsigned int i = 0; i < instance.m; i++) {
		// 		gradAvg[i] -= gradIdx[idx * instance.m + i];
		// 		gradIdx[idx * instance.m + i] = 0.0;
		// 	}
		// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
		// 		xTs += instance.A_csr_values[i] * w[instance.A_csr_col_idx[i]] * instance.b[idx];
		// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
		// 		gradIdx[idx * instance.m + instance.A_csr_col_idx[i]] = instance.A_csr_values[i]
		// 		        * instance.b[idx] * (xTs - instance.b[idx]);
		// 		gradAvg[instance.A_csr_col_idx[i]] += gradIdx[idx * instance.m + instance.A_csr_col_idx[i]];
		// 	}
		// 	for (unsigned int i = 0; i < instance.m; i++) {
		// 		w[i] -= eta * (1.0 / instance.n * gradAvg[i]  + diag * w[i]);
		// 	}
		// }

		std::vector<double> gradAvg(instance.m);
		std::vector<double> y(instance.n);
		std::vector<double> C(instance.n);
		std::vector<int> V(instance.m, 1);
		std::vector<double> z(instance.m);
		std::vector<double> S(nEpoch * instance.n);
		cblas_dcopy(instance.m, &w[0], 1, &z[0], 1);

		for (k = 1; k < instance.n * nEpoch; k++) {

			start = gettime_();

			unsigned int idx = floor(rand() / (0.0 + RAND_MAX) * instance.n);
			if (C[idx] == 0) {
				em++;
				C[idx] = 1;
			}

			// Just-in-time calculation of needed values of z
			xTs = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				z[instance.A_csr_col_idx[i]] = z[instance.A_csr_col_idx[i]]
				                               - (S[k - 1] - S[V[instance.A_csr_col_idx[i]] - 1]) * gradAvg[instance.A_csr_col_idx[i]];

				V[instance.A_csr_col_idx[i]] = k;
				xTs += instance.A_csr_values[i] * z[instance.A_csr_col_idx[i]] * instance.b[idx];
			}
			//Update the memory y and the direction
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				gradAvg[instance.A_csr_col_idx[i]] -= y[idx] * instance.A_csr_values[i] * instance.b[idx];
			}
			y[idx] = xTs * kappa - instance.b[idx];
			//y[idx] = -1.0 * exp(-xTs * instance.b[idx]) / (1.0 + exp(-xTs * instance.b[idx])) * kappa;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				gradAvg[instance.A_csr_col_idx[i]] += y[idx] * instance.A_csr_values[i] * instance.b[idx];
			}
			//Update kappa and the sum needed for z updates.
			kappa = kappa * (1.0 - eta * diag);
			S[k] = S[k - 1] + eta / kappa / em;

			if ( (k % instance.n) == 0) {

				for (unsigned int i = 0; i < instance.m; i++) {
					w[i] = kappa * (z[i] - (S[k - 1] - S[V[i] - 1]) * gradAvg[i]);
				}

				lossFunction->computeVectorTimesData(w, instance, xTw, world, mode);
				lossFunction->computeObjective(w, instance, xTw, objective[0], world, mode);
				lossFunction->computeGradient(w, gradient, xTw, instance, world, mode);
				double grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
				int inner_iter = 0;
				finish = gettime_();
				elapsedTime += finish - start;
				int ep = floor(k / instance.n);
				output(instance, ep, inner_iter, elapsedTime, constantSum, objective, grad_norm, logFile, world, mode);

			}

		}

	}




	void output(ProblemData<unsigned int, double> &instance, int &iter, int &inner_iter, double & elapsedTime,
	            std::vector<double> &constantSum, std::vector<double> &objective, double & grad_norm,
	            std::ofstream & logFile, boost::mpi::communicator & world, int &mode) {

		if (mode == 1) {
			if (world.rank() == 0) {
				printf("%ith: %i CG iters, time %f, norm of gradient %E, objective %E\n",
				       iter, 2 * inner_iter, elapsedTime, grad_norm, objective[0]);
				logFile << iter << "," << 2 * inner_iter << "," << elapsedTime << "," << grad_norm << "," << objective[0] << endl;
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
