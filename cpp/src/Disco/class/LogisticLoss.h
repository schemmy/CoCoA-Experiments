/*
 * LogisticLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOGISTICLOSS_H_
#define LOGISTICLOSS_H_


#include "LossFunction.h"

template<typename L, typename D>
class LogisticLoss : public LossFunction<L, D> {
public:

	LogisticLoss() {}

	virtual ~LogisticLoss() {}

	virtual void computeObjective(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &obj, int nPartition) {

		obj = 0.0;

		for (unsigned int idx = 0; idx < instance.n; idx++) {

			double w_x = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

			obj += log(1.0 + exp(-w_x * instance.b[idx]));
		}

		obj = 1.0 / instance.total_n * obj + 0.5 * instance.lambda * cblas_l2_norm(w.size(), &w[0], 1)
		      * cblas_l2_norm(w.size(), &w[0], 1) / nPartition;

	}
	virtual void computePrimalAndDualObjective(ProblemData<unsigned int, double> &instance,
	        std::vector<double> &alpha, std::vector<double> &w, double &rho, double &finalDualError,
	        double &finalPrimalError) {

		double localError = 0;
		for (unsigned int i = 0; i < instance.n; i++) {
			double tmp = 0;
			if (instance.b[i] == -1.0) {
				if (alpha[i] < 0) {
					tmp += -alpha[i] * log(-alpha[i]) ;
				}
				if (alpha[i] > -1) {
					tmp += (1.0 + alpha[i]) * log(1.0 + alpha[i]);
				}

			}
			if (instance.b[i] == 1.0) {
				if (alpha[i] > 0) {
					tmp += alpha[i] * log(alpha[i]) ;
				}
				if (alpha[i] < 1) {
					tmp += (1.0 - alpha[i]) * log(1.0 - alpha[i]);
				}
			}
			localError += tmp;
		}

		double localLogisticLoss = 0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			double dotProduct = 0;
			for (unsigned int i = instance.A_csr_row_ptr[idx];
			        i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += (w[instance.A_csr_col_idx[i]])
				              * instance.A_csr_values[i];
			}

			double tmp = -1.0 * instance.b[idx] * instance.b[idx] * dotProduct;
			localLogisticLoss += log(1.0 + exp(tmp));

		}
		finalPrimalError = 0;
		finalDualError = 0;

		double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
		finalDualError = 1.0 / instance.n * localError
		                 + 0.5 * rho * tmp2 * tmp2;
		finalPrimalError =  1.0 / instance.n * localLogisticLoss
		                    + 0.5 * rho * tmp2 * tmp2;

	}



	virtual void computeGradient(std::vector<double> &w, std::vector<double> &grad,
	                                     ProblemData<unsigned int, double> &instance) {

		cblas_set_to_zero(grad);
		double temp = 0.0;
		double w_x = 0.0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			w_x = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

			temp = exp(-1.0 * instance.b[idx] * w_x);
			temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.n;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i];
		}

		cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);

	}


	void computeAtimesW(ProblemData<unsigned int, double> &instance, std::vector<double> &w,
	                    std::vector<double> &wTx) {

		double temp;

		for (unsigned int idx = 0; idx < instance.n; idx++) {

			temp = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				temp += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i] * instance.b[idx];
			}
			wTx[idx] = temp;
		}

	}

	virtual void computeHessianTimesAU(std::vector<double> &w, std::vector<double> &wTx,
	        std::vector<double> &u, std::vector<double> &Hu,
	        ProblemData<unsigned int, double> &instance) {

		double temp, scalar;
		double r1, r2;
		cblas_set_to_zero(Hu);

		for (unsigned int idx = 0; idx < instance.n; idx++) {

			temp = exp(-wTx[idx]);
			scalar = temp / (temp + 1) / (temp + 1);
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				r1 = instance.A_csr_values[i];
				r2 = instance.A_csr_col_idx[i];
				for (unsigned int j = instance.A_csr_row_ptr[idx]; j < instance.A_csr_row_ptr[idx + 1]; j++) {
					Hu[r2] += (r1 * instance.A_csr_values[j]) * scalar //* instance.b[idx] * instance.b[idx]
					          * u[instance.A_csr_col_idx[j]] / instance.n;
				}
			}

		}

		for (unsigned int i = 0; i < instance.m; i++)
			Hu[i] += instance.lambda * u[i];

	}





};

#endif /* LOGISTICLOSS_H_ */
