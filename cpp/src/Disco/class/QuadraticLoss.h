/*
 * QuadraticLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef QUADRATICLOSS_H_
#define QUADRATICLOSS_H_


#include "LossFunction.h"

template<typename L, typename D>
class QuadraticLoss: public LossFunction<L, D>  {
public:
	QuadraticLoss() {

	}

	virtual ~QuadraticLoss() {}

	virtual void computeObjective(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &obj, int nPartition) {

		obj = 0.0;

		for (unsigned int idx = 0; idx < instance.n; idx++) {

			double w_x = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

			obj += 0.5 * (w_x * instance.b[idx] - instance.b[idx]) * (w_x * instance.b[idx] - instance.b[idx]);
		}

		obj = 1.0 / instance.total_n * obj + 0.5 * instance.lambda * cblas_l2_norm(w.size(), &w[0], 1)
		      * cblas_l2_norm(w.size(), &w[0], 1) / nPartition;

	}


	virtual void computePrimalAndDualObjective(ProblemData<unsigned int, double> &instance,
	        std::vector<double> &alpha, std::vector<double> &w, double &rho, double &finalDualError,
	        double &finalPrimalError) {

		double localError = 0.0;
		for (unsigned int i = 0; i < instance.n; i++) {
			double tmp = alpha[i] * alpha[i] * 0.5 - instance.b[i] * alpha[i];
			localError += tmp;
		}

		double localQuadLoss = 0.0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			double dotProduct = 0.0;
			for (unsigned int i = instance.A_csr_row_ptr[idx];
			        i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
			}
			double single = 1.0 * instance.b[idx] -  dotProduct * instance.b[idx];
			double tmp = 0.5 * single * single;

			localQuadLoss += tmp;
		}

		finalPrimalError = 0;
		finalDualError = 0;

		double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
		finalDualError = 1.0 / instance.n * localError
		                 + 0.5 * rho * tmp2 * tmp2;
		finalPrimalError =  1.0 / instance.n * localQuadLoss
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

			temp = (w_x * instance.b[idx] - instance.b[idx]);
			//temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.total_n;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i] * instance.b[idx] / instance.n;
		}

		cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);

	}



};

#endif /* QUADRATICLOSS_H_ */
