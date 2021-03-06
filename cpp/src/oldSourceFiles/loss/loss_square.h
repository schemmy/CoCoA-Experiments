/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

/*
 *  Created on: 19 Jan 2012
 *      Author: jmarecek and taki
 */

#ifndef SQUARE_LOSS_H
#define SQUARE_LOSS_H

#include "loss_abstract.h"

struct square_loss_traits: public loss_traits {
};

/*******************************************************************/
// a partial specialisation for square loss
template<typename L, typename D>
class Losses<L, D, square_loss_traits> {

public:

	static inline void set_residuals_for_zero_x(const ProblemData<L, D> &inst,
			std::vector<D> &residuals) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = -inst.b[i];
		}
	}

	static inline void bulkIterations(const ProblemData<L, D> &inst,
			std::vector<D> &residuals) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = -inst.b[i];
		}
		for (L i = 0; i < inst.n; i++) {
			for (L j = inst.A_csc_col_ptr[i]; j < inst.A_csc_col_ptr[i + 1];
					j++) {
				residuals[inst.A_csc_row_idx[j]] += inst.A_csc_values[j]
						* inst.x[i];
			}
		}
	}
	static inline void bulkIterations(const ProblemData<L, D> &inst,
			std::vector<D> &residuals, std::vector<D> &x) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = -inst.b[i];
		}
		for (L i = 0; i < inst.n; i++) {
			for (L j = inst.A_csc_col_ptr[i]; j < inst.A_csc_col_ptr[i + 1];
					j++) {
				residuals[inst.A_csc_row_idx[j]] += inst.A_csc_values[j] * x[i];
			}
		}
	}
	static inline void bulkIterations_for_my_instance_data(
			const ProblemData<L, D> &inst, std::vector<D> &residuals) {
		for (L row = 0; row < inst.m; row++) {
			residuals[row] = -inst.b[row];
			for (L col_tmp = inst.A_csr_row_ptr[row];
					col_tmp < inst.A_csr_row_ptr[row + 1]; col_tmp++) {
				residuals[row] += inst.A_csr_values[col_tmp]
						* inst.x[inst.A_csr_col_idx[col_tmp]];
			}
		}
	}

	static inline D do_single_iteration_serial(const ProblemData<L, D> &inst,
			const L idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		x[idx] += tmp;
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			residuals[inst.A_csc_row_idx[j]] += tmp * inst.A_csc_values[j];
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel(const ProblemData<L, D> &inst,
			const L idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residuals[inst.A_csc_row_idx[j]],
					tmp * inst.A_csc_values[j]);
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel_for_distributed(
			const ProblemData<L, D> &inst, const L idx,
			std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li, D* residual_updates) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residual_updates[inst.A_csc_row_idx[j]],
					tmp * inst.A_csc_values[j]);
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel_for_distributed(
			const ProblemData<L, D> &inst,
			const ProblemData<L, D> &inst_local, const L idx,
			std::vector<D> &residuals, std::vector<D> &residuals_local,
			std::vector<D> &x, const std::vector<D> &Li, D* residual_updates) {
		D tmp = 0;
		tmp = compute_update(inst, inst_local, residuals, residuals_local, idx,
				Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residual_updates[inst.A_csc_row_idx[j]],
					tmp * inst.A_csc_values[j]);
		}
		for (unsigned int j = inst_local.A_csc_col_ptr[idx];
				j < inst_local.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residuals_local[inst_local.A_csc_row_idx[j]],
					tmp * inst_local.A_csc_values[j]);
		}

		return abs(tmp);
	}

	static inline D compute_update(const ProblemData<L, D> &inst,
			const std::vector<D> &residuals, const L idx,
			const std::vector<D> &Li) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst.A_csc_values[j] * residuals[inst.A_csc_row_idx[j]];
		}
		tmp = compute_soft_treshold(Li[idx] * inst.lambda,
				inst.x[idx] - Li[idx] * tmp) - inst.x[idx];
		return tmp;
	}

	static inline D compute_update(const ProblemData<L, D> &inst,
			const ProblemData<L, D> &inst_local,
			const std::vector<D> &residuals,
			const std::vector<D> &residuals_local, const L idx,
			const std::vector<D> &Li) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst.A_csc_values[j] * residuals[inst.A_csc_row_idx[j]];
		}
		for (unsigned int j = inst_local.A_csc_col_ptr[idx];
				j < inst_local.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst_local.A_csc_values[j]
					* residuals_local[inst_local.A_csc_row_idx[j]];
		}
		tmp = compute_soft_treshold(Li[idx] * inst.lambda,
				inst.x[idx] - Li[idx] * tmp) - inst.x[idx];
		return tmp;
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals) {
		D resids = 0;
		D sumx = 0;
		for (L i = 0; i < part.m; i++) {
			resids += residuals[i] * residuals[i];
		}
		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}
		return 0.5 * resids + part.lambda * sumx;
	}

	static inline void compute_reciprocal_lipschitz_constants(
			const ProblemData<L, D> &inst, std::vector<D> &h_Li) {
		for (unsigned int i = 0; i < inst.n; i++) {
			h_Li[i] = 0;
			for (unsigned int j = inst.A_csc_col_ptr[i];
					j < inst.A_csc_col_ptr[i + 1]; j++) {
				h_Li[i] += inst.A_csc_values[j] * inst.A_csc_values[j];
			}
			if (h_Li[i] > 0)
				h_Li[i] = 1 / (inst.sigma * h_Li[i]); // Compute reciprocal Lipschitz Constants
		}
	}

	static inline void compute_reciprocal_lipschitz_constants(
			const ProblemData<L, D> &inst,
			const ProblemData<L, D> &inst_local, std::vector<D> &h_Li) {
#pragma omp parallel for
		for (unsigned int i = 0; i < inst.n; i++) {
			h_Li[i] = 0;
			for (unsigned int j = inst.A_csc_col_ptr[i];
					j < inst.A_csc_col_ptr[i + 1]; j++) {
				h_Li[i] += inst.A_csc_values[j] * inst.A_csc_values[j];
			}
			for (unsigned int j = inst_local.A_csc_col_ptr[i];
					j < inst_local.A_csc_col_ptr[i + 1]; j++) {
				h_Li[i] += inst_local.A_csc_values[j]
						* inst_local.A_csc_values[j];
			}
			if (h_Li[i] > 0)
				h_Li[i] = 1 / (inst.sigma * h_Li[i]); // Compute reciprocal Lipschitz Constants
		}
	}

};

#endif /* SQUARE_LOSS_H */
