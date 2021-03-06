/*
 * LossFunction.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_

template<typename L, typename D>
class LossFunction {
public:
	LossFunction() {

	}
	virtual ~LossFunction() {}

	virtual void init(ProblemData<L, D> & instance) {
	}

	virtual int getName() { return 0;}

	virtual void computeVectorTimesData(std::vector<double> &vec, ProblemData<unsigned int, double> &instance,
	                                    std::vector<double> &result, boost::mpi::communicator &world, int &mode) {
	}

	virtual void computeObjective(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                              std::vector<double> &xTw, double & obj, boost::mpi::communicator & world, int &mode) {
	}

	virtual void computePrimalAndDualObjective(ProblemData<unsigned int, double> &instance,
	        std::vector<double> &alpha, std::vector<double> &w, double &rho,
	        double &finalDualError, double &finalPrimalError) {
	}

	virtual void computeGradient(std::vector<double> &w, std::vector<double> &grad, std::vector<double> &xTw,
	                             ProblemData<unsigned int, double> &instance, boost::mpi::communicator & world, int &mode) {
	}

	virtual void computeLocalGradient(std::vector<double> &w, std::vector<double> &grad, std::vector<double> &xTw,
	                                  ProblemData<unsigned int, double> &instance, boost::mpi::communicator & world, int &mode) {

	}
	virtual void computeHessianTimesAU(std::vector<double> &u, std::vector<double> &Hu, std::vector<double> &xTw,
	                                   std::vector<double> &xTu, ProblemData<unsigned int, double> &instance,
	                                   unsigned int &batchSizeH,  std::vector<unsigned int> &randIdx,
	                                   boost::mpi::communicator & world, int &mode) {
	}

	virtual void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                             ProblemData<unsigned int, double> &preConData, double &mu,
	                             std::vector<double> &vk, double &deltak, unsigned int &batchSizeP, unsigned int &batchSizeH,
	                             boost::mpi::communicator &world, std::ofstream &logFile, int &mode) {
	}


	virtual void getWoodburyH(ProblemData<unsigned int, double> &instance,
	                          unsigned int &p, std::vector<double> &woodburyH, std::vector<double> &wTx,  double & diag) {
	}

	virtual void WoodburySolver(ProblemData<unsigned int, double> &instance, unsigned int &n,
	                            std::vector<unsigned int> &randIdx, unsigned int &p, std::vector<double> &woodburyH,
	                            std::vector<double> &b, std::vector<double> &x, std::vector<double> &wTx, double & diag,
	                            boost::mpi::communicator & world, int &mode) {
	}

	virtual void SAGSolver(ProblemData<unsigned int, double> &instance, unsigned int &n, std::vector<double> &xTw,
	                       std::vector<double> &b, std::vector<double> &x, int nEpoch, double &diag) {
	}

	virtual void oneCocoaSDCAUpdate(ProblemData<unsigned int, double> &instance, std::vector<double> &w,
	                                std::vector<double> &deltaAlpha, std::vector<double> &deltaW) {
	}

	virtual void computeInitialW(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                             double & rho, int rank) {
	}

	virtual void computeStoGrad_SVRG(int iter, int freq, unsigned int batchGrad,
	                                 std::vector<double> &w, std::vector<double> &wRec,
	                                 ProblemData<unsigned int, double> instance,
	                                 std::vector<double> &xTw, std::vector<double> &xTwRec,
	                                 std::vector<double> &gradientFull, std::vector<double> &gradientRec,
	                                 std::vector<double> &gradient, std::vector<unsigned int> &randIdxGrad) {

	}

	virtual void StoWoodburyHGet(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
	                             unsigned int &p, std::vector<double> &woodburyH,
	                             std::vector<double> &wTx, std::vector<unsigned int> &randIdx, double & diag) {
	}
	
	virtual void StoWoodburySolve(unsigned int batchHessian, std::vector<double> &w, ProblemData<unsigned int, double> instance,
	                              std::vector<double> &woodburyH,
	                              std::vector<double> &gradient, std::vector<double> &woodburyZHVTy,
	                              std::vector<double> &woodburyVTy, std::vector<double> &woodburyHVTy,
	                              std::vector<double> &vk, std::vector<unsigned int> &randIdx) {


	}


};

#endif /* LOSSFUNCTION_H_ */
