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
	LossFunction(){

	}
	virtual ~LossFunction() {}

	virtual void init(ProblemData<L, D> & instance){
	}

	virtual void computeObjective(std::vector<double> &w, 
		ProblemData<unsigned int, double> &instance, double &obj, int nPartition) {
	}

	virtual void computePrimalAndDualObjective(ProblemData<unsigned int, double> &instance,
        std::vector<double> &alpha, std::vector<double> &w, double &rho, 
        double &finalDualError, double &finalPrimalError) {
	}

	virtual void computeGradient(std::vector<double> &w, std::vector<double> &grad,
        ProblemData<unsigned int, double> &instance) {		
	}


    virtual void computeHessianTimesAU(std::vector<double> &w, std::vector<double> &u, 
    	std::vector<double> &Hu, ProblemData<unsigned int, double> &instance) {
    }

	virtual void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
        std::vector<double> &vk, double &deltak, unsigned int &batchSize,
        boost::mpi::communicator &world, std::ofstream &logFile) {
	}
	
};

#endif /* LOSSFUNCTION_H_ */
