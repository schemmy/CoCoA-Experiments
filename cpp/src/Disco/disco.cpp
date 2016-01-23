#include "../solver/distributed/distributed_include.h"
#include "../class/Context.h"
#include "../helpers/option_console_parser.h"
#include "../solver/settingsAndStatistics.h"
#include "../utils/file_reader.h"
#include "../solver/Solver.h"
#include "../helpers/utils.h"
#include <math.h>
#include "../utils/distributed_instances_loader.h"
#include "../utils/matrixvector.h"
#include "../solver/distributed/distributed_structures.h"
#include "../helpers/option_distributed_console_parser.h"
#include "class/readWholeData.h"
#include "../solver/distributed/distributed_essentials.h"

#include "class/LossFunction.h"
#include "class/QuadraticLoss.h"
#include "class/LogisticLoss.h"
#include "class/CG.h"
#include  <sstream>

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	mpi::environment env(argc, argv);
	mpi::communicator world;
	DistributedSettings distributedSettings;
	Context ctx(distributedSettings);
	consoleHelper::parseDistributedOptions(ctx, distributedSettings, argc,
	                                       argv);

	ctx.settings.verbose = true;
	if (world.rank() != 0) {
		ctx.settings.verbose = false;
	}

	unsigned int batchsizeP = distributedSettings.iters_communicate_count;

	ProblemData<unsigned int, double> instance;
	instance.theta = ctx.tmp;
	instance.lambda = ctx.lambda;
	ProblemData<unsigned int, double> preConData;


	int mode = distributedSettings.LocalMethods;
	if (mode == 1 || mode == 3) {
		loadDistributedSparseSVMRowData(ctx.matrixAFile, world.rank(), world.size(), instance, false);
		unsigned int finalM;
		vall_reduce_maximum(world, &instance.m, &finalM, 1);
		instance.m = finalM;
		vall_reduce(world, &instance.n, &instance.total_n, 1);
	}
	else if (mode == 2) {
		loadDistributedByFeaturesSVMRowData(ctx.matrixAFile, world.rank(), world.size(), instance, false);
		readPartDataForPreCondi(ctx.matrixAFile, preConData, batchsizeP, false);
		instance.total_n = instance.n;
	}

	unsigned int batchsizeH = floor(instance.n / distributedSettings.iterationsPerThread);

	std::vector<double> w(instance.m);
	//for (unsigned int i = 0; i < instance.m; i++) w[i] = 0.5;
	double rho = 1.0 / instance.n;
	double mu = 0.1;

	int loss = distributedSettings.lossFunction;

	std::stringstream ss;
	ss << ctx.matrixAFile << "_"<< loss <<"_" << mode <<"_" << batchsizeP <<"_" << distributedSettings.iterationsPerThread
			 <<"_" << world.size() << ".log";
	std::ofstream logFile;
	logFile.open(ss.str().c_str());


	LossFunction<unsigned int, double> * lf;

	switch (loss) {
	case 1:
		lf = new QuadraticLoss<unsigned int, double>();
		lf->init(instance);
		break;
	case 2:
		lf = new LogisticLoss<unsigned int, double>();
		lf->init(instance);
		break;
	default:
		break;
	}
	
	CG<unsigned int, double> CGmethod(instance, batchsizeP, batchsizeH, mu, lf);
	switch (mode) {
	case 1:
		CGmethod.CGDistributedBySamples(w, instance, preConData, world, logFile);
		break;
	case 2:
		CGmethod.CGDistributedByFeatures(w, instance, preConData, world, logFile);
		break;
	case 3:
		CGmethod.CG_SAG(w, instance, preConData, world, logFile);
		break;
	default:
		break;
	}	

	//lf->distributed_PCG(w, instance, preConData, mu, vk, deltak, batchsizeP, batchsizeH, world, logFile, mode);

	// if (mode == 1) {
	// 	if (world.rank() == 0)
	// 		lf->computeInitialW(w, instance, rho, world.rank());
	// }

	//double nn = cblas_l2_norm(instance.m, &w[0], 1) * cblas_l2_norm(instance.m, &w[0], 1); cout<<nn<<endl;
	logFile.close();
	MPI::Finalize();
	return 0;

}