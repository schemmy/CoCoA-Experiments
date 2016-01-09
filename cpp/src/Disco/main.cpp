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
#include "readWholeData.h"
#include "../solver/distributed/distributed_essentials.h"

#include "class/LossFunction.h"
#include "class/QuadraticLoss.h"
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

	unsigned int batchsize = 100;

	ProblemData<unsigned int, double> instance;
	instance.theta = ctx.tmp;
	instance.lambda = ctx.lambda;
	loadDistributedSparseSVMRowData(ctx.matrixAFile, world.rank(), world.size(), instance, false);

	ProblemData<unsigned int, double> preConData;

	int mode = 1;
	if (mode == 1) {
		unsigned int finalM;
		vall_reduce_maximum(world, &instance.m, &finalM, 1);
		instance.m = finalM;
		vall_reduce(world, &instance.n, &instance.total_n, 1);
	}
	else if (mode == 2) {
		readPartDataForPreCondi(ctx.matrixAFile, preConData, batchsize, false);
		instance.total_n = instance.n;
	}


	std::vector<double> w(instance.m);
	std::vector<double> vk(instance.m);
	double rho = 1.0 / instance.n;
	double mu = 0.0001;
	double deltak = 0.0;

	std::stringstream ss;
	ss << ctx.matrixAFile << "_1_" << world.size() << ".log";
	std::ofstream logFile;
	logFile.open(ss.str().c_str());

	int loss = distributedSettings.lossFunction;

	LossFunction<unsigned int, double> * lf;
	lf = new QuadraticLoss<unsigned int, double>();
	lf->init(instance);


	if (mode == 1) {
		if (world.rank() == 0)
			lf->computeInitialW(w, instance, rho, world.rank());
	}

	lf->distributed_PCGByD(w, instance, preConData, mu, vk, deltak, batchsize, world, logFile, mode);


	logFile.close();
	MPI::Finalize();
	return 0;

}