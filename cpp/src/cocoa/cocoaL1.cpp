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
#include "cocoaHelper.h"
#include "../solver/distributed/distributed_essentials.h"

#include "class/HingeLossCD.h"


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

	ProblemData<unsigned int, double> forB;
	loadDistributedSparseSVMRowDataForL1CoCoA_readLabels(ctx.matrixAFile, forB, false);

	ProblemData<unsigned int, double> instance;
	// instance.theta = ctx.tmp;
	cout << world.rank() << " going to load data" << endl;
	//ctx.matrixAFile = "/Users/Schemmy/Desktop/ac-dc/cpp/data/a1a.4/a1a";
	//cout<< ctx.matrixAFile<<endl;

	loadDistributedSparseSVMRowDataForL1CoCoA(ctx.matrixAFile, world.rank(), world.size(),
	                                instance, false);
	instance.b = forB.b;
	unsigned int finalM;

	vall_reduce_maximum(world, &instance.m, &finalM, 1);

	// cout << "Local m " << instance.m << "   global m " << finalM << endl;

	instance.m = finalM;
	vall_reduce(world, &instance.n, &instance.total_n, 1);
	cout << "Local n " << instance.n << " Golbal n "<< instance.total_n <<endl;

	instance.lambda = ctx.lambda;
	// instance.oneOverLambdaN = 1 / (0.0 + instance.total_n * instance.lambda);
	instance.oneOverLambdaN = 1.0;

	std::vector<double> wBuffer(instance.m);
	std::vector<double> deltaW(instance.m);
	std::vector<double> deltaAlpha(instance.n);
	std::vector<double> w(instance.m);

	instance.x.resize(instance.n);
	cblas_set_to_zero(instance.x);
	cblas_set_to_zero(w);
	cblas_set_to_zero(deltaW);

	// compute local w
	// vall_reduce(world, deltaW, w);
	double gamma;
	if (distributedSettings.APPROX) {
		gamma = 1;
		instance.penalty = world.size() + 0.0;
	} else {
		gamma = 1. / (world.size() + 0.0);
		instance.penalty = 1;
	}

	LossFunction<unsigned int, double> * lf;
	instance.experimentName = ctx.experimentName;

	int loss = distributedSettings.lossFunction;

	switch (loss) {
	case 1:
		lf = new HingeLossCD<unsigned int, double>();
		break;
	default:
		break;
	}

	lf->init(instance);


	std::stringstream ss;
	int localsolver = distributedSettings.LocalMethods;

	ss <<ctx.matrixAFile << "_"<< "L1_" 
	   << localsolver << "_"
	   << distributedSettings.iters_communicate_count << "_"
	   << distributedSettings.iterationsPerThread << "_"
	   << instance.lambda << "_"
	   << distributedSettings.APPROX
	   << ".log";
	std::ofstream logFile;

	if (ctx.settings.verbose) {
		logFile.open(ss.str().c_str());
		cout<<ss.str().c_str()<<endl;
	}

	distributedSettings.iters_bulkIterations_count = 5;


	switch (localsolver) {
	case 1:
		lf->L1CoCoA_subproblem_solver_SDCA(instance, deltaAlpha, w, wBuffer, deltaW,
		                           distributedSettings, world, gamma, ctx, logFile);
		break;
	case 2:
		lf->L1CoCoA_Acce_subproblem_solver_SDCA(instance, deltaAlpha, w, wBuffer, deltaW,
		                                distributedSettings, world, gamma, ctx, logFile);
		break;

	default:
		break;
	}

	// for(int i = 0; i<instance.n; i++){
	// 	if (abs(instance.x[i])>1e-16)
	// 	cout<<i<<"  "<<instance.x[i]<<endl;
	// }

	if (ctx.settings.verbose) {
		logFile.close();
	}

	MPI::Finalize();

	return 0;
}
