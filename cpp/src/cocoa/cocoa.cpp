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

#include "class/LogisticLossCD.h"
#include "class/HingeLossCD.h"
#include "class/HingeLossCDCaseC.h"

#include "class/QuadraticLossCD.h"
#include "class/SquareHingeLossCD.h"

//#ifdef MATLAB
//
//#include "class/QuadraticLossLbfgs.h"
//#include "class/LogisticLossMatlab.h"
//
//#endif
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
	ProblemData<unsigned int, double> instance;
	instance.theta = ctx.tmp;
	cout << "XXXXXXXx   " << instance.theta << endl;
	cout << world.rank() << " going to load data" << endl;
	//ctx.matrixAFile = "/Users/Schemmy/Desktop/ac-dc/cpp/data/a1a.4/a1a";
	//cout<< ctx.matrixAFile<<endl;

	loadDistributedSparseSVMRowData(ctx.matrixAFile, world.rank(), world.size(),
	                                instance, false);
	unsigned int finalM;

	vall_reduce_maximum(world, &instance.m, &finalM, 1);

//	cout << "Local m " << instance.m << "   global m " << finalM << endl;

	instance.m = finalM;

	//	for (unsigned int i = 0; i < instance.m; i++) {
	//		for (unsigned int j = instance.A_csr_row_ptr[i];
	//				j < instance.A_csr_row_ptr[i + 1]; j++) {
	//			instance.A_csr_values[j] = instance.A_csr_values[j] * instance.b[i];
	//		}
	//		instance.b[i] = 1;
	//	}

	instance.lambda = ctx.lambda;

	std::vector<double> wBuffer(instance.m);
	std::vector<double> deltaW(instance.m);
	std::vector<double> deltaAlpha(instance.n);
	std::vector<double> w(instance.m);

	instance.x.resize(instance.n);
	cblas_set_to_zero(instance.x);
	cblas_set_to_zero(w);
	cblas_set_to_zero(deltaW);

	// compute local w
	vall_reduce(world, deltaW, w);

//	cout << " Local n " << instance.n << endl;

	vall_reduce(world, &instance.n, &instance.total_n, 1);

	instance.oneOverLambdaN = 1 / (0.0 + instance.total_n * instance.lambda);

	double gamma;
	if (distributedSettings.APPROX) {
		gamma = 1;
		instance.penalty = world.size() + 0.0;
	} else {
		gamma = 1 / (world.size() + 0.0);
		instance.penalty = 1;
	}

	LossFunction<unsigned int, double> * lf;

	instance.experimentName = ctx.experimentName;

	int loss = distributedSettings.lossFunction;
	//			+ distributedSettings.APPROX * 100;

	switch (loss) {
	case 0:
		lf = new LogisticLossCD<unsigned int, double>();
		break;
	case 1:
		lf = new HingeLossCD<unsigned int, double>();
		break;

	case 2:
		lf = new SquareHingeLossCD<unsigned int, double>();
		break;
	case 3:
		lf = new QuadraticLossCD<unsigned int, double>();
		break;


	default:
		break;
	}

	lf->init(instance);


	std::stringstream ss;
	int localsolver = distributedSettings.LocalMethods;
	// ss << ctx.matrixAFile << "_" << instance.lambda << "_"
	// 		<< distributedSettings.lossFunction << "_"
	// 		<< distributedSettings.iters_communicate_count << "_"
	// 		<< distributedSettings.iterationsPerThread << "_"
	// 		<< instance.experimentName << "_" << distributedSettings.APPROX
	// 		<< "_" << instance.theta << "_.log";

	ss << ctx.matrixAFile << "_"
	   << distributedSettings.lossFunction << "_"
	   << localsolver << "_"
	   << distributedSettings.iters_communicate_count << "_"
	   << distributedSettings.iterationsPerThread << "_"
	   << instance.lambda << "_"
	   << distributedSettings.APPROX
	   << ".log";
	std::ofstream logFile;
	if (ctx.settings.verbose) {
		logFile.open(ss.str().c_str());
	}

	distributedSettings.iters_bulkIterations_count = 1;

	// distributedSettings.iters_communicate_count =
	// 		distributedSettings.iters_communicate_count
	// 		/ distributedSettings.iters_bulkIterations_count;

//	cout << "BULK "<<distributedSettings.iters_bulkIterations_count<<" "<< distributedSettings.iters_communicate_count<<endl;


//	cout<< instance.A_csr_row_ptr[instance.A_csr_row_ptr.size()-1] <<endl;


	switch (localsolver) {
	case 0:

		lf->subproblem_solver_SDCA(instance, deltaAlpha, w, wBuffer, deltaW,
		                           distributedSettings, world, gamma, ctx, logFile);
		break;
	case 1:

		lf->subproblem_solver_accelerated_SDCA(instance, deltaAlpha, w, wBuffer, deltaW,
		                                       distributedSettings, world, gamma, ctx, logFile);
		break;
	case 2:

		lf->subproblem_solver_steepestdescent(instance, deltaAlpha, w, wBuffer, deltaW,
		                                      distributedSettings, world, gamma, ctx, logFile);
		break;

	case 3:
		lf->subproblem_solver_LBFGS(instance, deltaAlpha, w, wBuffer, deltaW,
		                            distributedSettings, world, gamma, ctx, logFile);
		break;

	case 4:
		lf->subproblem_solver_CG(instance, deltaAlpha, w, wBuffer, deltaW,
		                         distributedSettings, world, gamma, ctx, logFile);
		break;

	case 5:
		lf->subproblem_solver_BB(instance, deltaAlpha, w, wBuffer, deltaW,
		                         distributedSettings, world, gamma, ctx, logFile);
		break;
	case 6:
		lf->subproblem_solver_FISTA(instance, deltaAlpha, w, wBuffer, deltaW,
		                            distributedSettings, world, gamma, ctx, logFile);
		break;
	case 7:
		lf->subproblem_solver_SDCA_without_duality(instance, deltaAlpha, w, wBuffer, deltaW,
		        distributedSettings, world, gamma, ctx, logFile);
	case 8:
		lf->Acce_subproblem_solver_SDCA(instance, deltaAlpha, w, wBuffer, deltaW,
		                                distributedSettings, world, gamma, ctx, logFile);
		break;

	default:
		break;
	}
	// for(int i = 0; i<instance.n; i++){
	// 	cout<<i<<"  "<<instance.x[i]<<endl;
	// }
	// std::vector<double> trainLabel(instance.n);
	// std::vector<unsigned int> trainError(2);
	// std::vector<unsigned int> totalTrainError(2);
	// for (unsigned int idx = 0; idx < instance.n; idx++) {
	// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
	// 		trainLabel[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
	// 	}
	// 	if (trainLabel[idx] * instance.b[idx] <= 0 )
	// 		trainError[0] += 1;
	// }
	// vall_reduce(world, trainError, totalTrainError);
	// cout << 1.0*totalTrainError[0] /instance.total_n << endl;

	//-------------------------------------------------------

	if (ctx.settings.verbose) {
		logFile.close();
	}

	MPI::Finalize();

	return 0;
}
