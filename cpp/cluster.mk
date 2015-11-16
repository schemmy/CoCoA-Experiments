

DISTRIBUTED_INCLUDE=-I.    
DISTRIBUTED_LIB_PATH=   



DISTRIBUTED_COMPILER_OPTIONS=-fopenmp -O3 -fpermissive -DHAVE_CONFIG_H -DAUTOTOOLS_BUILD   -DMPICH_IGNORE_CXX_SEEK 
#DISTRIBUTED_LINK= -lboost_mpi -lzoltan -lboost_timer -lboost_serialization -lboost_thread -lgsl -lgslcblas -lm 
DISTRIBUTED_LINK= -lgslcblas  -lboost_system -lboost_timer -lboost_chrono -lrt  -lboost_mpi -lboost_serialization -fopenmp  -lgsl -lm -lboost_thread

#===========================

#MKL_LIBS=    $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm


# compiler which should be used
MPICC = mpicc
MPICPP = mpicxx
#MPICPP = CC

#============================================================================================
# You should not modify the lines below

# Cluster Consolve Solver
buildClusterSolver:
	$(MPICPP)  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) \
	-o $(BUILD_FOLDER)ClusterSolver \
 	$(FRONTENDS)solvers/ClusterSolver.cpp \
 	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)  


dt: buildClusterSolver
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A /tmp/nesterov   -b /tmp/nesterov_b \
	-l 0.1 -t 3 -f 0 -c 1 -i 1 -B 30 -I 10 -C 20 -T 10 -a 1 -S 0 

gyorgy: buildClusterSolver
#	mpirun -c 1 release/ClusterSolver  -A data/test.1/test.svm -b data/test.1/test.svm_b -l 0.1 -f 0 -c 1 -i 1 -I 2 -C 1000  -t 1  -B 50 -S 0 -a 0 -F 0
	mpirun -c 1 release/ClusterSolver  -A ../../news/news20.binary -b ../../news/news20.binary_b -l 12.3306 -f 0 -c 1 -i 1 -I 2 -C 1000  -t 1  -B 5 -S 0 -a 0 -F 0



distributed_APPROX: buildClusterSolver
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A data/a1a.4/a1a  \
	-l 0.1 -t 3 -f 2 -c 1 -i 1 -I 1000 -T 2 -a 1 -S 0
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A data/a1a.4/a1a  \
	-l 0.1 -t 3 -f 2 -c 1 -i 1 -I 1000 -T 2

clusterTest: buildClusterSolver
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A ../data/unittests/distributed \
	-b ../data/unittests/distributed \
	-l 0.1   -f 0 -c 1 -i 1 -I 2 -C 10  -t 8  -B 15 \
	  -S 0 -a 0  
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A ../data/unittests/distributed \
	-b ../data/unittests/distributed \
	-l 0.1  -f 0 -c 1 -i 1 -I 2  -C 10 -t 8 -B 15  \
	 -a 1   -S 0	
	
 	
 	
 	
testClusterSolver: buildClusterSolver
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A ../data/unittests/distributed \
	-b ../data/unittests/distributed \
	-l 0.1 -t 3 -f 0 -c 1 -i 1 -I 1000
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A ../data/unittests/distributed \
	-b ../data/unittests/distributed \
	-l 0.1 -t 3 -f 1 -c 1 -i 1 -I 1000	
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A data/a1a.4/a1a  \
	-l 0.1 -t 3 -f 2 -c 1 -i 1 -I 1000 -T 2


clusterNesterovTest: buildClusterSolver
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A /tmp/test_problem \
	-b  /tmp/test_problem_b \
	-l 0.1   -f 0 -c 1 -i 1 -I 2 -C 20  -t 8  -B 50 \
	  -S 0 -a 0   -F 0


testResidualShift: buildClusterSolver
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A ../data/unittests/distributed \
	-b ../data/unittests/distributed \
	-l 0.1 -t 3 -f 0 -c 1 -i 1 -T 20 -C 20 -S 0 -E testName
	mpirun -c 4 $(BUILD_FOLDER)ClusterSolver \
	 -A ../data/unittests/distributed \
	-b ../data/unittests/distributed \
	-l 0.1 -t 3 -f 0 -c 1 -i 1 -T 20 -C 20 -S 1 -E testName
 	
	 
#============================================================================================

distributed_unit_test:
	reset
	mpicxx  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -o $(BUILDFOLDER)distributed_unit_test $(SRCFOLDER)test/distributed_unit_test.cpp $(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)  
	mpirun -c 6 $(BUILDFOLDER)distributed_unit_test
	
	
distributed_huge_experimentOLD:	
	reset
	mpicxx  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -o $(BUILDFOLDER)distributed_huge_experiment $(EXPFOLDER)distributed_huge_experiment.cpp $(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)  
	mpirun -c 4 $(BUILDFOLDER)distributed_huge_experiment

distributed_huge_experiment_strategies:	
	reset
	mpicxx  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -o $(BUILDFOLDER)distributed_huge_experiment $(EXPFOLDER)distributed_huge_strategies.cpp $(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)  
	mpirun -c 4 $(BUILDFOLDER)distributed_huge_experiment



distributed_huge_experiment:	
	reset
	/opt/scorep/bin/scorep --mpi --openmp  --nocompiler  --verbose=1 mpicxx -g  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -o $(BUILDFOLDER)distributed_huge_experiment $(EXPFOLDER)distributed_huge_experiment.cpp $(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)
	export SCOREP_ENABLE_TRACING=1  
	#mpirun -c 4 $(BUILDFOLDER)distributed_huge_experiment

distributed_svm_solver:
	reset
	mpicxx  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -o $(BUILDFOLDER)distributed_svm_solver $(EXPFOLDER)distributed_svm_solver.cpp $(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)  
	mpirun -c 4 $(BUILDFOLDER)distributed_svm_solver
	
distributed_dual_svm_solver:
	reset
	mpicxx  $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -o $(BUILDFOLDER)distributed_dual_svm_solver $(EXPFOLDER)distributed_dual_svm_solver.cpp $(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)  
	mpirun -c 4 $(BUILDFOLDER)distributed_dual_svm_solver
	







distributed_huge_problem_generator:	
	reset
	g++  -o $(BUILDFOLDER)distributed_problem_generator $(EXPFOLDER)distributed_problem_generator.cpp  -fopenmp -lgsl -lgslcblas -lm   
	./$(BUILDFOLDER)distributed_problem_generator
	
distributed_svm_parser:	
	reset
	g++  -o $(BUILDFOLDER)distributed_svm_parser $(EXPFOLDER)distributed_svm_parser.cpp  -fopenmp -lgsl -lgslcblas -lm   
	./$(BUILDFOLDER)distributed_svm_parser
	
