
DISTRIBUTED_INCLUDE=-I.   -I /data/gsl/include     -I   /home/chm514/gsl/include 
DISTRIBUTED_LIB_PATH=   -L /data/gsl/lib   -L /home/chm514/gsl/lib 


CC= g++  #change to use a different compiler
 
CFLAGS =	-Wreorder  -Wall -fmessage-length=0  -O3 

LIBS =  -lgsl -lgslcblas


#============================================================================================
# You should not modify the lines below




cocoa:
	g++ $(CXXFLAGS) -O3    -fopenmp   -lm \
	-I $(MATLABROOT)/extern/include   $(DISTRIBUTED_INCLUDE)  \
	$(DISTRIBUTED_LIB_PATH) $(DISTRIBUTED_LINK)     -Wl,-rpath=/home/chm514/boost/lib \
	$(EXPFOLDER)cocoa.cpp $(LIBS)   -DMKL -o $(BUILD_FOLDER)cocoa    
	./$(BUILD_FOLDER)cocoa  -A data/covertype/covtype.libsvm.binary   -K 1

	


KMP:
	reset
	export KMP_AFFINITY=verbose,granularity=fine,scatter	

#---------------------  Truss Topology Design, see  http://code.google.com/p/ac-dc/wiki/TTD
ttd_build_generator: KMP
	$(CC) $(CFLAGS) $(INCLUDE)  $(FRONTENDS)ttd/TTDProblemGenerator.cpp -c  -o $(OBJFOLDER)TTDProblemGenerator.o 
	$(CC) $(LFLAGS) $(OBJFOLDER)TTDProblemGenerator.o  $(LIBS) -o $(BUILD_FOLDER)TTDProblemGenerator

ttd: ttd_build_generator
	./$(BUILD_FOLDER)TTDProblemGenerator -x 5 -y 5 -z 4 -r  ../data/ttd/3d_problem  -e 1
	./$(BUILD_FOLDER)TTDProblemGenerator -x 20 -y 10  -r  ../data/ttd/2d_problem  -e 1

#---------------------  Partially Separable Solver, see http://code.google.com/p/ac-dc/wiki/PartiallySeparableProblems
buildMulticoreConsoleSolver:
	g++ $(CFLAGS) -fopenmp $(FRONTENDS)solvers/MultiCoreSolver.cpp $(LIBS) -o $(BUILD_FOLDER)MultiCoreSolver

testMulticoreConsoleSolver: buildMulticoreConsoleSolver
	./$(BUILD_FOLDER)MultiCoreSolver -A ../data/svm/rcv1_train.binary -T ../data/svm/rcv1_train.binary -l 0.1 -t 2 -f 3 -i 1 -I 1000000
	
test:
	./$(BUILD_FOLDER)MultiCoreSolver -A ../data/unittests/matrixA -b ../data/unittests/vectorB.txt -l 0.1 -t 2 -f 0 -i 1 -I 1000	
	./$(BUILD_FOLDER)MultiCoreSolver -A ../data/unittests/matrixA -b ../data/unittests/vectorB.txt -l 0.1 -t 2 -f 1 -i 1 -I 1000
	./$(BUILD_FOLDER)MultiCoreSolver -A data/a1a   -l 0.1 -t 2 -f 2 -i 1 -I 1000	

tt:
	g++ $(CFLAGS) -I/home/w03/mtakac/gsl/include -fopenmp $(EXPFOLDER)svm_max_eigenvalue.cpp -L/home/w03/mtakac/gsl/lib $(LIBS) -o $(BUILD_FOLDER)maxSVMEIGENVALUE



#---------------------  Matrix Completion Solver, see http://code.google.com/p/ac-dc/wiki/MatrixCompletion
matrixCompletion:
	g++ $(CXXFLAGS) -fopenmp $(MATRIXCOMPLETITIONFOLDER)matrixCompletitionExperiment.cpp $(LIBS) -o $(BUILD_FOLDER)matrixCompletitionExperiment
	./$(BUILD_FOLDER)matrixCompletitionExperiment 

mcNetflix:
	g++ $(CXXFLAGS) -fopenmp $(FRONTENDS)matrixCompletion/mc_netflix.cpp $(LIBS) -o $(BUILD_FOLDER)mc_netflix
	./$(BUILD_FOLDER)mc_netflix  -A /work/software/yelp/yelp_academic_dataset_review.dat -T /work/software/yelp/yelp_academic_dataset_review.dat -t 0  -I 10000 -d 0 -h 4



#  -llapacke
MKLROOT=/opt/intel/mkl
cdnOLD:
	/opt/intel/bin/icc $(CXXFLAGS) -O3     -openmp -I$(MKLROOT)/include \
	 -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm \
	   $(EXPFOLDER)cdn.cpp $(LIBS)   -DMKL -o $(BUILD_FOLDER)cdn
	./$(BUILD_FOLDER)cdn

cdn:
	g++ $(CXXFLAGS) -O3    -fopenmp -m64 -I$(MKLROOT)/include \
	 -Wl,--start-group \
	 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
	 $(MKLROOT)/lib/intel64/libmkl_core.a \
	 $(MKLROOT)/lib/intel64/libmkl_blacs_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_sgimpt_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_sgimpt_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blas95_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blas95_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_cdft_core.a\
	 $(MKLROOT)/lib/intel64/libmkl_core.a\
	 $(MKLROOT)/lib/intel64/libmkl_gf_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_gf_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a\
	 $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_intel_thread.a\
	 $(MKLROOT)/lib/intel64/libmkl_lapack95_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_lapack95_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_scalapack_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm \
	 $(EXPFOLDER)cdn.cpp $(LIBS)   -DMKL -o $(BUILD_FOLDER)cdn  -llapacke 
	./$(BUILD_FOLDER)cdn  -A data/a1a  -R 0
	
	
	

mSDCA:
	g++ $(CXXFLAGS) -O3    -fopenmp -m64 -I$(MKLROOT)/include \
	 -Wl,--start-group \
	 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
	 $(MKLROOT)/lib/intel64/libmkl_core.a \
	 $(MKLROOT)/lib/intel64/libmkl_blacs_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_sgimpt_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blacs_sgimpt_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blas95_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_blas95_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_cdft_core.a\
	 $(MKLROOT)/lib/intel64/libmkl_core.a\
	 $(MKLROOT)/lib/intel64/libmkl_gf_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_gf_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a\
	 $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_intel_thread.a\
	 $(MKLROOT)/lib/intel64/libmkl_lapack95_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_lapack95_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_scalapack_ilp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a\
	 $(MKLROOT)/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm \
	 $(EXPFOLDER)mSDCA.cpp $(LIBS)   -DMKL -o $(BUILD_FOLDER)mSDCA  -llapacke 
	./$(BUILD_FOLDER)mSDCA  -A data/a1a  -R 0	 -T data/a1a  
	
	
asfddsa:	
	 -Wl,--start-group \
	 $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a \
	 $(MKLROOT)/lib/intel64/libmkl_core.a  \
	 $(MKLROOT)/lib/intel64/libmkl_sequential.a  \
	 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
	 $(MKLROOT)/lib/intel64/libmkl_lapack95_ilp64.a  \
	  $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm \
	   $(EXPFOLDER)cdn.cpp $(LIBS)   -DMKL -o $(BUILD_FOLDER)cdn  -llapacke
	./$(BUILD_FOLDER)cdn

 


#---------------------  Experiment from paper http://www.optimization-online.org/DB_HTML/2012/11/3688.html

large_scale_experiment:
	g++ $(CXXFLAGS) -fopenmp $(EXPFOLDER)large_scale_experiment.cpp $(LIBS) -o $(BUILD_FOLDER)large_scale_experiment
	./$(BUILD_FOLDER)large_scale_experiment

#---------------------  SDCA Experiment
large_scale_sdca_experiment:
	g++ $(CXXFLAGS) -fopenmp $(EXPFOLDER)large_scale_sdca_experiment.cpp $(LIBS) -o $(BUILD_FOLDER)large_scale_sdca_experiment
	./$(BUILD_FOLDER)large_scale_sdca_experiment

sigma:
	g++ $(CXXFLAGS) -fopenmp $(EXPFOLDER)svm_max_eigenvalue.cpp $(LIBS) -o $(BUILD_FOLDER)svm_max_eigenvalue
	./$(BUILD_FOLDER)svm_max_eigenvalue -f 2 -A ./data/a1a

  
#---------------------  SDCA Experiment
minibatch_sdca_experiment:
	g++ $(CXXFLAGS) -fopenmp $(EXPFOLDER)minibatch_sdca_experiment.cpp $(LIBS) -o $(BUILD_FOLDER)minibatch_sdca_experiment
	./$(BUILD_FOLDER)minibatch_sdca_experiment

#---------------------  Distributed generator
nesterov_distributed_generator:
	g++ $(CXXFLAGS) -fopenmp $(EXPFOLDER)distributedNesterovGenerator.cpp $(LIBS) -o $(BUILD_FOLDER)distributedNesterovGenerator
	./$(BUILD_FOLDER)distributedNesterovGenerator

	
	