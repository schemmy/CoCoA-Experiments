

DISTRIBUTED_INCLUDE=-I.  -I /usr/local/include/ -I /usr/local/Cellar/boost/1.57.0/include/  -I /usr/local/Cellar/gcc/5.2.0/lib/gcc/5/gcc/x86_64-apple-darwin14.5.0/5.2.0/include 

DISTRIBUTED_LIB_PATH=  -L /usr/local/lib/ -L /usr/local/Cellar/boost/1.57.0/lib  -L /usr/local/Cellar/gcc/5.2.0/lib/gcc/5/ 




DISTRIBUTED_COMPILER_OPTIONS=-fopenmp -O3 -fpermissive -DHAVE_CONFIG_H -DAUTOTOOLS_BUILD   -DMPICH_IGNORE_CXX_SEEK 
#DISTRIBUTED_LINK= -lboost_mpi -lzoltan -lboost_timer -lboost_serialization -lboost_thread -lgsl -lgslcblas -lm -stdlib=libstdc++ -lstdc++
DISTRIBUTED_LINK= -lgslcblas  -lboost_system-mt -lboost_timer-mt -lboost_chrono-mt -lboost_mpi-mt -lboost_serialization-mt -fopenmp  -lgsl -lm -lboost_thread-mt  


	
#MATLABROOT=/opt/programs/MATLAB/R2011a/
MATLABROOT=/usr/local/matlab/latest/

#===========================

#MKL_LIBS=    $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm
	

# compiler which should be used
MPICC = mpicc
MPICPP = mpicxx
#MPICPP = CC

#============================================================================================
# You should not modify the lines below


# Cluster Consolve Solver            
c:
	$(MPICPP) -O3 $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE)   \
	-o $(BUILD_FOLDER)Cocoa \
	$(FRONTENDS)../cocoa/cocoa.cpp \
	-I $(MATLABROOT)/extern/include     \
	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)   -I ~/Downloads/dlib-18.18/
	#mpirun -np 4  $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.001 -C 50 -I 10 -f 3 -a 1  -p 0.001 -M 3
	#mpirun -np 4  $(BUILD_FOLDER)Cocoa -A data/rcv1_train.binary.4/rcv1_train.binary -l 0.001 -C 20 -I 1000 -f 3 -a 1  -p 0.001 -M 5
	mpirun -np 4  $(BUILD_FOLDER)Cocoa -A data/rcv.4/rcv -l 0.0001 -C 20 -I 10 -f 3 -a 1  -p 0.001 -M 2

disco:
	$(MPICPP) -O3 $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) \
	-o $(BUILD_FOLDER)DISCO \
	$(FRONTENDS)../Disco/disco.cpp \
	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)   
	mpirun -np 4 $(BUILD_FOLDER)DISCO -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 8 $(BUILD_FOLDER)DISCO -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 16 $(BUILD_FOLDER)DISCO -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 32 $(BUILD_FOLDER)DISCO -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 64 $(BUILD_FOLDER)DISCO -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
 
ocsid:
	$(MPICPP) -O3 $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) \
	-o $(BUILD_FOLDER)OCSID \
	$(FRONTENDS)../Disco/ocsid.cpp \
	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)   
	mpirun -np 4 $(BUILD_FOLDER)OCSID -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 8 $(BUILD_FOLDER)OCSID -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 16 $(BUILD_FOLDER)OCSID -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 32 $(BUILD_FOLDER)OCSID -A data/eps.4/eps -l 0.1  -a 1  -p 0.001
	mpirun -np 64 $(BUILD_FOLDER)OCSID -A data/eps.4/eps -l 0.1  -a 1  -p 0.001

cocoa:
	$(MPICPP) -O3 $(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE) -DMATLAB  \
	-o $(BUILD_FOLDER)Cocoa \
	$(FRONTENDS)../Disco/cocoa.cpp \
	-I $(MATLABROOT)/extern/include     \
	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)   
	mpirun -np 4  $(BUILD_FOLDER)Cocoa -A data/eps.4/eps -l 0.1 -C 100 -I 1000 -f 0 -a 1  -p 0.001 -M 0
	mpirun -np 8  $(BUILD_FOLDER)Cocoa -A data/eps.4/eps -l 0.1 -C 100 -I 1000 -f 0 -a 1  -p 0.001 -M 0
	mpirun -np 16  $(BUILD_FOLDER)Cocoa -A data/eps.4/eps -l 0.1 -C 100 -I 1000 -f 0 -a 1  -p 0.001 -M 0
	mpirun -np 32  $(BUILD_FOLDER)Cocoa -A data/eps.4/eps -l 0.1 -C 100 -I 1000 -f 0 -a 1  -p 0.001 -M 0
	

	
asfdsafsafdfsa:	
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/covertype.4/covertype -l 0.1 -C 10 -I 1 -f 4	
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/rcv1_train.binary.4/rcv1_train.binary -l 0.1 -C 5 -I 1 -f 5	
	
	
asfddsafsa:	
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 10 -I 1 -f 4	-E sd
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 10 -I 1 -f 4	-E csd
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 10 -I 1 -f 4	-E cg

# 

asfsfa:
	mpirun -np 1 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 100 -I 500 -f 4

	
runall:	
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 100 -I 500 -f 0
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 100 -I 500 -f 1
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 100 -I 500 -f 2
	mpirun -np 4 $(BUILD_FOLDER)Cocoa -A data/a1a.4/a1a -l 0.1 -C 100 -I 500 -f 3
	