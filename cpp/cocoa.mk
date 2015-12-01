

DISTRIBUTED_INCLUDE= -I /usr/local/include/ -I /usr/local/boost-1.56.0/include 
DISTRIBUTED_LIB_PATH=  -L /usr/local/lib/ -L /usr/local/boost-1.56.0/lib/ -L /usr/lib/x86_64-linux-gnu/ -L /home/mcx/Home/R/bbinutils_sb/lib/ 


DISTRIBUTED_COMPILER_OPTIONS= -fpermissive -DHAVE_CONFIG_H -DAUTOTOOLS_BUILD   -DMPICH_IGNORE_CXX_SEEK 
#DISTRIBUTED_LINK= -lboost_mpi -lzoltan -lboost_timer -lboost_serialization -lboost_thread -lgsl -lgslcblas -lm -stdlib=libstdc++ -lstdc++
DISTRIBUTED_LINK= -lgslcblas  -lboost_system -lboost_timer -lboost_chrono -lboost_mpi -lboost_serialization -fopenmp  -lgsl -lboost_thread  

mpiP_root=/home/mcx/R/mpiP_sb/


#===========================

#MKL_LIBS=    $(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group -lpthread -lm
	

# compiler which should be used
MPICC = mpicc
MPICPP = mpic++
#MPICPP = CC

#============================================================================================
# You should not modify the lines below


# Cluster Consolve Solver            
c:
	$(MPICPP) -L${mpiP_root}/lib  \
	$(FRONTENDS)../cocoa/cocoa.cpp \
	$(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE)   \
	-o $(BUILD_FOLDER)Cocoa \
	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)   
	#mpirun -n 4 ./$(BUILD_FOLDER)Cocoa  -A data/rcv.4/rcv -l 0.0001 -C 20 -I 10 -f 3 -a 0  -p 0.001 -M 2
	mpirun -n 4 ./$(BUILD_FOLDER)Cocoa  -A data/rcv1_train.binary.4/rcv1_train.binary -l 0.0001 -C 20 -I 10000 -f 2 -a 0  -p 0.001 -M 0

cocoa-mpip:
	export MPIP="-y -p"
	$(MPICPP) -g 	-L${mpiP_root}/lib  \
	$(FRONTENDS)../cocoa/cocoa.cpp \
	-lmpiP -lm  -lbfd -liberty  -lunwind \
	$(DISTRIBUTED_COMPILER_OPTIONS) $(DISTRIBUTED_INCLUDE)   \
	-o $(BUILD_FOLDER)Cocoa \
	$(DISTRIBUTED_LIB_PATH)  $(DISTRIBUTED_LINK)   
	mpirun -n 4 ./$(BUILD_FOLDER)Cocoa  -A data/a1a.4/a1a -l 0.1 -C 500 -I 10 -f 3 -a 1  -p 0.001 -M 0
	

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
	
