/*

kernel.cu: ginSODA's main file.

ginSODA is black-box integrator of ODEs that exploits the remarkable memory 
bandwidth and computational capability of GPUs. ginSODA allows to efficiently 
execute in parallel large numbers of integrations, which are usually required 
to analyze the emergent behaviour of a complex system (e.g., i biochemical 
reaction system) under different conditions. ginSODA works by automatically 
deriving the Jacobian matrix associated to the system of coupled ODEs, and then 
exploiting the numerical integration algorithm for stiff systems LSODA.

See file COPYING for copyright and licensing information.

Bibliography:

- Petzold L.: Automatic selection of methods for solving stiff and nonstiff 
systems of ordinary differential equations. SIAM Journal of Scientific and
Statistical Computing, 4(1):136-148, 1983

*/

#define NOMINMAX

#include <algorithm>
#include <stdio.h>
#include <cmath>		// std::min, std::max
#include <iostream>
#include <fstream>
#include <vector>
#include "lsoda.h"
#include "ginSODA.h"
#include "hparser.hpp"
#include <cuda.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <string>
#include <sstream>


/* Int2String conversion utility */
std::string convertInt(int const number)
{
   std::stringstream ss;
   ss << number;
   return ss.str();
}

int main(int argc, const char** argv)  
{

	bool just_fitness = false;
	bool ACTIVATE_SHARED_MEMORY = false;
	bool ACTIVATE_CONSTANT_MEMORY = false;
	int DUMP_DEBUG = 0;
	
	bool print_fitness = false;
	bool print_dynamics = false;

	ArgumentParser parser;
	parser.addArgument("-i", "--input_model", 1, false);
	parser.addArgument("-o", "--output_folder", 1, false);
	parser.addArgument("-p", "--output_prefix", 1, false);
	parser.addArgument("-b", "--blocks", 1, true);
	parser.addArgument("-g", "--gpu", 1, true);
	parser.addArgument("-d", "--debug", 1, true);
	parser.addArgument("-t", "--print_dynamics", 1, true);
	parser.addArgument("-h", "--shared_memory", 1, true);
	parser.parse(argc, argv);

	DUMP_DEBUG = atoi(parser.retrieve<std::string>("d").c_str());
	if (DUMP_DEBUG) std::cout << " * Debug mode on" << std::endl;

	std::string input_folder = parser.retrieve<std::string>("i");
	if (DUMP_DEBUG) std::cout << " * Opening directory " << input_folder << std::endl;

	ACTIVATE_SHARED_MEMORY = atoi(parser.retrieve<std::string>("h").c_str()) > 0;
	if (ACTIVATE_SHARED_MEMORY && DUMP_DEBUG) std::cout << " * Shared memory: enabled" << std::endl;	

	std::string output_folder = parser.retrieve<std::string>("o");
	std::string output_prefix = parser.retrieve<std::string>("p");
	if (DUMP_DEBUG) std::cout << " * Results will be saved in directory " << output_folder << " using the prefix " << output_prefix << std::endl;

	unsigned int GPU = atoi(parser.retrieve<std::string>("g").c_str());
	if (DUMP_DEBUG) std::cout << " * Using GPU #" << GPU << std::endl;

	unsigned int blocks = atoi(parser.retrieve<std::string>("b").c_str());
	if (blocks==0) {
		if (DUMP_DEBUG) std::cout << " * Requested " << blocks << "blocks" << std::endl;
	} else {
		if (DUMP_DEBUG) std::cout << " * The number of CUDA blocks was not specified. ginSODA will attempt to automatically calculate the optimal distribution of resources." << std::endl;
	}
	
	print_dynamics = atoi(parser.retrieve<std::string>("t").c_str()) >0 ;
	if (print_dynamics) {
		if (DUMP_DEBUG) std::cout << " * Redirecting output dynamics to stdout" << std::endl;
	}
		
	double*			host_X;
	unsigned int*	device_species_to_sample;
	conc_t*			device_X;
		
	// Load model data
	InputLoader IL = InputLoader();
	IL.output_folder = output_folder;
	IL.output_prefix = output_prefix;
	IL.load_initial_conditions(input_folder+"/MX_0");
	IL.load_parameterizations(input_folder+"/c_matrix");
	IL.load_time_instants(input_folder+"/t_vector");
	IL.load_species_to_sample(input_folder+"/cs_vector");
	IL.load_tolerances(input_folder+"/atol_vector", input_folder+"/rtol");
	if (DUMP_DEBUG) std::cout << " * Detected " << IL.actual_threads << " actual threads.\n";

	IL.blocks=blocks;
	IL.calculate_tpb();

	unsigned int TOTAL_CUDA_THREADS = IL.tpb*IL.blocks;

	IL.check_memory_requirements(TOTAL_CUDA_THREADS, IL.time_instants.size(), GPU, DUMP_DEBUG);
	
	// This function sets all the constant values in the GPU, storing them in constant memory
	SetConstants(IL.variables, IL.parameters, IL.time_instants.size(), IL.species_to_sample.size(), IL.actual_threads, DUMP_DEBUG);
	
	// Allocates the memory space for samples and the target time-series 	
	cudaMalloc((void**)&device_X, sizeof( conc_t ) * IL.species_to_sample.size() * IL.time_instants.size() * TOTAL_CUDA_THREADS );	
	CudaCheckErrorRelease();
	host_X = (conc_t*) malloc ( sizeof( conc_t ) * IL.species_to_sample.size() * IL.time_instants.size() * TOTAL_CUDA_THREADS );
	if (host_X==NULL) {
		perror("ERROR allocating states\n");
		exit(-14);
	}

	// All samples are reset to zero
	memset( host_X, 0, sizeof( conc_t ) * IL.species_to_sample.size() * IL.time_instants.size() * TOTAL_CUDA_THREADS );
	
	// The first sample is the initial condition
	for (unsigned int ss=0; ss<IL.species_to_sample.size(); ss++) {
		for (unsigned int t=0; t<TOTAL_CUDA_THREADS; t++) {
			if ( t>=IL.actual_threads) break;
			host_X[IL.species_to_sample.size()*t + ss] = IL.initial_amounts[t][IL.species_to_sample[ss]];
		}
	}	
	cudaMemcpy(device_X, host_X, sizeof(conc_t) * IL.species_to_sample.size() * IL.time_instants.size() * TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice );
	CudaCheckError();	

	// Allocate and store the species to be sampled 
	cudaMalloc((void**)&device_species_to_sample, sizeof( unsigned int ) * IL.species_to_sample.size() );
	cudaMemcpy(device_species_to_sample,&IL.species_to_sample[0],sizeof(unsigned int) * IL.species_to_sample.size(),cudaMemcpyHostToDevice);
	CudaCheckError();	
		
	///// DEBUG //////
	int* h_debug = (int*) malloc ( sizeof(int)*TOTAL_CUDA_THREADS );
	int* d_debug;
	cudaMalloc((void**)&d_debug,sizeof(int)*TOTAL_CUDA_THREADS );	
	CudaCheckError();  
	///// DEBUG //////
		
	/* Local variables */
	double* constants = (double*)malloc(sizeof(double)*TOTAL_CUDA_THREADS*IL.parameters);     
	double* y         = (double*)malloc(sizeof(double)*IL.variables*TOTAL_CUDA_THREADS);
     
	int *itol =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
	int *jt =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
    int *neq =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
	int *liw =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
	int *lrw =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);     
	int *iopt =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
    int *iout =			(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
    int *itask =		(int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
	 
	double *atol =		(double*)malloc(sizeof(double)*IL.variables*TOTAL_CUDA_THREADS);
    double *rtol =		(double*)malloc(sizeof(double)*TOTAL_CUDA_THREADS);
    double *tout =		(double*)malloc(sizeof(double)*TOTAL_CUDA_THREADS);
	double *t =			(double*)malloc(sizeof(double)*TOTAL_CUDA_THREADS);
     	 	 
	const int lrs = 22+9*IL.variables + (IL.variables*IL.variables);
	const int LRW = 22+IL.variables*std::max(16,(int)(IL.variables+9));	 
	const int LIW = 20+IL.variables;	  
	 
	double *rwork = (double*)malloc(sizeof(double)*LRW*TOTAL_CUDA_THREADS);
	memset(rwork, 0, sizeof(double)*LRW*TOTAL_CUDA_THREADS); 
	int *iwork = (int*) malloc(sizeof(int)*LIW*TOTAL_CUDA_THREADS);     
	memset(iwork, 0, sizeof(int)*LIW*TOTAL_CUDA_THREADS); 

	int *istate = (int*)malloc(sizeof(int)*TOTAL_CUDA_THREADS);
	struct ginSODACommonBlock* common = (struct ginSODACommonBlock*)malloc(sizeof(struct ginSODACommonBlock)*TOTAL_CUDA_THREADS);
	struct ginSODACommonBlock *Hcommon = common;
	// printf("size of ginsoda common block: %d.\n", (int) sizeof ginSODACommonBlock);

	// Pointers to GPU's global memory areas of LSODA data structures 
	double* device_constants;
	double	*_Dt;
	double	*_Dy;	
	int	*_Djt;
	int	*_Dneq;
	int	*_Dliw;
	int	*_Dlrw;
    double	*_Datol;
    int	*_Ditol;
	int	*_Diopt;
    double	*_Drtol;
    int	*_Diout;
    double	*_Dtout;
    int	*_Ditask;
	int	*_Diwork;	
    double	*_Drwork;	
	int	*_Distate;
	struct ginSODACommonBlock *_Dcommon;	
	
	// Assignment of initial values to locals 
	for (unsigned int i=0; i<TOTAL_CUDA_THREADS; i++) {

		if (i>=IL.actual_threads) break;
		
		iwork[i*LIW+5] = IL.max_steps;

		// Parameters of the ODE systems
		for (unsigned r=0; r<IL.parameters; r++) 
			constants[i*IL.parameters+r] = IL.parameterizations[i][r];

		// Number of ODEs
		neq[i] = IL.variables;

		// Initial quantities 
		for (unsigned int s=0; s<IL.variables; s++)
			y[i*IL.variables+s] = IL.initial_amounts[i][s];

		/* Initial time */
		t[i] = (double)0.;
				
		/* Error tolerances */
		itol[i] = 2;
		rtol[i] = IL.rtol;						
		for (unsigned int s=0; s<IL.variables; s++) {
			atol[i*IL.variables+s] = IL.atol[s];			
		}		

		/* Standard LSODA execution */
		itask[i] = 1;
		istate[i] = 1;
		iopt[i] = 0;
		lrw[i] = LRW;
		liw[i] = LIW;
		jt[i] = 2;
	}
	ginSODA_Init(Hcommon, TOTAL_CUDA_THREADS);
	

	// Allocate LSODA data structures, replicated for each thread
	
	cudaMalloc((void**)&device_constants, sizeof(double)*TOTAL_CUDA_THREADS*IL.parameters);	
	cudaMemcpy(device_constants, constants, sizeof(double)*TOTAL_CUDA_THREADS*IL.parameters, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dt, sizeof(double)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Dt, t, sizeof(double)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dy, sizeof(double)*IL.variables*TOTAL_CUDA_THREADS);				
	cudaMemcpy(_Dy, y, sizeof(double)*IL.variables*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Djt, sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Djt, jt, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dneq, sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Dneq, neq, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dliw, sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Dliw, liw, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dlrw,sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Dlrw,lrw,sizeof(int)*TOTAL_CUDA_THREADS,cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Datol, sizeof(double)*IL.variables*TOTAL_CUDA_THREADS);				
	cudaMemcpy(_Datol, atol, sizeof(double)*IL.variables*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Drtol, sizeof(double)*TOTAL_CUDA_THREADS);					
	cudaMemcpy(_Drtol, rtol, sizeof(double)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Ditol, sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Ditol, itol, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Diopt, sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Diopt, iopt, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();
	
	cudaMalloc((void**)&_Diout, sizeof(int)*TOTAL_CUDA_THREADS);								
	cudaMemcpy(_Diout, iout, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dtout, sizeof(double)*TOTAL_CUDA_THREADS);
	cudaMemcpy(_Dtout, tout, sizeof(double)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Ditask, sizeof(int)*TOTAL_CUDA_THREADS);							
	cudaMemcpy(_Ditask, itask, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Diwork, sizeof(int)*LIW*TOTAL_CUDA_THREADS);						
	cudaMemcpy(_Diwork, iwork, sizeof(int)*LIW*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Drwork, sizeof(double)*LRW*TOTAL_CUDA_THREADS);
	cudaMemcpy(_Drwork, rwork, sizeof(double)*LRW*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Distate, sizeof(int)*TOTAL_CUDA_THREADS);		
	cudaMemcpy(_Distate, istate, sizeof(int)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();

	cudaMalloc((void**)&_Dcommon, sizeof(struct ginSODACommonBlock)*TOTAL_CUDA_THREADS);
	cudaMemcpy(_Dcommon, Hcommon, sizeof(struct ginSODACommonBlock)*TOTAL_CUDA_THREADS, cudaMemcpyHostToDevice);
	CudaCheckError();
	
	
	unsigned int sh_memory_bytes;

	// Check for available shared memory: if the execution hierarchy (i.e., number of threads per block)
	// cannot be launched with the current configuration, then abort. 	
	if (ACTIVATE_SHARED_MEMORY) {
		sh_memory_bytes = sizeof(double)*IL.variables*IL.tpb + sizeof(double)*IL.tpb;
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, GPU);
		if (sh_memory_bytes > prop.sharedMemPerBlock ) {
			printf("ERROR: insufficient shared memory (%d).\n", sh_memory_bytes);
			exit(ERROR_INSUFF_SHARED_MEMORY);
		}
	} else {
		sh_memory_bytes = 0;
	}

	// SetODEarray( s2d );

	// Code for profiling
	cudaEvent_t start, stop;
	if (!just_fitness) {
		cudaEventCreate(&start);  
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}
	
	// LSODA documentation reads: "write a main program which calls subroutine lsoda once for each point at which answers are desired"
	// For this reason, I use a for cycle that goes through the set of sampling temporal instants.
	for (unsigned int ti=0; ti<IL.time_instants.size(); ti++) 
	{

		// Si potrebbe parallelizzare
		for (unsigned int i=0; i<IL.tpb*IL.blocks; i++) tout[i] = IL.time_instants[ti];		
		cudaMemcpy(_Dtout,tout,sizeof(double)*IL.tpb*IL.blocks,cudaMemcpyHostToDevice);

		// printf(" * Time step to %f.\n", tout[0]);
					
		dim3 BlocksPerGrid(IL.blocks,1,1);
		dim3 ThreadsPerBlock(IL.tpb,1,1);
		
		ginSODA<<<BlocksPerGrid,ThreadsPerBlock,sh_memory_bytes>>>
			( _Dneq, _Dy, _Dt, _Dtout, _Ditol, _Drtol, _Datol, _Ditask, _Distate, _Diopt, _Drwork, _Dlrw, _Diwork, _Dliw, 
			  _Djt, _Dcommon, d_debug, device_constants, device_X, ti, device_species_to_sample, ACTIVATE_SHARED_MEMORY, ACTIVATE_CONSTANT_MEMORY);
		CudaCheckError();

		/* Print debug information (if requested), for each thread */
		if (DUMP_DEBUG==2) {
			cudaMemcpy(istate,_Distate, sizeof(int)*IL.tpb*IL.blocks,cudaMemcpyDeviceToHost);
			printf(" * Dumping istates:\n");
			for (unsigned int th=0; th<IL.actual_threads; th++) {				
				printf("Thread %d: istate %d", th, istate[th]);
				switch(istate[th]) {
				case 1: printf(" (First step) "); break; 
				case 2: printf(" (OK!) "); break; 
				case -1: printf (" (excess of word done) "); break;
				case -2: printf (" (excess of accuracy requested) "); break;
				case -3: printf (" (illegal input detected) "); break;
				case -4: printf (" (repeated error test failures) "); break;
				case -5: printf (" (convergence failure) "); break;
				case -6: printf (" (error weight became zero) "); break;
				case -7: printf (" (work space insufficient to finish) "); break;
				default:
					printf(" (UNKNOWN LSODA ERROR) "); break;
				};
				printf("\n");
			}

			cudaMemcpy(iwork,_Diwork, sizeof(int)*LIW*IL.tpb*IL.blocks,cudaMemcpyDeviceToHost);

			
			for (unsigned int th=0; th<IL.actual_threads; th++) {
				printf("[Thr%d] steps so far: %d, ODE evals: %d, Jac evals: %d.\n", th, iwork[th*20+10], iwork[th*20+11], iwork[th*20+12]);
			}
			

			printf("\n");
		}

    }

	if ((!just_fitness) && (!print_dynamics)) {
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		float tempo;
		cudaEventElapsedTime( &tempo, start, stop );
		tempo /= 1000;
		printf("Running time: %f seconds\n", tempo);
		cudaEventDestroy(start); 
		cudaEventDestroy(stop); 
	}
	
	unsigned int DEV_CONST_SAMPLESLUN = (unsigned int)IL.species_to_sample.size();

	cudaThreadSynchronize();

	// Calculate the FT of the signals (i.e., time-series)
	/*
	if (dump_fft) {

		// calculate_fft(s2d, device_X, "prova_fft");
		exit(0);

	}
	*/

	/* If we are just calculating a fitness value, avoid creating and dumping to output files of simulations */
	/*
	if (just_fitness) {

		host_fitness = (double*) malloc ( sizeof(double) * IL.threads );

		cudaMalloc((void**)&device_fitness,sizeof(double)*IL.threads);	
		cudaMalloc((void**)&device_swarms, sizeof(char)  *IL.threads);	
		CudaCheckError();

		cudaMemcpy(device_swarms, s2d->thread2experiment, sizeof(char)*IL.threads, cudaMemcpyHostToDevice);
		CudaCheckError();
		
		dim3 BlocksPerGrid(s2d->blocks,1,1);
		dim3 ThreadsPerBlock(s2d->tpb,1,1);

		if (print_yukifitness)  {
			// calculateFitnessYuki<<<BlocksPerGrid,ThreadsPerBlock>>>( device_X, device_fitness);
			CudaCheckError();			
		} else if (print_nomanfitness) {
			calculateFitnessNoman<<<BlocksPerGrid,ThreadsPerBlock>>>( device_X, device_fitness, EA_ITERATION);
			CudaCheckError();			
		} 	else {
			calculateFitness<<<BlocksPerGrid,ThreadsPerBlock>>>( device_X, device_target, device_fitness, device_swarms );
			CudaCheckError();
		}

		// cudaMemcpy(device_swarms,s2d->thread2experiment,sizeof(char)*IL.threads, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_fitness,device_fitness,sizeof(double)*IL.threads, cudaMemcpyDeviceToHost);
		CudaCheckError();

		// if we are not just printing on video the fitness values, open the output file "pref_allfit"
		if ( !(print_fitness | print_yukifitness | print_nomanfitness) ) {

		#ifdef _WIN32
			std::string comando_crea_cartella("md ");
			comando_crea_cartella.append(s2d->DEFAULT_OUTPUT);		
			system(comando_crea_cartella.c_str());
		#else
			std::string comando_crea_cartella("mkdir ");
			comando_crea_cartella.append(s2d->DEFAULT_OUTPUT);
			system(comando_crea_cartella.c_str());
		#endif


			std::string outputfile(s2d->DEFAULT_OUTPUT);
			outputfile.append("/pref_allfit");
		
			std::ofstream dump2(outputfile.c_str());

			if (!dump2.is_open()) {
				printf("Path: %s.\n", outputfile.c_str());
				perror("Unable to save fitness file 'pref_allfit', aborting.");
				exit(-17);
			}

			// verify!!
			for (unsigned int sw=0; sw<s2d->blocks; sw++) {
				for (unsigned int part=0; part<s2d->tpb; part++) {
					dump2 << host_fitness[sw*s2d->tpb + part] << "\t";					
				}
				dump2 << "\n";
			}
			dump2.close();

		} else {

			// verify!!
			for (unsigned int sw=0; sw<s2d->blocks; sw++) {
				for (unsigned int part=0; part<s2d->tpb; part++) {
					std::cout << host_fitness[sw*s2d->tpb + part] << "\t";					
				}
				std::cout << "\n";
			}

		}

		free(host_fitness);
		cudaFree(device_fitness);	
		cudaFree(device_swarms);
		cudaFree(device_target);


	}  // end if just fitness

	*/

	if (!just_fitness) {

		// No fitness: let's save the output of simulations on the hard disk
		cudaMemcpy(host_X,device_X, sizeof(conc_t) * IL.species_to_sample.size() * IL.tpb*IL.blocks * IL.time_instants.size(), cudaMemcpyDeviceToHost);
		CudaCheckError();

		if (print_dynamics) {

			for ( unsigned int tid=0; tid<IL.actual_threads; tid++ ) {
				
				std::cout << std::setprecision(15);
		
				unsigned int larg = TOTAL_CUDA_THREADS;
				unsigned int DEV_CONST_SAMPLESPECIES = (unsigned int) IL.species_to_sample.size();		
	
				for (unsigned int campione=0; campione<IL.time_instants.size(); campione++) {
					std::cout << IL.time_instants[campione] << "\t";
					for (unsigned int s=0; s<IL.species_to_sample.size(); s++) {				
						std::cout << host_X[ ACCESS_SAMPLE ];
						if ( s!=IL.species_to_sample.size()-1 )
							std::cout << "\t";
					}
					std::cout << "\n";
				}
				std::cout << "\n";

			}

		} else {

			// """crossplatform""" folder creation (TODO)
			#ifdef _WIN32
				std::string comando_crea_cartella("md ");
				comando_crea_cartella.append(IL.output_folder);		
				system(comando_crea_cartella.c_str());
			#else
				std::string comando_crea_cartella("mkdir ");
				comando_crea_cartella.append(IL.output_folder);
				system(comando_crea_cartella.c_str());
			#endif

			// No fitness: let's save the output of simulations on the hard disk
			cudaMemcpy(host_X,device_X, sizeof(conc_t) * IL.species_to_sample.size() * TOTAL_CUDA_THREADS * IL.time_instants.size(), cudaMemcpyDeviceToHost);
			CudaCheckError();

			for ( unsigned int tid=0; tid<IL.actual_threads; tid++ ) {
				
				std::string outputfile(IL.output_folder);
				outputfile.append("/");
				outputfile.append(IL.output_prefix);
				outputfile.append("_");
				outputfile.append( convertInt(tid) );

				std::ofstream dump2(outputfile.c_str());

				if (! dump2.good() ) {
					perror("ERROR: cannot save dynamics");
					exit(-2);
				}
	 
				dump2 << std::setprecision(15);
		
				unsigned int larg = TOTAL_CUDA_THREADS;
				unsigned int DEV_CONST_SAMPLESPECIES = (unsigned int) IL.species_to_sample.size();		
	
				for (unsigned int campione=0; campione<IL.time_instants.size(); campione++) {
					dump2 << IL.time_instants[campione] << "\t";
					for (unsigned int s=0; s<IL.species_to_sample.size(); s++) {										
						dump2 << host_X[ ACCESS_SAMPLE ];
						if ( s!=IL.species_to_sample.size()-1 )
							dump2 << "\t";
					}
					dump2 << "\n";
				}
				dump2.close();

			} // end for

		} // end print fitness

	} // end not just fitness


	// release memory on the CPU
	free(host_X);
	// free(host_fitness);
	// delete &IL;
		
	// release memory on the GPU
	cudaFree(device_X);
	cudaFree(device_constants);
	cudaFree(device_species_to_sample);	

	    return 0;
} 


