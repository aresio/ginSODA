#ifndef __GINSODA__
#define __GINSODA__

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <cuda.h>

#include "constants.h"

template< typename T >
T operator^( T x, T y ) {
    return std::pow( x, y );
}

class InputLoader {

public:

	/// Constructor.
	InputLoader() {
		this->variables = 0;
		this->parameters = 0;
		// this->threads = 0;
		this->max_steps = 1000;
		this->rtol = 1e-6;
		this->output_folder = ".";
		this->output_prefix = "output";
		this->blocks = 0;
		this->actual_threads = 0;
	}

	// Destructor.
	~ InputLoader () {
		this->initial_amounts.clear();
		this->parameterizations.clear();
		this->atol.clear();
		this->species_to_sample.clear();
	}

	bool calculate_tpb(int verbose=false) {
		if (this->actual_threads==0) {
			std::cerr << "ERROR: no threads selected." << std::endl;
			exit(1);
		}

		// did the user request a specific blocks subdivision?
		if (this->blocks!=0) {

			this->tpb = this->actual_threads % this->blocks? this->actual_threads/this->blocks +1  : this->actual_threads/this->blocks;
			if (verbose) {
				std::cout << "CUDA execution hierarchy: " << std::endl;
				std::cout << " * " << this->actual_threads << " requested threads" << std::endl;
				std::cout << " * " << this->blocks << " blocks" << std::endl;
				std::cout << " * " << this->tpb << " threads per block" << std::endl;
			}

		}  else { 

			// TODO
			std::cerr << "ERROR: blocks not specified.\n";
			exit(100);

		}

		return true;

	}


	/// Loads the initial conditions for all threads from the specified file.
	bool load_initial_conditions(std::string path, int verbose=0) {

		std::string row;
		std::vector<double> values;

		std::ifstream conditions(path.c_str());
		if (conditions.is_open()) {

			this->initial_amounts.clear();

			// step #1: detect the number of variables in the system
			getline(conditions, row);
			size_t variables = std::count(row.begin(), row.end(), '\t') + 1;
			if (verbose) std::cout << " * Detected " << variables << " variables" << std::endl;
			if (variables==0) {
				std::cerr << "ERROR: cannot integrated with no ODEs, aborting." << std::endl;
				exit(ERROR_NO_VARIABLES);
			}

			// reset the file to the beginning
			conditions.clear(); conditions.seekg(0);

			// parse
			while( 1 ) {				
				std::vector<double> thread_amounts;
				this->initial_amounts.push_back(thread_amounts);
				// int threads = this->initial_amounts.size();
				for (double v=0; v<variables-1; v++) {					
					getline(conditions, row, '\t');
					if (row.size()<2) break;
					this->initial_amounts.back().push_back(atof(row.c_str()));
					// std::cout << " * Threads " << this->initial_amounts.size() << ", variable" << v << ": " << row << std::endl;
				}
				getline(conditions, row, '\n');  // empty read to skip \n
				this->initial_amounts.back().push_back(atof(row.c_str()));
				// std::cout << " * Threads " << this->initial_amounts.size() << ", variable" << variables-1 << ": " << row << std::endl;
				if (conditions.eof()) break;
			}
			
			if (this->initial_amounts.back().size()<variables) 	this->initial_amounts.pop_back();			

			this->variables = variables;
			this->actual_threads = this->initial_amounts.size();
		} else {
			std::cout << path.c_str() << "\n";
			perror("cannot find initial conditions file (MX_0).");
			return false;
		}

		conditions.close();
		return true;

	} // end load amounts


	/// Loads the parameters of the system.
	bool load_parameterizations(std::string path, int verbose=0) {
		
		std::string row;
		std::vector<double> values;

		std::ifstream file_params(path.c_str());
		if (file_params.is_open()) {

			this->parameterizations.clear();
				
			// step #1: detect the number of parameters in the system
			getline(file_params, row);			
			size_t pars = std::count(row.begin(), row.end(), '\t') + 1;
			if (verbose) std::cout << " * Detected " << pars << " parameterizations" << std::endl;
			if (pars == 0) {
				std::cerr << "ERROR: cannot integrated with no parameters, aborting." << std::endl;
				exit(ERROR_NO_PARAMETERS);
			}
			
			// reset the file to the beginning
			file_params.clear(); file_params.seekg(0);

			// parse
			while( 1 ) {				
				std::vector<double> thread_parameters;
				this->parameterizations.push_back(thread_parameters);				
				for (double v=0; v<pars-1; v++) {					
					getline(file_params, row, '\t');
					if (row.size()<2) break;
					this->parameterizations.back().push_back(atof(row.c_str()));
					// std::cout << " * Thread " << threads << ", parameter" << v << ": " << row << std::endl;
				}
				getline(file_params, row, '\n');				
				this->parameterizations.back().push_back(atof(row.c_str()));
				if (file_params.eof()) break;
			}

			if (this->parameterizations.back().size()<pars) 	this->parameterizations.pop_back();
			
			this->parameters= pars ;
			this->actual_threads = this->parameterizations.size();

		} else {
			perror("cannot find parameters file (c_vector).");
			return false;
		}
		
		file_params.close();
		return true;

	} // parameters


	/// Loads the time instants for the sampling of dynamics.
	bool load_time_instants(std::string path, int verbose=0) {

		std::string row;

		std::ifstream time_vector(path.c_str());

		if( time_vector.is_open() ) {

			this->time_instants.clear();

			while(getline(time_vector, row)) {
				this->time_instants.push_back(atof(row.c_str()));
			}

		} else {
			perror("cannot find initial conditions file (t_vector).");
			return false;
		}

		if (verbose) std::cout << " * " << this->time_instants.size() << " time instants requested" << std::endl;

		return true;
	}


	/// Loads the species to be sampled.
	bool load_species_to_sample(std::string path, int verbose=0) {
		
		std::string row;

		std::ifstream cs_vector(path.c_str());

		if( cs_vector.is_open() ) {

			this->species_to_sample.clear();

			while(getline(cs_vector, row)) {
				this->species_to_sample.push_back(atof(row.c_str()));
			}

		} else {
			if (verbose) std::cout << "WARNING: cannot find species to sample file (cs_vector).\n";
			for (unsigned int i=0; i<this->variables; i++) this->species_to_sample.push_back(i);
			// return false;
		}

		if (verbose) std::cout << " * " << this->species_to_sample.size() << " species will be sampled" << std::endl;

		return true;
	}


	/// Loads the atol, rtol and max_steps information.
	bool load_tolerances(std::string atol_path, std::string rtol_path, int verbose=0) {
		
		std::string row;
		std::ifstream atol_vector(atol_path.c_str());

		if( atol_vector.is_open() ) {
			
			this->atol.clear();
			while(getline(atol_vector, row)) {
				this->atol.push_back(atof(row.c_str()));
			}

			atol_vector.close();

		} else {

			if (verbose) std::cout << "WARNING: cannot find absolute tolerances file (atol_vector), using default 1e-2.\n";
			for (unsigned int i=0; i<this->variables; i++) this->atol.push_back(1e-12);
			
		}


		std::ifstream rtol_file(rtol_path.c_str());

		if( rtol_file.is_open() ) {
			
			this->rtol=0;
			getline(rtol_file, row);
			this->rtol = atof(row.c_str());			
			rtol_file.close();

		} else {

			if (verbose) std::cout << "WARNING: cannot find relative tolerance file (rtol), using default 1e-2.\n";
			for (int i=0; i<this->variables; i++) this->rtol = 1e-6;
			
		}


		std::ifstream maxsteps_file(rtol_path.c_str());

		if( maxsteps_file.is_open() ) {
			
			this->max_steps=0;
			getline(maxsteps_file, row);
			this->max_steps = atoi(row.c_str());			
			maxsteps_file.close();

		} else {

			if (verbose) std::cout << "WARNING: cannot find max internal steps file (max_steps), using default 10000.\n";
			for (int i=0; i<this->variables; i++) this->rtol = 10000;
			
		}
		
		return true;
	}

	bool check_memory_requirements(unsigned int THREADS, unsigned int time_instants, unsigned int GPU, bool verbose=false) {

		unsigned int var_bytes = this->variables*8*THREADS*2;
		unsigned int params_bytes = this->parameters*8*THREADS;
		unsigned int jacob_bytes = this->variables*this->variables*8*THREADS;
		unsigned int lsoda_bytes = THREADS*(4*8 + 3*8 + 8*this->variables);
		unsigned int lrw = 22+this->variables*std::max((unsigned int)16, this->variables+9); 
		unsigned int rwork_bytes = THREADS*8*lrw;
		unsigned int liw = 20+this->variables; 
		unsigned int iwork_bytes = THREADS*liw*4;
		unsigned int output_bytes = time_instants*THREADS*8*this->variables;
		unsigned int commonblock = 4304*THREADS*8;
		unsigned int total = var_bytes + params_bytes + jacob_bytes + rwork_bytes + lsoda_bytes + iwork_bytes + output_bytes + commonblock;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, GPU);
		if (verbose) printf(" * Device name: %s\n", prop.name);

        size_t free_byte;
        size_t total_byte;
        cudaMemGetInfo( &free_byte, &total_byte ) ;
		unsigned long long fb, tb;
		fb = (unsigned long long) free_byte;
		tb = (unsigned long long) total_byte;
		if (fb<total) {
			std::cout << "ERROR: insufficient GPU memory" << std::endl;
			return false;
		}

		// printf(" * Free memory: %llu\tTotal memory: %llu.\n", fb, tb);

		double percentage_free = (double)fb/tb;
		double percentage_busy = 1.-percentage_free;
		double percentage_needed = (double)total/total_byte;

		/*
		printf(" * Percentage of free GPU %d memory: %f.\n", GPU, percentage_free);
		printf(" * Percentage of busy GPU %d memory: %f.\n", GPU, percentage_busy);
		printf(" * Percentage of GPU %d memory needed for ginSODA: %f.\n", GPU, percentage_needed);
		*/

		if (verbose) {
			printf(" * ginSODA's GPU %d memory consumption: [", GPU);
			for (unsigned int i=0; i<percentage_busy*25; i++) printf("%c", 219);
			for (unsigned int i=0; i<percentage_needed*25; i++) printf("%c", 178);
			for (unsigned int i=0; i<(1.-(percentage_needed+percentage_busy))*25; i++) printf("%c", 176);
			printf("]\n");
		}		

		return true;

	}


	unsigned int variables;
	unsigned int parameters;
	// unsigned int threads;
	unsigned int actual_threads;
	unsigned int max_steps;
	double rtol;

	unsigned int tpb;
	unsigned int blocks;

	std::string output_folder;
	std::string output_prefix;

	std::vector< double > atol;
	std::vector< unsigned int > species_to_sample;
	std::vector< std::vector<double> > initial_amounts;
	std::vector< std::vector<double> > parameterizations;
	std::vector<double> time_instants;

};


#define DEEP_ERROR_CHECK
#define CUDA_ERROR_CHECK

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK

#ifdef DEEP_ERROR_CHECK
	cudaThreadSynchronize();
#endif 

#ifdef _DEBUG
	
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
		system("pause");
        exit( -1 );
    }
	
#endif 

#endif
    return;
}
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )



inline void __cudaCheckErrorRelease( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK

	cudaThreadSynchronize();
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
		system("pause");
        exit( -1 );
    }
    return;
}
#define CudaCheckErrorRelease()    __cudaCheckErrorRelease( __FILE__, __LINE__ )
#endif




#endif 
