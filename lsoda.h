/*
lsoda.h: porting of LSODA to CUDA. Implementation of device-side
           parsers for ODEs and Jacobian matrix.
See file COPYING for copyright and licensing information.
*/

#ifndef __CULSODO__
#define __CULSODO__


extern "C" __device__ void myFex(int *neq, double *t, double *y, double *ydot, double* k)	;
extern "C" __device__ void myJex(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd,  double* k);


// #define KERNEL_VERBOSE

// #include "input_reader.h"
// #include "stoc2det.h"
#include <cmath>

typedef double param_t;
typedef double conc_t;

// #define min(a,b) ((a) <= (b) ? (a) : (b))
// #define max(a,b) ((a) >= (b) ? (a) : (b))

/* Pointer to compressed ODE on the GPU */
extern char* device_compressed_odes;

/* Number of threads per block. */
__device__ __constant__ unsigned int NUM_THREADS = 0;

/* ODE in constant memory */
// __constant__ double ODE_new[MAX_ODE_LUN];
// __constant__ double JAC_new[MAX_JAC_LUN];

/* Size of compressed matrix for ODEs. */
__constant__ unsigned int COMP_ODE_SIZE = 0;

/* Shared memory global container. */
extern __shared__ double shared[];

/* information for ODE reconstruction */
__constant__ unsigned int DEV_CONST_VARIABLES = 0;
__constant__ unsigned int DEV_CONST_PARAMETERS = 0;
// __constant__ unsigned int DEV_CONST_SAMPLESLUN = 0;
__constant__ unsigned int DEV_CONST_TIMESLUN = 0;

#define ACCESS_SAMPLE  larg*DEV_CONST_SAMPLESPECIES*campione + larg*s + tid  

__constant__ unsigned int DEV_CONST_EXPERIMENTS = 0;
__constant__ unsigned int DEV_CONST_REPETITIONS = 0;
// __constant__ unsigned int DEV_CONST_PAR_REPETITIONS = 0;
__constant__ unsigned int DEV_CONST_TARGET_QUANTITIES = 0;
// __constant__ unsigned int DEV_CONST_SAMPLES = 0;
__constant__ unsigned int DEV_CONST_SAMPLESPECIES = 0;

// threads that actually must perform calculations 
__constant__ unsigned int DEV_ACTUAL_THREADS = 0;


void SetConstants( unsigned int , unsigned int , unsigned int cs_lun, unsigned int time_ins, unsigned int act_threads, bool);

/*void save_constants(unsigned int);
void load_compressed_odes( InputReader* ir );
void LoadSystem( st2det* system );
bool CheckArguments(unsigned int argc, char** argv);
void SetODEarray(st2det* system  );
void calculate_fft( st2det* s2d, double* device_X, std::string exportfile="" );
*/

/* Common Block Declarations */
struct ginSODACommonBlock
{
	double /*rowns[209],*/ CM_conit, CM_crate, CM_ccmax, CM_el0, CM_h__, CM_hmin, CM_hmxi, CM_hu, CM_rc, CM_tn;
	double CM_uround, CM_pdest, CM_pdlast, CM_ratio, CM_hold, CM_rmax;
	double  CM_el[13], CM_elco[156]	/* was [13][12] */, CM_tesco[36]	/* was [3][12] */;
	double CM_rls[218];
	double CM_tsw, /*rowns2[20],*/ CM_pdnorm;
	double /*rownd2,*/ CM_cm1[12], CM_cm2[5];
	double CM_rlsa[22];
	double CM_sm1[12];
	int CM_init, CM_mxstep, CM_mxhnil, CM_nhnil, CM_nslast, CM_nyh, /*iowns[6],*/ CM_icf, 
	CM_ierpj, CM_iersl, CM_jcur, CM_jstart, CM_kflag, CM_l, CM_lyh, CM_lewt, CM_lacor, CM_lsavf,
	CM_lwm, CM_liwm, CM_meth, CM_miter, CM_maxord, CM_maxcor, CM_msbp, CM_mxncf, CM_n, CM_nq, 
	CM_nst, CM_nfe, CM_nje, CM_nqu;
	int /*iownd[6],*/ CM_ialth, CM_ipup, CM_lmax, /*meo,*/ CM_nqnyh, CM_nslp;
	int CM_ils[37];
	int CM_insufr, CM_insufi, CM_ixpr, /*iowns2[2],*/ CM_jtyp, CM_mused, CM_mxordn, CM_mxords; 
	int /*iownd2[3],*/ CM_icount, CM_irflag;
	int CM_ilsa[9];
};


/*
#define Fex_and_Jex_definition_old
struct myFex_old
{
	__device__ void operator()(int *neq, double *t, double *y, double *ydot, char* comp_ode, param_t* flattenODE, unsigned int* offsetODE, double* costanti)
	{
		
		
		// double costanti[3] = { 0.0025, 0.1, 5.0 };
		unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

		unsigned int num_ode = 0;		

		for ( unsigned int o = 0; o < NUM_ODES; o++ ) 
			ydot[o]=0;

		unsigned int i=0;
		while ( i<COMP_ODE_SIZE ) {
		
			// trovato terminatore: salta a nuova ODE
			if (comp_ode[i] == 0) {
				num_ode++;
			} else {

				double temp = 0;
				
				// il primo numero mi dice quanti elementi ho 				
				// il secondo numero è l'indice della costante. se >0, si somma la costante. else viceversa.
				// es.: 3 -1 0 1 2 2 2 0
				//      significa addendo di 3 elementi, -c1 * y[0] * y[1] , addendo di 2 elementi, +c2 * y[2] , fine espressione
				
				// costante
				if ( comp_ode[i+1]>0 ) {
					temp =  costanti[ tid*KINCONSTANTS + comp_ode[i+1] -1 ];
				} else {
					temp = -costanti[ tid*KINCONSTANTS + abs(comp_ode[i+1])-1 ];
				}

				i++;

				unsigned int j=comp_ode[i-1]-1;
				
				while( j>0 ) { 
					i++;
					temp *= y[comp_ode[i]];
					j--;
				}
				
				ydot[num_ode] += temp;								
				
			} // endif

			i++;

		} 




	}
};

struct myJex_old
{
	__device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*)
	{
		return;
	}
};

*/



/*
#define Fex_and_Jex_definition
struct myFex
{
	__device__ void operator()(int *neq, double *t, double *y, double *ydot, char* comp_ode, param_t* ODE, unsigned int* ODE_offset, double* costanti)
	{
		
		// unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
		// init ODEs
		for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) 		ydot[s]=0;
	
		unsigned int pos=0;
		
		for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {

#ifdef KERNEL_VERBOSE
			if ( threadIdx.x==0 ) 
				printf("dX[%d]/dt = ", s);
#endif

			param_t temp = 0;
					
			while( pos<ODE_offset[s] ) {

				unsigned int i;
				
				// variazione
#ifdef KERNEL_VERBOSE
				if ( threadIdx.x==0 ) {
					if (ODE[pos]>0) printf("+");
					printf(" %d ", (int)ODE[pos]);
					printf(" * k[%d] ", (int)ODE[pos+1]);
					printf(" * %f ", ODE[pos+2]);
				}
#endif

				// ydot[s] = ODE[pos];
				temp = ODE[pos];
								
				// costante cinetica				
				// ydot[s] *= costanti[ (int)ODE[pos+1] ];
				temp *= costanti[ (int)ODE[pos+1] ];

				// fattore conversione				
				// ydot[s] *= ODE[pos+2];
				temp *= ODE[pos+2];

				// concentrazioni
				for( i=0; i<ODE[pos+3]; i++ ) {
					
#ifdef KERNEL_VERBOSE
					if ( threadIdx.x==0 ) 
						printf(" * X[%d] ", (int)ODE[pos+4+i]);
#endif
					
					// ydot[s] *= y[ (int)ODE[pos+4+i] ];
					temp *= y[ (int)ODE[pos+4+i] ];
				}
		
				pos += 4 + i;

				ydot[s] += temp;

			} 
#ifdef KERNEL_VERBOSE
			if ( threadIdx.x==0 ) printf("\n\n");
#endif
		}

	}
};
*/

/*

struct myJex
{
	__device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd, param_t* JAC, unsigned int* JAC_offset, double* costanti)
	{

		// PD stores the Jacobian	
		for (unsigned int s1=0; s1<DEV_CONST_SPECIES; s1++) 
			for (unsigned int s2=0; s2<DEV_CONST_SPECIES; s2++) 
				pd[s1*DEV_CONST_SPECIES+s2] = 0;

		// return;
		
		unsigned int pos = 0;
	
		for (unsigned int s1=0; s1<DEV_CONST_SPECIES; s1++) { 

			for (unsigned int s2=0; s2<DEV_CONST_SPECIES; s2++) { 

				unsigned int indice = s1 * DEV_CONST_SPECIES + s2;
				param_t temp = 0;

				// printf("dX[%d][%d] = ", s2, s1);
				// pd[ s2 * nropwd + s1 ] = 0;

				// if ( pos == this->JAC_offset[s1*this->species+s2]  )				
				if ( pos == JAC_offset[indice] )				
					pd[ indice ] = 0;
					// printf("0");

				// while( pos<this->JAC_offset[s1*this->species+s2] ) {
				while( pos<JAC_offset[indice] ) {

					unsigned int i;
					
					// variazione
					// temp = (int)JAC[pos];
					temp = JAC[pos];

					// costante cinetica
					// printf(" * k[%d] ", (int)this->JAC[pos+1]);
					// temp *= (int)JAC[pos+1];
					temp *= costanti[(int)JAC[pos+1]];

					// fattore conversione
					// printf(" * %f ", this->JAC[pos+2]);
					temp *= JAC[pos+2];

					// concentrazioni
					for( i=0; i<JAC[pos+3]; i++ ) {
						// printf(" * X[%d] ", (int)this->JAC[pos+4+i]);
						temp *= y[ (int)JAC[pos+4+i] ];
					}
		
					pos += 4 + i;

				}

				pd[ indice ] += temp;

				// printf("\n");

			}				
		}

		return;
	}
};

*/




// #define Fex_and_Jex_definition




__device__ int dlsoda_( int *, double *, double *, double *, int *, double *, double *, int *, int *, int *, double *, int *, int *, int *,  int *, struct ginSODACommonBlock *, int* debug, double* costanti);
__device__ int dstoda_(int *neq, double *y, double *yh, int *NOT_nyh, double *yh1, double *ewt, double *savf, double *acor, double *wm, int *iwm,  struct ginSODACommonBlock *common, double* costanti);
//  dprja_(           int*,     double*,   double*,    int*,         double*,     double*,      double*,      double*,    int*,     myFex,   myJex,            ginSODACommonBlock*,        double*)
__device__ int dprja_(int *neq, double *y, double *yh, int *NOT_nyh, double* ewt, double* ftem, double* savf, double* wm, int* iwm,  struct ginSODACommonBlock* common, double* costanti);
__device__ int dsolsy_(double *wm, int *iwm, double *x, double *tem, struct ginSODACommonBlock *common);
__device__ int dintdy_(double *t, int k, double *yh, int *NOT_nyh, double *dky, int *iflag, struct ginSODACommonBlock *common);
__device__ int dcfode_(int meth, double *DCFODE_elco, double *DCFODE_tesco, struct ginSODACommonBlock *common);
__device__ int dsolsy_(double *wm, int *iwm, double *x, double *tem, struct ginSODACommonBlock *common);
__device__ int dewset_(int *PARAM_n, int *itol, double *rtol, double *atol, double *ycur, double *ewt, struct ginSODACommonBlock *common);
__device__ double dmnorm_(int *PARAM_n, double *v, double *w, struct ginSODACommonBlock *common);
__device__ double dfnorm_(int *PARAM_n, double *a, double *w, struct ginSODACommonBlock *common);
__device__ double dbnorm_(int *PARAM_n, double *a, int *nra, int *ml, int *mu, double *w, struct ginSODACommonBlock *common);
__device__ int dgefa_(double *a, int *lda, int *PARAM_n, int *ipvt, int *info, struct ginSODACommonBlock *common);
__device__ int dgesl_(double *a, int *lda, int *PARAM_n, int *ipvt, double *b, int job, struct ginSODACommonBlock *common);
__device__ int dgbfa_(double *abd, int *lda, int *PARAM_n, int *ml, int *mu, int *ipvt, int *info, struct ginSODACommonBlock *common);
__device__ int dgbsl_(double *abd, int *lda, int *PARAM_n, int *ml, int *mu, int *ipvt, double *b, int job, struct ginSODACommonBlock *common);
__device__ double dumach_( struct ginSODACommonBlock *common);
__device__ int idamax_(int *PARAM_n, double *dx, int incx, struct ginSODACommonBlock *common);
__device__ int daxpy_(int *PARAM_n, double *da, double *dx, int incx, double *dy, int incy, struct ginSODACommonBlock *common);
__device__ int dumsum_(double a, double b, double *c__, struct ginSODACommonBlock *common);
__device__ int dscal_(int *PARAM_n, double *da, double *dx, int incx, struct ginSODACommonBlock *common);
__device__ double ddot_(int *PARAM_n, double *dx, int incx, double *dy, int incy, struct ginSODACommonBlock *common);
__device__ double d_sign(double *a, double *b);
__host__ __device__ void ginSODA_Init(struct ginSODACommonBlock *common, unsigned int);


/*
	This is the entrypoint of ginSODA.
	All these state variables must be changed to multidimensional arrays, in order to fit the SIMD architecture.
*/
__global__ void ginSODA(int *neq, double *y, double *t, double *tout, int *itol, double *rtol, 
	double *atol, int *itask, int *istate, int *iopt, double *rwork, int *lrw, int *iwork, int *liw,  int *jt, 
	struct ginSODACommonBlock *common, int* debug,  double* costanti, conc_t* device_X, unsigned int campione, unsigned int* s2s,
	bool ACTIVATE_SHARED_MEMORY, bool ACTIVATE_CONSTANT_MEMORY);

//__global__ void calculateFitnessPRR( double* samples, double* target, double* fitness, char* swarm );
__global__ void calculateFitness( double* samples, double* target, double* fitness, char* swarm );
//__global__ void calculateFitnessYuki( double* samples, double* fitness );
__global__ void calculateFitnessNoman( double* samples, double* fitness, double const iteration );



#endif