/*
Constants and defines used by cupSODA.
See file COPYING for copyright and licensing information.
*/


#ifndef __CONSTANTS__
#define __CONSTANTS

/* cuLSODO variables */
// #define THREADS	16
// #define BLOCKS 1
// #define VARIABLES 4
// #define KINCONSTANTS 3

#define USE_SHARED_MEMORY
#define USE_CONSTANT_MEMORY

#define ERROR_INSUFF_CONSTANT_MEMORY_ODE 100
#define ERROR_INSUFF_CONSTANT_MEMORY_JAC 101
#define ERROR_INSUFF_SHARED_MEMORY		 102
#define ERROR_INSUFF_GLOBAL_MEMORY	     103
#define ERROR_NO_VARIABLES				 105
#define ERROR_NO_PARAMETERS				 106

#define ERROR_TIMESERIES_NOT_SPECIFIED   1001

#endif