/*
 * dslash_vectype_sycl.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */

#pragma once
#include "sycl_dslash_config.h"

#ifndef MG_FORTRANLIKE_COMPLEX
#pragma message ( "Using Complex Vector Type A (RRRR)(IIII)" )
#define MG_TESTING_VECTYPE_A 1
#include "dslash/dslash_vectype_sycl_a.h"

#else


#pragma message ( "Using Fortranlike Complex Vector Type B (RIRIRIRI)" )
#define MG_TESTING_WECTYPE_B 1
#include "dslash/dslash_vectype_sycl_b.h"
#endif
