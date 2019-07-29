/*
 * dslash_vectype_sycl.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */

#pragma once
#include "sycl_dslash_config.h"

#ifndef MG_FORTRANLIKE_COMPLEX
#ifdef MG_DEBUG_INCLUDES
#pragma message ( "Using Complex Vector Type A (RRRR)(IIII)" )
#endif
#define MG_TESTING_VECTYPE_A 1
#include "dslash/dslash_vectype_sycl_a.h"

#else

#ifdef MG_DEBUG_INCLUDES
#pragma message ( "Using Fortranlike Complex Vector Type B (RIRIRIRI)" )
#endif

#define MG_TESTING_WECTYPE_B 1
#include "dslash/dslash_vectype_sycl_b.h"
#endif
