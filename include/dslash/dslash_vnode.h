/*
 * dslash_vnode.h
 *
 *  Created on: Jul 8, 2019
 *      Author: bjoo
 */

#ifndef INCLUDE_DSLASH_DSLASH_VNODE_H_
#define INCLUDE_DSLASH_DSLASH_VNODE_H_

#pragma once
#include "sycl_dslash_config.h"
#include "dslash_vectype_sycl.h"

#ifndef MG_FORTRANLIKE_COMPLEX

#ifdef MG_DEBUG_INCLUDES
#pragma message ( "Using vnode type A" )
#endif

#include "dslash/dslash_vnode_a.h"

#else

#ifdef MG_DEBUG_INCLUDES
#pragma message ( "Using vnode type B" )
#endif
#include "dslash/dslash_vnode_b.h"
#endif




#endif /* INCLUDE_DSLASH_DSLASH_VNODE_H_ */
