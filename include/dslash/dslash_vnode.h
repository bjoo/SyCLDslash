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
#pragma message ( "Using vnode type A" )
#include "dslash/dslash_vnode_a.h"
#else
#pragma message ( "Using vnode type B" )
#include "dslash/dslash_vnode_b.h"
#endif




#endif /* INCLUDE_DSLASH_DSLASH_VNODE_H_ */
