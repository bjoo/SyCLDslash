/*
 * dslash_vnode.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */
// FIXME:: Check the masks.
// FIXME:: Check we can call the permute calls from a test...

#pragma once
#include <array>
#include "dslash_complex.h"
#include "dslash/dslash_vectype_sycl_a.h"
#include "dslash/dslash_vectype_sycl_subgroup.h"
#include "lattice/constants.h"
#include "CL/sycl.hpp"
// This file is specific to fermion storeage type a)
//
//  (RR...)(II...)
namespace MG {

// Forward declaration of VNode
template<typename T, int N>
struct VNode;



template<typename T>
struct VNode<T,8> {


	using VecType =  SIMDComplexSyCL<typename BaseType<T>::Type,8>;
	using SIMDScalarType = ThreadPrivateSIMDComplex<typename BaseType<T>::Type,8>;

	static constexpr int VecLen = 8;
	static constexpr int NDim = 3;

	static constexpr int Dim0 = 1;
	static constexpr int Dim1 = 2;
	static constexpr int Dim2 = 2;
	static constexpr int Dim3 = 2;


	//     Vec  Index      0 1 2 3 4 5 6 7
	//       X  coord      0 0 0 0 0 0 0 0
	//       Y  coord      0 1 0 1 0 1 0 1
	//       Z  coord      0 0 1 1 0 0 1 1
	//       T  coord      0 0 0 0 1 1 1 1
	//
	//   Y permute: 0<->1, 2<->3, 4<->5, 6<->7 so swizzle: 1,0,3,2,5,4,7,6
	//   Z permute: (0,1)<->(2,3) (4,5) <-> (6,7) so swizzle: 2,3,0,1,6,7,4,5
	//   T permute: (0,1,2,3) <-> (4,5,6,7) so swizzle: 4,5,6,7,0,1,2,3
	static constexpr std::array<int,8> x_mask = {0,1,2,3,4,5,6,7};
	static constexpr std::array<int,8> y_mask = {1,0,3,2,5,4,7,6};
	static constexpr std::array<int,8> z_mask = {2,3,0,1,6,7,4,5};
	static constexpr std::array<int,8> t_mask = {4,5,6,7,0,1,2,3};
	static constexpr std::array<int,8> nopermute_mask = {0,1,2,3,4,5,6,7};



	static inline VecType permuteX(const VecType& vec_in) {
		return vec_in;
	}


	static inline VecType permuteY(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6>(),
				(vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6>() );
	}


	static inline VecType permuteZ(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<2,3,0,1,6,7,4,5>(),
				(vec_in.imag()).template swizzle<2,3,0,1,6,7,4,5>() );
	}


	static inline VecType permuteT(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<4,5,6,7,0,1,2,3>(),
				(vec_in.imag()).template swizzle<4,5,6,7,0,1,2,3>() );
	}

}; // struct vector length = 8

template<typename T>
struct VNode<T,16> {


	using VecType =  SIMDComplexSyCL<typename BaseType<T>::Type,16>;
	using SIMDScalarType = ThreadPrivateSIMDComplex<typename BaseType<T>::Type,16>;

	static constexpr int VecLen = 16;
	static constexpr int NDim = 4;

	static constexpr int Dim0 = 2;
	static constexpr int Dim1 = 2;
	static constexpr int Dim2 = 2;
	static constexpr int Dim3 = 2;

	template <IndexType dir>
	static inline VecType permute(const VecType& vec_in) {
		return vec_in;

	}
	// Swizzle both with 1,0
	//                    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
	//            X =     0 1 0 1 0 1 0 1 0 1 0  1  0  1   0  1
	//            Y =     0 0 1 1 0 0 1 1 0 0 1  1  0  0   1  1
	//            Z =     0 0 0 0 1 1 1 1 0 0 0  0  1  1   1  1
	//            T =     0 0 0 0 0 0 0 0 1 1 1  1  1  1   1  1
	static constexpr std::array<int,16> x_mask = {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};
	static constexpr std::array<int,16> y_mask = {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13};
	static constexpr std::array<int,16> z_mask = {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11};
	static constexpr std::array<int,16> t_mask = {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7};
	static constexpr std::array<int,16> nopermute_mask = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

	// X: 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13, 14-15
	// so permute: 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14

	static inline VecType permuteX(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>(),
				(vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>() );

	}

	// Y: 0,1-2,3  4,5-6,7  8,9-10,11 12,13-14,15
	// so permute: 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13

	static inline VecType permuteY(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>(),
				(vec_in.imag()).template swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>() );

	}


	// Z: 0,1,2,3-4,5,6,7  8,9,10,11-12,13,14,15
	// so permute: 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11

	static inline VecType permuteZ(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>(),
				(vec_in.imag()).template swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>() );

	}

	// X: 0,1,2,3,4,5,6,7 - 8,9,10,11,12,13,14,15
	// so permute: 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7

	static inline VecType permuteT(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>(),
				(vec_in.imag()).template swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());

	}

}; // struct vector length = 16

// Generic Subgroup shuffle
template<typename T, int N>
static inline MGComplex<T> permute(const std::array<int,N> mask,
									 const MGComplex<T>& in,
									 const cl::sycl::intel::sub_group& sg)
{
	MGComplex<T> ret_val;
	ret_val.real( sg.shuffle(in.real(), mask[ sg.get_local_id() ]) );
	ret_val.imag( sg.shuffle(in.imag(), mask[ sg.get_local_id() ]) );
}
} // Namespace






