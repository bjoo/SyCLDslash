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
#include "lattice/constants.h"
// This file is specific to fermion storeage type a)
//
//  (RR...)(II...)
namespace MG {

// Forward declaration of VNode
template<typename T, int N>
struct VNode;

template<typename T>
  struct VNode<T,1> {
  using VecType =  SIMDComplexSyCL<typename BaseType<T>::Type,1>;

  static constexpr int VecLen = 1 ;
  static constexpr int nDim = 0;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 1;


  template<IndexType dir>
  static VecType permute(const VecType& vec_in) {
	  return vec_in;

  }
}; // Struct Vector Length = 1



template<typename T>
struct VNode<T,2> {
	using FloatType = typename BaseType<T>::Type;
	using VecType =  SIMDComplexSyCL<FloatType,2>;
	static constexpr int VecLen =  2;
	static constexpr int NDim = 1;

	static constexpr int Dim0 = 1;
	static constexpr int Dim1 = 1;
	static constexpr int Dim2 = 1;
	static constexpr int Dim3 = 2;

	template<IndexType dir>
	static inline VecType permute(const VecType& vec_in) {
		return vec_in;
	}


	//     Vec  Index      0 1
	//       X  coord      0 0
	//       Y  coord      0 0
	//       Z  coord      0 0
	//       T  coord      0 1
	//
	//       T - permute: 0 <-> 1 so swizzle <1,0>
	template<>
	static inline VecType permute<T_DIR>(const VecType& vec_in) {
		return VecType( (vec_in.real()).template swizzle<1,0>(),
				(vec_in.imag()).template swizzle<1,0>() );
	}
}; // Struct Vector Length = 2


template<typename T>
struct VNode<T,4> {

  using VecType = SIMDComplexSyCL<typename BaseType<T>::Type,4>;

  static constexpr int VecLen =  4;
  static constexpr int NDim = 2;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;


	//     Vec  Index      0 1 2 3
	//       X  coord      0 0 0 0
	//       Y  coord      0 0 0 0
	//       Z  coord      0 1 0 1
	//       T  coord      0 0 1 1
	//
    //
    //       Z permute:  vec-lanes 0<->1 2<->3     so swizzle 1,0,3,2
	//       T permute:  vec-lanes (0,1) <-> (2,3) so swizzle 2,3,0,1
    template<IndexType dir>
    static inline VecType permute(const VecType& vec_in) {
    	return vec_in;
    }

    template<>
    static inline VecType permute<Z_DIR>(const VecType& vec_in) {
    	return VecType( (vec_in.real()).template swizzle<1,0,3,2>(),
    			(vec_in.imag()).template swizzle<1,0,3,2>() );
    }

    template<>
    static inline VecType permute<T_DIR>(const VecType& vec_in) {
    	return VecType( (vec_in.real()).template swizzle<2,3,0,1>(),
    			(vec_in.imag()).template swizzle<2,3,0,1>() );
    }
};   // struct vector length = 4


template<typename T>
struct VNode<T,8> {

  using VecType = SIMDComplexSyCL<typename BaseType<T>::Type,8>;

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
  template<IndexType dir>
  static inline VecType permute(const VecType& vec_in) {
  	return vec_in;
  }

  template<>
  static inline VecType permute<Y_DIR>(const VecType& vec_in) {
  	return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6>(),
  			(vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6>() );
  }

  template<>
  static inline VecType permute<Z_DIR>(const VecType& vec_in) {
  	return VecType( (vec_in.real()).template swizzle<2,3,0,1,6,7,4,5>(),
  			(vec_in.imag()).template swizzle<2,3,0,1,6,7,4,5>() );
  }

  template<>
  static inline VecType permute<T_DIR>(const VecType& vec_in) {
  	return VecType( (vec_in.real()).template swizzle<4,5,6,7,0,1,2,3>(),
  			(vec_in.imag()).template swizzle<4,5,6,7,0,1,2,3>() );
  }

}; // struct vector length = 8

template<typename T>
struct VNode<T,16> {

  using VecType = SIMDComplexSyCL<typename BaseType<T>::Type,16>;

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

  // X: 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13, 14-15
  // so permute: 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14
  template <>
  static inline VecType permute<X_DIR>(const VecType& vec_in) {
	  return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>(),
			  (vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>() );

  }

  // Y: 0,1-2,3  4,5-6,7  8,9-10,11 12,13-14,15
  // so permute: 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13
  template <>
  static inline VecType permute<Y_DIR>(const VecType& vec_in) {
	  return VecType( (vec_in.real()).template swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>(),
			  (vec_in.imag()).template swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>() );

  }


  // Z: 0,1,2,3-4,5,6,7  8,9,10,11-12,13,14,15
  // so permute: 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
  template <>
  static inline VecType permute<Z_DIR>(const VecType& vec_in) {
	  return VecType( (vec_in.real()).template swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>(),
			  (vec_in.imag()).template swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>() );

  }

  // X: 0,1,2,3,4,5,6,7 - 8,9,10,11,12,13,14,15
  // so permute: 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
  template <>
  static inline VecType permute<T_DIR>(const VecType& vec_in) {
	  return VecType( (vec_in.real()).template swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>(),
			  (vec_in.imag()).template swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());

  }

}; // struct vector length = 16
} // Namespace






