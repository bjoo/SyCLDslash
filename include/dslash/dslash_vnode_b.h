/*
 * dslash_vnode.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */
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
#include "dslash/dslash_vectype_sycl_b.h"
#include "lattice/constants.h"
// This file is specific to fermion storeage type B)
//
//  (R,I,R,I,...)
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
  static inline VecType permute(const VecType& vec_in) {
	  return vec_in;
  }
}; // Struct Vector Length = 1



template<typename T>
struct VNode<T,2> {
	using VecType =  SIMDComplexSyCL<typename BaseType<T>::Type,2>;
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

    //                     r i  r  i
	//     Vec  Index      0 1  2  3
	//       X  coord      0 0  1  1
	//       Y  coord      0 0  0  0
	//       Z  coord      0 0  0  0
	//       T  coord      0 0  1  1
	//
	//   T permuted:  2,3  0,1
  	template<>
	static inline VecType permute<T_DIR>(const VecType& vec_in) {
		return VecType{ vec_in._data.template swizzle<2,3,0,1>() };
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

    //                     r i r i r i r i
	//     Vec  Index      0 1 2 3 4 5 6 7
	//       X  coord      0 0 0 0 0 0 0 0
	//       Y  coord      0 0 0 0 0 0 0 0
	//       Z  coord      0 0 1 1 0 0 1 1
	//       T  coord      0 0 0 0 1 1 1 1
	//
    //
    //       Z permute:  2,3 - 0,1 and 6,7 - 4,5    so swizzle 2,3,0,1,6,7,4,5
  	//       T permute:  vec-lanes (4,5)(6,7) - (0,1),(2,3) so swizzle 4,5,6,7,0,1,2,3
    template<IndexType dir>
    static inline  VecType permute(const VecType& vec_in) {
    	return vec_in;
    }

    template<>
    static inline VecType permute<Z_DIR>(const VecType& vec_in) {
    	return VecType{ vec_in._data.template swizzle<2,3,0,1,6,7,4,5>() };

    }

    template<>
    static inline VecType permute<T_DIR>(const VecType& vec_in) {
    	return VecType{ vec_in._data.template swizzle<4,5,6,7,0,1,2,3>() };
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

    //                     r i r i r i r i r i  r  i  r  i  r  i
	//     Vec  Index      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16
	//       X  coord      0 0 0 0 0 0 0 0 0 0  0  0  0  0  0  0
	//       Y  coord      0 0 1 1 0 0 1 1 0 0  1  1  0  0  1  1
	//       Z  coord      0 0 0 0 1 1 1 1 0 0  0  0  1  1  1  1
	//       T  coord      0 0 0 0 0 0 0 0 1 1  1  1  1  1  1  1
	//
  	//   Y permute: 0,1-2,3 4,5-6,7 8,9-10,11 12,13-14,15      so swizzle: 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13
    //   Z permute: 0,1,2,3 - 4,5,6,7  8,9,10,11 - 12,13,14,15 so swizzle: 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
	//   T permute: 0,1,2,3,4,5,6,7 - 8,9,10,11,12,13,14,15    so swizzle: 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7

  // No swizzle in non Y,Z,T dir
  template<IndexType dir>
  static inline VecType permute(const VecType& vec_in) {
  	return vec_in;
  }


  //   Y permute: 0,1-2,3 4,5-6,7 8,9-10,11 12,13-14,15      so swizzle: 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13
  template<>
  static inline VecType permute<Y_DIR>(const VecType& vec_in) {
  	return VecType{ vec_in._data.template swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>()  };
  }

  //   Z permute: 0,1,2,3 - 4,5,6,7  8,9,10,11 - 12,13,14,15 so swizzle: 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11
  template<>
  static inline VecType permute<Z_DIR>(const VecType& vec_in) {
  	return VecType{ vec_in._data.template swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>() };
   }

  //   T permute: 0,1,2,3,4,5,6,7 - 8,9,10,11,12,13,14,15    so swizzle: 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
  template<>
  static inline VecType permute<T_DIR>(const VecType& vec_in) {
  	return VecType{ vec_in._data.template swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>() };
  }

}; // struct vector length = 8



} // Namespace











