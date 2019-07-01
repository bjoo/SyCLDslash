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
  using VecTypeGlobal = SIMDComplexSyCL<typename BaseType<T>::Type,1>;
  using VecType =  SIMDComplexSyCL<typename BaseType<T>::Type,1>;

  static constexpr int VecLen = 1 ;
  static constexpr int nDim = 0;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 1;

  template<IndexType dir>
  struct DirPermutes {
	  constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
		  return vec_in;
	  }
  };
}; // Struct Vector Length = 1



template<typename T>
struct VNode<T,2> {
	using FloatType = typename BaseType<T>::Type;
	using VecTypeGlobal = SIMDComplexSyCL<FloatType,2>;
	using VecType =  SIMDComplexSyCL<FloatType,2>;
	static constexpr int VecLen =  2;
	static constexpr int NDim = 1;

	static constexpr int Dim0 = 1;
	static constexpr int Dim1 = 1;
	static constexpr int Dim2 = 1;
	static constexpr int Dim3 = 2;

	template<IndexType dir>
	struct DirPermutes {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			return vec_in;
		}
	};


	template<>
	struct DirPermutes<T_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<1,0>(),
					        (vec_in.imag()).template swizzle<1,0>() );

		}
	};
}; // Struct Vector Length = 2


template<typename T>
struct VNode<T,4> {

  using VecTypeGlobal = SIMDComplexSyCL<typename BaseType<T>::Type,4>;
  using VecType = SIMDComplexSyCL<typename BaseType<T>::Type,4>;

  static constexpr int VecLen =  4;
  static constexpr int NDim = 2;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;
#if 0
  using MaskType = MaskArray<4>;

  static constexpr MaskType XPermuteMask  = {0,1,2,3};
  static constexpr MaskType YPermuteMask  = {0,1,2,3};
  static constexpr MaskType ZPermuteMask  = {1,0,3,2};
  static constexpr MaskType TPermuteMask  = {2,3,0,1};
  static constexpr MaskType NoPermuteMask = {0,1,2,3};
#endif

	template<IndexType dir>
	struct DirPermutes {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			return vec_in;
		}
	};

	template<>
	struct DirPermutes<Z_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<1,0,3,2>(),
					        (vec_in.imag()).template swizzle<1,0,3,2>() );

		}
	};
	template<>
	struct DirPermutes<T_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<2,3,0,1>(),
					        (vec_in.imag()).template swizzle<2,3,0,1>() );

		}
	};

};   // struct vector length = 4


template<typename T>
struct VNode<T,8> {
  using VecTypeGlobal = SIMDComplexSyCL<typename BaseType<T>::Type,8>;
  using VecType = SIMDComplexSyCL<typename BaseType<T>::Type,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

#if 0
  using MaskType = MaskArray<8>;

  static constexpr MaskType NoPermuteMask = {0,1,2,3,4,5,6,7};
  static constexpr MaskType XPermuteMask  = {0,1,2,3,4,5,6,7};
  static constexpr MaskType YPermuteMask  = {1,0,3,2,5,4,7,6};
  static constexpr MaskType ZPermuteMask  = {2,3,0,1,6,7,4,5};
  static constexpr MaskType TPermuteMask  = {4,5,6,7,0,1,2,3};
#endif

	template<IndexType dir>
	struct DirPermutes {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			return vec_in;
		}
	};

	template<>
	struct DirPermutes<Y_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6>(),
					        (vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6>() );

		}
	};

	template<>
	struct DirPermutes<Z_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<2,3,0,1,6,7,4,5>(),
					        (vec_in.imag()).template swizzle<2,3,0,1,6,7,4,5>() );

		}
	};
	template<>
	struct DirPermutes<T_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<4,5,6,7,0,1,2,3>(),
					        (vec_in.imag()).template swizzle<4,5,6,7,0,1,2,3>() );

		}
	};

}; // struct vector length = 8

template<typename T>
struct VNode<T,16> {
  using VecTypeGlobal = SIMDComplexSyCL<typename BaseType<T>::Type,16>;
  using VecType = SIMDComplexSyCL<typename BaseType<T>::Type,16>;

  static constexpr int VecLen = 16;
  static constexpr int NDim = 4;

  static constexpr int Dim0 = 2;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

#if 0
  static constexpr MaskType NoPermuteMask = {0,1,2,3,4,5,6,7};
  static constexpr MaskType XPermuteMask  = {0,1,2,3,4,5,6,7};
  static constexpr MaskType YPermuteMask  = {1,0,3,2,5,4,7,6};
  static constexpr MaskType ZPermuteMask  = {2,3,0,1,6,7,4,5};
  static constexpr MaskType TPermuteMask  = {4,5,6,7,0,1,2,3};
#endif

	template<IndexType dir>
	struct DirPermutes {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			return vec_in;
		}
	};

	template<>
	struct DirPermutes<X_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>(),
					(vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>() );
		}
	};

	// Change Swizzle...
	template<>
	struct DirPermutes<Y_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<1,0,3,2,5,4,7,6>(),
					        (vec_in.imag()).template swizzle<1,0,3,2,5,4,7,6>() );

		}
	};

	// Change Swizzle
	template<>
	struct DirPermutes<Z_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<2,3,0,1,6,7,4,5>(),
					        (vec_in.imag()).template swizzle<2,3,0,1,6,7,4,5>() );

		}
	};

	//Change swizzle
	template<>
	struct DirPermutes<T_DIR> {
		constexpr VecType operator()(const VecTypeGlobal& vec_in) const {
			// Swizzle both with 1,0
			return VecType( (vec_in.real()).template swizzle<4,5,6,7,0,1,2,3>(),
					        (vec_in.imag()).template swizzle<4,5,6,7,0,1,2,3>() );

		}
	};

}; // struct vector length = 16
} // Namespace






