/*
 * dslash_vnode.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */

#pragma once
#include <array>
#include "dslash/dslash_vectype_sycl.h"


namespace MG {

// An array for our masks.
template<int N>
struct MaskArray {
	std::array<int,N> _data;

	int& operator()(int i) { return _data[i]; }
	const int& operator()(int i) const { return _data[i]; }
};

// Forward declaration of VNode
template<typename T, int N>
struct VNode;

template<typename T>
  struct VNode<T,1> {
  using VecTypeGlobal = SIMDComplexSyCL<T,1>;
  using VecType =  SIMDComplexSyCL<T,1>;
  using MaskType = MaskArray<1>;

  static constexpr MaskType XPermuteMask = {0};
  static constexpr MaskType YPermuteMask = {0};
  static constexpr MaskType ZPermuteMask = {0};
  static constexpr MaskType TPermuteMask = {0};
  static constexpr MaskType NoPermuteMask = {0};

  static constexpr int VecLen = 1 ;
  static constexpr int nDim = 0;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 1;

  static
  KOKKOS_FORCEINLINE_FUNCTION
  VecType permute(const MaskType& mask, const VecTypeGlobal& vec_in)
  {
	  VecType vec_out;
	  ComplexCopy( vec_out, vec_in);
	  return vec_out;
  }

 }; // Struct Vector Length = 1


template<typename T>
struct VNode<T,2> {
  using FloatType = typename BaseType<T>::Type;
  using VecTypeGlobal = SIMDComplex<typename BaseType<T>::Type,2>;
  using VecType =  SIMDComplex<typename BaseType<T>::Type,2>;
  static constexpr int VecLen =  2;
  static constexpr int NDim = 1;

  using MaskType = MaskArray<2>;

  static constexpr MaskType XPermuteMask  = {0,1};
  static constexpr MaskType YPermuteMask  = {0,1};
  static constexpr MaskType ZPermuteMask  = {0,1};
  static constexpr MaskType TPermuteMask  = {1,0};
  static constexpr MaskType NoPermuteMask = {0,1};

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 1;
  static constexpr int Dim3 = 2;

  static
  KOKKOS_FORCEINLINE_FUNCTION
  VecType permute(const MaskType& mask, const VecTypeGlobal& vec_in)
  {
      VecType vec_out;
	  Kokkos::parallel_for(VectorPolicy(2),[&](const int& i){
		  ComplexCopy( vec_out(i), vec_in( mask(i) ));
	  });
	 return vec_out;
  }


}; // Struct Vector Length = 2



template<typename T>
struct VNode<T,4> {

  using VecTypeGlobal = SIMDComplex<typename BaseType<T>::Type,4>;
  using VecType = SIMDComplex<typename BaseType<T>::Type,4>;

  static constexpr int VecLen =  4;
  static constexpr int NDim = 2;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 1;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

  using MaskType = MaskArray<4>;

  static constexpr MaskType XPermuteMask  = {0,1,2,3};
  static constexpr MaskType YPermuteMask  = {0,1,2,3};
  static constexpr MaskType ZPermuteMask  = {1,0,3,2};
  static constexpr MaskType TPermuteMask  = {2,3,0,1};
  static constexpr MaskType NoPermuteMask = {0,1,2,3};

  static
  KOKKOS_FORCEINLINE_FUNCTION
  VecType permute(const MaskType& mask, const VecTypeGlobal& vec_in)
  {
	  VecType vec_out;
	  Kokkos::parallel_for(VectorPolicy(4),[&](const int& i){
		  ComplexCopy( vec_out(i), vec_in( mask(i) ));
	  });
	  return vec_out;
  }


};   // struct vector length = 4


template<typename T>
struct VNode<T,8> {
  using VecTypeGlobal = SIMDComplex<typename BaseType<T>::Type,8>;
  using VecType = SIMDComplex<typename BaseType<T>::Type,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

  using MaskType = MaskArray<8>;

  static constexpr MaskType NoPermuteMask = {0,1,2,3,4,5,6,7};
  static constexpr MaskType XPermuteMask  = {0,1,2,3,4,5,6,7};
  static constexpr MaskType YPermuteMask  = {1,0,3,2,5,4,7,6};
  static constexpr MaskType ZPermuteMask  = {2,3,0,1,6,7,4,5};
  static constexpr MaskType TPermuteMask  = {4,5,6,7,0,1,2,3};

  static
  KOKKOS_FORCEINLINE_FUNCTION
  VecType permute(const MaskType& mask, const VecTypeGlobal& vec_in)
  {
	  VecType vec_out;
	  Kokkos::parallel_for(VectorPolicy(8),[&](const int& i){
		  ComplexCopy( vec_out(i), vec_in( mask(i) ));
	  });
	  return vec_out;
  }

}; // struct vector length = 8


} // namespace

#if defined(MG_USE_AVX512)

#include <immintrin.h>

namespace MG {

struct PermMaskAVX512 {
 union {
   unsigned int maskdata[16];
   __m512i maskvalue;
 };
};

template<>
struct VNode<MGComplex<float>,8> {

  using VecTypeGlobal = SIMDComplex<float,8>;
  using VecType = SIMDComplex<float,8>;

  static constexpr int VecLen = 8;
  static constexpr int NDim = 3;

  static constexpr int Dim0 = 1;
  static constexpr int Dim1 = 2;
  static constexpr int Dim2 = 2;
  static constexpr int Dim3 = 2;

  using MaskType = PermMaskAVX512;

  // These initializations rely on __m512i being a union type
  static constexpr MaskType NoPermuteMask = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static constexpr MaskType XPermuteMask = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static constexpr MaskType YPermuteMask = {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13};
  static constexpr MaskType ZPermuteMask = {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11};
  static constexpr MaskType TPermuteMask = {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7};

  static
  KOKKOS_FORCEINLINE_FUNCTION
  VecType permute(const MaskType& mask, const VecTypeGlobal& vec_in)
  {
	VecType vec_out;
	vec_out._vdata = _mm512_permutexvar_ps(mask.maskvalue, vec_in._vdata);
	return vec_out;
  }
}; // struct vector length = 8

} // Namespace
#endif // AVX512





