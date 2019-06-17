/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */
#pragma once
#ifndef TEST_KOKKOS_KOKKOS_VECTPYE_H_
#define TEST_KOKKOS_KOKKOS_VECTYPE_H_

#include "kokkos_defaults.h"

#include "MG_config.h"
#if defined(MG_USE_AVX512)
#include <immintrin.h>
#endif

#if defined(KOKKOS_HAVE_CUDA)
#include"cuda.h"
#include "cuda_runtime.h"
#endif


#include <Kokkos_Core.hpp>


namespace MG
{


// General
template<typename T, int N>
  struct  SIMDComplex {
  MGComplex<T> _data[N]; 
  constexpr static int len() { return N; }
  
  KOKKOS_INLINE_FUNCTION
  void set(int l, const MGComplex<T>& value)
  {
    _data[l] = value;
  }
  
  KOKKOS_INLINE_FUNCTION
  const MGComplex<T>& operator()(int i) const
  {
    return _data[i];
  }
  
  KOKKOS_INLINE_FUNCTION
  MGComplex<T>& operator()(int i) {
    return _data[i];
  }
};

 // On the GPU only one elemen per 'VectorThread'
template<typename T, int N>
  struct GPUThreadSIMDComplex {

  MGComplex<T> _data;

  // This is the vector length so still N
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr static  int len() { return N; }
  
  // Ignore l
  KOKKOS_INLINE_FUNCTION
  void set(int l, const MGComplex<T>& value)
  {
    _data = value;
  }
  
  // Ignore i
  KOKKOS_FORCEINLINE_FUNCTION
  const MGComplex<T>& operator()(int i) const
  {
    return _data;
  }
  
  // Ignore i
  KOKKOS_FORCEINLINE_FUNCTION
  MGComplex<T>& operator()(int i) {
    return _data;
  }
};

 
 // THIS IS WHERE WE INTRODUCE SOME NONPORTABILITY
  // ThreadSIMDComplex ***MUST** only be instantiated in 
  // a Kokkos parallel region
#ifdef KOKKOS_HAVE_CUDA
  template<typename T, int N> 
  using ThreadSIMDComplex = GPUThreadSIMDComplex<T,N>;
#else
  template<typename T, int N>
  using ThreadSIMDComplex = SIMDComplex<T,N>;
#endif





#ifndef KOKKOS_HAVE_CUDA

// GENERAL THREADVECTORRANGE
// T1 must support indexing with operator()
  template<typename T, int N, template <typename,int> class T1, template <typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void ComplexCopy(T1<T,N>& result, const T2<T,N>& source)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(i);
      result(i) = _s;
    });
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void Load(T1<T,N>& result, const T2<T,N>& source)
{
  
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(i);
      result(i) = _s;;

    });
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void Store(T1<T,N>& result, const T2<T,N>& source)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(i);
      result(i) = _s;;
    });
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
   KOKKOS_FORCEINLINE_FUNCTION
 void Stream(T1<T,N>& result, const T2<T,N>& source)
 {
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(i);
      result(i) = _s;
    });
}

  template<typename T, int N, template<typename,int> class T1>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(T1<T,N>& result)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(i)=MGComplex<T>(0,0);
    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(T1<T,N>& res, const T2<T,N>& a)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {

      auto _a = a(i);
      auto _r = res(i);

      res(i) = MGComplex<T>(_r.real() + _a.real() ,_r.imag() + _a.imag());
    });
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  auto _a = a;
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {

      auto _b = b(i);
      auto _res = res(i);


      T res_re =  _res.real();
      res_re +=  _a.real()*_b.real();
      res_re -=  _a.imag()*_b.imag();

      T res_im =  _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  +=  _a.imag()*_b.real();

      res(i) = MGComplex<T>( res_re, res_im);

    });
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  auto _a = a;
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {

      auto _b = b(i);
      auto _res = res(i);


      T res_re =  _res.real();
      res_re  += _a.real()*_b.real();
      res_re  += _a.imag()*_b.imag();

      T res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(i) = MGComplex<T>(res_re,res_im);

    });

}



  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      auto _b = b(i);
      auto _res = res(i);
      auto _a = a(i);

      T res_re = _res.real();
      res_re += _a.real()*_b.real();
      res_re -= _a.imag()*_b.imag();
      T res_im =  _res.imag();
      res_im +=  _a.real()*_b.imag();
      res_im +=  _a.imag()*_b.real();

      res(i) = MGComplex<T>( res_re, res_im);

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void
  ComplexConjMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      auto _b = b(i);
      auto _res = res(i);
      auto _a = a(i);

      T res_re =  _res.real();
      res_re +=   _a.real()*_b.real(); 
      res_re +=   _a.imag()*_b.imag();
      T res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(i) = MGComplex<T>(res_re,res_im);

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      //      res(i).real() = a(i).real() + sign*b(i).real();
      //      res(i).imag() = a(i).imag() + sign*b(i).imag();
      auto _a = a(i);
      auto _b = b(i);
      auto _res = res(i);

      T res_re = _a.real();
      res_re +=  sign*_b.real();
      T res_im = _a.imag();
      res_im +=  sign*_b.imag();

      res(i) = MGComplex<T>(res_re,res_im);

    });
}

  template<typename T, int N, 
    template <typename,int> class T1, 
    template<typename,int> class T2, 
    template<typename,int> class T3, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{

//  printf(".");
  const T fsign = static_cast<T>(sign);
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      //      res(i).real() = a(i).real() + sign*b(i).real();
      //      res(i).imag() = a(i).imag() + sign*b(i).imag();
      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real();
      res_re +=  fsign*_b.real();
      T res_im = _a.imag();
      res_im +=  fsign*_b.imag();

      res(i) = MGComplex<T>(res_re,res_im);

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      //res(i).real() = a(i).real() - sign*b(i).imag();
      //res(i).imag() = a(i).imag() + sign*b(i).real();

      auto _a = a(i);
      auto _b = b(i);
      auto _res = res(i);

      T res_re = _a.real() ;
      res_re -= sign*_b.imag();
      T res_im = _a.imag();
      res_im += sign*_b.real();

  res(i) = MGComplex<T>(res_re, res_im);

    });
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
  const T fsign=static_cast<T>(sign);

  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {

      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real() ;
      res_re -= fsign*_b.imag();
      T res_im = _a.imag();
      res_im += fsign*_b.real();

      res(i) = MGComplex<T>(res_re, res_im);

    });
}

// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {

      auto _a = a(i);
      auto _b = b(i);
      T res_re = _a.real();
      res_re += sign*_b.imag();
      T res_im = _a.imag();
      res_im -= sign*_b.real();
      a(i) = MGComplex<T>(res_re,res_im );
  
});
}

// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( T1<T,N>& a, const T2<T,N>& b)
{
  const T fsign=static_cast<T>(sign);
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {

      auto _a = a(i);
      auto _b = b(i);
      T res_re = _a.real();
      res_re += fsign*_b.imag();
      T res_im = _a.imag();
      res_im -= fsign*_b.real();
      a(i) = MGComplex<T>(res_re,res_im );
  
});
}

    
// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
  void A_peq_sign_B( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      // a(i).real() += sign*b(i).real();
      // a(i).imag() += sign*b(i).imag();

      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real();
      res_re += sign*_b.real();
      T res_im = _a.imag();
      res_im += sign*_b.imag();

      a(i) = MGComplex<T>( res_re,res_im );
    });
}

// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, int sign>
KOKKOS_FORCEINLINE_FUNCTION
  void A_peq_sign_B( T1<T,N>& a, const T2<T,N>& b)
{
  const T fsign = static_cast<T>(sign);
  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) { 
      // a(i).real() += sign*b(i).real();
      // a(i).imag() += sign*b(i).imag();

      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real();
      res_re += fsign*_b.real();
      T res_im = _a.imag();
      res_im += fsign*_b.imag();

      a(i) = MGComplex<T>( res_re,res_im );
    });
}
#else

 // Hacked for CUDA just work with thread IDX.x
// T1 must support indexing with operator()
  template<typename T, int N, template <typename,int> class T1, template <typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void ComplexCopy(T1<T,N>& result, const T2<T,N>& source)
{
  // Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(threadIdx.x);
      result(threadIdx.x) = _s;
      // });
}

 template<>
 KOKKOS_FORCEINLINE_FUNCTION
 void ComplexCopy<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>&result, const SIMDComplex<float,1>& source)
 {
	result(0) = source(0);
 } 

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void Load(T1<T,N>& result, const T2<T,N>& source)
{
  
  //Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(threadIdx.x);
      result(threadIdx.x) = _s;;

      //});
}

template<>
 KOKKOS_FORCEINLINE_FUNCTION
 void Load<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>&result, const SIMDComplex<float,1>& source)
 {
        result(0) = source(0);
 } 


 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
   void Store(T1<T,N>& result, const T2<T,N>& source)
{
  //  Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(threadIdx.x);
      result(threadIdx.x) = _s;;
      // });
}

template<>
 KOKKOS_FORCEINLINE_FUNCTION
 void Store<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>&result, const SIMDComplex<float,1>& source)
 {
        result(0) = source(0); 
 }

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
   KOKKOS_FORCEINLINE_FUNCTION
 void Stream(T1<T,N>& result, const T2<T,N>& source)
 {
   // Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      auto _s = source(threadIdx.x);
      result(threadIdx.x) = _s;
      //});
}

template<>
 KOKKOS_FORCEINLINE_FUNCTION 
 void Stream<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>&result, const SIMDComplex<float,1>& source)
 {
        result(0) = source(0);
 } 


  template<typename T, int N, template<typename,int> class T1>
KOKKOS_FORCEINLINE_FUNCTION
void ComplexZero(T1<T,N>& result)
{
  // Kokkos::parallel_for(VectorPolicy(N),[&](const int& i) {
      result(threadIdx.x)=MGComplex<T>(0,0);
      // });
}

template<>
 KOKKOS_FORCEINLINE_FUNCTION
 void ComplexZero<float,1,SIMDComplex>(SIMDComplex<float,1>&result)
 {
        result(0) = MGComplex<float>(0,0);
 }


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq(T1<T,N>& res, const T2<T,N>& a)
{
      auto _a = a(threadIdx.x);
      auto _r = res(threadIdx.x);
	res(threadIdx.x) = MGComplex<T>(_r.real() + _a.real() ,_r.imag() + _a.imag());
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexPeq<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a)
{
      auto _a = a(0);
      auto _r = res(0);
      res(0) = MGComplex<float>(_r.real() + _a.real() ,_r.imag() + _a.imag());
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  auto _a = a;
      auto _b = b(threadIdx.x);
      auto _res = res(threadIdx.x);


      T res_re =  _res.real();
      res_re +=  _a.real()*_b.real();
      res_re -=  _a.imag()*_b.imag();

      T res_im =  _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  +=  _a.imag()*_b.real();

      res(threadIdx.x) = MGComplex<T>( res_re, res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>& res, const MGComplex<float>& a, const SIMDComplex<float,1>& b)
{
      auto _a = a;
      auto _b = b(0);
      auto _res = res(0);


      float res_re =  _res.real();
      res_re +=  _a.real()*_b.real();
      res_re -=  _a.imag()*_b.imag();

      float res_im =  _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  +=  _a.imag()*_b.real();

      res(0) = MGComplex<float>( res_re, res_im);
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  auto _a = a;
  auto _b = b(threadIdx.x);
  auto _res = res(threadIdx.x);


      T res_re =  _res.real();
      res_re  += _a.real()*_b.real();
      res_re  += _a.imag()*_b.imag();

      T res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(threadIdx.x) = MGComplex<T>(res_re,res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd<float,1,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>& res, const MGComplex<float>& a, const SIMDComplex<float,1>& b)
{
  auto _a = a;
  auto _b = b(0);
  auto _res = res(0);


      float res_re =  _res.real();
      res_re  += _a.real()*_b.real();
      res_re  += _a.imag()*_b.imag();

      float res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(0) = MGComplex<float>(res_re,res_im);
}



  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
      auto _b = b(threadIdx.x);
      auto _res = res(threadIdx.x);
      auto _a = a(threadIdx.x);

      T res_re = _res.real();
      res_re += _a.real()*_b.real();
      res_re -= _a.imag()*_b.imag();
      T res_im =  _res.imag();
      res_im +=  _a.real()*_b.imag();
      res_im +=  _a.imag()*_b.real();

      res(threadIdx.x) = MGComplex<T>( res_re, res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd<float,1,SIMDComplex,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a, const SIMDComplex<float,1>& b)
{
      auto _b = b(0);
      auto _res = res(0);
      auto _a = a(0);

      float res_re = _res.real();
      res_re += _a.real()*_b.real();
      res_re -= _a.imag()*_b.imag();
      float res_im =  _res.imag();
      res_im +=  _a.real()*_b.imag();
      res_im +=  _a.imag()*_b.real();

      res(0) = MGComplex<float>( res_re, res_im);
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void
  ComplexConjMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
      auto _b = b(threadIdx.x);
      auto _res = res(threadIdx.x);
      auto _a = a(threadIdx.x);

      T res_re =  _res.real();
      res_re +=   _a.real()*_b.real(); 
      res_re +=   _a.imag()*_b.imag();
      T res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(threadIdx.x) = MGComplex<T>(res_re,res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd<float,1,SIMDComplex,SIMDComplex,SIMDComplex>(SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a, const SIMDComplex<float,1>& b)
{
      auto _b = b(0);
      auto _res = res(0);
      auto _a = a(0);

      float res_re =  _res.real();
      res_re +=   _a.real()*_b.real();
      res_re +=   _a.imag()*_b.imag();
      float res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(0) = MGComplex<float>(res_re,res_im);
}


template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{

      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);

      T res_re = _a.real();
      res_re +=  sign*_b.real();
      T res_im = _a.imag();
      res_im +=  sign*_b.imag();

      res(threadIdx.x) = MGComplex<T>(res_re,res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B<float,1,SIMDComplex,SIMDComplex,SIMDComplex>( SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a, const float& sign, 
	const SIMDComplex<float,1>& b)
{

      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real();
      res_re +=  sign*_b.real();
      float res_im = _a.imag();
      res_im +=  sign*_b.imag();

      res(0) = MGComplex<float>(res_re,res_im);
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3, int sign >
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a,  const T3<T,N>& b)
{
      const T fsign = static_cast<T>(sign);
      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);

      T res_re = _a.real();
      res_re +=  fsign*_b.real();
      T res_im = _a.imag();
      res_im +=  fsign*_b.imag();

      res(threadIdx.x) = MGComplex<T>(res_re,res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B<float,1,SIMDComplex,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a,  const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real();
      res_re +=  _b.real();
      float res_im = _a.imag();
      res_im +=  _b.imag();

      res(0) = MGComplex<float>(res_re,res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_B<float,1,SIMDComplex,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a,  const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);
      
      float res_re = _a.real();
      res_re -=  _b.real();
      float res_im = _a.imag();
      res_im -=  _b.imag();

      res(0) = MGComplex<float>(res_re,res_im);
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);

      T res_re = _a.real() ;
      res_re -= sign*_b.imag();
      T res_im = _a.imag();
      res_im += sign*_b.real();

  res(threadIdx.x) = MGComplex<T>(res_re, res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB<float,1,SIMDComplex,SIMDComplex,SIMDComplex>( SIMDComplex<float,1>& res, const SIMDComplex<float,1>& a, 
	const float& sign, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real() ;
      res_re -= sign*_b.imag();
      float res_im = _a.imag();
      res_im += sign*_b.real();

  res(0) = MGComplex<float>(res_re, res_im);
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a,  const T3<T,N>& b)
{
  const T fsign = static_cast<T>(sign);
      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);

      T res_re = _a.real() ;
      res_re -= fsign*_b.imag();
      T res_im = _a.imag();
      res_im += fsign*_b.real();

  res(threadIdx.x) = MGComplex<T>(res_re, res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB<float,1,SIMDComplex,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,1>& res, 
	const SIMDComplex<float,1>& a,  const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real() ;
      res_re -= _b.imag();
      float res_im = _a.imag();
      res_im += _b.real();

  res(0) = MGComplex<float>(res_re, res_im);
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB<float,1,SIMDComplex,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,1>& res, 
        const SIMDComplex<float,1>& a,  const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real() ;
      res_re += _b.imag();
      float res_im = _a.imag();
      res_im -= _b.real();

  res(0) = MGComplex<float>(res_re, res_im);
}

// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);
      T res_re = _a.real();
      res_re += sign*_b.imag();
      T res_im = _a.imag();
      res_im -= sign*_b.real();
      a(threadIdx.x) = MGComplex<T>(res_re,res_im );
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,1,SIMDComplex,SIMDComplex>( SIMDComplex<float,1>& a, const float& sign, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);
      float res_re = _a.real();
      res_re += sign*_b.imag();
      float res_im = _a.imag();
      res_im -= sign*_b.real();
      a(0) = MGComplex<float>(res_re,res_im );
}



// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, int sign>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB( T1<T,N>& a, const T2<T,N>& b)
{

  const T fsign = static_cast<T>(sign);

      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);
      T res_re = _a.real();
      res_re += fsign*_b.imag();
      T res_im = _a.imag();
      res_im -= fsign*_b.real();
      a(threadIdx.x) = MGComplex<T>(res_re,res_im );
  
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,1,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,1>& a, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);
      float res_re = _a.real();
      res_re += _b.imag();
      float res_im = _a.imag();
      res_im -= _b.real();
      a(0) = MGComplex<float>(res_re,res_im );

}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,1,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,1>& a, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);
      float res_re = _a.real();
      res_re -= _b.imag();
      float res_im = _a.imag();
      res_im += _b.real();
      a(0) = MGComplex<float>(res_re,res_im );

}
 
// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
KOKKOS_FORCEINLINE_FUNCTION
    void A_peq_sign_B( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);

      T res_re = _a.real();
      res_re += sign*_b.real();
      T res_im = _a.imag();
      res_im += sign*_b.imag();

      a(threadIdx.x) = MGComplex<T>( res_re,res_im );

}   

template<>
KOKKOS_FORCEINLINE_FUNCTION
    void A_peq_sign_B<float,1,SIMDComplex,SIMDComplex>( SIMDComplex<float,1>& a, const float& sign, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real();
      res_re += sign*_b.real();
      float res_im = _a.imag();
      res_im += sign*_b.imag();

      a(0) = MGComplex<float>( res_re,res_im );

}


// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, int sign>
KOKKOS_FORCEINLINE_FUNCTION
  void A_peq_sign_B( T1<T,N>& a, const T2<T,N>& b)
{
  const T fsign = static_cast<T>(sign);
      auto _a = a(threadIdx.x);
      auto _b = b(threadIdx.x);

      T res_re = _a.real();
      res_re += fsign*_b.real();
      T res_im = _a.imag();
      res_im += fsign*_b.imag();

      a(threadIdx.x) = MGComplex<T>( res_re,res_im );

}


  template<>
KOKKOS_FORCEINLINE_FUNCTION
  void A_peq_sign_B<float,1,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,1>& a, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);

      float res_re = _a.real();
      res_re += _b.real();
      float res_im = _a.imag();
      res_im += _b.imag();

      a(0) = MGComplex<float>( res_re,res_im );

}

  template<>
KOKKOS_FORCEINLINE_FUNCTION
  void A_peq_sign_B<float,1,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,1>& a, const SIMDComplex<float,1>& b)
{
      auto _a = a(0);
      auto _b = b(0);
  
      float res_re = _a.real();
      res_re -= _b.real();
      float res_im = _a.imag();
      res_im -= _b.imag();
      
      a(0) = MGComplex<float>( res_re,res_im );
      
}

#endif

 


#if defined(MG_USE_AVX512)

// ----***** SPECIALIZED *****
template<>
  struct SIMDComplex<float,8> {


  explicit SIMDComplex<float,8>() {}
  SIMDComplex<float,8>(const SIMDComplex<float,8>& in)
  {
	  _vdata = in._vdata;
  }

  SIMDComplex<float,8>& operator=(const SIMDComplex<float,8>& in)
  {
	  _vdata = in._vdata;
	  return (*this);
  }

  __m512 _vdata;

  constexpr static int len() { return 8; }

  inline
    void set(int l, const MGComplex<float>& value)
  {
      MGComplex<float> *data = reinterpret_cast<MGComplex<float>*>(&_vdata);
      data[l] = value;
  }

  inline
    const MGComplex<float>& operator()(int i) const
  { 
     const MGComplex<float>* data = reinterpret_cast<const MGComplex<float>*>(&_vdata);  
    return data[i];
  }

  inline
  MGComplex<float>& operator()(int i) {
    MGComplex<float>* data  = reinterpret_cast<MGComplex<float>*>(&_vdata);
    return data[i];
  }
};

 

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Load<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
		     const SIMDComplex<float,8>& source)
  {
    void const* src = reinterpret_cast<void const*>(&(source._vdata));
    
    result._vdata = _mm512_load_ps(src);
  }
  
  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void ComplexCopy<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
			    const SIMDComplex<float,8>& source)
  {
    result._vdata  = source._vdata;
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Store<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
		      const SIMDComplex<float,8>& source)
  {
    void* dest = reinterpret_cast<void*>(&(result._vdata));
    _mm512_store_ps(dest,source._vdata);
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void Stream<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& result, 
		      const SIMDComplex<float,8>& source)
  {
    void* dest = reinterpret_cast<void*>(&(result._vdata));
     _mm512_stream_ps(dest,source._vdata);
   // _mm512_store_ps(dest,source._vdata);
  }
  

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void ComplexZero<float,8,SIMDComplex>(SIMDComplex<float,8>& result)
  {
    result._vdata = _mm512_setzero_ps();
  }

  template<>
  KOKKOS_FORCEINLINE_FUNCTION
  void
  ComplexPeq<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, 
		      const SIMDComplex<float,8>& a)
  {
    res._vdata = _mm512_add_ps(res._vdata,a._vdata);
  }



template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
A_add_sign_B<float,8,SIMDComplex,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& res, 
			    const SIMDComplex<float,8>& a, 
			    const float& sign, 
			    const SIMDComplex<float,8>& b)
{
  __m512 sgnvec = _mm512_set1_ps(sign);
  res._vdata = _mm512_fmadd_ps(sgnvec,b._vdata,a._vdata);
}

// sign == 1
template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
  A_add_sign_B<float,8,SIMDComplex,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,8>& res, 
			    const SIMDComplex<float,8>& a, 
			    const SIMDComplex<float,8>& b)
{
  res._vdata = _mm512_add_ps(a._vdata,b._vdata);
}

// sign == -1 
template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
  A_add_sign_B<float,8,SIMDComplex,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,8>& res, 
			    const SIMDComplex<float,8>& a, 
			    const SIMDComplex<float,8>& b)
{
  res._vdata = _mm512_sub_ps(a._vdata,b._vdata);
}



// a += b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
A_peq_sign_B<float,8,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& a, 
		   const float& sign, 
		   const SIMDComplex<float,8>& b)
{
  __m512 sgnvec=_mm512_set1_ps(sign);
  a._vdata = _mm512_fmadd_ps( sgnvec, b._vdata, a._vdata);
}

// a += b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
  A_peq_sign_B<float,8,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,8>& a, 
						   const SIMDComplex<float,8>& b)
{
  a._vdata = _mm512_add_ps(a._vdata, b._vdata);
}

// a -= b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void 
  A_peq_sign_B<float,8,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,8>& a, 
						   const SIMDComplex<float,8>& b)
{
  a._vdata = _mm512_sub_ps(a._vdata, b._vdata);
}


template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, 
	     const MGComplex<float>& a,
	     const SIMDComplex<float,8>& b)
{
  __m512 a_vec_re = _mm512_set1_ps( a.real() );
  __m512 a_vec_im = _mm512_set1_ps( a.imag() );
  __m512 b_perm = _mm512_shuffle_ps( b._vdata, b._vdata,0xb1 );
  __m512 t = _mm512_fmaddsub_ps( a_vec_im, b_perm, res._vdata);
  res._vdata = _mm512_fmaddsub_ps( a_vec_re, b._vdata, t );
}

  template<>
KOKKOS_FORCEINLINE_FUNCTION
void
  ComplexConjMadd<float,8,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, const MGComplex<float>& a,
		const SIMDComplex<float,8>& b)
{

  __m512 a_vec_re = _mm512_set1_ps( a.real() );
  __m512 a_vec_im = _mm512_set1_ps( a.imag() );
  __m512 b_perm = _mm512_shuffle_ps(b._vdata,b._vdata, 0xb1);
  __m512 t = _mm512_fmsubadd_ps(a_vec_im, b_perm, res._vdata);
  res._vdata = _mm512_fmsubadd_ps( a_vec_re, b._vdata, t);

}


template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexCMadd<float,8,SIMDComplex,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res, 
		      const SIMDComplex<float,8>& a, 
		      const SIMDComplex<float,8>& b)
{
  __m512 a_vec_re = _mm512_shuffle_ps( a._vdata, a._vdata, 0xa0 );
  __m512 a_vec_im = _mm512_shuffle_ps( a._vdata, a._vdata, 0xf5 );
  __m512 b_perm = _mm512_shuffle_ps( b._vdata, b._vdata,0xb1 );
  __m512 t = _mm512_fmaddsub_ps( a_vec_im, b_perm, res._vdata);
  res._vdata = _mm512_fmaddsub_ps( a_vec_re, b._vdata, t );
}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void
ComplexConjMadd<float,8,SIMDComplex,SIMDComplex,SIMDComplex>(SIMDComplex<float,8>& res,
		const SIMDComplex<float,8>& a,
		const SIMDComplex<float,8>& b)
{
  __m512 a_vec_re = _mm512_shuffle_ps( a._vdata, a._vdata, 0xa0 );
  __m512 a_vec_im = _mm512_shuffle_ps( a._vdata, a._vdata, 0xf5 );
  __m512 b_perm = _mm512_shuffle_ps(b._vdata,b._vdata, 0xb1);
  __m512 t = _mm512_fmsubadd_ps(a_vec_im, b_perm, res._vdata);
  res._vdata = _mm512_fmsubadd_ps( a_vec_re, b._vdata, t);

}

template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_add_sign_iB<float,8,SIMDComplex,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& res, 
			     const SIMDComplex<float,8>& a, 
			     const float& sign, 
			     const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  __m512 sgnvec = _mm512_set1_ps(sign);
  res._vdata = _mm512_mul_ps(sgnvec,_mm512_fmaddsub_ps( sgnvec, a._vdata, perm_b));
}

// sign == 1
template<>
KOKKOS_FORCEINLINE_FUNCTION
  void A_add_sign_iB<float,8,SIMDComplex,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,8>& res, 
								     const SIMDComplex<float,8>& a, 
								     const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  res._vdata = _mm512_fmaddsub_ps( _mm512_set1_ps(1), a._vdata, perm_b);
}

// sign == -1
template<>
KOKKOS_FORCEINLINE_FUNCTION
  void A_add_sign_iB<float,8,SIMDComplex,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,8>& res, 
								     const SIMDComplex<float,8>& a, 
								     const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  res._vdata = _mm512_fmsubadd_ps( _mm512_set1_ps(1),a._vdata, perm_b);
}


// a = -i b
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,8,SIMDComplex,SIMDComplex>( SIMDComplex<float,8>& a, 
			      const float& sign, 
			      const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  __m512 sgnvec =_mm512_set1_ps(sign);
  a._vdata = _mm512_mul_ps( sgnvec, _mm512_fmsubadd_ps(sgnvec, a._vdata, perm_b));
}

// a += -i b or a -= ib
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,8,SIMDComplex,SIMDComplex,1>( SIMDComplex<float,8>& a, 
							const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  a._vdata = _mm512_fmsubadd_ps(_mm512_set1_ps(1),a._vdata, perm_b);;
}

// a -= -i b or a += ib
template<>
KOKKOS_FORCEINLINE_FUNCTION
void A_peq_sign_miB<float,8,SIMDComplex,SIMDComplex,-1>( SIMDComplex<float,8>& a, 
							const SIMDComplex<float,8>& b)
{
  __m512 perm_b = _mm512_shuffle_ps( b._vdata, b._vdata, 0xb1);
  a._vdata = _mm512_fmaddsub_ps(_mm512_set1_ps(1),a._vdata, perm_b);;
}

#endif



} // namespace



#endif /* TEST_KOKKOS_KOKKOS_OPS_H_ */
