/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */
#pragma once

#include <dslash/dslash_complex.h>
#include <complex>
#include <array>

namespace MG
{


template<typename T, int N, template <typename,int> class T1, template <typename,int> class T2>
inline
void ComplexCopy(T1<T,N>& result, const T2<T,N>& source)
{
#pragma omp simd simdlen(N)
	for(int i=0; i < N; ++i) {
      auto _s = source(i);
      result(i) = _s;
    }
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
inline
 void Load(T1<T,N>& result, const T2<T,N>& source)
{

#pragma omp simd
	 for(int i=0; i < N; ++i) {
      auto _s = source(i);
      result(i) = _s;
    }
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
 inline
 void Store(T1<T,N>& result, const T2<T,N>& source)
{
#pragma omp simd
	 for(int i=0; i < N; ++i) {
		 auto _s = source(i);
		 result(i) = _s;;
    }
}

 template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
inline
 void Stream(T1<T,N>& result, const T2<T,N>& source)
 {
#pragma omp simd
	 for(int i=0; i < N; ++i) {
		 auto _s = source(i);
		 result(i) = _s;
	 }
}

template<typename T, int N, template<typename,int> class T1>
inline
void ComplexZero(T1<T,N>& result)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {
		result(i)=MGComplex<T>(0,0);
    }
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
inline
  void
ComplexPeq(T1<T,N>& res, const T2<T,N>& a)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {

	  auto _a = a(i);
      auto _r = res(i);

      res(i) = MGComplex<T>(_r.real() + _a.real() ,_r.imag() + _a.imag());
    }
}


template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
inline void
ComplexCMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  auto _a = a;
#pragma omp simd
	for(int i=0; i <N; ++i) {

      auto _b = b(i);
      auto _res = res(i);


      T res_re =  _res.real();
      res_re +=  _a.real()*_b.real();
      res_re -=  _a.imag()*_b.imag();

      T res_im =  _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  +=  _a.imag()*_b.real();

      res(i) = MGComplex<T>( res_re, res_im);

    }
}


  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
inline
void
ComplexConjMadd(T1<T,N>& res, const MGComplex<T>& a, const T2<T,N>& b)
{
  auto _a = a;
#pragma omp simd
	for(int i=0; i <N; ++i) {

	  auto _b = b(i);
      auto _res = res(i);

      T res_re =  _res.real();
      res_re  += _a.real()*_b.real();
      res_re  += _a.imag()*_b.imag();

      T res_im = _res.imag();
      res_im  +=  _a.real()*_b.imag();
      res_im  -=  _a.imag()*_b.real();

      res(i) = MGComplex<T>(res_re,res_im);

    }

}



  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
inline
  void
ComplexCMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {

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

    }
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
inline
  void
  ComplexConjMadd(T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {

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

    }
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
inline
void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {

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

    }
}

  template<typename T, int N,
    template <typename,int> class T1,
    template<typename,int> class T2,
    template<typename,int> class T3, int sign>
inline
  void A_add_sign_B( T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{

//  printf(".");
  const T fsign = static_cast<T>(sign);

#pragma omp simd
	for(int i=0; i <N; ++i) {

      //      res(i).real() = a(i).real() + sign*b(i).real();
      //      res(i).imag() = a(i).imag() + sign*b(i).imag();
      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real();
      res_re +=  fsign*_b.real();
      T res_im = _a.imag();
      res_im +=  fsign*_b.imag();

      res(i) = MGComplex<T>(res_re,res_im);

    }
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3>
inline
  void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a, const T& sign, const T3<T,N>& b)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {

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

    }
}

  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, template<typename,int> class T3, int sign>
inline
  void A_add_sign_iB( T1<T,N>& res, const T2<T,N>& a, const T3<T,N>& b)
{
  const T fsign=static_cast<T>(sign);

#pragma omp simd
	for(int i=0; i <N; ++i) {


      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real() ;
      res_re -= fsign*_b.imag();
      T res_im = _a.imag();
      res_im += fsign*_b.real();

      res(i) = MGComplex<T>(res_re, res_im);

    }
}

// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
  inline
  void A_peq_sign_miB( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{
#pragma omp simd
	for(int i=0; i <N; ++i) {

      auto _a = a(i);
      auto _b = b(i);
      T res_re = _a.real();
      res_re += sign*_b.imag();
      T res_im = _a.imag();
      res_im -= sign*_b.real();
      a(i) = MGComplex<T>(res_re,res_im );

}
}

// a = -i b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, int sign>
inline
  void A_peq_sign_miB( T1<T,N>& a, const T2<T,N>& b)
{
  const T fsign=static_cast<T>(sign);
#pragma omp simd
	for(int i=0; i <N; ++i) {
      auto _a = a(i);
      auto _b = b(i);
      T res_re = _a.real();
      res_re += fsign*_b.imag();
      T res_im = _a.imag();
      res_im -= fsign*_b.real();
      a(i) = MGComplex<T>(res_re,res_im );

}
}


// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2>
 inline
 void A_peq_sign_B( T1<T,N>& a, const T& sign, const T2<T,N>& b)
{

#pragma omp simd
	for(int i=0; i <N; ++i) {

	  // a(i).real() += sign*b(i).real();
      // a(i).imag() += sign*b(i).imag();

      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real();
      res_re += sign*_b.real();
      T res_im = _a.imag();
      res_im += sign*_b.imag();

      a(i) = MGComplex<T>( res_re,res_im );
    }
}

// a = b
  template<typename T, int N, template <typename,int> class T1, template<typename,int> class T2, int sign>
inline
  void A_peq_sign_B( T1<T,N>& a, const T2<T,N>& b)
{
  const T fsign = static_cast<T>(sign);

  #pragma omp simd
	for(int i=0; i <N; ++i) {

  // a(i).real() += sign*b(i).real();
      // a(i).imag() += sign*b(i).imag();

      auto _a = a(i);
      auto _b = b(i);

      T res_re = _a.real();
      res_re += fsign*_b.real();
      T res_im = _a.imag();
      res_im += fsign*_b.imag();

      a(i) = MGComplex<T>( res_re,res_im );
    }
}
}
