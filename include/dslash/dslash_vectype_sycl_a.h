/*
 * dslash_vectype_sycl_b.h
 *
 *  Created on: Jun 28, 2019
 *      Author: bjoo
 */

#pragma once
#include <dslash/dslash_complex.h>
#include <CL/sycl.hpp>

namespace MG {

template<typename T, int N>
using SIMDComplexSyCL = std::complex< typename cl::sycl::vec<T,N> >;

template<typename T, int N, template <typename,int> class SIMD>
struct VectorTraits
{};

template<typename T, int N>
struct VectorTraits<T,N,SIMDComplexSyCL> {
	static constexpr int len() { return N; }
	static constexpr int num_fp() { return 2*N; }
	using BaseType = T;
	using VecType = cl::sycl::vec<T,N>;
};

template<typename T, int N>
static constexpr int len( const SIMDComplexSyCL<T,N>& a)
{
	return N;
}

template<typename T, int N>
static constexpr int num_fp( const SIMDComplexSyCL<T,N>& a)
{
	return 2*N;
}



//! FIXME: These guys should take accessors and derive their own
//  pointers? Then we could maybe use enable_if<> to check that
//  the accessors are appropriate e.g. read/read_write for load
//  write/read_write etc for Store, discard versions for stream?

template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Load(SIMDComplexSyCL<T,N>& result, size_t offset, cl::sycl::multi_ptr<T,Space> ptr)
{
#if 1
	// Works
	using VecType =  typename VectorTraits<T,N, SIMDComplexSyCL>::VecType;

	VecType r,i;

	r.load(2*offset,ptr);
	i.load(2*offset+1,ptr);

	/* This fails:
	 *
		result.real() = r;
		result.imag() = i;
	*/

	/* This works:
	 *
	 */
	result.real(r);
	result.imag(i);
#else

	// DOesn't work: Why? Bug? or Feature?

	(result.real()).load(2*offset,ptr);
	(result.real()).load(2*offset+1,ptr);
#endif
}

template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Store(size_t offset, cl::sycl::multi_ptr<T,Space> ptr, const SIMDComplexSyCL<T,N>& out)
{

	(out.real()).store(2*offset,ptr);
	(out.imag()).store(2*offset+1,ptr);
}

template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Stream(size_t offset, cl::sycl::multi_ptr<T,Space> ptr, const SIMDComplexSyCL<T,N>& out)
{

	(out.real()).store(2*offset,ptr);
	(out.imag()).store(2*offset+1,ptr);
}

template<typename T, int N, cl::sycl::access::mode mode,
	cl::sycl::access::target target=cl::sycl::access::target::global_buffer,
	cl::sycl::access::placeholder isPlaceHolder=cl::sycl::access::placeholder::false_t>
MGComplex<T> LoadLane(size_t lane, size_t vector,
		cl::sycl::accessor<T,1,mode,target,isPlaceHolder> buf_access)
{
	MGComplex<T> res(static_cast<T>(0),static_cast<T>(0));
	res.real( buf_access[ N*(2*vector) + lane] );
	res.imag( buf_access[ N*(2*vector + 1) + lane]);
	return res;
}

template<typename T, int N, cl::sycl::access::mode mode,
	cl::sycl::access::target target=cl::sycl::access::target::global_buffer,
	cl::sycl::access::placeholder isPlaceHolder=cl::sycl::access::placeholder::false_t>
void StoreLane(size_t lane, size_t vector,
		cl::sycl::accessor<T,1,mode,target,isPlaceHolder> buf_access, const MGComplex<T>& value)
{
	buf_access[ N*(2*vector) + lane]= value.real();
	buf_access[ N*(2*vector + 1) + lane] = value.imag();
}

template<typename T, int N>
inline void ComplexZero(SIMDComplexSyCL<T,N>& result)
{
	using VecType = typename VectorTraits<T,N,SIMDComplexSyCL>::VecType;
	result.real( VecType(static_cast<T>(0)) );
	result.imag( VecType(static_cast<T>(0)) );
}

template<typename T, int N>
inline
void ComplexCopy(SIMDComplexSyCL<T,N>& result,
			const SIMDComplexSyCL<T,N>& source)
{
	// This works
	result.real(source.real());
	result.imag(source.imag());
}

template<typename T, int N>
inline
void
ComplexCMadd(SIMDComplexSyCL<T,N>& res, const MGComplex<T>& a, const SIMDComplexSyCL<T,N>& b)
{
	// Works
   res.real( res.real() + a.real()*b.real() - a.imag()*b.imag());
   res.imag( res.imag() + a.real()*b.imag() + a.imag()*b.real());
}

template<typename T, int N>
inline
void
ComplexCMadd(SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{
	// Works
	using VecType =  typename VectorTraits<T,N, SIMDComplexSyCL>::VecType;

   res.real( res.real() + a.real()*b.real()  - a.imag()*b.imag());
   res.imag( res.imag() + a.real()*b.imag()  + a.imag()*b.real());
}


template<typename T, int N>
inline
void
ComplexPeq(SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a)
{

	res.real( res.real() + a.real() );
	res.imag( res.imag() + a.imag() );

}

template<typename T, int N>
inline
void
ComplexConjMadd(SIMDComplexSyCL<T,N>& res, const MGComplex<T>& a, const SIMDComplexSyCL<T,N>& b)
{
	res.real( res.real() + a.real()*b.real()  + a.imag()*b.imag() );
    res.imag( res.imag() + a.real()*b.imag()  - a.imag()*b.real() );
}

template<typename T, int N>
inline
void
ComplexConjMadd(SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{
	res.real( res.real() + a.real()*b.real() + a.imag()*b.imag() );
    res.imag( res.imag() + a.real()*b.imag() - a.imag()*b.real() );
}

template<typename T, int N>
inline
void A_add_sign_B( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a,
			const T& sign, const SIMDComplexSyCL<T,N>& b)
{
	res.real( a.real() + sign*b.real() );
	res.imag( a.imag() + sign*b.imag() );
}

template<typename T, int N, int sign>
inline
void A_add_sign_B( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a,
			const SIMDComplexSyCL<T,N>& b)
{
  const T fsign = static_cast<T>(sign);
   res.real( a.real() + fsign*b.real() );
   res.imag( a.imag() + fsign*b.imag() );
}

template<typename T, int N>
inline
void A_add_B( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a,
			const SIMDComplexSyCL<T,N>& b)
{
   res.real( a.real() + b.real() );
   res.imag( a.imag() + b.imag() );
}

template<typename T, int N>
inline
void A_sub_B( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a,
			const SIMDComplexSyCL<T,N>& b)
{
   res.real( a.real() - b.real() );
   res.imag( a.imag() - b.imag() );
}


template<typename T, int N>
inline
void A_add_sign_iB( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a, const T& sign, const SIMDComplexSyCL<T,N>& b)
{
  res.real( a.real()-sign*b.imag() );
  res.imag( a.imag()+sign*b.real() );
}

 template<typename T, int N, int sign>
inline
void A_add_sign_iB( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{

  const T fsign=static_cast<T>(sign);
  res.real( a.real()-sign*b.imag() );
  res.imag( a.imag()+sign*b.real() );
}

 template<typename T, int N>
inline
void A_add_iB( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{
	 res.real( a.real()-b.imag() );
	 res.imag( a.imag()+b.real() );
}

 template<typename T, int N>
inline
void A_sub_iB( SIMDComplexSyCL<T,N>& res, const SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{
	res.real( a.real() + b.imag() );
	res.imag( a.imag() - b.real() );
}

// a = -i b
template<typename T, int N>
inline
void A_peq_sign_miB( SIMDComplexSyCL<T,N>& a, const T& sign, const SIMDComplexSyCL<T,N>& b)
{
	a.real(a.real() + sign*b.imag());
    a.imag(a.imag() - sign*b.real());
}

// a = -i b
 template<typename T, int N, int sign>
inline
void A_peq_sign_miB( SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{
  const T fsign = static_cast<T>(sign);
  a.real(a.real() + fsign*b.imag());
  a.imag(a.imag() - fsign*b.real());
}

 // a = -i b
 template<typename T, int N>
 inline
 void A_peq_miB( SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
 {
   a.real(a.real() + b.imag());
   a.imag(a.imag() - b.real());
 }

 // a = -i b
 template<typename T, int N>
 inline
 void A_meq_miB( SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
 {
   a.real(a.real() - b.imag());
   a.imag(a.imag() + b.real());
 }


// a = b
template<typename T, int N>
inline
void A_peq_sign_B( SIMDComplexSyCL<T,N>& a, const T& sign, const SIMDComplexSyCL<T,N>& b)
{
  a.real( a.real() + sign*b.real() );
  a.imag( a.imag() + sign*b.imag() );
}

// a = b
template<typename T, int N, int sign>
inline
void A_peq_sign_B( SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{
	const T fsign = static_cast<T>(sign);

	a.real( a.real() + sign*b.real() );
	a.imag( a.imag() + sign*b.imag() );
}

// a = b
template<typename T, int N>
inline
void A_peq_B( SIMDComplexSyCL<T,N>& a,  const SIMDComplexSyCL<T,N>& b)
{
	a.real( a.real() + b.real() );
	a.imag( a.imag() + b.imag() );
}

// a = b
template<typename T, int N>
inline
void A_meq_B( SIMDComplexSyCL<T,N>& a, const SIMDComplexSyCL<T,N>& b)
{


	a.real( a.real() - b.real() );
	a.imag( a.imag() - b.imag() );
}


} // Namesapce


