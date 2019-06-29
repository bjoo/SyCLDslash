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
	cl::sycl::vec<T,N> r,i;

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


template<typename T, int N>
inline void ComplexZero(SIMDComplexSyCL<T,N>& result)
{
	cl::sycl::vec<T,N> z(static_cast<T>(0));
	result.real( z );
	result.imag( z );
}

template<typename T, int N>
inline
void ComplexCopy(SIMDComplexSyCL<T,N>& result,
			const SIMDComplexSyCL<T,N>& source)
{
#if 0
	// This fails
	result.real() = source.real();
	result.imag() = source.imag();
#else
	// This works
	result.real(source.real());
	result.imag(source.imag());
#endif
}

} // Namesapce


