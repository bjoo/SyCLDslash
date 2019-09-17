/*
 * dslash_vectype_sycl_subblock.h
 *
 *  Created on: Sep 13, 2019
 *      Author: bjoo
 */

#ifndef INCLUDE_DSLASH_DSLASH_VECTYPE_SYCL_SUBGROUP_H_
#define INCLUDE_DSLASH_DSLASH_VECTYPE_SYCL_SUBGROUP_H_
#include "dslash/dslash_complex.h"
#include "dslash/dslash_vectype_sycl_a.h"
#include <CL/sycl.hpp>
namespace MG {


// A thread private SIMD Complex
template< typename T, unsigned int N>
struct ThreadPrivateSIMDComplex : MGComplex<T> {

	using load_store_type = typename cl::sycl::vec<T,2>;


	constexpr int len(void) const {
	  return N;
	}

	constexpr int num_fp(void) const {
		return (2*N);
	}
};


template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Store(size_t offset, cl::sycl::multi_ptr<T,Space> ptr, const ThreadPrivateSIMDComplex<T,N>& out)
{
	ThreadPrivateSIMDComplex::load_store_type store_vec = { out.real(), out.imag() };
	cl::sycl::intel::sub_group::store(ptr,store_vec);
}

template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Stream(size_t offset, cl::sycl::multi_ptr<T,Space> ptr, const ThreadPrivateSIMDComplex<T,N>& out)
{
	ThreadPrivateSIMDComplex::load_store_type store_vec = { out.real(), out.imag() };
	cl::sycl::intel::sub_group::store(ptr,store_vec);
}



// Call Only in Subblock
template<typename T, int N>
inline
void ComplexCopy(ThreadPrivateSIMDComplex<T,N>& result,
			const SIMDComplexSyCL<T,N>& source)
{
	cl::sycl::id<1> lane = cl::sycl::intel::sub_group::get_local_id();
	size_t lane_id = lane[0];
	result._data = LaneOps<T,N>::extract(source, lane);
}


// Call Only in Subblock
template<typename T, int N>
inline
void ComplexCopy(SIMDComplexSyCL<T,N>& result,
			const ThreadPrivateSIMDComplex<T,N>& source)
{
	cl::sycl::id<1> lane = cl::sycl::intel::sub_group::get_local_id();
	size_t lane_id = lane[0];
	LaneOps<T,N>::insert(result,source, lane);
}





} // Namespace


#endif /* INCLUDE_DSLASH_DSLASH_VECTYPE_SYCL_SUBGROUP_H_ */
