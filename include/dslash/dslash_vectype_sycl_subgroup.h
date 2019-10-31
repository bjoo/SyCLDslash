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

using namespace cl;

namespace MG {


// A thread private SIMD Complex
template< typename T, unsigned int N>
struct ThreadPrivateSIMDComplex : MGComplex<typename BaseType<T>::Type > {


	using load_store_type = typename sycl::vec<typename BaseType<T>::Type,2>;


	constexpr int len(void) const {
	  return N;
	}

	constexpr int num_fp(void) const {
		return (2*N);
	}
};


// Assumption: This function is called in a SUBGROUP
// Store a SIMD from its local type to a pointer and an offset.
template<typename T,sycl::access::address_space Space>
inline void
Store(size_t offset, sycl::multi_ptr<T,Space> ptr, const MGComplex<typename BaseType<T>::Type>& out,
		const sycl::intel::sub_group& sg)
{
	sycl::vec<typename BaseType<T>::Type,2> store_vec = { out.real(), out.imag() };
	sg.store(ptr+offset,store_vec);
}

template<typename T, sycl::access::address_space Space>
inline  MGComplex< typename BaseType<T>::Type>
Load(size_t offset, const sycl::multi_ptr<T,Space> ptr, const sycl::intel::sub_group& sg)
{
	sycl::vec<typename BaseType<T>::Type ,2> load_vec = sg.load<2, typename BaseType<T>::Type,Space>(ptr + offset);
	return MGComplex<typename BaseType<T>::Type>(load_vec.s0(),load_vec.s1());
}




#if 0
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
#endif




} // Namespace


#endif /* INCLUDE_DSLASH_DSLASH_VECTYPE_SYCL_SUBGROUP_H_ */
