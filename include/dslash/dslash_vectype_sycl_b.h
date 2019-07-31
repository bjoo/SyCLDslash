/*
 * dslash_vectype_sycl.h
 *
 *  Created on: Jun 24, 2019
 *      Author: bjoo
 */

#pragma once
#include <dslash/dslash_complex.h>
#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

namespace MG {


template<typename T, int N>
struct SIMDComplexSyCL {
	SIMDComplexSyCL() { static_assert( (N != 1)
			&& (N != 2)
			&& (N != 4)
			&& (N != 8), "SIMDComplex General N not allowed");
	}

};


 // Partial specializations, to allow
 // N-dependent masks
 template<typename T>
  struct SIMDComplexSyCL<T,1> {

	 static constexpr int len() { return 1; }
 	 static constexpr int num_fp() { return 2*len(); }
 	 using ElemType = cl::sycl::vec<T,num_fp()>;
 	 ElemType _data;

 	 static constexpr ElemType mask_even() {
 		 return ElemType(static_cast<T>(1),static_cast<T>(-1));
 	 }

 	 static constexpr ElemType mask_odd() {
 		 return ElemType(static_cast<T>(-1),static_cast<T>(1));
 	 }

 	 static constexpr ElemType permute_evenodd(const ElemType& t) {
 		 return ElemType( t.template swizzle<1,0>() );
 	 }

 	 static constexpr ElemType real_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<0,0>() );
 	 }

 	 static constexpr ElemType imag_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<1,1>() );
 	 }
  };


 template<typename T>
 struct SIMDComplexSyCL<T,2> {
	 static constexpr int len() { return 2; }
	 static constexpr int num_fp() { return 2*len(); }
	 using ElemType = cl::sycl::vec<T,num_fp()>;
	 ElemType _data;

	 static constexpr ElemType mask_even() {
		 return ElemType(static_cast<T>(1),static_cast<T>(-1),
				         static_cast<T>(1),static_cast<T>(-1));
	 }
	 static constexpr ElemType mask_odd() {
		 return ElemType(static_cast<T>(-1),static_cast<T>(1),
				         static_cast<T>(-1),static_cast<T>(1));
	 }

	 static constexpr ElemType permute_evenodd(const ElemType& t) {
 		 return ElemType( t.template swizzle<1,0,3,2>() );
 	 }

 	 static constexpr ElemType real_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<0,0,2,2>() );
 	 }

 	 static constexpr ElemType imag_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<1,1,3,3>() );
 	 }
 };

 template<typename T>
   struct SIMDComplexSyCL<T,4> {
  	 static constexpr int len() { return 4; }
  	 static constexpr int num_fp() { return 2*len(); }
  	 using ElemType = cl::sycl::vec<T,num_fp()>;
  	 ElemType _data;

  	 static constexpr ElemType mask_even() {
  		 return ElemType(static_cast<T>(1),static_cast<T>(-1),
  				 	 	 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1));
  	 }

 	 static constexpr ElemType mask_odd() {
  		 return ElemType(static_cast<T>(-1),static_cast<T>(1),
  				 	 	 static_cast<T>(-1),static_cast<T>(1),
						 static_cast<T>(-1),static_cast<T>(1),
						 static_cast<T>(-1),static_cast<T>(1));
  	 }

 	 static constexpr ElemType permute_evenodd(const ElemType& t) {
 	 		 return ElemType( t.template swizzle<1,0,3,2,5,4,7,6>() );
 	 	 }

 	 static constexpr ElemType real_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<0,0,2,2,4,4,6,6>() );
 	 }

 	 static constexpr ElemType imag_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<1,1,3,3,5,5,7,7>() );
 	 }


 };

 template<typename T>
   struct SIMDComplexSyCL<T,8> {
  	 static constexpr int len() { return 8; }
  	 static constexpr int num_fp() { return 2*len(); }
  	 using ElemType = cl::sycl::vec<T,num_fp()>;
  	 ElemType _data;

  	 static constexpr ElemType mask_even() {
  		 return ElemType(static_cast<T>(1),static_cast<T>(-1),
  				 	 	 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1),
						 static_cast<T>(1),static_cast<T>(-1));
  	 }

 	 static constexpr ElemType mask_odd() {
 		 return ElemType(static_cast<T>(-1),static_cast<T>(1),
 				 static_cast<T>(-1),static_cast<T>(1),
				 static_cast<T>(-1),static_cast<T>(1),
				 static_cast<T>(-1),static_cast<T>(1),
				 static_cast<T>(-1),static_cast<T>(1),
				 static_cast<T>(-1),static_cast<T>(1),
				 static_cast<T>(-1),static_cast<T>(1),
				 static_cast<T>(-1),static_cast<T>(1));
  	 }

 	 static constexpr ElemType permute_evenodd(const ElemType& t) {
 	 	 		 return ElemType( t.template swizzle<1,0,
 	 	 				 	 	 	 	 	 	 	 3,2,
													 5,4,
													 7,6,
													 9,8,
													 11,10,
													 13,12,
													 15,14>() );
 	 	 	 }

 	 static constexpr ElemType real_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<0,0,2,2,4,4,6,6,
 				 	 	 	 	 	 	 	 8,8,10,10,12,12,14,14>() );
 	 }

 	 static constexpr ElemType imag_vals(const ElemType& t) {
 		 return ElemType( t.template swizzle<1,1,3,3,5,5,7,7,
 				 	                         9,9,11,11,13,13,15,15>() );
 	 }

 };


 template<typename T, int N, template <typename,int> class SIMD>
 struct VectorTraits
 {};
 template<typename T, int N>
 struct VectorTraits<T,N,SIMDComplexSyCL> {
 	static constexpr int len() { return SIMDComplexSyCL<T,N>::len(); }
 	static constexpr int num_fp() { return 2*N; }
 	using BaseType = T;
 	using VecType = SIMDComplexSyCL<T,N>;
 };


template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Load(SIMDComplexSyCL<T,N>& result, size_t offset, cl::sycl::multi_ptr<T,Space> ptr)
{
	result._data.load(offset,ptr);
}

template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Store(size_t offset, cl::sycl::multi_ptr<T,Space> ptr, const SIMDComplexSyCL<T,N>& source)
{
	source._data.store(offset,ptr);
}

// The only real difference here is we would need a write only
// iterator
template<typename T, int N, cl::sycl::access::address_space Space>
inline void
Stream(size_t offset, cl::sycl::multi_ptr<T,Space> ptr, const SIMDComplexSyCL<T,N>& source)
{
	source._data.store(offset,ptr);
}

template<typename T, int N, cl::sycl::access::mode mode,
	cl::sycl::access::target target=cl::sycl::access::target::global_buffer,
	cl::sycl::access::placeholder isPlaceHolder=cl::sycl::access::placeholder::false_t>
MGComplex<T> LoadLane(size_t lane, size_t vector,
		cl::sycl::accessor<T,1,mode,target,isPlaceHolder> buf_access)
{
	MGComplex<T> res(static_cast<T>(0),static_cast<T>(0));
	res.real( buf_access[ 2*(N*vector + lane) ] );
	res.imag( buf_access[ 2*(N*vector + lane) + 1 ]);
	return res;
}

template<typename T, int N, cl::sycl::access::mode mode,
	cl::sycl::access::target target=cl::sycl::access::target::global_buffer,
	cl::sycl::access::placeholder isPlaceHolder=cl::sycl::access::placeholder::false_t>
void StoreLane(size_t lane, size_t vector,
		cl::sycl::accessor<T,1,mode,target,isPlaceHolder> buf_access, const MGComplex<T>& value)
{
	buf_access[ 2*(N*vector + lane)]= value.real();
	buf_access[ 2*(N*vector + lane) + 1] = value.imag();
}


template<typename T, int N>
struct LaneOps;

template<typename T>
struct LaneOps<T,1> {
	static inline
	void insert(SIMDComplexSyCL<T,1>&simd, const MGComplex<T>& val, size_t lane) {
		assert(lane == 0);
		simd._data.s0() = val.real();
		simd._data.s1() = val.imag();
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,1>& in, size_t lane){
		MGComplex<T> ret_val;
		assert(lane < 1);
		ret_val.real(in._data.s0());
		ret_val.imag(in._data.s1());
		return ret_val;
	}
};

template<typename T>
struct LaneOps<T,2> {
	static inline
	void insert(SIMDComplexSyCL<T,2>&simd, const MGComplex<T>& val, size_t lane) {
		assert(lane < 2);
		switch(lane) {
		case 0:
			simd._data.s0() = val.real();
			simd._data.s1() = val.imag();
			break;
		case 1:
			simd._data.s2() = val.real();
			simd._data.s3() = val.imag();
			break;
		};
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,2>& in, size_t lane){
		MGComplex<T> ret_val;
		assert(lane < 2);
		switch( lane ) {
		case 0:
			ret_val.real( in._data.s0() );
			ret_val.imag( in._data.s1() );
			break;
		case 1:
			ret_val.real( in._data.s2() );
			ret_val.imag( in._data.s3() );
			break;
		};
		return ret_val;
	}
};

template<typename T>
struct LaneOps<T,4> {
	static inline
	void insert(SIMDComplexSyCL<T,4>&simd, const MGComplex<T>& val, size_t lane) {
		assert(lane < 4);
		switch(lane) {
		case 0:
			simd._data.s0() = val.real();
			simd._data.s1() = val.imag();
			break;
		case 1:
			simd._data.s2() = val.real();
			simd._data.s3() = val.imag();
			break;
		case 2:
			simd._data.s4() = val.real();
			simd._data.s5() = val.imag();
			break;
		case 3:
			simd._data.s6() = val.real();
			simd._data.s7() = val.imag();
			break;
		};
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,4>& in, size_t lane){
		MGComplex<T> ret_val;
		assert(lane < 4);
		switch( lane ) {
		case 0:
			ret_val.real( in._data.s0() );
			ret_val.imag( in._data.s1() );
			break;
		case 1:
			ret_val.real( in._data.s2() );
			ret_val.imag( in._data.s3() );
			break;
		case 2:
			ret_val.real( in._data.s4() );
			ret_val.imag( in._data.s5() );
			break;
		case 3:
			ret_val.real( in._data.s6() );
			ret_val.imag( in._data.s7() );
			break;

		};
		return ret_val;
	}
};

template<typename T>
struct LaneOps<T,8> {
	static inline
	void insert(SIMDComplexSyCL<T,8>&simd, const MGComplex<T>& val, size_t lane) {
		assert(lane < 8);
		switch(lane) {
		case 0:
			simd._data.s0() = val.real();
			simd._data.s1() = val.imag();
			break;
		case 1:
			simd._data.s2() = val.real();
			simd._data.s3() = val.imag();
			break;
		case 2:
			simd._data.s4() = val.real();
			simd._data.s5() = val.imag();
			break;
		case 3:
			simd._data.s6() = val.real();
			simd._data.s7() = val.imag();
			break;
		case 4:
			simd._data.s8() = val.real();
			simd._data.s9() = val.imag();
			break;
		case 5:
			simd._data.sA() = val.real();
			simd._data.sB() = val.imag();
			break;
		case 6:
			simd._data.sC() = val.real();
			simd._data.sD() = val.imag();
			break;
		case 7:
			simd._data.sE() = val.real();
			simd._data.sF() = val.imag();
			break;
		};
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,8>& in, size_t lane){
		MGComplex<T> ret_val;
		assert(lane < 8);
		switch( lane ) {
		case 0:
			ret_val.real( in._data.s0() );
			ret_val.imag( in._data.s1() );
			break;
		case 1:
			ret_val.real( in._data.s2() );
			ret_val.imag( in._data.s3() );
			break;
		case 2:
			ret_val.real( in._data.s4() );
			ret_val.imag( in._data.s5() );
			break;
		case 3:
			ret_val.real( in._data.s6() );
			ret_val.imag( in._data.s7() );
			break;
		case 4:
			ret_val.real( in._data.s8() );
			ret_val.imag( in._data.s9() );
			break;
		case 5:
			ret_val.real( in._data.sA() );
			ret_val.imag( in._data.sB() );
			break;
		case 6:
			ret_val.real( in._data.sC() );
			ret_val.imag( in._data.sD() );
			break;
		case 7:
			ret_val.real( in._data.sE() );
			ret_val.imag( in._data.sF() );
			break;
		};
		return ret_val;
	}
};

template<typename T, int N>
  inline void ComplexZero(SIMDComplexSyCL<T,N>& result)
  {
 	 result._data = static_cast<T>(0);
  }


template<typename T, int N>
inline
void ComplexCopy(SIMDComplexSyCL<T,N>& result,
			const SIMDComplexSyCL<T,N>& source)
{
	result._data = source._data;
}

template<typename T, int N>
inline
void ComplexPeq(SIMDComplexSyCL<T,N>& res,
				const SIMDComplexSyCL<T,N>& a)
{
	res._data += a._data;
}

template<typename T, int N>
inline
void A_add_sign_B(SIMDComplexSyCL<T,N>&res,
				  const SIMDComplexSyCL<T,N>& a,
				  const T& sign,
				  const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	RepT sgnvec = RepT(sign);
	res._data = a._data + sgnvec*b._data;
}


template<typename T, int N>
inline
void A_add_B(SIMDComplexSyCL<T,N>&res,
				  const SIMDComplexSyCL<T,N>& a,
				  const SIMDComplexSyCL<T,N>& b)
{
	res._data = a._data + b._data;

}

template<typename T, int N>
inline
void A_sub_B(SIMDComplexSyCL<T,N>&res,
				  const SIMDComplexSyCL<T,N>& a,
				  const SIMDComplexSyCL<T,N>& b)
{
	res._data = a._data - b._data;
}

template<typename T, int N>
inline
void A_peq_sign_B( SIMDComplexSyCL<T,N>& a,
				   const float& sign,
				   const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	RepT sgnvec = RepT(sign);
	a._data += sgnvec*b._data;
}

template<typename T, int N>
inline
void A_peq_B( SIMDComplexSyCL<T,N>& a,
				   const SIMDComplexSyCL<T,N>& b)
{
	a._data += b._data;
}

template<typename T, int N>
inline
void A_meq_B( SIMDComplexSyCL<T,N>& a,
				   const SIMDComplexSyCL<T,N>& b)
{
	a._data -= b._data;
}

template<typename T, int N>
inline
void ComplexCMadd(SIMDComplexSyCL<T,N>& res,
				  const MGComplex<T>& a,
				  const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;
	RepT a_vec_re(a.real());
	RepT a_vec_im(a.imag());

	RepT b_perm = CType::permute_evenodd(b._data);
	RepT t = a_vec_im*b_perm + CType::mask_odd()*res._data;
	res._data = a_vec_re*b._data + CType::mask_odd()*t;

}

template<typename T, int N>
inline
void ComplexConjMadd(SIMDComplexSyCL<T,N>& res,
				  const MGComplex<T>& a,
				  const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;
	RepT a_vec_re(a.real());
	RepT a_vec_im(a.imag());

	RepT b_perm = CType::permute_evenodd(b._data);
	RepT t = a_vec_im*b_perm + CType::mask_even()*res._data;
	res._data = a_vec_re*b._data + CType::mask_even()*t;

}

template<typename T, int N>
inline
void ComplexCMadd(SIMDComplexSyCL<T,N>& res,
				  const SIMDComplexSyCL<T,N>& a,
				  const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;
	RepT a_vec_re = CType::real_vals(a._data);
	RepT a_vec_im = CType::imag_vals(a._data);
	RepT b_perm = CType::permute_evenodd(b._data);

	// addsub: mask_odd
	RepT t = a_vec_im*b_perm + CType::mask_odd()*res._data;
	res._data = a_vec_re*b._data + CType::mask_odd()*t;

}

template<typename T, int N>
inline
void ComplexConjMadd(SIMDComplexSyCL<T,N>& res,
				  const SIMDComplexSyCL<T,N>& a,
				  const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;
	RepT a_vec_re = CType::real_vals(a._data);
	RepT a_vec_im = CType::imag_vals(a._data);
	RepT b_perm = CType::permute_evenodd(b._data);

	// subadd: mask_even
	RepT t = a_vec_im*b_perm + CType::mask_even()*res._data;
	res._data = a_vec_re*b._data + CType::mask_even()*t;

}

template<typename T, int N>
inline
void A_add_sign_iB(SIMDComplexSyCL<T,N>& res,
				   const SIMDComplexSyCL<T,N>& a,
				   const float& sign,
				   const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;

	RepT sgnvec(sign);
	RepT perm_b = sgnvec*CType::permute_evenodd(b._data);

	res._data = a._data + CType::mask_odd()*perm_b;

}

template<typename T, int N>
inline
void A_add_iB(SIMDComplexSyCL<T,N>& res,
				   const SIMDComplexSyCL<T,N>& a,
				   const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;

	RepT perm_b = CType::permute_evenodd(b._data);
	res._data = a._data + CType::mask_odd()*perm_b;

}

template<typename T, int N>
inline
void A_sub_iB(SIMDComplexSyCL<T,N>& res,
				   const SIMDComplexSyCL<T,N>& a,
				   const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;


	RepT perm_b = CType::permute_evenodd(b._data);
	res._data = a._data - CType::mask_odd()*perm_b;
}

template<typename T, int N>
inline
void A_peq_sign_miB(SIMDComplexSyCL<T,N>& a,
				   const float& sign,
				   const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;

	RepT sgnvec(sign);
	RepT perm_b = sgnvec*CType::permute_evenodd(b._data);

	a._data -= CType::mask_odd()*perm_b;

}

template<typename T, int N>
inline
void A_peq_miB(SIMDComplexSyCL<T,N>& a,
		const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;

	RepT perm_b = CType::permute_evenodd(b._data);
	a._data -= CType::mask_odd()*perm_b;

}

template<typename T, int N>
inline
void A_meq_miB(SIMDComplexSyCL<T,N>& a,
			  const SIMDComplexSyCL<T,N>& b)
{
	using RepT = typename SIMDComplexSyCL<T,N>::ElemType;
	using CType =  SIMDComplexSyCL<T,N>;


	RepT perm_b = CType::permute_evenodd(b._data);
	a._data += CType::mask_odd()*perm_b;
}

} // subspace

