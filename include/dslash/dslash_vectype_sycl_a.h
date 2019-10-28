/*
 * dslash_vectype_sycl_b.h
 *
 *  Created on: Jun 28, 2019
 *      Author: bjoo
 */

#pragma once
#include <complex>
#include <dslash/dslash_complex.h>
#include <CL/sycl.hpp>

namespace MG {

template<typename T, int N>
using SIMDComplexSyCL = MGComplex< typename cl::sycl::vec<T,N> >;

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
struct BaseType< SIMDComplexSyCL<T,N> > {
	using Type = typename BaseType<T>::Type;
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

template<typename T, int N>
struct LaneOps;

template<typename T>
struct LaneOps<T,1>
{
	static inline
	void insert(SIMDComplexSyCL<T,1>&simd, const MGComplex<T>& val, int lane) {
			using Vec = typename VectorTraits<T,1,SIMDComplexSyCL>::VecType;
			Vec real = simd.real();
			Vec imag = simd.imag();
			assert(lane < 1);
			switch(lane) {
			case 0:
				real.s0() = val.real();
				imag.s0() = val.imag();
				break;
			};
			simd.real(real);
			simd.imag(imag);
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,1>& in, size_t lane) {
			MGComplex<T> ret_val;
			assert(lane < 1);
			ret_val.real( in.real().s0());
			ret_val.imag( in.imag().s0());
			return ret_val;
	}

};

template<typename T>
struct LaneOps<T,2>
{
	static inline
	void insert(SIMDComplexSyCL<T,2>&simd, const MGComplex<T>& val, int lane) {
		using Vec = typename VectorTraits<T,2,SIMDComplexSyCL>::VecType;
		Vec real = simd.real();
		Vec imag = simd.imag();
		assert(lane < 2);
		switch(lane) {
		case 0:
			real.s0() = val.real();
			imag.s0() = val.imag();
			break;
		case 1:
			real.s1() = val.real();
			imag.s1() = val.imag();
			break;
		};
		simd.real(real);
		simd.imag(imag);
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,2>& in, size_t lane) {
		MGComplex<T> ret_val;
		assert(lane < 2);
		switch(lane) {
		case 0:
			ret_val.real( in.real().s0());
			ret_val.imag( in.imag().s0());
			break;
		case 1:
			ret_val.real( in.real().s1());
			ret_val.imag( in.imag().s1());
			break;
		}
		return ret_val;
	}
};

template<typename T>
struct LaneOps<T,4>
{
	static inline
	void insert(SIMDComplexSyCL<T,4>&simd, const MGComplex<T>& val, int lane) {
		using Vec = typename VectorTraits<T,4,SIMDComplexSyCL>::VecType;
		Vec real = simd.real(); Vec imag = simd.imag();
		assert(lane < 4);
		switch(lane) {
		case 0:
			real.s0() = val.real();
			imag.s0() = val.imag();
			break;
		case 1:
			real.s1() = val.real();
			imag.s1() = val.imag();
			break;
		case 2:
			real.s2() = val.real();
			imag.s2() = val.imag();
			break;
		case 3:
			real.s3() = val.real();
			imag.s3() = val.imag();
			break;
		};
		simd.real(real);
		simd.imag(imag);
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,4>& in, size_t lane) {
		MGComplex<T> ret_val;
		assert(lane < 4);
		switch(lane) {
		case 0:
			ret_val.real( in.real().s0());
			ret_val.imag( in.imag().s0());
			break;
		case 1:
			ret_val.real( in.real().s1());
			ret_val.imag( in.imag().s1());
			break;
		case 2:
			ret_val.real( in.real().s2());
			ret_val.imag( in.imag().s2());
			break;
		case 3:
			ret_val.real( in.real().s3());
			ret_val.imag( in.imag().s3());
			break;
		}
		return ret_val;
	}
};

template<typename T>
struct LaneOps<T,8>
{
	static inline
	void insert(SIMDComplexSyCL<T,8>&simd, const MGComplex<T>& val, int lane) {
		using Vec = typename VectorTraits<T,8,SIMDComplexSyCL>::VecType;
		Vec real = simd.real(); Vec imag = simd.imag();
		assert(lane < 8);
		switch(lane) {
		case 0:
			real.s0() = val.real();
			imag.s0() = val.imag();
			break;
		case 1:
			real.s1() = val.real();
			imag.s1() = val.imag();
			break;
		case 2:
			real.s2() = val.real();
			imag.s2() = val.imag();
			break;
		case 3:
			real.s3() = val.real();
			imag.s3() = val.imag();
			break;
		case 4:
			real.s4() = val.real();
			imag.s4() = val.imag();
			break;
		case 5:
			real.s5() = val.real();
			imag.s5() = val.imag();
			break;
		case 6:
			real.s6() = val.real();
			imag.s6() = val.imag();
			break;
		case 7:
			real.s7() = val.real();
			imag.s7() = val.imag();
			break;
		};
		simd.real(real);
		simd.imag(imag);
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,8>& in, size_t lane) {
		MGComplex<T> ret_val;
		assert(lane < 8);
		switch(lane) {
		case 0:
			ret_val.real( in.real().s0());
			ret_val.imag( in.imag().s0());
			break;
		case 1:
			ret_val.real( in.real().s1());
			ret_val.imag( in.imag().s1());
			break;
		case 2:
			ret_val.real( in.real().s2());
			ret_val.imag( in.imag().s2());
			break;
		case 3:
			ret_val.real( in.real().s3());
			ret_val.imag( in.imag().s3());
			break;
		case 4:
			ret_val.real( in.real().s4());
			ret_val.imag( in.imag().s4());
			break;
		case 5:
			ret_val.real( in.real().s5());
			ret_val.imag( in.imag().s5());
			break;
		case 6:
			ret_val.real( in.real().s6());
			ret_val.imag( in.imag().s6());
			break;
		case 7:
			ret_val.real( in.real().s7());
			ret_val.imag( in.imag().s7());
			break;
		}
		return ret_val;
	}
};

template<typename T>
struct LaneOps<T,16>
{
	static inline
	void insert(SIMDComplexSyCL<T,16>&simd, const MGComplex<T>& val, int lane) {
		using Vec = typename VectorTraits<T,16,SIMDComplexSyCL>::VecType;
		Vec real = simd.real(); Vec imag = simd.imag();
		assert(lane < 16);
		switch(lane) {
		case 0:
			real.s0() = val.real();
			imag.s0() = val.imag();
			break;
		case 1:
			real.s1() = val.real();
			imag.s1() = val.imag();
			break;
		case 2:
			real.s2() = val.real();
			imag.s2() = val.imag();
			break;
		case 3:
			real.s3() = val.real();
			imag.s3() = val.imag();
			break;
		case 4:
			real.s4() = val.real();
			imag.s4() = val.imag();
			break;
		case 5:
			real.s5() = val.real();
			imag.s5() = val.imag();
			break;
		case 6:
			real.s6() = val.real();
			imag.s6() = val.imag();
			break;
		case 7:
			real.s7() = val.real();
			imag.s7() = val.imag();
			break;
		case 8:
			real.s8() = val.real();
			imag.s8() = val.imag();
			break;
		case 9:
			real.s9() = val.real();
			imag.s9() = val.imag();
			break;
		case 10:
			real.sA() = val.real();
			imag.sA() = val.imag();
			break;
		case 11:
			real.sB() = val.real();
			imag.sB() = val.imag();
			break;
		case 12:
			real.sC() = val.real();
			imag.sC() = val.imag();
			break;
		case 13:
			real.sD() = val.real();
			imag.sD() = val.imag();
			break;
		case 14:
			real.sE() = val.real();
			imag.sE() = val.imag();
			break;
		case 15:
			real.sF() = val.real();
			imag.sF() = val.imag();
			break;
		};
		simd.real(real);
		simd.imag(imag);
	}

	static inline
	MGComplex<T> extract(const SIMDComplexSyCL<T,16>& in, size_t lane) {
		MGComplex<T> ret_val;
		assert(lane < 16);
		switch(lane) {
		case 0:
			ret_val.real( in.real().s0());
			ret_val.imag( in.imag().s0());
			break;
		case 1:
			ret_val.real( in.real().s1());
			ret_val.imag( in.imag().s1());
			break;
		case 2:
			ret_val.real( in.real().s2());
			ret_val.imag( in.imag().s2());
			break;
		case 3:
			ret_val.real( in.real().s3());
			ret_val.imag( in.imag().s3());
			break;
		case 4:
			ret_val.real( in.real().s4());
			ret_val.imag( in.imag().s4());
			break;
		case 5:
			ret_val.real( in.real().s5());
			ret_val.imag( in.imag().s5());
			break;
		case 6:
			ret_val.real( in.real().s6());
			ret_val.imag( in.imag().s6());
			break;
		case 7:
			ret_val.real( in.real().s7());
			ret_val.imag( in.imag().s7());
			break;
		case 8:
			ret_val.real( in.real().s8());
			ret_val.imag( in.imag().s8());
			break;
		case 9:
			ret_val.real( in.real().s9());
			ret_val.imag( in.imag().s9());
			break;
		case 10:
			ret_val.real( in.real().sA());
			ret_val.imag( in.imag().sA());
			break;
		case 11:
			ret_val.real( in.real().sB());
			ret_val.imag( in.imag().sB());
			break;
		case 12:
			ret_val.real( in.real().sC());
			ret_val.imag( in.imag().sC());
			break;
		case 13:
			ret_val.real( in.real().sD());
			ret_val.imag( in.imag().sD());
			break;
		case 14:
			ret_val.real( in.real().sE());
			ret_val.imag( in.imag().sE());
			break;
		case 15:
			ret_val.real( in.real().sF());
			ret_val.imag( in.imag().sF());
			break;
		}
		return ret_val;
	}
};

#if 1
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
#endif

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
  res.real( a.real()-fsign*b.imag() );
  res.imag( a.imag()+fsign*b.real() );
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

	a.real( a.real() + fsign*b.real() );
	a.imag( a.imag() + fsign*b.imag() );
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


