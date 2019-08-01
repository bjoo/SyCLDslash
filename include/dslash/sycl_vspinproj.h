/*
 * kokkos_spinproj.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#pragma once

#include "CL/sycl.hpp"


#include "dslash/dslash_defaults.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"

namespace MG {


// Will need sfinae to make sure access::target is only read
template<typename T, typename VN, typename T2, int isign,
	cl::sycl::access::mode accessMode = cl::sycl::access::mode::read,
	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
void SyCLProjectDir0( const SyCLVSpinorViewAccessor<T, VN, accessMode, accessTarget>& in,
			  HalfSpinorSiteView<T2>& spinor_out,
			  const size_t& i)
{
   using FType = typename BaseType<T>::Type;
   
  
	/*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
	 *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
	 *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
	 *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
	 *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
	 *      ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r )
	 */

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// in(i,color,0) is a 
		//   VSpinorView<T1,VN>::VecType =
		//      VSpinorView<T1,VN>::Container<BaseType<T>,VN::VecLen>
		//
		// out(color,0) is a T2 which is TST whichh needs to be 
	        // a SIMDComplexSyCL< of some base type >

                // Old Code: A_add_sign_iB(spinor_out(color,0), in(i,color,0), sign, in(i,color,3) );
		A_add_sign_iB<FType, VN::VecLen, isign>(spinor_out(color,0), in(i,0,color),in(i,3,color) );

	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		//	spinor_out(color,1,K_RE) = in(i,color,1,K_RE)-sign*in(i,color,2,K_IM);
		//	spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_RE);
		// A_add_sign_iB(spinor_out(color,1), in(i,color,1), sign, in(i,color,2));
		A_add_sign_iB<FType,VN::VecLen,isign>(spinor_out(color,1), in(i,1,color), in(i,2,color));

	}
}


 template<typename T, typename VN,  typename T2, int isign,
 	cl::sycl::access::mode accessMode = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
 void SyCLProjectDir0Perm( const  SyCLVSpinorViewAccessor<T,VN,accessMode,accessTarget>& in,
		 HalfSpinorSiteView<T2>& spinor_out,
			  const size_t& i)
 {
   using FType = typename BaseType<T>::Type;
  

 	/*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
 	 *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
 	 *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
 	 *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )
 	 * Therefore the top components are
 	 *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
 	 *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )
 	 * The bottom components of be may be reconstructed using the formula
 	 *      ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
 	 *      ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r )
 	 */

#pragma unroll
 	for(size_t color=0; color < 3; ++color) {
 		//		spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_IM);
 		//		spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,3,K_RE);

                // VN::VecType is SIMDComplexSyCL<T>
 		A_add_sign_iB<FType,VN::VecLen,isign>(spinor_out(color,0),
 				   	   VN::permute<X_DIR>(in(i,0,color)),
					   VN::permute<X_DIR>(in(i,3,color)) );

 	}

#pragma unroll
 	for(size_t color=0; color < 3; ++color) {
 		//	spinor_out(color,1,K_RE) = in(i,color,1,K_RE)-sign*in(i,color,2,K_IM);
 		//	spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_RE);
 		A_add_sign_iB<FType,VN::VecLen,isign>(spinor_out(color,1),
 				     	 	 	 	 	 	 	 VN::permute<X_DIR>(in(i,1,color)),
												 VN::permute<X_DIR>(in(i,2,color)));
 	}
 }



 // Will need sfinae to make sure access::target is only read
 template<typename T, typename VN, typename T2, int isign,
 	cl::sycl::access::mode accessMode = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
inline
    void SyCLProjectDir1(const SyCLVSpinorViewAccessor<T, VN, accessMode, accessTarget>& in,
    		HalfSpinorSiteView<T2>& spinor_out,
		const size_t& i)
{
	  using FType = typename BaseType<T>::Type;
	

	/*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
	 *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
	 *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
	 *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )

	 * Therefore the top components are

	 *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
	 *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
	 */
#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)-sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,-isign>(spinor_out(color,0),in(i,0,color),in(i,3,color));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType,VN::VecLen,isign>(spinor_out(color,1), in(i,1,color),in(i,2,color));
	}
}

 // Will need sfinae to make sure access::target is only read
 template<typename T, typename VN, typename T2, int isign,
 	cl::sycl::access::mode accessMode  = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
inline
    void SyCLProjectDir1Perm(const SyCLVSpinorViewAccessor<T, VN,accessMode,accessTarget>& in,
		HalfSpinorSiteView<T2>& spinor_out,
		const size_t& i)
{
	  using FType = typename BaseType<T>::Type;
	  

	/*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
	 *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
	 *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
	 *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )

	 * Therefore the top components are

	 *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
	 *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
	 */
#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)-sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,-isign>(spinor_out(color,0),
				     VN::permute<Y_DIR>(in(i,0,color)),
				     VN::permute<Y_DIR>(in(i,3,color)) );
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType, VN::VecLen,+isign>(spinor_out(color,1),
				    VN::permute<Y_DIR>(in(i,1,color)),
				    VN::permute<Y_DIR>(in(i,2,color)) );
	}
}

 // Will need sfinae to make sure access::target is only read
 template<typename T, typename VN, typename T2, int isign,
 	cl::sycl::access::mode accessMode  = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
inline
void SyCLProjectDir2(const SyCLVSpinorViewAccessor<T, VN, accessMode, accessTarget>& in,
		HalfSpinorSiteView<T2>& spinor_out,
		const size_t& i)
{

  using FType =  typename BaseType<T>::Type;
 

	/*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
	 *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
	 *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
	 *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
	 */
#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		//spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,2,K_IM);
		//spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_RE);
		A_add_sign_iB<FType,VN::VecLen,isign>(spinor_out(color,0),in(i,0,color),in(i,2,color));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color ) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_IM);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)-sign*in(i,color,3,K_RE);
		A_add_sign_iB<FType,VN::VecLen,-isign>(spinor_out(color,1), in(i,1,color),in(i,3,color));
	}
}


 // Will need sfinae to make sure access::target is only read
 template<typename T, typename VN, typename T2, int isign,
 	cl::sycl::access::mode accessMode = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
 inline
 void SyCLProjectDir2Perm(const SyCLVSpinorViewAccessor<T, VN, accessMode, accessTarget>& in,
 		HalfSpinorSiteView<T2>& spinor_out,
		const size_t& i,
		const typename VN::MaskType& mask)
 {

   using FType =  typename BaseType<T>::Type;

 	/*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
 	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
 	 *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
 	 *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )
 	 * Therefore the top components are
 	 *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
 	 *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
 	 */
#pragma unroll
 	for(size_t color=0; color < 3; ++color) {
 		//spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,2,K_IM);
 		//spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_RE);
 		A_add_sign_iB<FType,VN::VecLen,isign>(spinor_out(color,0),
				     VN::permute<Z_DIR>(in(i,0,color)),
				     VN::permute<Z_DIR>(in(i,2,color)) );
 	}

#pragma unroll
 	for(size_t color=0; color < 3; ++color ) {
 		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_IM);
 		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)-sign*in(i,color,3,K_RE);
 		A_add_sign_iB<FType,VN::VecLen,-isign>(spinor_out(color,1),
				      VN::permute<Z_DIR>(in(i,1,color)),
				      VN::permute<Z_DIR>(in(i,3,color)));
 	}
 }

 // Will need sfinae to make sure access::target is only read
 template<typename T, typename VN, typename T2, int isign,
 	cl::sycl::access::mode accessMode = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
inline
void SyCLProjectDir3(const SyCLVSpinorViewAccessor<T, VN, accessMode, accessTarget>& in,
		       HalfSpinorSiteView<T2>& spinor_out,
			   const size_t& i)

{
	  using FType = typename BaseType<T>::Type;
	 
	/*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
	 *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
	 *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
	 *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
	 */
#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType,VN::VecLen,isign>(spinor_out(color,0),
				    in(i,0,color), in(i,2,color));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,isign>(spinor_out(color,1), in(i,1,color), in(i,3,color));
	}
}

 // Will need sfinae to make sure access::target is only read
 template<typename T, typename VN, typename T2, int isign,
 	cl::sycl::access::mode accessMode  = cl::sycl::access::mode::read,
 	cl::sycl::access::target accessTarget = cl::sycl::access::target::global_buffer>
inline
void SyCLProjectDir3Perm(const SyCLVSpinorViewAccessor<T, VN, accessMode>& in,
		       HalfSpinorSiteView<T2>& spinor_out,
			   const size_t& i)
{
	  using FType = typename BaseType<T2>::Type;
	  
	/*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
	 *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
	 *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
	 *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
	 */
#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType,VN::VecLen, isign>(spinor_out(color,0),
				    							VN::permute<T_DIR>(in(i,0,color)),
												VN::permute<T_DIR>(in(i,2,color)));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,isign>(spinor_out(color,1),
				    VN::permute<T_DIR>(in(i,1,color)),
				    VN::permute<T_DIR>(in(i,3,color))) ;
	}
}



#if 0
template<typename T, typename VN, int isign>
inline
void SyCLRecons23Dir0(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{

  using FType = typename BaseType<T>::Type;
 
	/*                              ( 1  0  0 +i)  ( a0 )    ( a0 + i a3 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1 +i  0)  ( a1 )  = ( a1 + i a2 )
	 *                    0         ( 0 -i  1  0)  ( a2 )    ( a2 - i a1 )
	 *                              (-i  0  0  1)  ( a3 )    ( a3 - i a0 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a3i} + i{a0i + a3r} )
	 *      ( b1r + i b1i )     ( {a1r - a2i} + i{a1i + a2r} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
	 *      ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r )
	 */
#if 0
#pragma unroll
  for(size_t spin=0; spin < 2; ++spin ){

#pragma unroll
    for(size_t color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }

#else
#pragma unroll
  for(size_t color=0; color < 3; ++color ){

#pragma unroll
    for(size_t spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif

	// Spin 2
#pragma unroll
	for(size_t color=0; color < 3; ++color ) {
		//	spinor_out(color,2).real() = sign*hspinor_in(color,1).imag();
		//	spinor_out(color,2).imag() = -sign*hspinor_in(color,1).real();
		A_peq_sign_miB<FType,VN::VecLen,
		              SIMDComplexSyCL,
		              SIMDComplexSyCL,
		              isign>(spinor_out(color,2),  hspinor_in(color,1));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		//	spinor_out(color,3).real() = sign*hspinor_in(color,0).imag();
		//	spinor_out(color,3).imag() = -sign*hspinor_in(color,0).real();
		A_peq_sign_miB<FType,VN::VecLen,
		               SIMDComplexSyCL,
		               SIMDComplexSyCL,
		               isign>(spinor_out(color,3),hspinor_in(color,0));
	}
}

 template<typename T, typename VN, int isign>
inline
void SyCLRecons23Dir1(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{
  using FType = typename BaseType<T>::Type;
  
  /*                              ( 1  0  0 -1)  ( a0 )    ( a0 - a3 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  1  0)  ( a1 )  = ( a1 + a2 )
	 *                    1         ( 0  1  1  0)  ( a2 )    ( a2 + a1 )
	 *                              (-1  0  0  1)  ( a3 )    ( a3 - a0 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a3r} + i{a0i - a3i} )
	 *      ( b1r + i b1i )     ( {a1r + a2r} + i{a1i + a2i} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
	 *      ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i )
	 */

#if 0
#pragma unroll
  for(size_t spin=0; spin < 2; ++spin ){

#pragma unroll
    for(size_t color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#else
#pragma unroll
  for(size_t color=0; color < 3; ++color ){

#pragma unroll
    for(size_t spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif

	// Spin 2
#pragma unroll
	for(size_t color=0; color < 3; ++color ) {
		// spinor_out(color,2).real() = sign*hspinor_in(color,1).real();
		// spinor_out(color,2).imag() = sign*hspinor_in(color,1).imag();
		A_peq_sign_B<FType,VN::VecLen,
		             SIMDComplexSyCL,
		             SIMDComplexSyCL,
		             isign>(spinor_out(color,2),hspinor_in(color,1));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,3).real() = -sign*hspinor_in(color,0).real();
		// spinor_out(color,3).imag() = -sign*hspinor_in(color,0).imag();
	  A_peq_sign_B<FType,VN::VecLen,
	               SIMDComplexSyCL,
	               SIMDComplexSyCL,
	               -isign>(spinor_out(color,3),hspinor_in(color,0));
	}
}

 template<typename T, typename VN, int isign>
inline
void SyCLRecons23Dir2(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{
	  using FType = typename BaseType<T>::Type;
		
	/*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
	 *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
	 *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
	 *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
	 *      ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r )
	 */

#if 0
#pragma unroll
  for(size_t spin=0; spin < 2; ++spin ){

#pragma unroll
    for(size_t color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#else
#pragma unroll
  for(size_t color=0; color < 3; ++color ){

#pragma unroll
    for(size_t spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif


	// Spin 2
#pragma unroll
	for(size_t color=0; color < 3; ++color ) {
		// spinor_out(color,2).real() = sign*hspinor_in(color,0).imag();
		// spinor_out(color,2).imag() = -sign*hspinor_in(color,0).real();
		A_peq_sign_miB<FType, VN::VecLen,
		               SIMDComplexSyCL,
		               SIMDComplexSyCL,
		               isign>(spinor_out(color,2), hspinor_in(color,0));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,3,K_RE) = -sign*hspinor_in(color,1,K_IM);
		// spinor_out(color,3,K_IM) = sign*hspinor_in(color,1,K_RE);
		A_peq_sign_miB<FType,VN::VecLen,
		               SIMDComplexSyCL,
		               SIMDComplexSyCL,
		               -isign>(spinor_out(color,3), hspinor_in(color,1));
	}
}

 template<typename T, typename VN, int isign>
inline
void SyCLRecons23Dir3(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{
	  using FType = typename BaseType<T>::Type;
		

	/*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
	 *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
	 *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
	 *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
	 *      ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i )
	 */

#if 0
#pragma unroll
  for(size_t spin=0; spin < 2; ++spin ){

#pragma unroll
    for(size_t color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#else
#pragma unroll
  for(size_t color=0; color < 3; ++color ){

#pragma unroll
    for(size_t spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif

	// Spin 2
#pragma unroll
	for(size_t color=0; color < 3; ++color ) {
		// spinor_out(color,2,K_RE) = sign*hspinor_in(color,0,K_RE);
		// spinor_out(color,2,K_IM) = sign*hspinor_in(color,0,K_IM);
		A_peq_sign_B<FType, VN::VecLen,
		             SIMDComplexSyCL,
		             SIMDComplexSyCL,
		             isign>(spinor_out(color,2),hspinor_in(color,0));
	}

#pragma unroll
	for(size_t color=0; color < 3; ++color) {
		// spinor_out(color,3,K_RE) = sign*hspinor_in(color,1,K_RE);
		// spinor_out(color,3,K_IM) = sign*hspinor_in(color,1,K_IM);
		A_peq_sign_B<FType, VN::VecLen,
		             SIMDComplexSyCL,
		             SIMDComplexSyCL,
		             isign>(spinor_out(color,3), hspinor_in(color,1));
	}
}

#if 0
 template<typename T, typename T2, int dir, int isign>
void SyCLVReconsLattice(const SyCLCBFineSpinor<T,2>& kokkos_hspinor_in,
			 SyCLCBFineSpinor<T,4>& kokkos_spinor_out, int _sites_per_team = 2)
{
	const size_t num_sites = kokkos_hspinor_in.GetInfo().GetNumCBSites();
	SpinorView<T>& spinor_out = kokkos_spinor_out.GetData();
	const HalfSpinorView<T>& hspinor_in_view = kokkos_hspinor_in.GetData();

	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,SyCL::AUTO(),Veclen<T>::value);
	SyCL::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const size_t start_idx = team.league_rank()*_sites_per_team;
		    const size_t end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    SyCL::parallel_for(SyCL::TeamThreadRange(team,start_idx,end_idx),[=](const size_t i) {



		HalfSpinorSiteView<T2> hspinor_in;
		SpinorSiteView<T2> res;

		// Stream in top 2 components.
		for(size_t color=0; color < 3; ++color) {
		  for(size_t spin=0; spin < 2; ++spin ) {
		    Load(hspinor_in(color,spin),hspinor_in_view(i,color,spin));
		  }
		}

		for(size_t color=0; color < 3; ++color) {
		  for(size_t spin=0; spin < 4; ++spin ) {
		    ComplexZero(res(color,spin));
		  }
		}

		// Reconstruct size_to a SpinorSiteView
		if (dir == 0 ) {
		  SyCLRecons23Dir0<T2,isign>(hspinor_in,
					     res);
		}
		else if (dir == 1 ) {
		  SyCLRecons23Dir1<T2,isign>(hspinor_in,
						     res);

		}
		else if ( dir == 2 ) {
		  SyCLRecons23Dir2<T2,isign>(hspinor_in,
						     res);
		}
		else {
		  SyCLRecons23Dir3<T2,isign>(hspinor_in,
						     res);

		}

		// Stream out size_to a spinor
		for(size_t color=0; color < 3; ++color ) {
		  for(size_t spin=0; spin < 4; ++spin) {

		    Store( spinor_out(i,color,spin), res(color,spin));

		  }


		}

	});
	  });

}
#endif
#endif
}
