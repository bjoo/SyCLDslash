/*
 * kokkos_spinproj.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_VSPINPROJ_H_
#define TEST_KOKKOS_KOKKOS_VSPINPROJ_H_

#include "Kokkos_Core.hpp"


#include "kokkos_defaults.h"
#include "kokkos_traits.h"
#include "kokkos_types.h"
#include "kokkos_vtypes.h"
#include "kokkos_vnode.h"
#include "kokkos_vectype.h"

namespace MG {


template<typename T, typename VN, typename T2, int isign>
KOKKOS_FORCEINLINE_FUNCTION
  void KokkosProjectDir0( const  VSpinorView<T, VN>& in,
			  HalfSpinorSiteView<T2>& spinor_out,
			  const int& i)
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
	for(int color=0; color < 3; ++color) {
		// in(i,color,0) is a 
		//   VSpinorView<T1,VN>::VecType =
		//      VSpinorView<T1,VN>::Container<BaseType<T>,VN::VecLen>
		//
		// out(color,0) is a T2 which is TST whichh needs to be 
	        // a SIMDComplex< of some base type >

                // Old Code: A_add_sign_iB(spinor_out(color,0), in(i,color,0), sign, in(i,color,3) );
		A_add_sign_iB<FType,
		              VN::VecLen,
		              SIMDComplex,
		              SIMDComplex,
		              SIMDComplex,
		  isign>(spinor_out(color,0), in(i,0,color),in(i,3,color) );

	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		//	spinor_out(color,1,K_RE) = in(i,color,1,K_RE)-sign*in(i,color,2,K_IM);
		//	spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_RE);
		// A_add_sign_iB(spinor_out(color,1), in(i,color,1), sign, in(i,color,2));
		A_add_sign_iB<FType,
		              VN::VecLen,
		              SIMDComplex,
		              SIMDComplex,
		              SIMDComplex,
		  isign>(spinor_out(color,1), in(i,1,color), in(i,2,color));
	}
}

  template<typename T, typename VN, typename T2, int isign>
 KOKKOS_FORCEINLINE_FUNCTION
   void KokkosProjectDir0Perm( const  VSpinorView<T,VN>& in,
 			  HalfSpinorSiteView<T2>& spinor_out,
			  const int& i,
			  const typename VN::MaskType& mask)
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
 	for(int color=0; color < 3; ++color) {
 		//		spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_IM);
 		//		spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,3,K_RE);

                // VN::VecType is SIMDComplex<T>
 		A_add_sign_iB<FType,VN::VecLen,
 		              SIMDComplex,
 		              SIMDComplex,
 		              SIMDComplex,
 		              isign>(spinor_out(color,0),
 				     VN::permute(mask,in(i,0,color)),
			             VN::permute(mask,in(i,3,color)) );

 	}

#pragma unroll
 	for(int color=0; color < 3; ++color) {
 		//	spinor_out(color,1,K_RE) = in(i,color,1,K_RE)-sign*in(i,color,2,K_IM);
 		//	spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_RE);
 		A_add_sign_iB<FType,VN::VecLen,
 		              SIMDComplex, // output
 		              SIMDComplex, // result of VN::permute
 		              SIMDComplex, // result of VN::permute 
 		              isign>(spinor_out(color,1),
 				     VN::permute(mask,in(i,1,color)),	
			             VN::permute(mask,in(i,2,color)));
 	}
 }

  template<typename T, typename VN, typename T2, int isign>
KOKKOS_FORCEINLINE_FUNCTION
    void KokkosProjectDir1(const VSpinorView<T, VN>& in,
		HalfSpinorSiteView<T2>& spinor_out,
		const int& i)
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
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)-sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		            SIMDComplex, // Output
		            SIMDComplex,       // Input
		            SIMDComplex,       // Input
		            -isign>(spinor_out(color,0),
				    in(i,0,color),in(i,3,color));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,1), 
				    in(i,1,color),in(i,2,color));
	}
}

  template<typename T, typename VN, typename T2, int isign>
KOKKOS_FORCEINLINE_FUNCTION
    void KokkosProjectDir1Perm(const VSpinorView<T, VN>& in,
		HalfSpinorSiteView<T2>& spinor_out,
		const int& i,
		const typename VN::MaskType& mask)
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
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)-sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		             SIMDComplex, // output
		             SIMDComplex, // output of VN::permute
		             SIMDComplex, // output of VN::permute
		             -isign>(spinor_out(color,0),
				     VN::permute(mask,in(i,0,color)),
				     VN::permute(mask,in(i,3,color)) );
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType, VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,1),
				    VN::permute(mask,in(i,1,color)),
				    VN::permute(mask,in(i,2,color)) );
	}
}
  template<typename T, typename VN, typename T2, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir2(const VSpinorView<T, VN>& in,
		HalfSpinorSiteView<T2>& spinor_out,
		const int& i)
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
	for(int color=0; color < 3; ++color) {
		//spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,2,K_IM);
		//spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_RE);
		A_add_sign_iB<FType,VN::VecLen,
		              SIMDComplex,
		              SIMDComplex,
		              SIMDComplex,
		              isign>
		  (spinor_out(color,0),in(i,0,color),in(i,2,color));
	}

#pragma unroll
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_IM);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)-sign*in(i,color,3,K_RE);
		A_add_sign_iB<FType,VN::VecLen,
		              SIMDComplex,
		              SIMDComplex,
		              SIMDComplex,
		  -isign>(spinor_out(color,1), in(i,1,color),in(i,3,color));
	}
}


  template<typename T, typename VN, typename T2, int isign>
 KOKKOS_FORCEINLINE_FUNCTION
 void KokkosProjectDir2Perm(const VSpinorView<T, VN>& in,
 		HalfSpinorSiteView<T2>& spinor_out,
		const int& i,
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
 	for(int color=0; color < 3; ++color) {
 		//spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,2,K_IM);
 		//spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_RE);
 		A_add_sign_iB<FType,VN::VecLen,
 		              SIMDComplex,
 		              SIMDComplex,
 		              SIMDComplex,
 		              isign>(spinor_out(color,0),
				     VN::permute(mask,in(i,0,color)),
				     VN::permute(mask,in(i,2,color)) );
 	}

#pragma unroll
 	for(int color=0; color < 3; ++color ) {
 		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_IM);
 		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)-sign*in(i,color,3,K_RE);
 		A_add_sign_iB<FType,VN::VecLen,
 		              SIMDComplex,
 		              SIMDComplex,
 		              SIMDComplex,
 		              -isign>(spinor_out(color,1),
				      VN::permute(mask, in(i,1,color)),
				      VN::permute(mask, in(i,3,color)));
 	}
 }

  template<typename T, typename VN, typename T2, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir3(const VSpinorView<T, VN>& in,
		       HalfSpinorSiteView<T2>& spinor_out,
			   const int& i)

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
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,0), 
				    in(i,0,color), in(i,2,color));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             SIMDComplex,
		  isign>(spinor_out(color,1), in(i,1,color), in(i,3,color));
	}
}

  template<typename T, typename VN, typename T2,  int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir3Perm(const VSpinorView<T, VN>& in,
		       HalfSpinorSiteView<T2>& spinor_out,
			   const int& i,
			   const typename VN::MaskType& mask)
		
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
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,0),
				    VN::permute(mask, in(i,0,color)),
				    VN::permute(mask, in(i,2,color)));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,3,K_IM);
		A_add_sign_B<FType,VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,1),
				    VN::permute(mask, in(i,1,color)), 
				    VN::permute(mask, in(i,3,color))) ;
	}
}


  template<typename T, typename VN, typename T2, int dir, int isign>
    void KokkosVProjectLattice(const KokkosCBFineVSpinor<T,VN,4>& kokkos_in,
			      KokkosCBFineVSpinor<T,VN,2>& kokkos_hspinor_out, int _sites_per_team = 2)
{
	int num_sites = kokkos_in.GetInfo().GetNumCBSites();
	const VSpinorView<T,VN>& spinor_in = kokkos_in.GetData();
	VHalfSpinorView<T,VN>& hspinor_out = kokkos_hspinor_out.GetData();

	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),VN::VecLen);
	  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {

			HalfSpinorSiteView<T2> res;



			if( dir == 0) {
			  KokkosProjectDir0<T,VN,T2,isign>(spinor_in, res,i);
			}
			else if (dir == 1) {
			  //			KokkosProjectDir<T,1>(spinor_in,plus_minus,res,i);
			  KokkosProjectDir1<T,VN,T2,isign>(spinor_in, res,i);
			}
			else if (dir == 2 ) {
			  KokkosProjectDir2<T,VN,T2,isign>(spinor_in, res,i);
			}
			else {
			  KokkosProjectDir3<T,VN,T2,isign>(spinor_in, res,i);
			}
			
			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin<2; ++spin) {
			    
			    //hspinor_out(i,spin,color,reim) = res(spin,color,reim);
			    Store(hspinor_out(i,spin,color), res(color,spin));
			  }
			}
		  });
	    });
}

  template<typename T, typename VN, typename T2, int dir, int isign>
    void KokkosVProjectLatticePerm(const KokkosCBFineVSpinor<T,VN,4>& kokkos_in,
			      KokkosCBFineVSpinor<T,VN,2>& kokkos_hspinor_out, int _sites_per_team = 2)
{
	int num_sites = kokkos_in.GetInfo().GetNumCBSites();
	const VSpinorView<T,VN>& spinor_in = kokkos_in.GetData();
	VHalfSpinorView<T,VN>& hspinor_out = kokkos_hspinor_out.GetData();

	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),VN::VecLen);
	  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {

			HalfSpinorSiteView<T2> res;



			if( dir == 0) {
			  KokkosProjectDir0Perm<T,VN,T2,isign>(spinor_in, res,i, VN::XPermuteMask);
			}
			else if (dir == 1) {
			  //			KokkosProjectDir<T,1>(spinor_in,plus_minus,res,i);
			  KokkosProjectDir1Perm<T,VN,T2,isign>(spinor_in, res,i, VN::YPermuteMask);
			}
			else if (dir == 2 ) {
			  KokkosProjectDir2Perm<T,VN,T2,isign>(spinor_in, res,i, VN::ZPermuteMask);
			}
			else {
			  KokkosProjectDir3Perm<T,VN,T2,isign>(spinor_in, res,i, VN::TPermuteMask);
			}

			for(int color=0; color < 3; ++color) {
			  for(int spin=0; spin<2; ++spin) {

			    //hspinor_out(i,spin,color,reim) = res(spin,color,reim);
			    Store(hspinor_out(i,spin,color), res(color,spin));
			  }
			}
		  });
	    });
}

  template<typename T, typename VN, typename T2, int dir>
     void KokkosLatticeVHSpinorPerm(KokkosCBFineVSpinor<T,VN,2>& kokkos_in,
 			      int _sites_per_team = 2)
 {
 	int num_sites = kokkos_in.GetInfo().GetNumCBSites();
 	VHalfSpinorView<T,VN>& spinor = kokkos_in.GetData();

 	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),VN::VecLen);
 	  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
 		    const int start_idx = team.league_rank()*_sites_per_team;
 		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
 		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {

 			for(int color=0; color <3; ++color) {
 				for(int spin=0; spin <2; ++spin) {
 					T2 tmp;
 					if(dir ==0 ) {

 						tmp = VN::permute(VN::XPermuteMask, spinor(i,color,spin));
 					}

 					if(dir ==1 ) {
 						tmp = VN::permute(VN::YPermuteMask, spinor(i,color,spin));
 					}

 					if(dir ==2 ) {
 						tmp = VN::permute(VN::ZPermuteMask, spinor(i,color,spin));
 					}

 					if(dir ==3 ) {
 						tmp = VN::permute(VN::TPermuteMask, spinor(i,color,spin));
 					}

 					spinor(i,spin,color) = tmp;
 				}
 			}
 		  });
 	    });
 }


template<typename T, typename VN, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir0(const HalfSpinorSiteView<T>& hspinor_in,
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
  for(int spin=0; spin < 2; ++spin ){ 

#pragma unroll
    for(int color=0; color < 3; ++color) { 
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }

#else
#pragma unroll
  for(int color=0; color < 3; ++color ){

#pragma unroll
    for(int spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif

	// Spin 2
#pragma unroll
	for(int color=0; color < 3; ++color ) {
		//	spinor_out(color,2).real() = sign*hspinor_in(color,1).imag();
		//	spinor_out(color,2).imag() = -sign*hspinor_in(color,1).real();
		A_peq_sign_miB<FType,VN::VecLen,
		              SIMDComplex,
		              SIMDComplex,
		              isign>(spinor_out(color,2),  hspinor_in(color,1));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		//	spinor_out(color,3).real() = sign*hspinor_in(color,0).imag();
		//	spinor_out(color,3).imag() = -sign*hspinor_in(color,0).real();
		A_peq_sign_miB<FType,VN::VecLen,
		               SIMDComplex,
		               SIMDComplex,
		               isign>(spinor_out(color,3),hspinor_in(color,0));
	}
}

 template<typename T, typename VN, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir1(const HalfSpinorSiteView<T>& hspinor_in,
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
  for(int spin=0; spin < 2; ++spin ){

#pragma unroll
    for(int color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#else
#pragma unroll
  for(int color=0; color < 3; ++color ){

#pragma unroll
    for(int spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif

	// Spin 2
#pragma unroll
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,2).real() = sign*hspinor_in(color,1).real();
		// spinor_out(color,2).imag() = sign*hspinor_in(color,1).imag();
		A_peq_sign_B<FType,VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,2),hspinor_in(color,1));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,3).real() = -sign*hspinor_in(color,0).real();
		// spinor_out(color,3).imag() = -sign*hspinor_in(color,0).imag();
	  A_peq_sign_B<FType,VN::VecLen,
	               SIMDComplex,
	               SIMDComplex, 
	               -isign>(spinor_out(color,3),hspinor_in(color,0));
	}
}

 template<typename T, typename VN, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir2(const HalfSpinorSiteView<T>& hspinor_in,
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
  for(int spin=0; spin < 2; ++spin ){

#pragma unroll
    for(int color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#else
#pragma unroll
  for(int color=0; color < 3; ++color ){

#pragma unroll
    for(int spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif


	// Spin 2
#pragma unroll
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,2).real() = sign*hspinor_in(color,0).imag();
		// spinor_out(color,2).imag() = -sign*hspinor_in(color,0).real();
		A_peq_sign_miB<FType, VN::VecLen,
		               SIMDComplex,
		               SIMDComplex,
		               isign>(spinor_out(color,2), hspinor_in(color,0));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,3,K_RE) = -sign*hspinor_in(color,1,K_IM);
		// spinor_out(color,3,K_IM) = sign*hspinor_in(color,1,K_RE);
		A_peq_sign_miB<FType,VN::VecLen,
		               SIMDComplex,
		               SIMDComplex,
		               -isign>(spinor_out(color,3), hspinor_in(color,1));
	}
}

 template<typename T, typename VN, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir3(const HalfSpinorSiteView<T>& hspinor_in,
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
  for(int spin=0; spin < 2; ++spin ){

#pragma unroll
    for(int color=0; color < 3; ++color) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#else
#pragma unroll
  for(int color=0; color < 3; ++color ){

#pragma unroll
    for(int spin=0; spin < 2; ++spin) {
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }
#endif

	// Spin 2
#pragma unroll
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,2,K_RE) = sign*hspinor_in(color,0,K_RE);
		// spinor_out(color,2,K_IM) = sign*hspinor_in(color,0,K_IM);
		A_peq_sign_B<FType, VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,2),hspinor_in(color,0));
	}

#pragma unroll
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,3,K_RE) = sign*hspinor_in(color,1,K_RE);
		// spinor_out(color,3,K_IM) = sign*hspinor_in(color,1,K_IM);
		A_peq_sign_B<FType, VN::VecLen,
		             SIMDComplex,
		             SIMDComplex,
		             isign>(spinor_out(color,3), hspinor_in(color,1));
	}
}

#if 0
 template<typename T, typename T2, int dir, int isign>
void KokkosVReconsLattice(const KokkosCBFineSpinor<T,2>& kokkos_hspinor_in,
			 KokkosCBFineSpinor<T,4>& kokkos_spinor_out, int _sites_per_team = 2)
{
	const int num_sites = kokkos_hspinor_in.GetInfo().GetNumCBSites();
	SpinorView<T>& spinor_out = kokkos_spinor_out.GetData();
	const HalfSpinorView<T>& hspinor_in_view = kokkos_hspinor_in.GetData();

	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<T>::value);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {



		HalfSpinorSiteView<T2> hspinor_in;
		SpinorSiteView<T2> res;

		// Stream in top 2 components.
		for(int color=0; color < 3; ++color) {
		  for(int spin=0; spin < 2; ++spin ) {
		    Load(hspinor_in(color,spin),hspinor_in_view(i,color,spin));
		  }
		}

		for(int color=0; color < 3; ++color) {
		  for(int spin=0; spin < 4; ++spin ) {
		    ComplexZero(res(color,spin));
		  }
		}

		// Reconstruct into a SpinorSiteView
		if (dir == 0 ) {
		  KokkosRecons23Dir0<T2,isign>(hspinor_in,
					     res);
		}
		else if (dir == 1 ) {
		  KokkosRecons23Dir1<T2,isign>(hspinor_in,
						     res);

		}
		else if ( dir == 2 ) {
		  KokkosRecons23Dir2<T2,isign>(hspinor_in,
						     res);
		}
		else {
		  KokkosRecons23Dir3<T2,isign>(hspinor_in,
						     res);

		}

		// Stream out into a spinor
		for(int color=0; color < 3; ++color ) {
		  for(int spin=0; spin < 4; ++spin) {

		    Store( spinor_out(i,color,spin), res(color,spin));

		  }


		}

	});
	  });

}
#endif

}



#endif /* TEST_KOKKOS_KOKKOS_SPINPROJ_H_ */
