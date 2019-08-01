/*
 * sycl_lattice_spin_proj_utils.h
 *
 *  Created on: Aug 1, 2019
 *      Author: bjoo
 */

#pragma once
#include <CL/sycl.hpp>
#include <dslash/sycl_view.h>
#include <dslash/sycl_vtypes.h>
#include <dslash/sycl_vspinproj.h>
namespace MG {

template<typename T, typename VN, int dir, int isign>
class lattice_spin_proj;

template<typename T, typename VN,  int dir, int isign>
void SyCLVProjectLattice(const SyCLCBFineVSpinor<T,VN,4>& sycl_in,
		SyCLCBFineVSpinor<T,VN,2>& sycl_hspinor_out)
{
	size_t num_sites = sycl_in.GetInfo().GetNumCBSites();
	SyCLVSpinorView<T,VN> spinor_in_view = sycl_in.GetData();
	SyCLVHalfSpinorView<T,VN> hspinor_out_view = sycl_hspinor_out.GetData();

	cl::sycl::queue q;
	q.submit([&]( cl::sycl::handler& cgh ) {

	 SyCLVSpinorViewAccessor<T,VN,cl::sycl::access::mode::read> spinor_in = spinor_in_view.template get_access<cl::sycl::access::mode::read>(cgh);
     SyCLVHalfSpinorViewAccessor<T,VN, cl::sycl::access::mode::write> hspinor_out = hspinor_out_view.template get_access<cl::sycl::access::mode::write>(cgh);

		cgh.parallel_for<lattice_spin_proj<T,VN,dir,isign>>(
				cl::sycl::range<1>(num_sites), [=](cl::sycl::id<1> idx) {

			using T2 =  SIMDComplexSyCL< typename  BaseType<T>::Type, VN::VecLen >;
			size_t i = idx[0];
			HalfSpinorSiteView<T2> res;



			if( dir == 0) {
				SyCLProjectDir0<T,VN,T2,isign>(spinor_in, res,i);
			}
			else if (dir == 1) {
				//			SyCLProjectDir<T,1>(spinor_in,plus_minus,res,i);
				SyCLProjectDir1<T,VN,T2,isign>(spinor_in, res,i);
			}
			else if (dir == 2 ) {
				SyCLProjectDir2<T,VN,T2,isign>(spinor_in, res,i);
			}
			else {
				SyCLProjectDir3<T,VN,T2,isign>(spinor_in, res,i);
			}

			for(int color=0; color < 3; ++color) {
				for(int spin=0; spin<2; ++spin) {

					//hspinor_out(i,spin,color,reim) = res(spin,color,reim);
					hspinor_out(i,spin,color)=res(color,spin);
				}
			}

		}); // parallel for


	}); // queue
	q.wait_and_throw();
}

#if 0
  template<typename T, typename VN, typename T2, int dir, int isign>
    void SyCLVProjectLatticePerm(const SyCLCBFineVSpinor<T,VN,4>& sycl_in,
			      SyCLCBFineVSpinor<T,VN,2>& sycl_hspinor_out, int _sites_per_team = 2)
{
	int num_sites = sycl_in.GetInfo().GetNumCBSites();
	const VSpinorView<T,VN>& spinor_in = sycl_in.GetData();
	VHalfSpinorView<T,VN>& hspinor_out = sycl_hspinor_out.GetData();

	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,SyCL::AUTO(),VN::VecLen);
	  SyCL::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    SyCL::parallel_for(SyCL::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {

			HalfSpinorSiteView<T2> res;



			if( dir == 0) {
			  SyCLProjectDir0Perm<T,VN,T2,isign>(spinor_in, res,i, VN::XPermuteMask);
			}
			else if (dir == 1) {
			  //			SyCLProjectDir<T,1>(spinor_in,plus_minus,res,i);
			  SyCLProjectDir1Perm<T,VN,T2,isign>(spinor_in, res,i, VN::YPermuteMask);
			}
			else if (dir == 2 ) {
			  SyCLProjectDir2Perm<T,VN,T2,isign>(spinor_in, res,i, VN::ZPermuteMask);
			}
			else {
			  SyCLProjectDir3Perm<T,VN,T2,isign>(spinor_in, res,i, VN::TPermuteMask);
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
     void SyCLLatticeVHSpinorPerm(SyCLCBFineVSpinor<T,VN,2>& sycl_in,
 			      int _sites_per_team = 2)
 {
 	int num_sites = sycl_in.GetInfo().GetNumCBSites();
 	VHalfSpinorView<T,VN>& spinor = sycl_in.GetData();

 	const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,SyCL::AUTO(),VN::VecLen);
 	  SyCL::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
 		    const int start_idx = team.league_rank()*_sites_per_team;
 		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
 		    SyCL::parallel_for(SyCL::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {

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
#endif
} // namespace

