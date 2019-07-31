/*
 * kokkos_dslash.h
 *
 *  Created on: May 30, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_VDSLASH_H_
#define TEST_KOKKOS_KOKKOS_VDSLASH_H_
#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include "kokkos_defaults.h"
#include "kokkos_types.h"
#include "kokkos_vtypes.h"
#include "kokkos_spinproj.h"
#include "kokkos_vspinproj.h"
#include "kokkos_vnode.h"
#include "kokkos_vmatvec.h"
#include "kokkos_traits.h"
#include "kokkos_vneighbor_table.h"

#define MG_KOKKOS_USE_MDRANGE

namespace MG {



enum DirIdx { T_MINUS=0, Z_MINUS=1, Y_MINUS=2, X_MINUS=3, X_PLUS=4, Y_PLUS=5, Z_PLUS=6, T_PLUS=7 };





 template<typename VN,
   typename GT, 
   typename ST, 
   typename TGT, 
   typename TST, 
   const int isign, const int target_cb>
   struct VDslashFunctor { 

     VSpinorView<ST,VN> s_in;
     VGaugeView<GT,VN> g_in;
     VSpinorView<ST,VN> s_out;
     SiteTable<VN> neigh_table;

#ifdef MG_KOKKOS_USE_MDRANGE 
     KOKKOS_FORCEINLINE_FUNCTION
       void operator()(const int& xcb, const int& y, const int& z, const int& t) const
     {

       int site = neigh_table.coords_to_idx(xcb,y,z,t);
#else
     KOKKOS_FORCEINLINE_FUNCTION
     void operator()(const int site) const {
	int xcb,y,z,t;
	neigh_table.idx_to_coords(site,xcb,y,z,t);
#endif
       int n_idx;
       typename VN::MaskType mask;
     
      // Warning: GCC Alignment Attribute!
    		 // Site Sum: Not a true Kokkos View
#if 0
    		 SpinorSiteView<TST> res_sum __attribute__((aligned(64)));
    		 HalfSpinorSiteView<TST> proj_res  __attribute__((aligned(64)));
    		 HalfSpinorSiteView<TST> mult_proj_res  __attribute__((aligned(64)));
#else
                 SpinorSiteView<TST> res_sum ;
                 HalfSpinorSiteView<TST> proj_res ;
                 HalfSpinorSiteView<TST> mult_proj_res ;
#endif
                 // Zero Result
#pragma unroll
                 for(int spin=0; spin < 4; ++spin ) { 

#pragma unroll
                         for(int color=0; color < 3; ++color) {
                                 ComplexZero(res_sum(color,spin));
                         }
                 }


    		 // T - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborTMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,mask);
#endif
    		 KokkosProjectDir3Perm<ST,VN,TST,isign>(s_in, proj_res,n_idx,mask);
    		 mult_adj_u_halfspinor<GT,VN,TST,0>(g_in, proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir3<TST,VN,isign>(mult_proj_res,res_sum);

    		 // Z - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborZMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborZMinus(xcb,y,z,t,n_idx,mask);
#endif
    		 KokkosProjectDir2Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx, mask);
    		 mult_adj_u_halfspinor<GT,VN,TST,1>(g_in, proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir2<TST,VN,isign>(mult_proj_res,res_sum);

    		 // Y - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborYMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,mask);
#endif
    		 KokkosProjectDir1Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx,mask);
    		 mult_adj_u_halfspinor<GT,VN,TST,2>(g_in, proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir1<TST,VN,isign>(mult_proj_res,res_sum);

    		 // X - minus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborXMinus(site,target_cb,n_idx,mask);
#else
    		 neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,mask);
#endif
    		 KokkosProjectDir0Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx,mask);
    		 mult_adj_u_halfspinor<GT,VN,TST,3>(g_in, proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir0<TST,VN,isign>(mult_proj_res,res_sum);


    		 // X - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborXPlus(site, target_cb, n_idx, mask);
#else

    		 neigh_table.NeighborXPlus(xcb,y,z,t,target_cb,n_idx, mask);
#endif
    		 KokkosProjectDir0Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
    		 mult_u_halfspinor<GT,VN,TST,4>(g_in,proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir0<TST,VN,-isign>(mult_proj_res, res_sum);

    		 // Y - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
    		 neigh_table.NeighborYPlus(site, target_cb, n_idx, mask);
#else
    		 neigh_table.NeighborYPlus(xcb,y,z,t, n_idx, mask);
#endif
    		 KokkosProjectDir1Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
    		 mult_u_halfspinor<GT,VN,TST,5>(g_in,proj_res,mult_proj_res,site);
    		 KokkosRecons23Dir1<TST,VN,-isign>(mult_proj_res, res_sum);

    		 // Z - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
		        	 neigh_table.NeighborZPlus(site, target_cb, n_idx, mask);
#else
			 neigh_table.NeighborZPlus(xcb,y,z,t, n_idx, mask);
#endif
			 KokkosProjectDir2Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
			 mult_u_halfspinor<GT,VN,TST,6>(g_in,proj_res,mult_proj_res,site);
			 KokkosRecons23Dir2<TST,VN,-isign>(mult_proj_res, res_sum);

			 // T - plus
#if defined(MG_KOKKOS_USE_NEIGHBOR_TABLE)
			 neigh_table.NeighborTPlus(site,target_cb, n_idx, mask);
#else
			 neigh_table.NeighborTPlus(xcb,y,z,t, n_idx, mask);
#endif
			 KokkosProjectDir3Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,mask);
			 mult_u_halfspinor<GT,VN,TST,7>(g_in,proj_res,mult_proj_res,site);
			 KokkosRecons23Dir3<TST,VN,-isign>(mult_proj_res, res_sum);

			 // Stream out spinor
#pragma unroll
			 for(int spin=0; spin < 4; ++spin) {
#pragma unroll
			   for(int color=0; color < 3; ++color) {

			     Stream(s_out(site,spin,color),res_sum(color,spin));
			   }
			 }

     }

     
   };

 template<typename VN, typename GT, typename ST,  typename TGT, typename TST>
   class KokkosVDslash {
 public:
	const LatticeInfo& _info;
	SiteTable<VN> _neigh_table;
public:

 KokkosVDslash(const LatticeInfo& info) : _info(info),
	  _neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3])
	  {}
	
	void operator()(const KokkosCBFineVSpinor<ST,VN,4>& fine_in,
			const KokkosCBFineVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			KokkosCBFineVSpinor<ST,VN,4>& fine_out,
		      int plus_minus, const IndexArray& blocks) const
	{
	  int source_cb = fine_in.GetCB();
	  int target_cb = (source_cb == EVEN) ? ODD : EVEN;
	  const VSpinorView<ST,VN>& s_in = fine_in.GetData();
	  const VGaugeView<GT,VN>& g_in = gauge_in.GetData();
	  VSpinorView<ST,VN>& s_out = fine_out.GetData();

	  IndexArray cb_latdims = _info.GetCBLatticeDimensions();

#ifdef MG_KOKKOS_USE_MDRANGE
	  MDPolicy policy({0,0,0,0},
			  	  {cb_latdims[0],cb_latdims[1],cb_latdims[2],cb_latdims[3]},
			  	  {blocks[0],blocks[1],blocks[2],blocks[3]}
	  	  	  	  );
#else
	  int num_sites = fine_in.GetInfo().GetNumCBSites();
#endif

	  if( plus_minus == 1 ) {
	    if (target_cb == 0 ) {
	      VDslashFunctor<VN,GT,ST,TGT,TST,1,0> f = {s_in, g_in, s_out,
	    		  _neigh_table};

#ifdef MG_KOKKOS_USE_MDRANGE
	      Kokkos::Experimental::md_parallel_for(policy, f); // Outer Lambda
#else
	      Kokkos::parallel_for(SimpleRange(0,num_sites),f);
#endif

	    }
	    else {
	      VDslashFunctor<VN,GT,ST,TGT,TST,1,1> f = {s_in, g_in, s_out,
	    		   _neigh_table};

#ifdef MG_KOKKOS_USE_MDRANGE
	      Kokkos::Experimental::md_parallel_for(policy, f); // Outer Lambda
#else
	      Kokkos::parallel_for(SimpleRange(0,num_sites),f);
#endif
	    }
	  }
	  else {
	    if( target_cb == 0 ) { 
	      VDslashFunctor<VN,GT,ST,TGT,TST,-1,0> f = {s_in, g_in,  s_out,
	    		  _neigh_table};

#ifdef MG_KOKKOS_USE_MDRANGE
	      Kokkos::Experimental::md_parallel_for(policy, f); // Outer Lambda
#else
	      Kokkos::parallel_for(SimpleRange(0,num_sites),f);
#endif
	    }
	    else {
	      VDslashFunctor<VN,GT,ST,TGT,TST,-1,1> f = {s_in, g_in, s_out,
	    		  _neigh_table };

#ifdef MG_KOKKOS_USE_MDRANGE
	      Kokkos::Experimental::md_parallel_for(policy, f); // Outer Lambda
#else
	      Kokkos::parallel_for(SimpleRange(0,num_sites),f);
#endif
	    }
	  }
	  
	}

};




};




#endif /* TEST_KOKKOS_KOKKOS_DSLASH_H_ */
