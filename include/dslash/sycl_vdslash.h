/*
 * kokkos_dslash_paralell_for.h
 *
 *  Created on: May 30, 2017
 *      Author: bjoo
 */

#pragma once
#include "CL/sycl.hpp"                          // SyCL Defs
#include "sycl_dslash_config.h"                 // Compile time configs for vector types etc.
#include "dslash/dslash_defaults.h"             // Default layouts (depend on config)
#include "lattice/constants.h"                  // Constants, enums, and index types
#include "lattice/lattice_info.h"               // class LatticeInfo, geometry etc.
#include "dslash/dslash_vectype_sycl.h"         // Vector complex type
#include "dslash/dslash_vnode.h"                // Virtual node type
#include "dslash/sycl_view.h"                   // View Type, Layout indexing
#include "dslash/sycl_vtypes.h"                 // Spinors, Gauges
#include "dslash/sycl_vspinproj.h"              // Spin project/recons on a site
#include "dslash/sycl_vmatvec.h"                // MatVec/ AdjsMatVec on a site
#include "dslash/sycl_vneighbor_table.h"        // a 'site' table


namespace MG {


template<typename VN,
typename GT,
typename ST>
struct VDslashFunctor {

	SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::read> s_in;
	SyCLVGaugeViewAccessor<GT,VN, cl::sycl::access::mode::read> g_in;
	SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::write> s_out;
	SiteTableAccess neigh_table;

	VDslashFunctor(SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::read> _s_in,
			SyCLVGaugeViewAccessor<GT,VN, cl::sycl::access::mode::read> _g_in,
			SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::write> _s_out,
			SiteTableAccess _neigh_table )
	: s_in(_s_in), g_in(_g_in), s_out(_s_out), neigh_table(_neigh_table ) {}

	using FType = typename BaseType<ST>::Type;
	using TST = SIMDComplexSyCL<FType,VN::VecLen>;

	template<int isign, int target_cb>
	inline
	void dslash_func(cl::sycl::id<1> idx) const  {
		size_t site = idx[0];
		IndexArray site_coords=LayoutLeft::coords(site,neigh_table._cb_dims);
		size_t xcb=site_coords[0];
		size_t y = site_coords[1];
		size_t z = site_coords[2];
		size_t t = site_coords[3];

		size_t n_idx;
		bool do_perm;


		SpinorSiteView<TST> res_sum ;
		HalfSpinorSiteView<TST> proj_res ;
		HalfSpinorSiteView<TST> mult_proj_res ;

#pragma unroll
		for(int spin=0; spin < 4; ++spin ) {

#pragma unroll
			for(int color=0; color < 3; ++color) {
				ComplexZero<FType,VN::VecLen>(res_sum(color,spin));
			}
		}


		// T - minus
		neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_perm );


		if( do_perm ) {
			SyCLProjectDir3Perm<ST,VN,TST,isign>(s_in, proj_res,n_idx);
		}
		else {
			SyCLProjectDir3<ST,VN,TST,isign>(s_in, proj_res,n_idx);
		}
//      mult_adj_u_halfspinor<GT,VN,TST,0>(g_in, proj_res,mult_proj_res,site);
//		SyCLRecons23Dir3<TST,VN,isign>(mult_proj_res,res_sum);
		SyCLRecons23Dir3<TST,VN,isign>(proj_res,res_sum);

		// Z - minus
		neigh_table.NeighborZMinus(xcb,y,z,t,n_idx, do_perm );

		if( do_perm ) {
			SyCLProjectDir2Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		else {
			SyCLProjectDir2<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
//		mult_adj_u_halfspinor<GT,VN,TST,1>(g_in, proj_res,mult_proj_res,site);
//		SyCLRecons23Dir2<TST,VN,isign>(mult_proj_res,res_sum);
		SyCLRecons23Dir2<TST,VN,isign>(proj_res,res_sum);


		// Y - minus
		neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_perm);

		if( do_perm ) {
			SyCLProjectDir1Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		else {
			SyCLProjectDir1<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}

//		mult_adj_u_halfspinor<GT,VN,TST,2>(g_in, proj_res,mult_proj_res,site);
//		SyCLRecons23Dir1<TST,VN,isign>(mult_proj_res,res_sum);
		SyCLRecons23Dir1<TST,VN,isign>(proj_res,res_sum);
#if 0
		// X - minus
		neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_perm);

		if( do_perm ) {
			SyCLProjectDir0Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		else {
			SyCLProjectDir0<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
//		mult_adj_u_halfspinor<GT,VN,TST,3>(g_in, proj_res,mult_proj_res,site);
// 		SyCLRecons23Dir0<TST,VN,isign>(mult_proj_res,res_sum);
		SyCLRecons23Dir0<TST,VN,isign>(proj_res,res_sum);

		// X - plus
		neigh_table.NeighborXPlus(xcb,y,z,t,target_cb,n_idx, do_perm);

		if ( do_perm ) {
			SyCLProjectDir0Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir0<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
//		mult_u_halfspinor<GT,VN,TST,4>(g_in,proj_res,mult_proj_res,site);
	//	SyCLRecons23Dir0<TST,VN,-isign>(mult_proj_res, res_sum);
		SyCLRecons23Dir0<TST,VN,-isign>(proj_res, res_sum);
#endif

		// Y - plus
		neigh_table.NeighborYPlus(xcb,y,z,t, n_idx, do_perm);
		if( do_perm ) {
			SyCLProjectDir1Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir1<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
//		mult_u_halfspinor<GT,VN,TST,5>(g_in,proj_res,mult_proj_res,site);
//		SyCLRecons23Dir1<TST,VN,-isign>(mult_proj_res, res_sum);
		SyCLRecons23Dir1<TST,VN,-isign>(proj_res, res_sum);

		// Z - plus
		neigh_table.NeighborZPlus(xcb,y,z,t, n_idx, do_perm);
		if( do_perm ) {
			SyCLProjectDir2Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir2<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
//		mult_u_halfspinor<GT,VN,TST,6>(g_in,proj_res,mult_proj_res,site);
//		SyCLRecons23Dir2<TST,VN,-isign>(mult_proj_res, res_sum);
		SyCLRecons23Dir2<TST,VN,-isign>(proj_res, res_sum);

		// T- plus
		neigh_table.NeighborTPlus(xcb,y,z,t, n_idx, do_perm);
		if( do_perm ) {
			SyCLProjectDir3Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir3<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}

//		mult_u_halfspinor<GT,VN,TST,7>(g_in,proj_res,mult_proj_res,site);
//		SyCLRecons23Dir3<TST,VN,-isign>(mult_proj_res, res_sum);
		SyCLRecons23Dir3<TST,VN,-isign>(proj_res, res_sum);

		// Stream out spinor
#pragma unroll
		for(int spin=0; spin < 4; ++spin) {
#pragma unroll
			for(int color=0; color < 3; ++color) {

				//     Stream(s_out(site,spin,color),res_sum(color,spin));
				ComplexCopy<FType,VN::VecLen>(s_out(site,spin,color),res_sum(color,spin));
			}
		}

	}


};

template<typename VN, typename GT, typename ST, int dir, int cb>
class dslash1;

template<typename VN, typename GT, typename ST, int dir, int cb>
class dslash2;

template<typename VN, typename GT, typename ST, int dir, int cb>
class dslash3;

template<typename VN, typename GT, typename ST, int dir, int cb>
class dslash4;

template<typename VN, typename GT, typename ST>
   class SyCLVDslash {

	const LatticeInfo& _info;
	SiteTable _neigh_table;
public:

	SyCLVDslash(const LatticeInfo& info) : _info(info),
	_neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3]) {}
	
	void operator()(const SyCLCBFineVSpinor<ST,VN,4>& fine_in,
			const SyCLCBFineVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			SyCLCBFineVSpinor<ST,VN,4>& fine_out,
			int plus_minus)
	{
		int source_cb = fine_in.GetCB();
		int target_cb = (source_cb == EVEN) ? ODD : EVEN;
		SyCLVSpinorView<ST,VN> s_in = fine_in.GetData();
		SyCLVGaugeView<GT,VN> g_in = gauge_in.GetData();
		SyCLVSpinorView<ST,VN> s_out = fine_out.GetData();

		IndexArray cb_latdims = _info.GetCBLatticeDimensions();

		int num_sites = fine_in.GetInfo().GetNumCBSites();

		cl::sycl::queue q;

		if( plus_minus == 1 ) {
			if (target_cb == 0 ) {
				//	      VDslashFunctor<VN,GT,ST,1,0> f = {s_in, g_in,  s_out, _neigh_table};
				q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash1<VN,GT,ST,1,0>>(cl::sycl::range<1>(num_sites),
							[=](cl::sycl::id<1> idx) {
						f.template dslash_func<1,0>(idx);
					});

				});
			}
			else {
				//	      VDslashFunctor<VN,GT,ST,1,1> f = {s_in, g_in,  s_out, _neigh_table};
				q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash2<VN,GT,ST,1,1>>(cl::sycl::range<1>(num_sites),
							[=](cl::sycl::id<1> idx) {
										f.template dslash_func<1,1>(idx);
									});

				});


			}
		}
		else {
			if( target_cb == 0 ) {
				//	      VDslashFunctor<VN,GT,ST,-1,0> f = {s_in, g_in,  s_out, _neigh_table};
				q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash3<VN,GT,ST,-1,0>>(cl::sycl::range<1>(num_sites),			[=](cl::sycl::id<1> idx) {
						f.template dslash_func<-1,0>(idx);
					});

				});



			}
			else {
				//VDslashFunctor<VN,GT,ST,-1,1> f = {s_in, g_in, s_out, _neigh_table };
				q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash4<VN,GT,ST,-1,1>>(cl::sycl::range<1>(num_sites), 			[=](cl::sycl::id<1> idx) {
						f.template dslash_func<-1,1>(idx);
					});

				});

			}
		}

	}
 };




} // namespace
