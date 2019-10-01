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

#include <unordered_map>
#include <utility>
#include <chrono>
namespace MG {


template<typename VN,
typename GT,
typename ST,
int isign,
int target_cb>
struct VDslashFunctor {

	SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::read> s_in;
	SyCLVGaugeViewAccessor<GT,VN, cl::sycl::access::mode::read> g_in;
	SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::discard_write> s_out;
	SiteTableAccess neigh_table;

	VDslashFunctor(SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::read> _s_in,
			SyCLVGaugeViewAccessor<GT,VN, cl::sycl::access::mode::read> _g_in,
			SyCLVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::discard_write> _s_out,
			SiteTableAccess _neigh_table )
	: s_in(_s_in), g_in(_g_in), s_out(_s_out), neigh_table(_neigh_table ) {}

	using FType = typename BaseType<ST>::Type;
	using TST = SIMDComplexSyCL<FType,VN::VecLen>;


	inline
	void operator()(cl::sycl::nd_item<1> nd_idx) const {
		size_t site = nd_idx.get_global_id(0);
//		if (site >= nd_idx.get_global_range(0) ) return ;

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

		// Init output spinor to zero
#pragma unroll
		for(int spin=0; spin < 4; ++spin ) {

#pragma unroll
			for(int color=0; color < 3; ++color) {
				ComplexZero<FType,VN::VecLen>(res_sum(color,spin));
			}
		}

		// Accumulate the directions...

		// T - minus
		neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_perm );
		if( do_perm ) {
			SyCLProjectDir3Perm<ST,VN,TST,isign>(s_in, proj_res,n_idx);
		}
		else {
			SyCLProjectDir3<ST,VN,TST,isign>(s_in, proj_res,n_idx);
		}
		mult_adj_u_halfspinor<GT,VN,TST,0>(g_in, proj_res,mult_proj_res,site);
		SyCLRecons23Dir3<TST,VN,isign>(mult_proj_res,res_sum);

		// Z - minus
		neigh_table.NeighborZMinus(xcb,y,z,t,n_idx, do_perm );
		if( do_perm ) {
			SyCLProjectDir2Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		else {
			SyCLProjectDir2<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		mult_adj_u_halfspinor<GT,VN,TST,1>(g_in, proj_res,mult_proj_res,site);
		SyCLRecons23Dir2<TST,VN,isign>(mult_proj_res,res_sum);

		// Y - minus
		neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_perm);
		if( do_perm ) {
			SyCLProjectDir1Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		else {
			SyCLProjectDir1<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		mult_adj_u_halfspinor<GT,VN,TST,2>(g_in, proj_res,mult_proj_res,site);
		SyCLRecons23Dir1<TST,VN,isign>(mult_proj_res,res_sum);

		// X - minus
		neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_perm);
		if( do_perm ) {
			SyCLProjectDir0Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		else {
			SyCLProjectDir0<ST,VN,TST,isign>(s_in, proj_res, n_idx);
		}
		mult_adj_u_halfspinor<GT,VN,TST,3>(g_in, proj_res,mult_proj_res,site);
 		SyCLRecons23Dir0<TST,VN,isign>(mult_proj_res,res_sum);

		// X - plus
		neigh_table.NeighborXPlus(xcb,y,z,t,target_cb,n_idx, do_perm);
		if ( do_perm ) {
			SyCLProjectDir0Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir0<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		mult_u_halfspinor<GT,VN,TST,4>(g_in,proj_res,mult_proj_res,site);
		SyCLRecons23Dir0<TST,VN,-isign>(mult_proj_res, res_sum);

		// Y - plus
		neigh_table.NeighborYPlus(xcb,y,z,t, n_idx, do_perm);
		if( do_perm ) {
			SyCLProjectDir1Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir1<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		mult_u_halfspinor<GT,VN,TST,5>(g_in,proj_res,mult_proj_res,site);
		SyCLRecons23Dir1<TST,VN,-isign>(mult_proj_res, res_sum);

		// Z - plus
		neigh_table.NeighborZPlus(xcb,y,z,t, n_idx, do_perm);
		if( do_perm ) {
			SyCLProjectDir2Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir2<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		mult_u_halfspinor<GT,VN,TST,6>(g_in,proj_res,mult_proj_res,site);
		SyCLRecons23Dir2<TST,VN,-isign>(mult_proj_res, res_sum);

		// T- plus
		neigh_table.NeighborTPlus(xcb,y,z,t, n_idx, do_perm);
		if( do_perm ) {
			SyCLProjectDir3Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		else {
			SyCLProjectDir3<ST,VN, TST,-isign>(s_in,proj_res,n_idx);
		}
		mult_u_halfspinor<GT,VN,TST,7>(g_in,proj_res,mult_proj_res,site);
		SyCLRecons23Dir3<TST,VN,-isign>(mult_proj_res, res_sum);


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
class dslash_loop;

struct pair_hash
{
	template<class T1, class T2>
	inline
	size_t operator()(const std::pair<T1,T2>& p) const {
		return p.first + 2*(p.second + 1);
	}
};
template<typename VN, typename GT, typename ST>
   class SyCLVDslash {

	const LatticeInfo& _info;
	SiteTable _neigh_table;
	cl::sycl::queue _q;


	size_t _max_work_group_size;
	std::unordered_map< std::pair<int,int>, size_t, pair_hash > tunings;
public:

#if 0
	SyCLVDslash(const LatticeInfo& info) : _info(info),
	_neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3]),
	_q(cl::sycl::queue()){}
#endif



	SyCLVDslash(const LatticeInfo& info,  cl::sycl::queue& q ) : _info(info),
	_neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3]),
	_q(q){
		auto device = q.get_device();
		auto max_witem_sizes = device.get_info<cl::sycl::info::device::max_work_item_sizes>();
		_max_work_group_size = max_witem_sizes[0];
		if ( _max_work_group_size > 256 ) _max_work_group_size = 256;
		//_max_work_group_size = device.get_info<cl::sycl::info::device::max_work_group_size>();
	}
	
	size_t tune(const SyCLCBFineVSpinor<ST,VN,4>& fine_in,
			  const SyCLCBFineVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			  SyCLCBFineVSpinor<ST,VN,4>& fine_out,
			  int plus_minus)
	{
		std::cout << "Tuning:" << std::endl;
		for(size_t workgroup_size=1; workgroup_size <= _max_work_group_size; workgroup_size *=2 ) {
			std::cout << "Workgroup size: Compiling " << workgroup_size << std::endl;
			(*this)(fine_in,gauge_in,fine_out,plus_minus, workgroup_size);
		}
		int target_cb = fine_out.GetCB();

		double fastest_time = 3600; // Surely it won't take an hour ever....
		size_t fastest_wgroup = 0;

		for(size_t workgroup_size=1; workgroup_size <= _max_work_group_size; workgroup_size *=2 ) {
			std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
			{
				(*this)(fine_in,gauge_in,fine_out,plus_minus,workgroup_size);
			} // all queues finish here.
			std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
			double time_taken = (std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)).count();
			std::cout << "Tuning: workgroup_size=" << workgroup_size << " time = " << time_taken << " (sec)" << std::endl;
			if(time_taken <= fastest_time) {
				fastest_time = time_taken;
				fastest_wgroup = workgroup_size;
			}
		}
		std::cout << "Fastest workgroup size is " << fastest_wgroup << std::endl;
		tunings[{target_cb,plus_minus}] =  fastest_wgroup;
		return fastest_wgroup;


	}


	void operator()(const SyCLCBFineVSpinor<ST,VN,4>& fine_in,
			const SyCLCBFineVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			SyCLCBFineVSpinor<ST,VN,4>& fine_out,
			int plus_minus)
	{
		int target_cb = fine_out.GetCB();
		auto iterator = tunings.find({target_cb,plus_minus});
		size_t wgroup_size;
		if ( iterator == tunings.end() ) {
			wgroup_size = this->tune(fine_in,gauge_in,fine_out,plus_minus);
		}
		else {
			wgroup_size = (*iterator).second;
		}

		(*this)(fine_in,gauge_in,fine_out,plus_minus,wgroup_size);
	}

	void operator()(const SyCLCBFineVSpinor<ST,VN,4>& fine_in,
			const SyCLCBFineVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			SyCLCBFineVSpinor<ST,VN,4>& fine_out,
			int plus_minus, size_t workgroup_size)
	{
		int source_cb = fine_in.GetCB();
		int target_cb = (source_cb == EVEN) ? ODD : EVEN;


		SyCLVSpinorView<ST,VN> s_in = fine_in.GetData();
		SyCLVGaugeView<GT,VN> g_in = gauge_in.GetData();
		SyCLVSpinorView<ST,VN> s_out = fine_out.GetData();

		IndexArray cb_latdims = _info.GetCBLatticeDimensions();

		size_t num_sites = fine_in.GetInfo().GetNumCBSites();
		cl::sycl::nd_range<1> dispatch_space( cl::sycl::range<1>{num_sites},
											  cl::sycl::range<1>{workgroup_size});

		if( plus_minus == 1 ) {
			if (target_cb == 0 ) {

				_q.submit( [&](cl::sycl::handler& cgh) {


					VDslashFunctor<VN,GT,ST,1,0> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::discard_write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash_loop<VN,GT,ST,1,0>>(dispatch_space, f);
				});
			}
			else {

				_q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST,1,1> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::discard_write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash_loop<VN,GT,ST,1,1>>(dispatch_space,f);

				});


			}
		}
		else {
			if( target_cb == 0 ) {

				_q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST,-1,0> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::discard_write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash_loop<VN,GT,ST,-1,0>>(dispatch_space,f);

				});



			}
			else {

				_q.submit( [&](cl::sycl::handler& cgh) {

					VDslashFunctor<VN,GT,ST,-1,1> f{
							s_in.template get_access<cl::sycl::access::mode::read>(cgh),
							g_in.template get_access<cl::sycl::access::mode::read>(cgh),
							s_out.template get_access<cl::sycl::access::mode::discard_write>(cgh),
							_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<dslash_loop<VN,GT,ST,-1,1>>(dispatch_space, f);

				});

			}
		}
		_q.wait_and_throw();
	}
 };




} // namespace
