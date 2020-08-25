/*
 * sycl_vdslash_subgroup.h
 *
 *  Created on: Oct 28, 2019
 *      Author: bjoo
 */

#pragma once
#include "CL/sycl.hpp"                          // SyCL Defs
#include "sycl_dslash_config.h"                 // Compile time configs for vector types etc.
#include "dslash/dslash_defaults.h"             // Default layouts (depend on config)
#include "lattice/constants.h"                  // Constants, enums, and index types
#include "lattice/lattice_info.h"               // class LatticeInfo, geometry etc.
#include "dslash/dslash_vectype_sycl_subgroup.h"         // Vector complex type
#include "dslash/dslash_vnode.h"                // Virtual node type
#include "dslash/sycl_view.h"                   // View Type, Layout indexing
#include "dslash/sycl_vtypes.h"                 // Spinors, Gauges
#include "dslash/sycl_vspinproj_subgroup.h"              // Spin project/recons on a site
#include "dslash/sycl_vmatvec_subgroup.h"                // MatVec/ AdjsMatVec on a site
#include "dslash/sycl_vneighbor_table.h"        // a 'site' table

#include <unordered_map>
#include <utility>
#include <chrono>

namespace MG {

namespace SGVDSlashInternal {

// Hacktastic:
[[cl::intel_reqd_sub_group_size(8)]]
 void force_sub_group_size8(){}

#if 0
[[cl::intel_reqd_sub_group_size(16)]]
 void force_sub_group_size16(){}
#endif

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

} //namespace


template<typename VN,
typename GT,
typename ST,
int isign,
int target_cb>
struct SGVDslashFunctor {

	SyCLSGVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::read> s_in;
	SyCLSGVGaugeViewAccessor<GT,VN, cl::sycl::access::mode::read> g_in;
	SyCLSGVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::write> s_out;
	SiteTableAccess neigh_table;


	SGVDslashFunctor(SyCLSGVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::read> _s_in,
			SyCLSGVGaugeViewAccessor<GT,VN, cl::sycl::access::mode::read> _g_in,
			SyCLSGVSpinorViewAccessor<ST,VN,cl::sycl::access::mode::write> _s_out,
			SiteTableAccess _neigh_table)
	: s_in(_s_in), g_in(_g_in), s_out(_s_out), neigh_table(_neigh_table ) {}



	using FType = typename BaseType<ST>::Type;
	using TST = MGComplex<FType>;

        [[cl::intel_reqd_sub_group_size(VN::VecLen)]]
	inline
	void operator()(cl::sycl::nd_item<2> nd_idx) const {

		// VN Grid site id.
		size_t site = nd_idx.get_global_id(0);

		// Get the subgroup that I am a part of
		sycl::intel::sub_group sg = nd_idx.get_sub_group();

		IndexArray site_coords=LayoutLeft::coords(site,neigh_table._cb_dims);
		size_t xcb=site_coords[0];
		size_t y = site_coords[1];
		size_t z = site_coords[2];
		size_t t = site_coords[3];

		size_t n_idx;
	        bool do_permute;

		SpinorSiteView<TST> res_sum ;
		HalfSpinorSiteView<TST> proj_res ;
		HalfSpinorSiteView<TST> mult_proj_res ;

		// Init output spinor to zero
		for(int spin=0; spin < 4; ++spin ) {
			for(int color=0; color < 3; ++color) {
				ComplexZero(res_sum(color,spin));
			}
		}

		// Accumulate the directions...
		// T - minus
		neigh_table.NeighborTMinus(xcb,y,z,t,n_idx,do_permute );

		do_permute ? SyCLProjectDir3Perm<ST,VN,TST,isign>(s_in, proj_res,n_idx, sg) : SyCLProjectDir3<ST,VN,TST,isign>(s_in, proj_res,n_idx, sg);
		mult_adj_u_halfspinor<GT,VN,TST,0>(g_in, proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir3<TST,VN,isign>(mult_proj_res,res_sum);

		// Z - minus
		neigh_table.NeighborZMinus(xcb,y,z,t,n_idx, do_permute );

		do_permute ? SyCLProjectDir2Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx, sg) : SyCLProjectDir2<ST,VN,TST,isign>(s_in, proj_res, n_idx, sg);
		mult_adj_u_halfspinor<GT,VN,TST,1>(g_in, proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir2<TST,VN,isign>(mult_proj_res,res_sum);


		// Y - minus
		neigh_table.NeighborYMinus(xcb,y,z,t,n_idx,do_permute);
		do_permute ? SyCLProjectDir1Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx, sg) : SyCLProjectDir1<ST,VN,TST,isign>(s_in, proj_res, n_idx, sg);
		mult_adj_u_halfspinor<GT,VN,TST,2>(g_in, proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir1<TST,VN,isign>(mult_proj_res,res_sum);

		// X - minus
		neigh_table.NeighborXMinus(xcb,y,z,t,target_cb,n_idx,do_permute);
		do_permute ? SyCLProjectDir0Perm<ST,VN,TST,isign>(s_in, proj_res, n_idx, sg) : SyCLProjectDir0<ST,VN,TST,isign>(s_in, proj_res, n_idx, sg);
		mult_adj_u_halfspinor<GT,VN,TST,3>(g_in, proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir0<TST,VN,isign>(mult_proj_res,res_sum);

		// X - plus
		neigh_table.NeighborXPlus(xcb,y,z,t,target_cb,n_idx, do_permute);
		do_permute ? SyCLProjectDir0Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg) : SyCLProjectDir0<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg);
		mult_u_halfspinor<GT,VN,TST,4>(g_in,proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir0<TST,VN,-isign>(mult_proj_res, res_sum);

		// Y - plus
		neigh_table.NeighborYPlus(xcb,y,z,t, n_idx, do_permute);
		do_permute ? SyCLProjectDir1Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg) : SyCLProjectDir1<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg);
		mult_u_halfspinor<GT,VN,TST,5>(g_in,proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir1<TST,VN,-isign>(mult_proj_res, res_sum);

		// Z - plus
		neigh_table.NeighborZPlus(xcb,y,z,t, n_idx, do_permute);
		do_permute ? SyCLProjectDir2Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg) : SyCLProjectDir2<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg);
		mult_u_halfspinor<GT,VN,TST,6>(g_in,proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir2<TST,VN,-isign>(mult_proj_res, res_sum);


		// T- plus
		neigh_table.NeighborTPlus(xcb,y,z,t, n_idx, do_permute);
		do_permute ? SyCLProjectDir3Perm<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg) : SyCLProjectDir3<ST,VN, TST,-isign>(s_in,proj_res,n_idx,sg);
		mult_u_halfspinor<GT,VN,TST,7>(g_in,proj_res,mult_proj_res,site,sg);
		SyCLRecons23Dir3<TST,VN,-isign>(mult_proj_res, res_sum);



		// Stream out spinor
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {
				Store( s_out.offset(site,spin,color,0,0), s_out.get_pointer(), res_sum(color,spin), sg);
			}
		}

	}


};




template<typename VN, typename GT, typename ST>
class SyCLSGVDslash {

	const LatticeInfo& _info;
	SiteTable _neigh_table;
	cl::sycl::queue _q;
	size_t _max_team_size;
	std::unordered_map< std::pair<int,int>, size_t, SGVDSlashInternal::pair_hash > tunings;

public:
	SyCLSGVDslash(const LatticeInfo& info,  cl::sycl::queue& q ) : _info(info),
			_neigh_table(info.GetCBLatticeDimensions()[0],info.GetCBLatticeDimensions()[1],info.GetCBLatticeDimensions()[2],info.GetCBLatticeDimensions()[3]),
			_q(q) {
		auto device = q.get_device();
		size_t max_witem_sizex = device.get_info<cl::sycl::info::device::max_work_item_sizes>()[0];
		size_t max_wgroup_size = device.get_info<cl::sycl::info::device::max_work_group_size>();

		// Dunno which of these is the limiter, so take minimum
		_max_team_size = 256/VN::VecLen;
	}

	size_t tune(const SyCLCBFineSGVSpinor<ST,VN,4>& fine_in,
				  const SyCLCBFineSGVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
				  SyCLCBFineSGVSpinor<ST,VN,4>& fine_out,
				  int plus_minus)
		{
			size_t num_sites = fine_in.GetInfo().GetNumCBSites();
			size_t real_max_team_size = _max_team_size <= num_sites ? _max_team_size : num_sites;
			std::cout << "Tuning:" << std::endl;
			for(size_t team_size=1; team_size <= real_max_team_size; team_size *=2 ) {
				std::cout << "Compiling with team size = " << team_size << std::endl;
				(*this)(fine_in,gauge_in,fine_out,plus_minus, team_size);
			}
			int target_cb = fine_out.GetCB();

			double fastest_time = 3600; // Surely it won't take an hour ever....
			size_t fastest_team = 0;

			for(size_t team_size=1; team_size <= real_max_team_size; team_size *=2 ) {
				std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
				{
					(*this)(fine_in,gauge_in,fine_out,plus_minus,team_size);
				} // all queues finish here.
				std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
				double time_taken = (std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)).count();
				std::cout << "Tuning: team_size=" << team_size << " time = " << time_taken << " (sec)" << std::endl;
				if(time_taken <= fastest_time) {
					fastest_time = time_taken;
					fastest_team = team_size;
				}
			}
			std::cout << "Fastest team size is " << fastest_team << std::endl;
			tunings[{target_cb,plus_minus}] =  fastest_team;
			return fastest_team;


		}

	void operator()(const SyCLCBFineSGVSpinor<ST,VN,4>& fine_in,
			const SyCLCBFineSGVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			SyCLCBFineSGVSpinor<ST,VN,4>& fine_out,
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


	void operator()(const SyCLCBFineSGVSpinor<ST,VN,4>& fine_in,
			const SyCLCBFineSGVGaugeFieldDoubleCopy<GT,VN>& gauge_in,
			SyCLCBFineSGVSpinor<ST,VN,4>& fine_out,
			int plus_minus,
			size_t team_size)
	{
		int source_cb = fine_in.GetCB();
		int target_cb = (source_cb == EVEN) ? ODD : EVEN;


		SyCLSGVSpinorView<ST,VN> s_in = fine_in.GetData();
		SyCLSGVGaugeView<GT,VN> g_in = gauge_in.GetData();
		SyCLSGVSpinorView<ST,VN> s_out = fine_out.GetData();

		IndexArray cb_latdims = _info.GetCBLatticeDimensions();

		size_t num_sites = _info.GetNumCBSites();

		//cl::sycl::nd_range<1> dispatch_space( {VN::VecLen*num_sites},{team_size*VN::VecLen});
		cl::sycl::nd_range<2> dispatch_space( {num_sites,VN::VecLen},{team_size,VN::VecLen});
		if( plus_minus == 1 ) {
			if (target_cb == 0 ) {

				_q.submit( [&](cl::sycl::handler& cgh) {


					SGVDslashFunctor<VN,GT,ST,1,0> f{
						s_in.template get_access<cl::sycl::access::mode::read>(cgh),
								g_in.template get_access<cl::sycl::access::mode::read>(cgh),
								s_out.template get_access<cl::sycl::access::mode::write>(cgh),
								_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<SGVDSlashInternal::dslash_loop<VN,GT,ST,1,0>>(dispatch_space, f);
				});
			}
			else {

				_q.submit( [&](cl::sycl::handler& cgh) {

					SGVDslashFunctor<VN,GT,ST,1,1> f{
						s_in.template get_access<cl::sycl::access::mode::read>(cgh),
								g_in.template get_access<cl::sycl::access::mode::read>(cgh),
								s_out.template get_access<cl::sycl::access::mode::write>(cgh),
								_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<SGVDSlashInternal::dslash_loop<VN,GT,ST,1,1>>(dispatch_space,f);

				});


			}
		}
		else {
			if( target_cb == 0 ) {

				_q.submit( [&](cl::sycl::handler& cgh) {

					SGVDslashFunctor<VN,GT,ST,-1,0> f{
						s_in.template get_access<cl::sycl::access::mode::read>(cgh),
								g_in.template get_access<cl::sycl::access::mode::read>(cgh),
								s_out.template get_access<cl::sycl::access::mode::write>(cgh),
								_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<SGVDSlashInternal::dslash_loop<VN,GT,ST,-1,0>>(dispatch_space,f);

				});



			}
			else {

				_q.submit( [&](cl::sycl::handler& cgh) {

					SGVDslashFunctor<VN,GT,ST,-1,1> f{
						s_in.template get_access<cl::sycl::access::mode::read>(cgh),
								g_in.template get_access<cl::sycl::access::mode::read>(cgh),
								s_out.template get_access<cl::sycl::access::mode::write>(cgh),
								_neigh_table.template get_access<cl::sycl::access::mode::read>(cgh)
					};

					cgh.parallel_for<SGVDSlashInternal::dslash_loop<VN,GT,ST,-1,1>>(dispatch_space, f);

				});

			}
		}
		_q.wait_and_throw();

	}
};




} // namespace
