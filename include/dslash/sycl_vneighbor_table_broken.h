#pragma once

#include <cstddef>
#include <utility>
#include "sycl_dslash_config.h"
#include "CL/sycl.hpp"
#include "dslash/sycl_view.h"
#include "dslash/dslash_defaults.h"
#include "lattice/constants.h"
namespace MG { 

#if defined(MG_USE_NEIGHBOR_TABLE)

void ComputeSiteTable(IndexArray cb_dims, View<std::pair<IndexType,bool>,3,DefaultNeighborTableLayout> _table) {
	IndexType num_sites = BodySize::bodySize(cb_dims);
	IndexType _n_xh = cb_dims[0];
	IndexType _n_x = 2*_n_xh;
	IndexType _n_y = cb_dims[1];
	IndexType _n_z = cb_dims[2];
	IndexType _n_t = cb_dims[3];

	cl::sycl::queue MyQueue;

	MyQueue.submit([&](cl::sycl::handler& cgh) {

		auto table = _table.get_access<cl::sycl::access::mode::write>(cgh);

		cgh.parallel_for<class fill>( cl::sycl::range<1>(num_sites), [=](cl::sycl::id<1> site_id) {

			IndexType site=site_id[0];

			for(IndexType target_cb=0; target_cb < 2; ++target_cb) {

				// Break down site index IndexTypeo xcb, y,z and t
				IndexArray coords = LayoutLeft::coords(site, cb_dims);
				IndexType x_cb = coords[0];
				IndexType y = coords[1];
				IndexType z = coords[2];
				IndexType t = coords[3];


				// Global, uncheckerboarded x, assumes cb = (x + y + z + t ) & 1
				IndexType x = 2*x_cb + ((target_cb+y+z+t)&0x1);

				if( t > 0 ) {
					table(site,target_cb,T_MINUS) = std::make_pair( LayoutLeft::index( {x_cb,y,z,t-1},cb_dims ),false);
				}
				else {
					table(site,target_cb,T_MINUS) = std::make_pair( LayoutLeft::index( {x_cb,y,z,_n_t-1},cb_dims ),true);
				}

				if( z > 0 ) {
					table(site,target_cb,Z_MINUS) = std::make_pair( LayoutLeft::index( {x_cb,y,z-1,t}, cb_dims ), false);
				}
				else {
					table(site,target_cb,Z_MINUS) = std::make_pair( LayoutLeft::index( {x_cb,y,_n_z-1,t}, cb_dims ), true);
				}

				if( y > 0 ) {
					table(site,target_cb,Y_MINUS) = std::make_pair( LayoutLeft::index( {x_cb,y-1,z,t}, cb_dims), false);
				}
				else {
					table(site,target_cb,Y_MINUS) = std::make_pair( LayoutLeft::index( {x_cb,_n_y-1,z,t}, cb_dims), true);
				}

				if ( x > 0 ) {
					table(site,target_cb,X_MINUS)= std::make_pair(LayoutLeft::index( {(x-1)/2,y,z,t}, cb_dims), false);
				}
				else {
					table(site,target_cb,X_MINUS)= std::make_pair(LayoutLeft::index( {(_n_x-1)/2,y,z,t}, cb_dims),true);
				}

				if ( x < _n_x - 1 ) {
					table(site,target_cb,X_PLUS) = std::make_pair(LayoutLeft::index({(x+1)/2,y,z,t},cb_dims),false);
				}
				else {
					table(site,target_cb,X_PLUS) = std::make_pair(LayoutLeft::index({0,y,z,t},cb_dims),true);
				}

				if( y < _n_y-1 ) {
					table(site,target_cb,Y_PLUS) = std::make_pair(LayoutLeft::index( {x_cb,y+1,z,t}, cb_dims),false);
				}
				else {
					table(site,target_cb,Y_PLUS) = std::make_pair(LayoutLeft::index( {x_cb, 0, z,t}, cb_dims), true);
				}

				if( z < _n_z-1 ) {
					table(site,target_cb,Z_PLUS) = std::make_pair( LayoutLeft::index( {x_cb,y,z+1,t}, cb_dims),false);
				}
				else {
					table(site,target_cb,Z_PLUS) = std::make_pair( LayoutLeft::index( {x_cb,y,0,t},cb_dims), true);
				}

				if( t < _n_t-1 ) {
					table(site,target_cb,T_PLUS) = std::make_pair( LayoutLeft::index( {x_cb,y,z,t+1}, cb_dims),false);
				}
				else {
					table(site,target_cb,T_PLUS) = std::make_pair( LayoutLeft::index( {x_cb,y,z,0}, cb_dims),true);
				}
			} // target CB
		}); // parallel for
	}); // queue submit
}

class SiteAccessorHost;
class SiteAccessorGlobal;


struct SiteTable {
	  SiteTable(IndexType n_xh,
		    IndexType n_y,
		    IndexType n_z,
		    IndexType n_t) :
     _table("table",{n_xh*n_y*n_z*n_t,2,8} ),
     _cb_dims({n_xh,n_y,n_z,n_t}),
	 _n_x(2*n_xh),
	 _n_xh(n_xh),
	 _n_y(n_y),
	 _n_z(n_z),
	 _n_t(n_t) { ComputeSiteTable(_cb_dims, _table); }

	  View<std::pair<IndexType,bool>,3,DefaultNeighborTableLayout> _table;


	  using host_accessor = ViewAccessor<std::pair<std::size_t,bool>, 3, DefaultNeighborTableLayout,
		  	  cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>;

	  using global_accessor = ViewAccessor<std::pair<std::size_t,bool>, 3, DefaultNeighborTableLayout,
		  	  cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;

	  global_accessor get_access(cl::sycl::handler& cgh) {
		  	  return _table.template get_access<cl::sycl::access::mode::read>(cgh);
	  }

	  host_accessor get_access() {
			  	  return _table.template get_access<cl::sycl::access::mode::read>();
	  }

	  std::array<IndexType,4> _cb_dims;
	  IndexType _n_x;
	  IndexType _n_xh;
	  IndexType _n_y;
	  IndexType _n_z;
	  IndexType _n_t;

};

template<typename VA>
struct SiteTableAccess  {

	VA tab;
	SiteTableAccess(const VA& _tab) : tab(_tab) {}


	inline
	  void NeighborTMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,T_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborTPlus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,T_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborZMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,Z_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborZPlus(IndexType site, IndexType target_cb,  IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,Z_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborYMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,Y_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborYPlus(IndexType site, IndexType target_cb,  IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,Y_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}

	inline
	  void NeighborXMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,X_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}

	inline
	  void NeighborXPlus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = tab(site,target_cb,X_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}
}; // class


struct SiteTableAccessGlobal  {

	SiteTable::global_accessor tab;
	SiteTableAccessGlobal(const SiteTable::global_accessor& _tab) : tab(_tab) {}
	SiteTableAccessGlobal(const SiteTableAccessGlobal& g) : tab(g.tab){}


	inline
	  void NeighborTMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,T_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborTPlus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,T_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborZMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,Z_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborZPlus(IndexType site, IndexType target_cb,  IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,Z_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborYMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,Y_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}


	inline
	  void NeighborYPlus(IndexType site, IndexType target_cb,  IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,Y_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}

	inline
	  void NeighborXMinus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,X_MINUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}

	inline
	  void NeighborXPlus(IndexType site, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		const std::pair<IndexType,bool>& lookup = this->tab(site,target_cb,X_PLUS);
		n_idx = lookup.first;
		do_permute = lookup.second;
	}
}; // class

#endif

} // Namespace MG



