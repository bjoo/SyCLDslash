#pragma once

#include <cstddef>
#include <utility>
#include "sycl_dslash_config.h"
#include "CL/sycl.hpp"
#include "dslash/sycl_view.h"
#include "dslash/dslash_defaults.h"
#include "lattice/constants.h"
namespace MG { 

class SiteTable;

class SiteTableAccess {
public:
	SiteTableAccess(IndexArray cb_dims, IndexType n_x, IndexType n_y, IndexType n_z, IndexType n_t)
		: _cb_dims(cb_dims), _n_x(n_x), _n_xh(n_x/2), _n_y(n_y), _n_z(n_z), _n_t(n_t) {}


	inline
	  void NeighborTMinus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType& n_idx, bool& do_permute) const {
		if( t >  0) {
			n_idx=LayoutLeft::index({xcb,y,z,t-1},_cb_dims);
			do_permute=false;
		}
		else {
			n_idx=LayoutLeft::index({xcb,y,z,_n_t-1},_cb_dims);
			do_permute=true;
		}
	}

	inline
	 void NeighborZMinus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType& n_idx, bool& do_permute) const {
		if( z >  0 ) {
			n_idx=LayoutLeft::index({xcb,y,z-1,t},_cb_dims);
		    do_permute=false;
		}
		else {
			n_idx=LayoutLeft::index({xcb,y,_n_z-1,t},_cb_dims);
			do_permute=true;
		}
	}


	inline
	  void NeighborYMinus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType& n_idx, bool& do_permute) const {
		if ( y > 0 ) {
			n_idx=LayoutLeft::index({xcb,y-1,z,t},_cb_dims);
			do_permute  = false;
		}
		else {
			n_idx=LayoutLeft::index({xcb,_n_y-1,z,t},_cb_dims);
			do_permute  = true;
		}
	}


	inline
	void NeighborXMinus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);
		if ( x > 0 ) {
				n_idx=LayoutLeft::index({(x-1)/2,y,z,t},_cb_dims);
				do_permute  = false;
		}
		else {
				n_idx=LayoutLeft::index({(_n_x-1)/2,y-1,z,t},_cb_dims);
				do_permute = true;
		}
	}


	inline
	  void NeighborXPlus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType target_cb, IndexType& n_idx, bool& do_permute) const {
		IndexType x = 2*xcb + ((target_cb+y+z+t)&0x1);
		if ( x < _n_x - 1) {
			n_idx=LayoutLeft::index({(x+1)/2,y,z,t},_cb_dims);
			do_permute = false;
		}
		else {
			n_idx=LayoutLeft::index({0,y,z,t},_cb_dims);
			do_permute = true;

		}
	}



	inline
	  void NeighborYPlus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType& n_idx, bool& do_permute) const {
		if (y < _n_y - 1) {
			n_idx=LayoutLeft::index({xcb,y+1,z,t},_cb_dims);
			do_permute = false;
		}
		else {
			n_idx=LayoutLeft::index({xcb,0,z,t},_cb_dims);
			do_permute = true;
		}
	}


	inline
	  void NeighborZPlus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType& n_idx, bool& do_permute) const {


		if(z < _n_z - 1) {
			n_idx=LayoutLeft::index({xcb,y,z+1,t},_cb_dims);
			do_permute = false;
		}
		else {
			n_idx=LayoutLeft::index({xcb,y,0,t},_cb_dims);
			do_permute = true;
		}
	}


	inline
	  void NeighborTPlus(IndexType xcb, IndexType y, IndexType z, IndexType t, IndexType& n_idx, bool& do_permute) const {

		if (t < _n_t - 1) {
			n_idx = LayoutLeft::index({xcb,y,z,t+1}, _cb_dims);
			do_permute = false;
		}
		else {
			n_idx = LayoutLeft::index({xcb,y,z,0}, _cb_dims);
			do_permute = true;
		}
	}

	std::array<IndexType,4> _cb_dims;
	IndexType _n_x;
	IndexType _n_xh;
	IndexType _n_y;
	IndexType _n_z;
	IndexType _n_t;


};

struct SiteTable {
	  SiteTable(IndexType n_xh,
		    IndexType n_y,
		    IndexType n_z,
		    IndexType n_t) :
     _cb_dims({n_xh,n_y,n_z,n_t}),
	 _n_x(2*n_xh),
	 _n_xh(n_xh),
	 _n_y(n_y),
	 _n_z(n_z),
	 _n_t(n_t) {}



	  // Memory access semantics in case we every implement this with a table
	  template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget=cl::sycl::access::target::host_buffer>
	  SiteTableAccess get_access() {
		  SiteTableAccess ret_val( _cb_dims, _n_x, _n_y, _n_z, _n_t);
		  return ret_val;
	  }

	  // Memory access semantics in case we ever implement this with a table
	  template<cl::sycl::access::mode accessMode, cl::sycl::access::target accessTarget=cl::sycl::access::target::global_buffer>
	  SiteTableAccess get_access(cl::sycl::handler &cgh) {
		  SiteTableAccess ret_val( _cb_dims, _n_x, _n_y, _n_z, _n_t);
		  return ret_val;
	  }

	  std::array<IndexType,4> _cb_dims;
	  IndexType _n_x;
	  IndexType _n_xh;
	  IndexType _n_y;
	  IndexType _n_z;
	  IndexType _n_t;



};



} // Namespace MG



