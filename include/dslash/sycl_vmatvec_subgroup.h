/*
 * sycl_vmatvec_subgroup.h
 *
 *  Created on: Oct 29, 2019
 *      Author: bjoo
 */



#pragma once
/*
 * kokkos_matvec.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#pragma once

#include "dslash/sycl_view.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"
#include "lattice/constants.h"

namespace MG {

 template<typename GT, typename VN>
 using GaugeAccessor = SyCLSGVGaugeViewAccessor<GT,VN,cl::sycl::access::mode::read>;

 template<typename GT, typename VN, typename ST,int dir>
 inline
 void mult_u_halfspinor(const GaugeAccessor<GT,VN>& gauge_in,
 		const HalfSpinorSiteView<ST>& v_in,
 		HalfSpinorSiteView<ST>& v_out,
 		const size_t& i,
 		const sycl::intel::sub_group& sg)
 {
 	using FType = typename BaseType<GT>::Type;
 #pragma unroll
 	for(int spin=0; spin < 2; ++spin) {
 #pragma unroll
 		for(int row=0; row < 3; ++row) {

 			ComplexZero(v_out(row,spin));

 #pragma unroll
 			for(int col=0; col < 3; ++col) {
 				MGComplex<FType> u = Load( gauge_in.offset(i,dir,row,col,0,0), gauge_in.get_pointer(), sg );
 				ComplexCMadd( v_out(row,spin), u, v_in(col,spin));
 			}
 		}
 	}

 }


 template<typename GT, typename VN, typename ST, int dir>
 inline
 void mult_adj_u_halfspinor(const GaugeAccessor<GT,VN>& gauge_in,
 		const HalfSpinorSiteView<ST>& v_in,
 		HalfSpinorSiteView<ST>& v_out,
 		const size_t& i,
 		const sycl::intel::sub_group& sg)
 {
 	using FType = typename BaseType<GT>::Type;

 #pragma unroll
 	for(int spin=0; spin < 2; ++spin ) {
 #pragma unroll
 		for(int row=0; row < 3; ++row) {
 			ComplexZero(v_out(row,spin));
 		}

 #pragma unroll
 		for(int col=0; col < 3; ++col) {

 #pragma unroll
 			for(int row=0; row < 3; ++row) {
 				MGComplex<FType> u = Load( gauge_in.offset(i,dir,col,row,0,0), gauge_in.get_pointer(), sg );
 				ComplexConjMadd(v_out(row,spin), u, v_in(col,spin));
 			}
 		}
 	}

 }


 // Perm versions
 //
template<typename GT, typename VN, typename ST, int dir>
inline
void mult_u_halfspinor(const GaugeAccessor<GT,VN>& gauge_in,
		const HalfSpinorSiteView<ST>& v_in,
		HalfSpinorSiteView<ST>& v_out,
		const size_t& i,
		const std::array<int, VN::VecLen>& mask,
		const sycl::intel::sub_group& sg)
{
	using FType = typename BaseType<GT>::Type;
#pragma unroll
	for(int spin=0; spin < 2; ++spin) {
#pragma unroll
		for(int row=0; row < 3; ++row) {

			ComplexZero(v_out(row,spin));

#pragma unroll
			for(int col=0; col < 3; ++col) {
				MGComplex<FType> gauge_perm = permute<FType,VN::VecLen>( mask, Load( gauge_in.offset(i,dir,row,col,0,0), gauge_in.get_pointer(), sg ), sg);
				ComplexCMadd( v_out(row,spin), gauge_perm, v_in(col,spin));
			}
		}
	}

}


template<typename GT, typename VN, typename ST, int dir>
inline
void mult_adj_u_halfspinor(const GaugeAccessor<GT,VN>& gauge_in,
		const HalfSpinorSiteView<ST>& v_in,
		HalfSpinorSiteView<ST>& v_out,
		const size_t& i,
		const std::array<int,VN::VecLen>& mask,
		const sycl::intel::sub_group& sg)
{
	using FType = typename BaseType<GT>::Type;

#pragma unroll
	for(int spin=0; spin < 2; ++spin ) {
#pragma unroll
		for(int row=0; row < 3; ++row) {
			ComplexZero(v_out(row,spin));
		}

#pragma unroll
		for(int col=0; col < 3; ++col) {

#pragma unroll
			for(int row=0; row < 3; ++row) {
				MGComplex<FType> gauge_perm = permute<FType,VN::VecLen>( mask, Load( gauge_in.offset(i,dir,col,row,0,0), gauge_in.get_pointer(), sg ), sg);
				ComplexConjMadd(v_out(row,spin), gauge_perm, v_in(col,spin));
			}
		}
	}

}


} // namespace



