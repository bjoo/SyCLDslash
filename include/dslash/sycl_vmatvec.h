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
 using GaugeAccessor = SyCLVGaugeViewAccessor<GT,VN,cl::sycl::access::mode::read>;

template<typename GT, typename VN, typename ST, int dir>
inline
void mult_u_halfspinor(const GaugeAccessor<GT,VN>& gauge_in,
		const HalfSpinorSiteView<ST>& v_in,
		HalfSpinorSiteView<ST>& v_out,
		const size_t& i)
{
	using FType = typename BaseType<GT>::Type;

#pragma unroll
	for(int spin=0; spin < 2; ++spin) {
#pragma unroll
		for(int row=0; row < 3; ++row) {

			ComplexZero<FType,VN::VecLen>(v_out(row,spin));

#pragma unroll
			for(int col=0; col < 3; ++col) {
				ComplexCMadd<FType,VN::VecLen>(v_out(row,spin), gauge_in(i,dir,row,col), v_in(col,spin));
			}
		}
	}

}

// Permute Versions
template<typename GT, typename VN, typename ST, int dir, IndexType perm_dir>
inline
void mult_u_halfspinor_perm(const GaugeAccessor<GT,VN>& gauge_in,
		const HalfSpinorSiteView<ST>& v_in,
		HalfSpinorSiteView<ST>& v_out,
		const size_t& i)
{
	using FType = typename BaseType<GT>::Type;
#pragma unroll
	for(int spin=0; spin < 2; ++spin) {
#pragma unroll
		for(int row=0; row < 3; ++row) {

			ComplexZero<FType,VN::VecLen>(v_out(row,spin));

#pragma unroll
			for(int col=0; col < 3; ++col) {
				if( perm_dir == X_DIR ) {
					ComplexCMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteX(gauge_in(i,dir,row,col)), v_in(col,spin));
				}
				else if (perm_dir == Y_DIR ) {
					ComplexCMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteY(gauge_in(i,dir,row,col)), v_in(col,spin));
				}
				else if (perm_dir == Z_DIR ) {
					ComplexCMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteZ(gauge_in(i,dir,row,col)), v_in(col,spin));
				}
				else {
					ComplexCMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteT(gauge_in(i,dir,row,col)), v_in(col,spin));
				}
			}
		}
	}

}






template<typename GT, typename VN, typename ST, int dir>
inline
void mult_adj_u_halfspinor(const GaugeAccessor<GT,VN>& gauge_in,
		const HalfSpinorSiteView<ST>& v_in,
		HalfSpinorSiteView<ST>& v_out,
		const size_t i)
{

	using FType = typename BaseType<GT>::Type;

#pragma unroll
	for(int spin=0; spin < 2; ++spin ) {
#pragma unroll
		for(int row=0; row < 3; ++row) {
			ComplexZero<FType,VN::VecLen>(v_out(row,spin));
		}

#pragma unroll
		for(int col=0; col < 3; ++col) {

#pragma unroll
			for(int row=0; row < 3; ++row) {

				ComplexConjMadd<FType,VN::VecLen>(v_out(row,spin), gauge_in(i,dir,col,row), v_in(col,spin));
			}
		}
	}


}

template<typename GT, typename VN, typename ST, int dir, IndexType perm_dir>
inline
void mult_adj_u_halfspinor_perm(const GaugeAccessor<GT,VN>& gauge_in,
		const HalfSpinorSiteView<ST>& v_in,
		HalfSpinorSiteView<ST>& v_out,
		const size_t& i)
{
	using FType = typename BaseType<GT>::Type;

#pragma unroll
	for(int spin=0; spin < 2; ++spin ) {
#pragma unroll
		for(int row=0; row < 3; ++row) {
			ComplexZero<FType,VN::VecLen>(v_out(row,spin));
		}

#pragma unroll
		for(int col=0; col < 3; ++col) {

#pragma unroll
			for(int row=0; row < 3; ++row) {
				if( perm_dir == X_DIR ) {
					ComplexConjMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteX(gauge_in(i,dir,col,row)), v_in(col,spin));
				}
				else if ( perm_dir == Y_DIR ) {
					ComplexConjMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteY(gauge_in(i,dir,col,row)), v_in(col,spin));

				}
				else if ( perm_dir == Z_DIR ) {
					ComplexConjMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteZ(gauge_in(i,dir,col,row)), v_in(col,spin));

				}
				else {
					ComplexConjMadd<FType,VN::VecLen>(v_out(row,spin), VN::permuteY(gauge_in(i,dir,col,row)), v_in(col,spin));

				}
			}
		}
	}

}


} // namespace
