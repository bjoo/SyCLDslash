/*
 * sycl_qdp_utils.h
 *
 *  Created on: May 23, 2017
 *      Author: bjoo
 */

#pragma once

#include "qdp.h"
#include "dslash/sycl_vtypes.h"
#include <CL/sycl.hpp>
#include <utils/print_utils.h>
#include <lattice/constants.h>
#include <lattice/lattice_info.h>
#include "dslash/dslash_defaults.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_complex.h"
#include "dslash/sycl_vtypes.h"

//#include "lattice/geometry_utils.h"
namespace MG
{

// Single QDP++ Vector
template<typename T, typename VN, typename LF>
void
QDPLatticeFermionToSyCLCBVSpinor(const LF& qdp_in,
		SyCLCBFineVSpinor<MGComplex<T>,VN,4>& sycl_out)
{
	auto cb = sycl_out.GetCB();
	const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

	// Check conformance:
	IndexType num_gsites=static_cast<IndexType>(sycl_out.GetGlobalInfo().GetNumCBSites());

	if ( sub.numSiteTable() != num_gsites ) {
		MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
				__FUNCTION__);
	}

	IndexType num_sites = static_cast<IndexType>(sycl_out.GetInfo().GetNumCBSites());
	if ( num_sites * VN::VecLen != num_gsites ) {
		MasterLog(ERROR, "%s Veclen of Vector type x num_coarse_sites != num_fine_sites",
				__FUNCTION__);
	}

	auto h_out = sycl_out.GetData().template get_access<cl::sycl::access::mode::write>();

	IndexArray coarse_dims = sycl_out.GetInfo().GetCBLatticeDimensions();
	IndexArray fine_dims = sycl_out.GetGlobalInfo().GetCBLatticeDimensions();

#pragma omp parallel for
	for(size_t i=0; i < num_sites; ++i) {
		IndexArray c_coords =  LayoutLeft::coords(i,coarse_dims);

		for(IndexType color=0; color < 3; ++color) {
			for(IndexType spin=0; spin < 4; ++spin) {

				for(IndexType lane =0; lane < VN::VecLen; ++lane) {
					IndexArray p_coords = LayoutLeft::coords(lane,{VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3});

					IndexArray g_coords;
					for(IndexType mu=0; mu < 4; ++mu) {
						g_coords[mu] = c_coords[mu] + p_coords[mu]*coarse_dims[mu];
					}

					IndexType g_idx = LayoutLeft::index(g_coords, fine_dims);
					IndexType qdp_index = sub.siteTable()[g_idx];

					LaneOps<T,VN::VecLen>::insert(h_out(i,spin,color),
							MGComplex<T>(qdp_in.elem(qdp_index).elem(spin).elem(color).real(),
									qdp_in.elem(qdp_index).elem(spin).elem(color).imag()),
									lane);
				}//lane
			} // spin
		} // color

	}
}

// Single QDP++ Vector
template<typename T, typename VN, typename HF>
void
QDPLatticeHalfFermionToSyCLCBVSpinor2(const HF& qdp_in,
		SyCLCBFineVSpinor<MGComplex<T>,VN,2>& sycl_out)
{
	auto cb = sycl_out.GetCB();
	const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

	// Check conformance:
	IndexType num_gsites=static_cast<IndexType>(sycl_out.GetGlobalInfo().GetNumCBSites());

	if ( sub.numSiteTable() != num_gsites ) {
		MasterLog(ERROR, "%s QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
				__FUNCTION__);
	}

	IndexType num_sites = static_cast<IndexType>(sycl_out.GetInfo().GetNumCBSites());
	if ( num_sites * VN::VecLen != num_gsites ) {
		MasterLog(ERROR, "%s Veclen of Vector type x num_coarse_sites != num_fine_sites",
				__FUNCTION__);
	}

	auto h_out = sycl_out.GetData().template get_access<cl::sycl::access::mode::write>();
	IndexArray coarse_dims = sycl_out.GetInfo().GetCBLatticeDimensions();
	IndexArray fine_dims = sycl_out.GetGlobalInfo().GetCBLatticeDimensions();

#pragma omp parallel for
	for(size_t i=0; i < num_sites; ++i) {
		IndexArray c_coords = LayoutLeft::coords(i, coarse_dims);

		for(IndexType color=0; color < 3; ++color) {
			for(IndexType spin=0; spin < 2; ++spin) {
				for(IndexType lane =0; lane < VN::VecLen; ++lane) {

					IndexArray p_coords = LayoutLeft::coords(lane,{VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3} );
					IndexArray g_coords;
					for(IndexType mu=0; mu < 4; ++mu) {
						g_coords[mu] = c_coords[mu] + p_coords[mu]*coarse_dims[mu];
					}

					IndexType g_idx = LayoutLeft::index(g_coords, fine_dims);
					IndexType qdp_index = sub.siteTable()[g_idx];

					LaneOps<T,VN::VecLen>::insert(h_out(i,spin,color),
							MGComplex<T>(qdp_in.elem(qdp_index).elem(spin).elem(color).real(),
							             qdp_in.elem(qdp_index).elem(spin).elem(color).imag()),
							lane);
				}//lane
			} // spin
		} // color
	}
}

// Single QDP++ vector
template<typename T, typename VN, typename LF>
void
SyCLCBVSpinorToQDPLatticeFermion(const SyCLCBFineVSpinor<MGComplex<T>,VN, 4>& sycl_in,
		LF& qdp_out) {

	auto cb = sycl_in.GetCB();
	const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

	// Check conformance:
	IndexType num_csites=static_cast<IndexType>(sycl_in.GetInfo().GetNumCBSites());
	IndexType num_gsites=static_cast<IndexType>(sycl_in.GetGlobalInfo().GetNumCBSites());

	if ( sub.numSiteTable() != num_gsites ) {
		MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
				__FUNCTION__);
	}

	if( num_csites * VN::VecLen != num_gsites ) {
		MasterLog(ERROR, "%s: num_csites * veclen != num_gsites",
				__FUNCTION__);
	}

	typename SyCLCBFineVSpinor<MGComplex<T>,VN, 4>::DataType h_in_view  = sycl_in.GetData();
	auto h_in = h_in_view.template get_access<cl::sycl::access::mode::read>();


	IndexArray c_dims = sycl_in.GetInfo().GetCBLatticeDimensions();
	IndexArray g_dims = sycl_in.GetGlobalInfo().GetCBLatticeDimensions();

	cl::sycl::cpu_selector cpu;
	cl::sycl::queue q(cpu);

#pragma omp parallel for
	for(int i=0; i < num_csites; ++i ) {
		IndexArray c_coords = LayoutLeft::coords(i,c_dims);

		for(IndexType color=0; color < 3; ++color) {
			for(IndexType spin=0; spin < 4; ++spin) {
				for(IndexType lane=0; lane < VN::VecLen;++lane) {

					IndexArray p_coords=LayoutLeft::coords(lane,{VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3});
					IndexArray g_coords;

					for(IndexType mu=0; mu < 4; ++mu ) {
						g_coords[mu] = c_coords[mu] + p_coords[mu]*c_dims[mu];
					}
					IndexType g_index=LayoutLeft::index(g_coords,g_dims);

					// FIXME. IS the site table available or do I need to wrap it?
					// This is explicitly on the CPU
					IndexType qdp_index = sub.siteTable()[g_index];

					MGComplex<T> from = LaneOps<T,VN::VecLen>::extract(h_in(i,spin,color), lane);

					qdp_out.elem(qdp_index).elem(spin).elem(color).real() = from.real();
					qdp_out.elem(qdp_index).elem(spin).elem(color).imag() =from.imag();
				} // lane
			} // spin
		} // color
	}
}

// Single QDP++ vector
template<typename T, typename VN, typename HF>
void
SyCLCBVSpinor2ToQDPLatticeHalfFermion(const SyCLCBFineVSpinor<MGComplex<T>,VN, 2>& sycl_in,
		HF& qdp_out) {

	auto cb = sycl_in.GetCB();
	const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

	// Check conformance:
	IndexType num_csites=static_cast<IndexType>(sycl_in.GetInfo().GetNumCBSites());
	IndexType num_gsites=static_cast<IndexType>(sycl_in.GetGlobalInfo().GetNumCBSites());

	if ( sub.numSiteTable() != num_gsites ) {
		MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
				__FUNCTION__);
	}

	if( num_csites * VN::VecLen != num_gsites ) {
		MasterLog(ERROR, "%s: num_csites * veclen != num_gsites",
				__FUNCTION__);
	}

	auto h_in =  sycl_in.GetData().template get_access<cl::sycl::access::mode::read>();

	IndexArray c_dims = sycl_in.GetInfo().GetCBLatticeDimensions();
	IndexArray g_dims = sycl_in.GetGlobalInfo().GetCBLatticeDimensions();


#pragma omp parallel for
	for(int i=0; i< num_csites;++i) {
		IndexArray c_coords=LayoutLeft::coords(i,c_dims);

		for(IndexType color=0; color < 3; ++color) {
			for(IndexType spin=0; spin < 2; ++spin) {
				for(IndexType lane=0; lane < VN::VecLen;++lane) {
					IndexArray p_coords = LayoutLeft::coords(lane,{VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3});
					IndexArray g_coords;
					for(IndexType mu=0; mu < 4; ++mu ) {
						g_coords[mu] = c_coords[mu] + p_coords[mu]*c_dims[mu];
					}
					IndexType g_index=LayoutLeft::index(g_coords,g_dims);
					IndexType qdp_index = sub.siteTable()[g_index];

					MGComplex<T> v = LaneOps<T,VN::VecLen>::extract(h_in(i,spin,color),lane);
					qdp_out.elem(qdp_index).elem(spin).elem(color).real() = v.real();
					qdp_out.elem(qdp_index).elem(spin).elem(color).imag() = v.imag();
				} // lane
			} // spin
		} // color
	}
}


template<typename T, typename VN, typename GF>
void
QDPGaugeFieldToSyCLCBVGaugeField(const GF& qdp_in,
		SyCLCBFineVGaugeField<T,VN>& sycl_out)
{
	auto cb = sycl_out.GetCB();
	const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];

	using FType = typename BaseType<T>::Type;

	// Check conformance:
	IndexType num_gsites=static_cast<IndexType>(sycl_out.GetGlobalInfo().GetNumCBSites());

	if ( sub.numSiteTable() != num_gsites ) {
		MasterLog(ERROR, "%s QDP++ Gauge has different number of sites per checkerboard than the KokkosCBFineVGaugeField",
				__FUNCTION__);
	}

	IndexType num_sites = static_cast<IndexType>(sycl_out.GetInfo().GetNumCBSites());
	if ( num_sites * VN::VecLen != num_gsites ) {
		MasterLog(ERROR, "%s Veclen of Vector type x num_coarse_sites != num_fine_sites",
				__FUNCTION__);
	}


	auto h_out = sycl_out.GetData().template get_access<cl::sycl::access::mode::write>();

	IndexArray coarse_dims = sycl_out.GetInfo().GetCBLatticeDimensions();
	IndexArray fine_dims = sycl_out.GetGlobalInfo().GetCBLatticeDimensions();

	cl::sycl::cpu_selector cpu;
	cl::sycl::queue q(cpu);

#pragma omp parallel for
	for(size_t i=0; i < num_sites; ++i) {
		IndexArray c_coords = LayoutLeft::coords(i, coarse_dims);

		for(IndexType dir=0; dir < 4; ++dir) {
			for(IndexType color=0; color < 3; ++color) {
				for(IndexType color2=0; color2 < 3; ++color2) {
					for(IndexType lane =0; lane < VN::VecLen; ++lane) {
						IndexArray p_coords=LayoutLeft::coords(lane,{VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3});
						IndexArray g_coords;
						for(IndexType mu=0; mu < 4; ++mu) {
							g_coords[mu] = c_coords[mu] + p_coords[mu]*coarse_dims[mu];
						}

						IndexType g_idx = LayoutLeft::index(g_coords, fine_dims);
						IndexType qdp_index = sub.siteTable()[g_idx];

						LaneOps<FType,VN::VecLen>::insert( h_out(i,dir,color,color2),
								MGComplex<FType>(qdp_in[dir].elem(qdp_index).elem().elem(color,color2).real(),
										qdp_in[dir].elem(qdp_index).elem().elem(color,color2).imag()),
										lane);
					}//lane
				} // color2
			} // color
		} // dir
	}
}


template<typename T, typename VN, typename GF>
void
SyCLCBVGaugeFieldToQDPGaugeField(const SyCLCBFineVGaugeField<MGComplex<T>,VN>& sycl_in,
		GF& qdp_out)
{
	auto cb = sycl_in.GetCB();

	const QDP::Subset& sub = ( cb == EVEN ) ? QDP::rb[0] : QDP::rb[1];
	// Check conformance:
	IndexType num_csites=static_cast<IndexType>(sycl_in.GetInfo().GetNumCBSites());
	IndexType num_gsites=static_cast<IndexType>(sycl_in.GetGlobalInfo().GetNumCBSites());

	if ( sub.numSiteTable() != num_gsites ) {
		MasterLog(ERROR, "%s: QDP++ Spinor has different number of sites per checkerboard than the KokkosCBFineSpinor",
				__FUNCTION__);
	}

	if( num_csites * VN::VecLen != num_gsites ) {
		MasterLog(ERROR, "%s: num_csites * veclen != num_gsites",
				__FUNCTION__);
	}


	typename SyCLCBFineVGaugeField<MGComplex<T>,VN>::DataType h_in_view =  sycl_in.GetData();
	auto h_in = h_in_view.template get_access<cl::sycl::access::mode::read>();

	IndexArray c_dims = sycl_in.GetInfo().GetCBLatticeDimensions();
	IndexArray g_dims = sycl_in.GetGlobalInfo().GetCBLatticeDimensions();

#pragma omp parallel for
	for(size_t i=0; i < num_csites; ++i ) {
		IndexArray c_coords = LayoutLeft::coords(i,c_dims);

		for(IndexType dir=0; dir < 4; ++dir) {
			for(IndexType color=0; color < 3; ++color) {
				for(IndexType color2=0; color2 < 3; ++color2) {
					for(IndexType lane=0; lane < VN::VecLen;++lane) {

						IndexArray p_coords = LayoutLeft::coords(lane,{VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3});

						IndexArray g_coords;
						for(IndexType mu=0; mu < 4; ++mu ) {
							g_coords[mu] = c_coords[mu] + p_coords[mu]*c_dims[mu];
						}

						IndexType g_index=LayoutLeft::index(g_coords,g_dims);
						IndexType qdp_index = sub.siteTable()[g_index];

						MGComplex<T> v = LaneOps<T,VN::VecLen>::extract(h_in(i,dir,color,color2),lane );
						qdp_out[dir].elem(qdp_index).elem().elem(color,color2).real() = v.real();
						qdp_out[dir].elem(qdp_index).elem().elem(color,color2).imag() = v.imag();
					} // lane
				} // color2
			} // color
		}// mu
	}
}





template<typename T, typename VN, typename GF>
void
QDPGaugeFieldToSyCLVGaugeField(const GF& qdp_in,
		SyCLFineVGaugeField<T,VN>& sycl_out)
{
	QDPGaugeFieldToSyCLCBVGaugeField<T,VN,GF>( qdp_in, sycl_out(EVEN));
	QDPGaugeFieldToSyCLCBVGaugeField<T,VN,GF>( qdp_in, sycl_out(ODD));
}

template<typename T, typename VN, typename GF>
void
SyCLVGaugeFieldToQDPGaugeField(const SyCLFineVGaugeField<T,VN>& sycl_in,
		GF& qdp_out)
{
	SyCLCBVGaugeFieldToQDPGaugeField( sycl_in(EVEN),qdp_out);
	SyCLCBVGaugeFieldToQDPGaugeField( sycl_in(ODD), qdp_out);
}

} // namespace
