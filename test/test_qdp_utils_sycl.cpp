/*
 * test_qdp_utils.cpp
 *
 *  Created on: Jul 31, 2019
 *      Author: bjoo
 */

#include "sycl_dslash_config.h"
#include "gtest/gtest.h"
#include "test_env.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"

#include "sycl_dslash_config.h"   // Build options
#include "dslash/dslash_defaults.h" // Default layouts
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"     // Vector type s
#include "dslash/sycl_qdp_vutils.h" // Utils
#include "dslash/dslash_vectype_sycl.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

template<typename T>
class TestVNode :  public ::testing::Test{};

#ifdef MG_FORTRANLIKE_COMPLEX
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8> >;
#else
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8>,
		std::integral_constant<int,16>	>;

#endif

TYPED_TEST_CASE(TestVNode, test_types);

TYPED_TEST(TestVNode,TestVSpinor)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<float,VectorLength>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<float>,VN,4>;

	// 4 spins
	SpinorType vnode_spinor(info, MG::EVEN);

	const LatticeInfo& c_info = vnode_spinor.GetInfo();
	IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
	for(int mu=0; mu < 4;++mu) {
		ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
				info.GetLatticeDimensions()[mu]/VNDims[mu] );
	}

	bool same_global_vectype = std::is_same< typename SpinorType::VecType, SIMDComplexSyCL<float,VectorLength> >::value;
	ASSERT_EQ( same_global_vectype, true);
}

TYPED_TEST(TestVNode,TestVGauge)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using GaugeType = SyCLCBFineVGaugeField<MGComplex<float>,VN>;

	GaugeType vnode_gauge(info, MG::EVEN);

	const LatticeInfo& c_info = vnode_gauge.GetInfo();
	IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
	for(int mu=0; mu < 4;++mu) {
		ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
				info.GetLatticeDimensions()[mu]/VNDims[mu] );
	}
	bool same_global_vectype = std::is_same< typename GaugeType::VecType, SIMDComplexSyCL<float,VectorLength> >::value;
	ASSERT_EQ( same_global_vectype, true);



}

template<int VectorLength>
float computeLane(const IndexArray& coords, const IndexArray& cb_latdims)
{
	float value;
	if( VectorLength == 1) {
		value = 0;
		return value;
	}


	if (VectorLength == 2) {
		if( coords[3] < cb_latdims[3]/2 ) {
			value = 0;
		}
		else {
			value = 1;
		}
		return value;
	}

	if( VectorLength == 4 ) {
		if( coords[2] < cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 0;
		}
		if( coords[2] >= cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 1;
		}
		if( coords[2] < cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 2;
		}
		if( coords[2] >= cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 3;
		}
	}

	if( VectorLength == 8) {
		if( coords[1] < cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 0;
		}

		if(  coords[1] >= cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 1;
		}

		if(  coords[1] < cb_latdims[1]/2
				&& coords[2] >=  cb_latdims[2]/2
				&& coords[3] <  cb_latdims[3]/2 ) {
			value = 2;
		}

		if(  coords[1] >= cb_latdims[1]/2
				&& coords[2] >=  cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 3;
		}

		if(   coords[1] < cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 4;
		}

		if( coords[1] >= cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 5;
		}

		if( coords[1] < cb_latdims[1]/2
				&& coords[2] >=  cb_latdims[2]/2
				&& coords[3] >=  cb_latdims[3]/2 ) {
			value = 6;
		}

		if(    coords[1] >= cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 7;
		}

		return value;
	}





	if( VectorLength == 16 ) {
		if(    coords[0] < cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 0;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 1;
		}

		if(    coords[0] <  cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] <  cb_latdims[2]/2
				&& coords[3] <  cb_latdims[3]/2 ) {
			value = 2;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 3;
		}

		if(    coords[0] < cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 4;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 5;
		}

		if(    coords[0] <  cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] >=  cb_latdims[2]/2
				&& coords[3] <  cb_latdims[3]/2 ) {
			value = 6;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] < cb_latdims[3]/2 ) {
			value = 7;
		}

		if(    coords[0] < cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 8;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 9;
		}

		if(    coords[0] <  cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] <  cb_latdims[2]/2
				&& coords[3] >=  cb_latdims[3]/2 ) {
			value = 10;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] < cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 11;
		}

		if(    coords[0] < cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 12;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] < cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 13;
		}

		if(    coords[0] <  cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] >=  cb_latdims[2]/2
				&& coords[3] >=  cb_latdims[3]/2 ) {
			value = 14;
		}

		if(    coords[0] >= cb_latdims[0]/2
				&& coords[1] >= cb_latdims[1]/2
				&& coords[2] >= cb_latdims[2]/2
				&& coords[3] >= cb_latdims[3]/2 ) {
			value = 15;
		}
		return value;
	}


	return value;
}

TYPED_TEST(TestVNode, TestPackSpinor)
{

	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<float>,VN,4>;

	LatticeFermion qdp_in = zero;

	IndexArray cb_latdims=info.GetCBLatticeDimensions();
	size_t num_cbsites= info.GetNumCBSites();
	MasterLog(INFO, "Num_cbsites=%d", num_cbsites);
	MasterLog(INFO, "cb_latdims=(%d,%d,%d,%d)", cb_latdims[0], cb_latdims[1],cb_latdims[2],cb_latdims[3]);

	// Set QDP++ spinor with value we want in the SyCL Spinor
#pragma omp parallel for
    for(size_t i=0; i < num_cbsites; ++i) {
		IndexArray coords = LayoutLeft::coords(i,cb_latdims);

		float value=computeLane<VectorLength>(coords,cb_latdims);
		int qdp_idx = rb[EVEN].siteTable()[i];

		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {
				qdp_in.elem(qdp_idx).elem(spin).elem(color).real() = value;
				qdp_in.elem(qdp_idx).elem(spin).elem(color).imag() = 0;
			}
		}
	}

    // Transfer QDP spinor to SyCL Spinor
	SpinorType sycl_spinor(info,EVEN);
	QDPLatticeFermionToSyCLCBVSpinor(qdp_in, sycl_spinor);

	// Get host accessor
	auto sycl_h = sycl_spinor.GetData().template get_access<cl::sycl::access::mode::read>();

	// Check its vectype matches expectations
	bool same_global_vectype = std::is_same< typename SpinorType::VecType, SIMDComplexSyCL<float,VN::VecLen> >::value;
	ASSERT_EQ( same_global_vectype, true);

	const LatticeInfo& vinfo = sycl_spinor.GetInfo();
	int num_vcbsites = vinfo.GetNumCBSites();

	// Check that the data in the SyCL Spinor accessed via Host ViewAccessor

	for(size_t i=0; i < num_vcbsites; ++i) {
		for(int color=0; color <3; ++color) {
			for(int spin=0; spin < 4; ++spin) {

				// Extract appropriate vector from SyCL Spinor
				auto vec_data = sycl_h(i,spin,color);

				// Loop through vector lanes, and check...
				for(int lane=0; lane < VN::VecLen; ++lane) {
					float ref = lane;
					auto lane_complex = LaneOps<float,VectorLength>::extract(vec_data,lane);
					ASSERT_FLOAT_EQ( lane_complex.real(), ref );
					ASSERT_FLOAT_EQ( lane_complex.imag(), 0 );

				}//  lane
			} // spin
		} // color
	}

	// Transfer back into QDP++ (initialize it with gaussian noise first)
	LatticeFermion back_spinor;  gaussian(back_spinor);

	// Do the transfer
	SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor, back_spinor);

	// Check back_spinor is same as the original spinor
	for(size_t i=0; i < num_cbsites;++i ) {
	for(int color=0; color <3; ++color) {
			for(int spin=0; spin < 4; ++spin) {
				float ref_re = qdp_in.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).real();
				float ref_im = qdp_in.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).imag();

				float out_re = back_spinor.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).real();
				float out_im = back_spinor.elem(rb[EVEN].siteTable()[i]).elem(spin).elem(color).imag();

				ASSERT_FLOAT_EQ( ref_re, out_re);
				ASSERT_FLOAT_EQ( ref_im, out_im);
			} // spin
		} // color
	}
}

template<size_t N>
class times_two;

TYPED_TEST(TestVNode, TestSpinorSyCLAccess )
{
	static constexpr int VectorLength = TypeParam::value;
		IndexArray latdims={{8,8,8,8}};
		initQDPXXLattice(latdims);
		QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
		LatticeInfo info(latdims,4,3,NodeInfo());

		using VN = VNode<MGComplex<float>,VectorLength>;
		using SpinorType = SyCLCBFineVSpinor<MGComplex<float>,VN,4>;

		LatticeFermion qdp_in = zero; gaussian(qdp_in);

		SpinorType sycl_spinor(info,EVEN);
		QDPLatticeFermionToSyCLCBVSpinor(qdp_in, sycl_spinor);

		// Spinor is now packed
		auto spinor_view = sycl_spinor.GetData();
		const LatticeInfo& vinfo = sycl_spinor.GetInfo();
		int num_vcbsites = vinfo.GetNumCBSites();

		// Now double every value in a device loop:
		cl::sycl::queue q;
		q.submit([&](cl::sycl::handler& cgh ) {
			auto spinor_access = spinor_view.template get_access<cl::sycl::access::mode::read_write>(cgh);
			MGComplex<float> a(2.0,0);
			cgh.parallel_for<times_two<VectorLength>>(cl::sycl::range<1>(num_vcbsites), [=](cl::sycl::id<1> idx) {
				size_t site = idx[0];

				for(int spin=0; spin < 4; ++spin) {
					for(int color=0; color < 3; ++color ) {
						ComplexCMadd(spinor_access(site,spin,color),a,spinor_access(site,spin,color));
					}
				}
			});
		});
		q.wait_and_throw();

		LatticeFermion qdp_back;
		SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor,qdp_back);

		for(size_t site=0; site < rb[0].numSiteTable(); ++site) {

			for(int spin=0; spin < 4; ++spin ) {
				for(int color=0; color < 3; ++color) {

					ASSERT_FLOAT_EQ(3.0*qdp_in.elem(rb[0].start() + site).elem(spin).elem(color).real(),
							qdp_back.elem(rb[0].start() + site).elem(spin).elem(color).real() );

					ASSERT_FLOAT_EQ( 3.0*qdp_in.elem(rb[0].start() + site).elem(spin).elem(color).imag(),
												qdp_back.elem(rb[0].start() + site).elem(spin).elem(color).imag() );
				}
			}
		}
}

TYPED_TEST(TestVNode, TestPackSpinor2)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());
	int num_cbsites = info.GetNumCBSites();

	using VN = VNode<MGComplex<float>,VectorLength>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<float>,VN,4>;

	LatticeFermion qdp_in;
	gaussian(qdp_in);

	SpinorType sycl_spinor_e(info,EVEN);
	SpinorType sycl_spinor_o(info,ODD);

	// Import
	QDPLatticeFermionToSyCLCBVSpinor(qdp_in, sycl_spinor_e);
	QDPLatticeFermionToSyCLCBVSpinor(qdp_in, sycl_spinor_o);

	// Export
	LatticeFermion qdp_out;
	SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_e, qdp_out);
	SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_o, qdp_out);

	for(int cb=EVEN; cb <= ODD; ++cb) {
		for(size_t i=0; i < num_cbsites; ++i) {
			for(int color=0; color <3; ++color) {
				for(int spin=0; spin < 4; ++spin) {
					float ref_re = qdp_in.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).real();
					float ref_im = qdp_in.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).imag();

					float out_re = qdp_out.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).real();
					float out_im = qdp_out.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).imag();

					ASSERT_FLOAT_EQ( ref_re, out_re);
					ASSERT_FLOAT_EQ( ref_im, out_im);
				} // spin
			} // color
		}
	} // cb
}

TYPED_TEST(TestVNode, TestPackHalfSpinor2)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());
	int num_cbsites = info.GetNumCBSites();

	using VN = VNode<MGComplex<float>,VectorLength>;
	using HalfSpinorType = SyCLCBFineVSpinor<MGComplex<float>,VN,2>;

	LatticeFermion qdp_in;
	gaussian(qdp_in);

	HalfSpinorType sycl_spinor_e(hinfo,EVEN);
	HalfSpinorType sycl_spinor_o(hinfo,ODD);

	// Import
	QDPLatticeHalfFermionToSyCLCBVSpinor2(qdp_in, sycl_spinor_e);
	QDPLatticeHalfFermionToSyCLCBVSpinor2(qdp_in, sycl_spinor_o);

	// Export
	LatticeFermion qdp_out;
	SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_spinor_e, qdp_out);
	SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_spinor_o, qdp_out);

	for(int cb=EVEN; cb <= ODD; ++cb) {
		for(int i=0; i < num_cbsites; ++i) {
			for(int color=0; color <3; ++color) {
				for(int spin=0; spin < 2; ++spin) {
					float ref_re = qdp_in.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).real();
					float ref_im = qdp_in.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).imag();

					float out_re = qdp_out.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).real();
					float out_im = qdp_out.elem(rb[cb].siteTable()[i]).elem(spin).elem(color).imag();

					ASSERT_FLOAT_EQ( ref_re, out_re);
					ASSERT_FLOAT_EQ( ref_im, out_im);
				} // spin
			} // color
		}
	} // cb
}

TYPED_TEST(TestVNode, TestPackGauge)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using GaugeType = SyCLCBFineVGaugeField<MGComplex<float>,VN>;

	multi1d<LatticeColorMatrix> u(Nd);
	IndexArray cb_latdims=info.GetCBLatticeDimensions();
	int num_cbsites= info.GetNumCBSites();
	MasterLog(INFO, "Num_cbsites=%d", num_cbsites);
	MasterLog(INFO, "cb_latdims=(%d,%d,%d,%d)", cb_latdims[0], cb_latdims[1],cb_latdims[2],cb_latdims[3]);

	for(int mu=0; mu < 4; ++mu)  {
#pragma omp parallel for
		for(int i=0; i < num_cbsites; ++i) {
			IndexArray coords = LayoutLeft::coords(i,cb_latdims);
			float value=computeLane<VectorLength>(coords,cb_latdims);
			int qdp_idx = rb[EVEN].siteTable()[i];

			for(int color=0; color < 3; ++color) {
				for(int color2=0; color2 < 3; ++color2) {
					u[mu].elem(qdp_idx).elem().elem(color,color2).real() = value;
					u[mu].elem(qdp_idx).elem().elem(color,color2).imag() = -value;
				}
			}
		}
	}

	GaugeType sycl_u(info,EVEN);
	QDPGaugeFieldToSyCLCBVGaugeField(u, sycl_u);

	auto sycl_h = sycl_u.GetData().template get_access<cl::sycl::access::mode::read_write>();


	bool same_global_vectype = std::is_same< typename GaugeType::VecType, SIMDComplexSyCL<float,VN::VecLen> >::value;
	ASSERT_EQ( same_global_vectype, true);

	const LatticeInfo& vinfo = sycl_u.GetInfo();
	int num_vcbsites = vinfo.GetNumCBSites();

	// no parallel because GTest doesn't like assertions in parallel sections
	for(size_t i=0; i < num_vcbsites; ++i) {
		for(int dir=0; dir < 4; ++dir) {
			for(int color=0; color <3; ++color) {
				for(int color2=0; color2 < 3; ++color2) {

					auto vec_data = sycl_h(i,dir,color,color2);
					for(int lane=0; lane < VN::VecLen; ++lane) {
						float ref = lane;
						MGComplex<float> ve = LaneOps<float,VectorLength>::extract(vec_data,lane);

						ASSERT_FLOAT_EQ( ref, ve.real() );
						ASSERT_FLOAT_EQ( -ref , ve.imag() );

					}//  lane
				} // color2
			} // color
		} // dir
	}

	multi1d<LatticeColorMatrix> u_back(Nd);

	SyCLCBVGaugeFieldToQDPGaugeField(sycl_u, u_back);

	for(int mu=0; mu < Nd; ++mu) {

		// no parallel because GTest doesn't like assertions in parallel sections
		for(size_t i=0; i < num_cbsites; ++i) {

			for(int color=0; color <3; ++color) {
				for(int color2=0; color2 < 3; ++color2) {
					float ref_re = u[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).real();
					float ref_im = u[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).imag();

					float out_re = u_back[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).real();
					float out_im = u_back[mu].elem(rb[EVEN].siteTable()[i]).elem().elem(color,color2).imag();

					ASSERT_FLOAT_EQ( ref_re, out_re);
					ASSERT_FLOAT_EQ( ref_im, out_im);
				} // color2
			} // color
		}
	}// mu

}


TYPED_TEST(TestVNode, TestPackGauge2)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());
	int num_cbsites = info.GetNumCBSites();

	using VN = VNode<MGComplex<float>,VectorLength>;
	using GaugeType = SyCLCBFineVGaugeField<MGComplex<float>,VN>;

	multi1d<LatticeColorMatrix> qdp_in(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(qdp_in[mu]);
		reunit(qdp_in[mu]);
	}

	GaugeType sycl_u_e(info,EVEN);
	GaugeType sycl_u_o(info,ODD);

	// Import
	QDPGaugeFieldToSyCLCBVGaugeField(qdp_in, sycl_u_e);
	QDPGaugeFieldToSyCLCBVGaugeField(qdp_in, sycl_u_o);

	// Export
	multi1d<LatticeColorMatrix> qdp_out(Nd);
	SyCLCBVGaugeFieldToQDPGaugeField(sycl_u_e, qdp_out);
	SyCLCBVGaugeFieldToQDPGaugeField(sycl_u_o, qdp_out);

	for(int mu=0; mu < Nd; ++mu) {
		for(int cb=EVEN; cb <= ODD; ++cb) {
			// No parallel for because Googletest assertions complain if in a parallel region
			for(size_t i=0; i < num_cbsites; ++i) {
				for(int color=0; color <3; ++color) {
					for(int color2=0; color2 < 3; ++color2) {
						float ref_re = qdp_in[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).real();
						float ref_im = qdp_in[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).imag();

						float out_re = qdp_out[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).real();
						float out_im = qdp_out[mu].elem(rb[cb].siteTable()[i]).elem().elem(color,color2).imag();

						ASSERT_FLOAT_EQ( ref_re, out_re);
						ASSERT_FLOAT_EQ( ref_im, out_im);
					} // spin
				} // color
			}
		} // cb
	}//mu
}
