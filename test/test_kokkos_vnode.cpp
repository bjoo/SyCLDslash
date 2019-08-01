#include "kokkos_dslash_config.h"
#include "gtest/gtest.h"
#include "test_env.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"


#include "./kokkos_types.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_vspinproj.h"
#include "./kokkos_matvec.h"

#include "kokkos_dslash_config.h"
#include "kokkos_vnode.h"
#include "kokkos_vtypes.h"
#include "kokkos_qdp_vutils.h"
#include "kokkos_traits.h"
#include "kokkos_vdslash.h"
#include <type_traits>

#include <cmath>
using namespace MG;
using namespace MGTesting;
using namespace QDP;

#ifdef KOKKOS_HAVE_CUDA
static constexpr int VectorLength=1;
#else

#if defined(MG_USE_AVX512) || defined(MG_USE_AVX2)
#ifdef MG_USE_AVX512
static constexpr int VectorLength=8;
#endif
#ifdef MG_USE_AVX2
static constexpr int VectorLength=4;
#endif
#else
static constexpr int VectorLength=8;
#endif
#endif

TEST(TestVNode,TestVSpinor)
{

	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,4>;
	// 4 spins
	SpinorType vnode_spinor(info, MG::EVEN);

	const LatticeInfo& c_info = vnode_spinor.GetInfo();
	IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
	for(int mu=0; mu < 4;++mu) {
		ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
				info.GetLatticeDimensions()[mu]/VNDims[mu] );
	}
	bool same_global_vectype = std::is_same< SpinorType::VecType, SIMDComplex<float,VectorLength> >::value;
	ASSERT_EQ( same_global_vectype, true);

	bool same_thread_vectype = std::is_same< VN::VecType, SIMDComplex<float,VectorLength> >::value;
	ASSERT_EQ( same_thread_vectype, true);


}
#if 1
TEST(TestVNode,TestVGauge)
{

	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using GaugeType = KokkosCBFineVGaugeField<MGComplex<float>,VN>;

	GaugeType vnode_spinor(info, MG::EVEN);

	const LatticeInfo& c_info = vnode_spinor.GetInfo();
	IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
	for(int mu=0; mu < 4;++mu) {
		ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
				info.GetLatticeDimensions()[mu]/VNDims[mu] );
	}
	bool same_global_vectype = std::is_same< GaugeType::VecType, SIMDComplex<float,VectorLength> >::value;
	ASSERT_EQ( same_global_vectype, true);

	bool same_thread_vectype = std::is_same< VN::VecType, SIMDComplex<float,VectorLength> >::value;
	ASSERT_EQ( same_thread_vectype, true);


}
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

TEST(TestVNode, TestPackSpinor)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,4>;

	LatticeFermion qdp_in = zero;

	IndexArray cb_latdims=info.GetCBLatticeDimensions();
	int num_cbsites= info.GetNumCBSites();
	MasterLog(INFO, "Num_cbsites=%d", num_cbsites);
	MasterLog(INFO, "cb_latdims=(%d,%d,%d,%d)", cb_latdims[0], cb_latdims[1],cb_latdims[2],cb_latdims[3]);
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites),[&](int i) {
		IndexArray coords;
		IndexToCoords(i,cb_latdims,coords);
		float value=computeLane(coords,cb_latdims);
		int qdp_idx = rb[EVEN].siteTable()[i];

		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {
				qdp_in.elem(qdp_idx).elem(spin).elem(color).real() = value;
				qdp_in.elem(qdp_idx).elem(spin).elem(color).imag() = 0;
			}
		}
	});


	SpinorType kokkos_spinor(info,EVEN);
	QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_spinor);

	auto kokkos_h = Kokkos::create_mirror_view( kokkos_spinor.GetData() );
	Kokkos::deep_copy( kokkos_h, kokkos_spinor.GetData() );

	bool same_global_vectype = std::is_same< SpinorType::VecType, SIMDComplex<float,VN::VecLen> >::value;
	ASSERT_EQ( same_global_vectype, true);

	const LatticeInfo& vinfo = kokkos_spinor.GetInfo();
	int num_vcbsites = vinfo.GetNumCBSites();
	Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_vcbsites), [=](int i) {
		for(int color=0; color <3; ++color) {
			for(int spin=0; spin < 4; ++spin) {
				auto vec_data = kokkos_h(i,spin,color);
				for(int lane=0; lane < VN::VecLen; ++lane) {
					float ref = lane;
					ASSERT_FLOAT_EQ( ref, vec_data(lane).real() );
					ASSERT_FLOAT_EQ(  0 , vec_data(lane).imag() );

				}//  lane
			} // spin
		} // color
	});
	LatticeFermion back_spinor=zero;
	KokkosCBVSpinorToQDPLatticeFermion(kokkos_spinor, back_spinor);
	Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
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
	});
}

TEST(TestVNode, TestPackSpinor2)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());
	int num_cbsites = info.GetNumCBSites();

	using VN = VNode<MGComplex<float>,VectorLength>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,4>;

	LatticeFermion qdp_in;
	gaussian(qdp_in);

	SpinorType kokkos_spinor_e(info,EVEN);
	SpinorType kokkos_spinor_o(info,ODD);

	// Import
	QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_spinor_e);
	QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_spinor_o);

	// Export
	LatticeFermion qdp_out;
	KokkosCBVSpinorToQDPLatticeFermion(kokkos_spinor_e, qdp_out);
	KokkosCBVSpinorToQDPLatticeFermion(kokkos_spinor_o, qdp_out);

	for(int cb=EVEN; cb <= ODD; ++cb) {
		Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
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
		});
	} // cb
}


TEST(TestVNode, TestPackHalfSpinor2)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());
	int num_cbsites = info.GetNumCBSites();

	using VN = VNode<MGComplex<float>,VectorLength>;
	using HalfSpinorType = KokkosCBFineVSpinor<MGComplex<float>,VN,2>;

	LatticeFermion qdp_in;
	gaussian(qdp_in);

	HalfSpinorType kokkos_spinor_e(hinfo,EVEN);
	HalfSpinorType kokkos_spinor_o(hinfo,ODD);

	// Import
	QDPLatticeHalfFermionToKokkosCBVSpinor2(qdp_in, kokkos_spinor_e);
	QDPLatticeHalfFermionToKokkosCBVSpinor2(qdp_in, kokkos_spinor_o);

	// Export
	LatticeFermion qdp_out;
	KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_spinor_e, qdp_out);
	KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_spinor_o, qdp_out);

	for(int cb=EVEN; cb <= ODD; ++cb) {
		Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
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
		});
	} // cb
}


TEST(TestVNode, TestPackGauge)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using GaugeType = KokkosCBFineVGaugeField<MGComplex<float>,VN>;

	multi1d<LatticeColorMatrix> u(Nd);
	IndexArray cb_latdims=info.GetCBLatticeDimensions();
	int num_cbsites= info.GetNumCBSites();
	MasterLog(INFO, "Num_cbsites=%d", num_cbsites);
	MasterLog(INFO, "cb_latdims=(%d,%d,%d,%d)", cb_latdims[0], cb_latdims[1],cb_latdims[2],cb_latdims[3]);

	for(int mu=0; mu < 4; ++mu)  {
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites),[&](int i) {
			IndexArray coords;
			IndexToCoords(i,cb_latdims,coords);
			float value=computeLane(coords,cb_latdims);
			int qdp_idx = rb[EVEN].siteTable()[i];

			for(int color=0; color < 3; ++color) {
				for(int color2=0; color2 < 3; ++color2) {

					u[mu].elem(qdp_idx).elem().elem(color,color2).real() = value;
					u[mu].elem(qdp_idx).elem().elem(color,color2).imag() = -value;
				}
			}
		});
	}

	GaugeType kokkos_u(info,EVEN);
	QDPGaugeFieldToKokkosCBVGaugeField(u, kokkos_u);

	auto kokkos_h = Kokkos::create_mirror_view( kokkos_u.GetData() );
	Kokkos::deep_copy( kokkos_h, kokkos_u.GetData() );

	bool same_global_vectype = std::is_same< GaugeType::VecType, SIMDComplex<float,VN::VecLen> >::value;
	ASSERT_EQ( same_global_vectype, true);

	const LatticeInfo& vinfo = kokkos_u.GetInfo();
	int num_vcbsites = vinfo.GetNumCBSites();

	Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_vcbsites), [=](int i) {
		for(int dir=0; dir < 4; ++dir) {
			for(int color=0; color <3; ++color) {
				for(int color2=0; color2 < 3; ++color2) {

					auto vec_data = kokkos_h(i,dir,color,color2);
					for(int lane=0; lane < VN::VecLen; ++lane) {
						float ref = lane;
						ASSERT_FLOAT_EQ( ref, vec_data(lane).real() );
						ASSERT_FLOAT_EQ( -ref , vec_data(lane).imag() );

					}//  lane
				} // color2
			} // color
		} // dir
	});

	multi1d<LatticeColorMatrix> u_back(Nd);

	KokkosCBVGaugeFieldToQDPGaugeField(kokkos_u, u_back);

	for(int mu=0; mu < Nd; ++mu) {
		Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {

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
		});
	}// mu

}


TEST(TestVNode, TestPackGauge2)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());
	int num_cbsites = info.GetNumCBSites();

	using VN = VNode<MGComplex<float>,VectorLength>;
	using GaugeType = KokkosCBFineVGaugeField<MGComplex<float>,VN>;

	multi1d<LatticeColorMatrix> qdp_in(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(qdp_in[mu]);
		reunit(qdp_in[mu]);
	}

	GaugeType kokkos_u_e(info,EVEN);
	GaugeType kokkos_u_o(info,ODD);

	// Import
	QDPGaugeFieldToKokkosCBVGaugeField(qdp_in, kokkos_u_e);
	QDPGaugeFieldToKokkosCBVGaugeField(qdp_in, kokkos_u_o);

	// Export
	multi1d<LatticeColorMatrix> qdp_out(Nd);
	KokkosCBVGaugeFieldToQDPGaugeField(kokkos_u_e, qdp_out);
	KokkosCBVGaugeFieldToQDPGaugeField(kokkos_u_o, qdp_out);

	for(int mu=0; mu < Nd; ++mu) {
		for(int cb=EVEN; cb <= ODD; ++cb) {
			Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_cbsites), [=](int i) {
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
			});
		} // cb
	}//mu
}


TEST(TestKokkos, TestVSpinProject)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());


	LatticeFermion qdp_in;
	LatticeHalfFermion qdp_out;
	LatticeHalfFermion kokkos_out;

	using VN = VNode<MGComplex<REAL>,VectorLength>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<REAL>,VN,4>;
	using HalfSpinorType = KokkosCBFineVSpinor<MGComplex<REAL>,VN,2>;

	gaussian(qdp_in);
	SpinorType kokkos_in(info,EVEN);
	HalfSpinorType kokkos_hspinor_out(hinfo,EVEN);

	QDPLatticeFermionToKokkosCBVSpinor(qdp_in, kokkos_in);

	{
		// sign = -1 dir = 0
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,-1);
		qdp_out[rb[0]] = spinProjectDir0Minus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,0,-1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		// sign = -1 dir = 1
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,-1);
		qdp_out[rb[0]] = spinProjectDir1Minus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,1,-1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = -1 dir = 2
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,-1);
		qdp_out[rb[0]] = spinProjectDir2Minus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,2,-1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = -1 dir = 3
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,-1);
		qdp_out[rb[0]] = spinProjectDir3Minus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,3,-1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}


	{
		// sign = 1 dir = 0
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,1);
		qdp_out[rb[0]] = spinProjectDir0Plus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,0,1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		// sign = 1 dir = 1
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,1);
		qdp_out[rb[0]] = spinProjectDir1Plus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,1,1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = 1 dir = 2
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,1);
		qdp_out[rb[0]] = spinProjectDir2Plus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,2,1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = 1 dir = 3
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,1);
		qdp_out[rb[0]] = spinProjectDir3Plus(qdp_in);
		qdp_out[rb[1]] = zero;

		KokkosVProjectLattice<MGComplex<REAL>,
		VN, SIMDComplex<REAL,VN::VecLen>,3,1>(kokkos_in,kokkos_hspinor_out);
		KokkosCBVSpinor2ToQDPLatticeHalfFermion(kokkos_hspinor_out,kokkos_out);
		qdp_out[rb[0]] -= kokkos_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}


}


TEST(TestKokkos, TestDslash)
{
	IndexArray latdims={{32,32,32,32}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	using VN = VNode<MGComplex<REAL32>,VectorLength>;
	using SpinorType = KokkosCBFineVSpinor<MGComplex<REAL32>,VN,4>;
	using FullGaugeType = KokkosFineVGaugeField<MGComplex<REAL32>,VN>;

	using GaugeType = KokkosCBFineVGaugeFieldDoubleCopy<MGComplex<REAL32>,VN>;

	SpinorType  kokkos_spinor_even(info,EVEN);
	SpinorType  kokkos_spinor_odd(info,ODD);
	FullGaugeType  kokkos_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToKokkosVGaugeField(gauge_in, kokkos_gauge);


	// Double Stor Gauge field
	GaugeType  gauge_even(info,EVEN);
	import(gauge_even, kokkos_gauge(EVEN), kokkos_gauge(ODD));

	GaugeType  gauge_odd(info, ODD);
	import(gauge_odd, kokkos_gauge(ODD), kokkos_gauge(EVEN));

	KokkosVDslash<VN,MGComplex<REAL32>,MGComplex<REAL32>,
	SIMDComplex<REAL32,VN::VecLen>,SIMDComplex<REAL32,VN::VecLen>> D(kokkos_spinor_even.GetInfo());

#ifdef KOKKOS_HAVE_CUDA
	IndexArray blockings[6] = { { 1,1,1,1 },
			{ 2,2,2,4 },
			{ 4,4,1,2 },
			{ 4,2,8,4 },
			{ 8,4,1,4 },
			{ 16,4,1,1} };
#else
	IndexArray blockings[6] = { { 1,1,1,1 },
			{ 2,2,2,4 },
			{ 4,4,1,2 },
			{ 4,2,8,4 },
			{ 8,4,1,4 },
			{ 4,2,2,16} };
#endif

for(int b=0; b < 6; ++b) {

	IndexType bx=blockings[b][0];
	IndexType by=blockings[b][1];
	IndexType bz=blockings[b][2];
	IndexType bt=blockings[b][3];

#ifdef KOKKOS_HAVE_CUDA
	if( bx*by*bz*bt > 256 ) continue;
#endif

	LatticeFermion psi_out = zero;
	LatticeFermion  kokkos_out=zero;

	for(int cb=0; cb < 2; ++cb) {
		SpinorType& out_spinor = (cb == EVEN) ? kokkos_spinor_even : kokkos_spinor_odd;
		SpinorType& in_spinor = (cb == EVEN) ? kokkos_spinor_odd: kokkos_spinor_even;
		GaugeType& gauge = ( cb == EVEN ) ? gauge_even : gauge_odd;


		for(int isign=-1; isign < 2; isign+=2) {

			// In the Host
			psi_out = zero;

			// Target cb=1 for now.
			dslash(psi_out,gauge_in,psi_in,isign,cb);

			QDPLatticeFermionToKokkosCBVSpinor(psi_in, in_spinor);


			MasterLog(INFO, "D with blocking=(%d,%d,%d,%d)", bx,by,bz,bt);
			D(in_spinor,gauge,out_spinor,isign, {bx,by,bz,bt});


			kokkos_out = zero;
			KokkosCBVSpinorToQDPLatticeFermion(out_spinor, kokkos_out);

			// Check Diff on Odd
			psi_out[rb[cb]] -= kokkos_out;
			double norm_diff = toDouble(sqrt(norm2(psi_out,rb[cb])))/toDouble(rb[cb].numSiteTable());

			MasterLog(INFO, "norm_diff / site= %lf", norm_diff);
			int num_sites = info.GetNumCBSites();
#if 0
			for(int site=0; site < num_sites; ++site) {
				for(int spin=0; spin < 4; ++spin ) {
					for(int color=0; color < 3; ++color) {
						QDPIO::cout << "psi_out("<<site<<","<<color<<","<<spin<<") = (" <<	psi_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real() << "," << psi_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag() << ")     kokkos_out("<<site<<","<<color<<","<<spin<<") = (" <<
								kokkos_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real()
								<< " , " << kokkos_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag() << " ) " << std::endl;

					}
				}
			}

#endif

			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0,num_sites),
					[=](int i) {
				int qdp_idx = rb[cb].siteTable()[i];
				for(int spin=0; spin < 4; ++spin) {
					for(int color=0; color < 3; ++color ) {
						REAL32 re=std::abs(psi_out.elem(qdp_idx).elem(spin).elem(color).real());
						REAL32 im=std::abs(psi_out.elem(qdp_idx).elem(spin).elem(color).imag());
						ASSERT_LT( re, 3.5e-6);
						ASSERT_LT( im, 3.5e-6 );
					}
				}
			});

		}
	}
} // blockings


}

#endif

int main(int argc, char *argv[])
{
	return ::MGTesting::TestMain(&argc, argv);
}

