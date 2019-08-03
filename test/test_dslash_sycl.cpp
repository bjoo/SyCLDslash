/*
 * test_dslash_sycl.cpp
 *
 *  Created on: Aug 2, 2019
 *      Author: bjoo
 */



#include "sycl_dslash_config.h"
#include "gtest/gtest.h"
#include "test_env.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "utils/print_utils.h"

#include "sycl_dslash_config.h"   // Build options
#include "dslash/dslash_defaults.h" // Default layouts
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"     // Vector type s
#include "dslash/sycl_qdp_vutils.h" // Utils
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/sycl_vdslash.h"
#include "sycl_lattice_spin_proj_utils.h"
using namespace MG;
using namespace MGTesting;
using namespace QDP;

template<typename T>
class TestVDslash :  public ::testing::Test{};

#ifdef MG_FORTRANLIKE_COMPLEX
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8> >;
#else
#if 0
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8>,
		std::integral_constant<int,16>	>;
#endif

using test_types = ::testing::Types<
		std::integral_constant<int,1> >;
#endif

TYPED_TEST_CASE(TestVDslash, test_types);

TYPED_TEST(TestVDslash, TestVDslash)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,4,4,4}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
	//gaussian(gauge_in[mu]);
		// reunit(gauge_in[mu]);
		gauge_in[mu]=1;
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	using VN = VNode<MGComplex<float>,VectorLength>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<float>,VN,4>;
	using FullGaugeType = SyCLFineVGaugeField<MGComplex<float>,VN>;

	using GaugeType = SyCLCBFineVGaugeFieldDoubleCopy<MGComplex<float>,VN>;

	SpinorType  sycl_spinor_even(info,EVEN);
	SpinorType  sycl_spinor_odd(info,ODD);
	FullGaugeType  sycl_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToSyCLVGaugeField(gauge_in, sycl_gauge);
	GaugeType  gauge_even(info,EVEN);
	import(gauge_even, sycl_gauge(EVEN), sycl_gauge(ODD));

	GaugeType  gauge_odd(info, ODD);
	import(gauge_odd, sycl_gauge(ODD), sycl_gauge(EVEN));

	// Create the Dslash
	SyCLVDslash<VN,MGComplex<REAL32>,MGComplex<REAL32>> D(sycl_spinor_even.GetInfo());

	// QDP++ result
	LatticeFermion psi_out = zero;

	// SyCL  result
	LatticeFermion sycl_out=zero;

	//for(int cb=0; cb < 2; ++cb) {
	{
		int cb=0;
		// This could be done more elegantly
		SpinorType& out_spinor = (cb == EVEN) ? sycl_spinor_even : sycl_spinor_odd;
		SpinorType& in_spinor = (cb == EVEN) ? sycl_spinor_odd: sycl_spinor_even;
		GaugeType& gauge = ( cb == EVEN ) ? gauge_even : gauge_odd;


		for(int isign=-1; isign < 2; isign+=2) {

			// In the Host
			psi_out = zero;


			psi_out[rb[cb]] =
						spinReconstructDir3Minus(shift(spinProjectDir3Minus(psi_in),BACKWARD,3)) +
						spinReconstructDir2Minus(shift(spinProjectDir2Minus(psi_in),BACKWARD,2)) +
						spinReconstructDir1Minus(shift(spinProjectDir1Minus(psi_in),BACKWARD,1)) +
						spinReconstructDir1Plus(shift(spinProjectDir1Plus(psi_in),FORWARD,1)) +
						spinReconstructDir2Plus(shift(spinProjectDir2Plus(psi_in),FORWARD,2)) +
						spinReconstructDir3Plus(shift(spinProjectDir3Plus(psi_in),FORWARD,3));
	//					spinReconstructDir0Minus(shift(spinProjectDir0Minus(psi_in),BACKWARD,0))+
	//					spinReconstructDir0Plus(shift(spinProjectDir0Plus(psi_in),FORWARD,0));

		     // Target cb=1 for now.
			//dslash(psi_out,gauge_in,psi_in,isign,cb);

			QDPLatticeFermionToSyCLCBVSpinor(psi_in, in_spinor);


			MasterLog(INFO, "Applying D: cb=%d isign=%d\n", cb,isign);
			D(in_spinor,gauge,out_spinor,isign);


			sycl_out = zero;
			SyCLCBVSpinorToQDPLatticeFermion(out_spinor, sycl_out);

			// Check Diff on Odd
			//psi_out[rb[cb]] -= sycl_out;
			double norm_diff = toDouble(sqrt(norm2(psi_out-sycl_out,rb[cb])))/toDouble(rb[cb].numSiteTable());

			MasterLog(INFO, "norm_diff / site= %lf", norm_diff);
			int num_sites = info.GetNumCBSites();
#if 1
			for(int site=0; site < num_sites; ++site) {
				for(int spin=0; spin < 4; ++spin ) {
					for(int color=0; color < 3; ++color) {
						QDPIO::cout << "psi_out("<<site<<","<<color<<","<<spin<<") = (" <<	psi_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real() << "," << psi_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag() << ")     sycl_out("<<site<<","<<color<<","<<spin<<") = (" <<
								sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real()
								<< " , " << sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag() << " ) " << std::endl;

					}
				}
			}

#endif



		} //isign
	} // cb


} // TEST

