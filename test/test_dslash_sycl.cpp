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
using namespace cl::sycl;

template<typename T>
class TestVDslash :  public ::testing::Test{};

#ifdef MG_FORTRANLIKE_COMPLEX
#if 0
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8> >;

#else
using test_types = ::testing::Types<
		std::integral_constant<int,1>>;
#endif
#else

#if 0
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8>,
		std::integral_constant<int,16>	>;
#else

// Get Scalar (nonvectorized and AVX=2 (8) for now.
using test_types = ::testing::Types<
		std::integral_constant<int,1>>;
#endif
#endif

TYPED_TEST_CASE(TestVDslash, test_types);

TYPED_TEST(TestVDslash, TestVDslash)
{
	static constexpr int VectorLength = TypeParam::value;

	cl::sycl::queue q = TestEnv::getQueue();
	auto dev=q.get_device();
	std::cout << "Using Device: " << dev.get_info<info::device::name>() << " Driver: "
			<< dev.get_info<info::device::driver_version>() << std::endl;


	IndexArray latdims={{8,4,4,4}};
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
	SyCLVDslash<VN,MGComplex<REAL32>,MGComplex<REAL32>> D(sycl_spinor_even.GetInfo(),q);

	// QDP++ result
	LatticeFermion psi_out = zero;

	// SyCL  result
	LatticeFermion sycl_out=zero;

	for(int cb=0; cb < 2; ++cb) {
		// This could be done more elegantly
		SpinorType& out_spinor = (cb == EVEN) ? sycl_spinor_even : sycl_spinor_odd;
		SpinorType& in_spinor = (cb == EVEN) ? sycl_spinor_odd: sycl_spinor_even;
		GaugeType& gauge = ( cb == EVEN ) ? gauge_even : gauge_odd;


		for(int isign=-1; isign < 2; isign+=2) {

			// In the Host
			psi_out = zero;


		    // Reference Dslash
			dslash(psi_out,gauge_in,psi_in,isign,cb);

			// SyCL Dslash:
			// Import input vector
			QDPLatticeFermionToSyCLCBVSpinor(psi_in, in_spinor);


			MasterLog(INFO, "Applying D: cb=%d isign=%d\n", cb,isign);
			// Apply
			D(in_spinor,gauge,out_spinor,isign);

			// EXPORT OUTPUT VECTOR
			sycl_out = zero;
			SyCLCBVSpinorToQDPLatticeFermion(out_spinor, sycl_out);

			// Check Diff
			double norm_diff = toDouble(sqrt(norm2(psi_out-sycl_out,rb[cb])))/toDouble(rb[cb].numSiteTable());

			MasterLog(INFO, "norm_diff / site= %lf", norm_diff);
			int num_sites = info.GetNumCBSites();

			ASSERT_LT( norm_diff, 5.0e-7 ) ;
#if 0
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

