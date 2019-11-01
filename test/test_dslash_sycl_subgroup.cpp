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
#include "dslash/dslash_vectype_sycl_subgroup.h"
#include "dslash/sycl_vdslash_subgroup.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;
using namespace cl::sycl;

template<typename T>
class TestSGVDslash :  public ::testing::Test{
public:
	TestSGVDslash() : _q(nullptr){
		auto dlist = cl::sycl::device::get_devices();
		_devices.clear();
		_devices.insert(_devices.end(),dlist.begin(),dlist.end());

		int choice = TestEnv::getChosenDevice();
		if ( choice == -1 ) {
			_q.reset( new cl::sycl::queue );
		}
		else {
			_q.reset( new cl::sycl::queue( _devices[choice]));
		}
	}

	cl::sycl::queue& getQueue() const {
		return (*_q);
	}
private:
	std::vector<cl::sycl::device> _devices;
	std::unique_ptr<cl::sycl::queue> _q;
};

#ifdef MG_FORTRANLIKE_COMPLEX
#error "This code does not work for fortranlike complex"
#endif

using test_types = ::testing::Types<
		std::integral_constant<int,8>,
		std::integral_constant<int,16>	>;

TYPED_TEST_CASE(TestSGVDslash, test_types);

TYPED_TEST(TestSGVDslash, TestSGVDslash)
{
	static constexpr int VectorLength = TypeParam::value;

	cl::sycl::queue& q = this->getQueue();
	auto dev=q.get_device();
	std::cout << "Using Device: " << dev.get_info<info::device::name>() << " Driver: "
			<< dev.get_info<info::device::driver_version>() << std::endl;


	IndexArray latdims={{16,8,4,12}};
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
	using SpinorType = SyCLCBFineSGVSpinor<MGComplex<float>,VN,4>;
	using FullGaugeType = SyCLFineSGVGaugeField<MGComplex<float>,VN>;

	using GaugeType = SyCLCBFineSGVGaugeFieldDoubleCopy<MGComplex<float>,VN>;

	SpinorType  sycl_spinor_even(info,EVEN);
	SpinorType  sycl_spinor_odd(info,ODD);
	FullGaugeType  sycl_gauge(info);


	// Import Gauge Field
	QDPGaugeFieldToSyCLSGVGaugeField(gauge_in, sycl_gauge);
	GaugeType  gauge_even(info,EVEN);
	import(gauge_even, sycl_gauge(EVEN), sycl_gauge(ODD));

	GaugeType  gauge_odd(info, ODD);
	import(gauge_odd, sycl_gauge(ODD), sycl_gauge(EVEN));

	// Create the Dslash
	SyCLSGVDslash<VN,MGComplex<REAL32>,MGComplex<REAL32>> D(sycl_spinor_even.GetInfo(),q);

	// QDP++ result
	LatticeFermion psi_out = zero;

	// SyCL  result
	LatticeFermion sycl_out=zero;

	for(int cb=0; cb < 2; ++cb) {
		// This could be done more elegantly
		SpinorType& out_spinor = (cb == EVEN) ? sycl_spinor_even : sycl_spinor_odd;
		SpinorType& in_spinor = (cb == EVEN) ? sycl_spinor_odd: sycl_spinor_even;
		// SpinorType& in_spinor = (cb == EVEN) ? sycl_spinor_even: sycl_spinor_odd;
		GaugeType& gauge = ( cb == EVEN ) ? gauge_even : gauge_odd;

	for(int isign=-1; isign < 2; isign +=2) {



			MasterLog(INFO, "Applying D: cb=%d isign=%d\n", cb,isign);

				// EXPORT OUTPUT VECTOR
			gaussian(sycl_out); /// should get overwritten
			QDPLatticeFermionToSyCLCBSGVSpinor(psi_in, in_spinor);
			D(in_spinor,gauge,out_spinor,isign);
			SyCLCBSGVSpinorToQDPLatticeFermion(out_spinor, sycl_out);

			{
				dslash(psi_out,gauge_in,psi_in,isign,cb);
#if 0
				LatticeHalfFermion tmp,tmp2;
				int otherCB=1-cb;
				tmp[rb[otherCB]]  = spinProjectDir3Plus(psi_in);
				tmp2[rb[cb]] = shift(tmp, FORWARD, 3);
				psi_out[rb[cb]] += spinReconstructDir3Plus(gauge_in[3]*tmp2);
#endif
			}
			// Check Diff
			double norm_diff = toDouble(sqrt(norm2(psi_out-sycl_out,rb[cb])))/toDouble(rb[cb].numSiteTable());

			MasterLog(INFO, "norm_diff / site= %lf", norm_diff);
			int num_sites = info.GetNumCBSites();

#if 0
			for(int site=0; site < num_sites; ++site) {
				for(int spin=0; spin < 4; ++spin ) {
					for(int color=0; color < 3; ++color) {
						float po_r = psi_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real();
						float po_i = psi_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag();

						float so_r= sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real();
						float so_i= sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag();

						bool bad = false;
						if ( fabs(po_r -so_r) > 5.0e-7) bad = true;
						if ( fabs(po_i -so_i) > 5.0e-7) bad = true;
						QDPIO::cout << "psi_out("<<site<<","<<color<<","<<spin<<") = (" <<	po_r << "," << po_i << ")     sycl_out("
								<<site<<","<<color<<","<<spin<<") = (" << so_r << " , " << so_i << " ) ";
						if( bad )
							QDPIO::cout << "    <<--- BAD BAD BAD ";

						QDPIO::cout << std::endl;

					}
				}
			}

#endif
			ASSERT_LT( norm_diff, 5.0e-7);
		} //isign
	} // cb


} // TEST

