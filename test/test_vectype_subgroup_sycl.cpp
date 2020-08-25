/*
 * test_vectype_tests.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */

#include "sycl_dslash_config.h"   // Build options
#include "gtest/gtest.h"
#include "test_env.h"
#include "qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "utils/print_utils.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_vectype_sycl_subgroup.h"
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"
#include "dslash/sycl_qdp_vutils.h"
#include <CL/sycl.hpp>
#include <cstdio>
using namespace MG;
using namespace MGTesting;
using namespace cl;
using namespace QDP;
using std::printf;

template<typename T>
class SYCLSGVecTypeTest : public ::testing::Test {
public:
	SYCLSGVecTypeTest() : _q(nullptr){
		auto dlist = sycl::device::get_devices();
		devices.clear();
		devices.insert(devices.end(),dlist.begin(),dlist.end());

		int choice = TestEnv::getChosenDevice();
		if ( choice == -1 ) {
			_q.reset( new sycl::queue );
		}
		else {
			_q.reset( new sycl::queue( devices[choice]));
		}
	}

	sycl::queue& getQueue() const {
		return (*_q);
	}
private:
	std::vector<sycl::device> devices;
	std::unique_ptr<sycl::queue> _q;
};

using test_types = ::testing::Types<
		std::integral_constant<int,8>>;

TYPED_TEST_CASE(SYCLSGVecTypeTest, test_types);

// Hacktastic:
[[cl::intel_reqd_sub_group_size(8)]]
void force_sub_group_size8(){}


TYPED_TEST(SYCLSGVecTypeTest, SGManip1Test)
{
	// Set Vector Length
	static constexpr int V = TypeParam::value;

	// Get the Queue
	sycl::queue& q = this->getQueue();

	// Print device driver info
	auto dev=q.get_device();
	std::cout << "Using Device: " << dev.get_info<sycl::info::device::name>() << " Driver: "
				<< dev.get_info<sycl::info::device::driver_version>() << std::endl;

	// set up a basic lattice
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	LatticeInfo info(latdims,4,3,NodeInfo());

	LatticeFermion psi_in = zero;
	gaussian(psi_in);

	// Set up Internal Lattice stuff
	using VN=VNode<MGComplex<float>,V>; // Pick vector length
	using SpinorType = SyCLCBFineSGVSpinor<MGComplex<float>,VN,4>;

	SpinorType  in_spinor_even(info,EVEN);
	SpinorType  out_spinor_even(info,EVEN);
	// Import qdp spinor:
	QDPLatticeFermionToSyCLCBSGVSpinor(psi_in, in_spinor_even);

	// Setup done.


	// get the number of SIMD-ized sites
	size_t num_simd_sites = in_spinor_even.GetInfo().GetNumCBSites();
	size_t y_block_size = 1;

	q.submit([&](sycl::handler& cgh) {
		auto in_spinor = in_spinor_even.GetData().template get_access<sycl::access::mode::read>(cgh);
		auto out_spinor = out_spinor_even.GetData().template get_access<sycl::access::mode::write>(cgh);

		cgh.parallel_for( sycl::nd_range<1>({V*num_simd_sites}, {V}), [=](sycl::nd_item<1> nd_idx) [[cl::intel_reqd_sub_group_size(V)]] {
#if 0
			if constexpr (V==8) {
				force_sub_group_size8();
			}
#endif

			sycl::group<1>  gp = nd_idx.get_group();
			sycl::intel::sub_group sg = nd_idx.get_sub_group();
			// Pick the site
			size_t coarse_site = gp.get_id()[0];
			size_t lane =sg.get_local_id()[0];

			for(size_t spin=0; spin < 4; ++spin) {
				for(size_t color=0; color < 3; ++color) {
					sycl::vec<float,2> simd = sg.load<2,float,sycl::access::address_space::global_space>( in_spinor.get_pointer() + in_spinor.offset(coarse_site,spin,color,0,0) );
					sg.store( out_spinor.get_pointer() + out_spinor.offset(coarse_site,spin,color,0,0), simd);
				}
			}
		});

	});
	q.wait_and_throw();


	// Check results: Convert back to QDP++
	// EXPORT OUTPUT VECTOR
	LatticeFermion sycl_out;
	gaussian(sycl_out); /// should get overwritten
	SyCLCBSGVSpinorToQDPLatticeFermion(out_spinor_even, sycl_out);

	// Check Diff
	double norm_diff = toDouble(sqrt(norm2(psi_in-sycl_out,rb[0])))/toDouble(rb[0].numSiteTable());

//	MasterLog(INFO, "norm_diff / site= %lf", norm_diff);

#if 0
	size_t num_sites = info.GetNumCBSites();
	int cb=0;
	for(int site=0; site < num_sites; ++site) {
		for(int spin=0; spin < 4; ++spin ) {
			for(int color=0; color < 3; ++color) {
				bool pooh = false;
				size_t coarse_site = site / V;
				size_t lane = site % V;

				float in_re = psi_in.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real();
				float in_im = psi_in.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag();
				float out_re = sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real();
				float out_im = sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag();

				if ( fabs( in_re - out_re) > 5.0e-7  ) { pooh = true; }
				if ( fabs( in_im - out_im) > 5.0e-7  ) { pooh = true; }

				QDPIO::cout << "psi_in("<<coarse_site<<","<<color<<","<<spin<<"," << lane << ") = (" << in_re
								<< "," << in_im << ")     sycl_out(" <<coarse_site<<","<<color<<","<<spin<<","<< lane << ") = (" << out_re
									<< " , " << out_im << " ) ";

				if( pooh ) {
					QDPIO::cout << "      <---------- ERROR ERROR ERROR ";
				}
				QDPIO::cout <<std::endl;
			}
		}
	}

#endif
	ASSERT_LT( norm_diff, 5.0e-7 ) ;

}

TYPED_TEST(SYCLSGVecTypeTest, SGPermuteT)
{
	// Set Vector Length
	static constexpr int V = TypeParam::value;

	using VN = VNode<float,V>;

	// Get the Queue
	sycl::queue& q = this->getQueue();

	// Print device driver info
	auto dev=q.get_device();
	std::cout << "Using Device: " << dev.get_info<sycl::info::device::name>() << " Driver: "
				<< dev.get_info<sycl::info::device::driver_version>() << std::endl;

	// set up a basic lattice
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	LatticeInfo info(latdims,4,3,NodeInfo());

	LatticeFermion psi_in = zero;
	gaussian(psi_in);

	// Set up Internal Lattice stuff

	using SpinorType = SyCLCBFineSGVSpinor<MGComplex<float>,VN,4>;

	SpinorType  in_spinor_even(info,EVEN);
	SpinorType  in_spinor_odd(info,ODD);
	SpinorType  out_spinor_even(info,EVEN);
	// Import qdp spinor:
	QDPLatticeFermionToSyCLCBSGVSpinor(psi_in, in_spinor_even);
	QDPLatticeFermionToSyCLCBSGVSpinor(psi_in, in_spinor_odd);
	// Setup done.


	// get the number of SIMD-ized sites
	LatticeInfo coarse_info = in_spinor_even.GetInfo();
	size_t num_simd_sites = coarse_info.GetNumCBSites();
	IndexArray cdims = coarse_info.GetCBLatticeDimensions();
	SiteTable neigh(cdims[0],cdims[1],cdims[2],cdims[3]);

	q.submit([&](sycl::handler& cgh) {
		auto in_spinor = in_spinor_odd.GetData().template get_access<sycl::access::mode::read>(cgh);
		auto out_spinor = out_spinor_even.GetData().template get_access<sycl::access::mode::write>(cgh);
		auto neigh_table_access = neigh.template get_access<sycl::access::mode::read>(cgh);

		cgh.parallel_for( sycl::nd_range<1>({V*num_simd_sites}, {V}), [=](sycl::nd_item<1> nd_idx) {

			if constexpr (V==8) {
				force_sub_group_size8();
			}

			sycl::group<1>  gp = nd_idx.get_group();
			sycl::intel::sub_group  sg = nd_idx.get_sub_group();
			// Pick the site
			size_t coarse_site = gp.get_id()[0];
			size_t lane =sg.get_local_id()[0];

			// Get Forward Neighbor
			IndexArray my_coords = LayoutLeft::coords(coarse_site,cdims);
			size_t t_minus = 0;
			bool do_permute = false;
			neigh_table_access.NeighborTMinus(my_coords[0], my_coords[1],my_coords[2],my_coords[3],t_minus, do_permute);

			for(size_t spin=0; spin < 4; ++spin) {
				for(size_t color=0; color < 3; ++color) {



					MGComplex<float> simd = Load(in_spinor.offset(t_minus,spin,color,0,0), in_spinor.get_pointer(), sg);
					MGComplex<float> shuffled_simd = do_permute ? Permute<float,V>::permute_xor_T(simd,sg) : simd;
					Store( out_spinor.offset(coarse_site,spin,color,0,0), out_spinor.get_pointer(),shuffled_simd,sg);
				}
			}
		});

	});
	q.wait_and_throw();

	// Check results: Convert back to QDP++
	// EXPORT OUTPUT VECTOR
	LatticeFermion sycl_out;
	gaussian(sycl_out); /// should get overwritten
	SyCLCBSGVSpinorToQDPLatticeFermion(out_spinor_even, sycl_out);

	// Check Diff
	double norm_diff = toDouble(sqrt(norm2(shift(psi_in,BACKWARD, 3)-sycl_out,rb[0])))/toDouble(rb[0].numSiteTable());

	MasterLog(INFO, "norm_diff / site= %lf", norm_diff);

#if 0
	size_t num_sites = info.GetNumCBSites();
	int cb=0;
	for(int site=0; site < num_sites; ++site) {
		for(int spin=0; spin < 4; ++spin ) {
			for(int color=0; color < 3; ++color) {
				bool pooh = false;
				size_t coarse_site = site / V;
				size_t lane = site % V;

				float in_re = psi_in.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real();
				float in_im = psi_in.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag();
				float out_re = sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).real();
				float out_im = sycl_out.elem(rb[cb].siteTable()[site]).elem(spin).elem(color).imag();

				if ( fabs( in_re - out_re) > 5.0e-7  ) { pooh = true; }
				if ( fabs( in_im - out_im) > 5.0e-7  ) { pooh = true; }

				QDPIO::cout << "psi_in("<<coarse_site<<","<<color<<","<<spin<<"," << lane << ") = (" << in_re
								<< "," << in_im << ")     sycl_out(" <<coarse_site<<","<<color<<","<<spin<<","<< lane << ") = (" << out_re
									<< " , " << out_im << " ) ";

				if( pooh ) {
					QDPIO::cout << "      <---------- ERROR ERROR ERROR ";
				}
				QDPIO::cout <<std::endl;
			}
		}
	}

#endif
	ASSERT_LT( norm_diff, 5.0e-7 ) ;

}
