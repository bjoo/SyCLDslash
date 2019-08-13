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

#include <chrono>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

using namespace  std::chrono;


template<typename T>
class TimeVDslash :  public ::testing::Test{};

#ifdef MG_FORTRANLIKE_COMPLEX
#if 0
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8> >;
#else

//using test_types = ::testing::Types<
//		std::integral_constant<int,1>,
//		std::integral_constant<int,4> >;   // length 8 for AVX2

using test_types = ::testing::Types<
		std::integral_constant<int,1> >;
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
//using test_types = ::testing::Types<
//		std::integral_constant<int,1>,
//		std::integral_constant<int,8> >;  // lenth 8 for AVX2
using test_types = ::testing::Types<
		std::integral_constant<int,1> >;
#endif
#endif

TYPED_TEST_CASE(TimeVDslash, test_types);

TYPED_TEST(TimeVDslash, DslashTime)
{
	// Vector length
	static constexpr int V = TypeParam::value;

#if 0
	cl::sycl::cpu_selector cpu;
	cl::sycl::queue q(cpu);
#else
	cl::sycl::gpu_selector gpu;
	cl::sycl::queue q(gpu);
#endif
	IndexArray latdims={{24,24,24,24}};

	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
		reunit(gauge_in[mu]);
	}

	LatticeFermion psi_in=zero;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<MGComplex<REAL32>,V>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<REAL32>,VN,4>;
	using FullGaugeType = SyCLFineVGaugeField<MGComplex<REAL32>,VN>;
	using GaugeType = SyCLCBFineVGaugeFieldDoubleCopy<MGComplex<REAL32>,VN>;

	SpinorType  sycl_spinor_even(info,EVEN);
	SpinorType  sycl_spinor_odd(info,ODD);
	FullGaugeType  sycl_gauge(info);



	// Import Gauge Field
	QDPGaugeFieldToSyCLVGaugeField(gauge_in, sycl_gauge);


	// Double Store Gauge field. This benchmark is always even cb.
	GaugeType  gauge_even(info,EVEN);


	// Import gets the rear neighbors, and permutes them if needed
	import(gauge_even,  sycl_gauge(EVEN), sycl_gauge(ODD));

	// Import spinor
	QDPLatticeFermionToSyCLCBVSpinor(psi_in, sycl_spinor_even);


	SyCLVDslash<VN,	MGComplex<float>,MGComplex<float> > D(sycl_spinor_even.GetInfo(),q);

#if 0
	IndexArray cb_latdims = sycl_spinor_even.GetInfo().GetCBLatticeDimensions();
	double num_sites = static_cast<double>(V*cb_latdims[0]*cb_latdims[1]*cb_latdims[2]*cb_latdims[3]);
#endif

	MasterLog(INFO, "Running timing for VectorLength=%u", V);
	int isign=1;
	MasterLog(INFO, "isign=%d First run (JIT-ing)", isign);
	{
		D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign);
	}

	int iters=1;
	MasterLog(INFO, "Calibrating");
	{
		high_resolution_clock::time_point start_time = high_resolution_clock::now();
		{
			D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign);
		} // all queues finish here.
		high_resolution_clock::time_point end_time = high_resolution_clock::now();

		double time_taken = (duration_cast<duration<double>>(end_time - start_time)).count();
		MasterLog(INFO, "One application=%16.8e (sec)", time_taken);
		double rfo = 1.0;
		double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
		double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites);
		double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites);
		double rfo_bytes_out = (1.0 + rfo)*bytes_out;
		double flops = static_cast<double>(1320.0*num_sites);

		MasterLog(INFO,"isign=%d Performance: %lf GFLOPS", isign, flops/(time_taken*1.0e9));
		MasterLog(INFO,"isign=%d Effective BW (RFO=0): %lf GB/sec",isign, (bytes_in+bytes_out)/(time_taken*1.0e9));
		MasterLog(INFO,"isign=%d Effective BW (RFO=1): %lf GB/sec",  isign, (bytes_in+rfo_bytes_out)/(time_taken*1.0e9));

#if 0
		iters = static_cast<int>( 10.0 / time_taken );
		// Do at least one lousy iteration
		if ( iters == 0 ) iters = 1;
		if ( iters > 500 ) iters=500;
#endif
		iters=200;
		MasterLog(INFO, "Setting Timing iters=%d",iters);
	}

	for(int rep=0; rep < 3; ++rep ) {

			// Time it.
			high_resolution_clock::time_point start_time = high_resolution_clock::now();
			for(int i=0; i < iters; ++i) {
				D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign);
			}

			high_resolution_clock::time_point end_time = high_resolution_clock::now();

			double time_taken = (duration_cast<duration<double>>(end_time - start_time)).count();


			double rfo = 1.0;
			double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
			double bytes_in = static_cast<double>((8*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites*iters);
			double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites*iters);
			double rfo_bytes_out = (1.0 + rfo)*bytes_out;
			double flops = static_cast<double>(1320.0*num_sites*iters);

			MasterLog(INFO,"isign=%d Performance: %lf GFLOPS", isign, flops/(time_taken*1.0e9));
			MasterLog(INFO,"isign=%d Effective BW (RFO=0): %lf GB/sec",isign, (bytes_in+bytes_out)/(time_taken*1.0e9));
			MasterLog(INFO,"isign=%d Effective BW (RFO=1): %lf GB/sec",  isign, (bytes_in+rfo_bytes_out)/(time_taken*1.0e9));



		// } // isign
		MasterLog(INFO,"");
	} // rep
}

