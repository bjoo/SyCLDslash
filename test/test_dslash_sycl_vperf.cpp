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
using namespace cl::sycl;

template<typename T>
class TimeVDslash :  public ::testing::Test {
public:
	TimeVDslash() : _q(nullptr){
		auto dlist = cl::sycl::device::get_devices();
		devices.clear();
		devices.insert(devices.end(),dlist.begin(),dlist.end());

		int choice = TestEnv::getChosenDevice();
		if ( choice == -1 ) {
			_q.reset( new cl::sycl::queue );
		}
		else {
			_q.reset( new cl::sycl::queue( devices[choice]));
		}
	}

	cl::sycl::queue& getQueue() const {
		return (*_q);
	}
private:
	std::vector<cl::sycl::device> devices;
	std::unique_ptr<cl::sycl::queue> _q;
};

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



	cl::sycl::queue& q = this->getQueue();
	size_t wgroup_size = TestEnv::getChosenWorkgroupSize();
	bool use_wgroup = wgroup_size > 0;

	auto dev=q.get_device();
	std::cout << "Using Device: " << dev.get_info<info::device::name>() << " Driver: "
				<< dev.get_info<info::device::driver_version>() << std::endl;

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
		if( use_wgroup ) {
			MasterLog(INFO, "Using supplied wgroup size of %d", wgroup_size);
			D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign,wgroup_size);
		}
		else {
			MasterLog(INFO, "Will tune ...");
			D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign);
		}
	}

	int iters=1;
	MasterLog(INFO, "Calibrating");
	{

		high_resolution_clock::time_point start_time = high_resolution_clock::now();
		{
			if( use_wgroup ) {
				D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign,wgroup_size);
			}
			else {
				D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign);
			}
		} // all queues finish here.
		high_resolution_clock::time_point end_time = high_resolution_clock::now();

		double time_taken = (duration_cast<duration<double>>(end_time - start_time)).count();
		MasterLog(INFO, "One application=%16.8e (sec)", time_taken);
		
		double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
                double flops = static_cast<double>(1320.0*num_sites);
                MasterLog(INFO,"isign=%d Performance: %lf GFLOPS", isign, flops/(time_taken*1.0e9));

		double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites);
	        for(int R=0; R < 8; ++R) {
		  double bytes_in = static_cast<double>(((8-R)*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites);
		  double rfo_bytes_in = bytes_in + bytes_out;
                  MasterLog(INFO,"isign=%d  R=%d Effective READ BW (RFO=0): %lf GB/sec",isign,R, bytes_in/(time_taken*1.0e9));
                  MasterLog(INFO,"isign=%d  R=%d Effective READ BW (RFO=1): %lf GB/sec",isign,R, (bytes_in+bytes_out)/(time_taken*1.0e9));
                  MasterLog(INFO,"isign=%d  R=%d Effective WRITE BW: %lf GB/sec",isign, R, bytes_out/(time_taken*1.0e9));
                  MasterLog(INFO,"isign=%d  R=%d Effective Total BW (RFO=0): %lf GB/sec",isign,R, (bytes_in+bytes_out)/(time_taken*1.0e9));
		  MasterLog(INFO,"isign=%d  R=%d Effective Total BW (RFO=1): %lf GB/sec\n",isign,R, (rfo_bytes_in+bytes_out)/(time_taken*1.0e9));
	        }
		MasterLog(INFO,"");

#ifndef MG_USE_FIXED_ITERS
		iters = static_cast<int>( 10.0 / time_taken );
		// Do at least one lousy iteration
		if ( iters == 0 ) iters = 1;
#else 
		iters=180;
#endif
		MasterLog(INFO, "Setting Timing iters=%d",iters);
	}

	for(int rep=0; rep < 3; ++rep ) {

			// Time it.
			high_resolution_clock::time_point start_time = high_resolution_clock::now();
			if( use_wgroup ) {
				for(int i=0; i < iters; ++i) {
					D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign,wgroup_size);
				}
			}
			else {
				for(int i=0; i < iters; ++i) {
					D(sycl_spinor_even,gauge_even,sycl_spinor_odd,isign);
				}
			}
			high_resolution_clock::time_point end_time = high_resolution_clock::now();

			double time_taken = (duration_cast<duration<double>>(end_time - start_time)).count();
			double num_sites = static_cast<double>((latdims[0]/2)*latdims[1]*latdims[2]*latdims[3]);
			double flops = static_cast<double>(1320.0*num_sites*iters);
                        MasterLog(INFO,"isign=%d Performance: %lf GFLOPS\n", isign, flops/(time_taken*1.0e9));

			double bytes_out = static_cast<double>(4*3*2*sizeof(REAL32)*num_sites*iters);

			for(int R=0; R < 8; ++R) {
			  double bytes_in = static_cast<double>(((8-R)*4*3*2*sizeof(REAL32)+8*3*3*2*sizeof(REAL32))*num_sites*iters);
			  double rfo_bytes_in = bytes_in + bytes_out;
                        
			  MasterLog(INFO,"isign=%d  R=%d Effective READ BW (RFO=0): %lf GB/sec",isign,R, bytes_in/(time_taken*1.0e9));
                          MasterLog(INFO,"isign=%d  R=%d Effective READ BW (RFO=1): %lf GB/sec",isign,R, (bytes_in+bytes_out)/(time_taken*1.0e9));
	 		  MasterLog(INFO,"isign=%d  R=%d Effective WRITE BW: %lf GB/sec",isign, R, bytes_out/(time_taken*1.0e9));
		  	  MasterLog(INFO,"isign=%d  R=%d Effective Total BW (RFO=0): %lf GB/sec",isign,R, (bytes_in+bytes_out)/(time_taken*1.0e9));
			  MasterLog(INFO,"isign=%d  R=%d Effective Total BW (RFO=1): %lf GB/sec\n",isign,R, (rfo_bytes_in+bytes_out)/(time_taken*1.0e9));
	                }
		        MasterLog(INFO,""); 

		// } // isign
		MasterLog(INFO,"");
	} // rep
}

