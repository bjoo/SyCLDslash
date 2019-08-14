
#include "test_env.h"

#include "utils/initialize.h"
#include <CL/sycl.hpp>
#include <cstring>
#include <cstdlib>
using namespace MG;

using namespace cl::sycl;

namespace MGTesting {

	namespace {
		std::unique_ptr<cl::sycl::queue> _theQueue(nullptr);
		std::vector<cl::sycl::device> devices;
	}

	void listDevices();
	void printHelp(const std::string& progname);

	/** The Constructor to set up a test environment.
	 *   Its job is essentially to set up QMP
	 */
	TestEnv::TestEnv(int  *argc, char ***argv)
	{


		// Set up list of devices
		//
		auto dlist = cl::sycl::device::get_devices();
		devices.clear();
		devices.insert(devices.end(),dlist.begin(),dlist.end());

		// Process args
		for(int arg=0; arg < *argc; ++arg) {
			if ( std::strcmp((*argv)[arg], "-l") == 0 ) {
				listDevices();
				std::exit(EXIT_SUCCESS);
			}
			if ( std::strcmp((*argv)[arg], "-d") == 0) {
				int dselect=std::atoi((*argv)[arg+1]);
				auto name = devices[dselect].get_info<info::device::name>();
				std::cout << "Selecting device: " << dselect << " : " << name  << std::endl;
				_theQueue.reset( new cl::sycl::queue( devices[dselect]));

			}


			if ( std::strcmp((*argv)[arg], "-h") == 0 ) {
				printHelp(std::string{(*argv)[0]});
				std::cout << "-h help from various sub-systems follows" << std::endl;

			}
			if ( std::strcmp((*argv)[arg], "-help") == 0 ) {
				printHelp(std::string{(*argv)[0]});
				std::exit(EXIT_SUCCESS);
			}
		}
		::MG::initialize(argc,argv);

	}

	TestEnv::~TestEnv() {
		/* Tear down QMP */
		::MG::finalize();
	}

	cl::sycl::queue& TestEnv::getQueue() {
		if ( !_theQueue ) {
			// If user has not set a queue -- pick the default one
			_theQueue.reset(new cl::sycl::queue);
		}

		// Return the queue
		return *(_theQueue);
	}

	void listDevices() {
		std::cout << "The system contains " << devices.size() << " devices" << std::endl;
		for(int i=0; i < devices.size(); i++) {
			device& d = devices[i];


			auto name = d.get_info<info::device::name>();

			auto driver_version = d.get_info<info::device::driver_version>();
			bool is_accelerator = d.is_accelerator() || d.is_gpu();

			std::cout << "Device " << i << " : " << name <<  " Driver Version: "
					<< driver_version << " Is accelerator: " <<( is_accelerator ? "YES" : "NO" )<< std::endl;

		}

	}

	void printHelp(const std::string& progname) {
		std::cout << progname << " <options> " << std::endl;
		std::cout << "\t<options> can be:" << std::endl;
		std::cout << "\t\t -help      -- print this help " << std::endl;
		std::cout << "\t\t -h         -- this help followed by help from googletest and qdpxx" << std::endl;
		std::cout << "\t\t -l         -- list available OpenCL devices" <<std::endl;
		std::cout << "\t\t -d dev_id  -- select device dev_id" << std::endl;
	}

}

	/* This is a convenience routine to setup the test environment for GTest and its layered test environments */
	int main(int argc, char *argv[])
	{
		  ::testing::InitGoogleTest(&argc, argv);
		  ::testing::AddGlobalTestEnvironment(new MGTesting::TestEnv(&argc,&argv));
		  return RUN_ALL_TESTS();
	}




