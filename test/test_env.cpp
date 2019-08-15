
#include "test_env.h"

#include "utils/initialize.h"
#include <CL/sycl.hpp>
#include <cstring>
#include <cstdlib>
using namespace MG;

using namespace cl::sycl;

namespace MGTesting {


	void listDevices();
	void printHelp(const std::string& progname);

	/** The Constructor to set up a test environment.
	 *   Its job is essentially to set up QMP
	 */

	int TestEnv::chosen_device = -1;


	TestEnv::TestEnv(int  *argc, char ***argv)
	{

		// Process args
		for(int arg=0; arg < *argc; ++arg) {
			if ( std::strcmp((*argv)[arg], "-l") == 0 ) {
				listDevices();
				std::exit(EXIT_SUCCESS);
			}
			if ( std::strcmp((*argv)[arg], "-d") == 0) {
				int dselect=std::atoi((*argv)[arg+1]);
				chosen_device = dselect;
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

	int TestEnv::getChosenDevice() {
		return chosen_device;
	}

	void listDevices() {

		std::vector<cl::sycl::device> devices;
		auto dlist = cl::sycl::device::get_devices();
		devices.insert(devices.end(),dlist.begin(),dlist.end());

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




