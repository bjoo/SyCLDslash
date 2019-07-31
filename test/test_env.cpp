
#include "test_env.h"

#include "utils/initialize.h"

using namespace MG;

namespace MGTesting {

	/** The Constructor to set up a test environment.
	 *   Its job is essentially to set up QMP
	 */
	TestEnv::TestEnv(int  *argc, char ***argv)
	{
		::MG::initialize(argc,argv);
	}

	TestEnv::~TestEnv() {
		/* Tear down QMP */
		::MG::finalize();
	}
}

	/* This is a convenience routine to setup the test environment for GTest and its layered test environments */
	int main(int argc, char *argv[])
	{
		  ::testing::InitGoogleTest(&argc, argv);
		  ::testing::AddGlobalTestEnvironment(new MGTesting::TestEnv(&argc,&argv));
		  return RUN_ALL_TESTS();
	}




