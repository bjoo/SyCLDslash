#ifndef TEST_ENV_H
#define TEST_ENV_H

#include "gtest/gtest.h"
#include <vector>
#include <CL/sycl.hpp>
/** A Namespace for testing utilities */
namespace MGTesting {

/** A Test Environment to set up QMP */
class TestEnv : public ::testing::Environment {
public:
	TestEnv(int *argc, char ***argv);
	~TestEnv();

	static int getChosenDevice();
private:
	static int chosen_device;

};

} // Namespace MGTesting

int main(int argc, char *argv[]);
#endif
