#include "gtest/gtest.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_sycl_a.h"
#include "dslash/dslash_vnode_a.h"

#include <CL/sycl.hpp>
using namespace MG;
using namespace cl::sycl;

#if 0
class SyCLVecTypeTest : public ::testing::Test {
public:
	static constexpr size_t num_float_elem() { return 1024; }
	static constexpr size_t num_cmpx_elem() { return num_float_elem()/2; }
	static constexpr size_t N=4;

	cpu_selector my_cpu;
	queue MyQueue;
	buffer<float,1> f_buf;
	SyCLVecTypeTest() : f_buf{range<1>{num_float_elem()}}, MyQueue{my_cpu} {}

protected:
	void SetUp() override
	{

		{

			std::cout << "Filling" << std::endl;
			range<1> N_vecs{num_cmpx_elem()/N};

			// Fill the buffers
			MyQueue.submit([&](handler& cgh) {
				auto write_fbuf = f_buf.get_access<access::mode::write>(cgh);

				cgh.parallel_for<class prefill>(N_vecs, [=](id<1> vec_id) {
					for(size_t lane=0; lane < N; ++lane) {


						MGComplex<float> fval( static_cast<float>(vec_id[0]*2*N + 2*lane) ,
								vec_id[0]*2*N + 2*lane + 1 );
						StoreLane<float,N>(lane,vec_id[0],write_fbuf, fval);
					}
				}); // parallel for
			}); // queue submit

			MyQueue.wait();
		} // End of scope
	} // SetUp

};
#endif


TEST(SyCLVNodeTest, Compile)
{
	VNode<MGComplex<float>,1> vn1;
}
