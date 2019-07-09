#include "gtest/gtest.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_vnode.h"

#include <CL/sycl.hpp>
#include <type_traits>
using namespace MG;
using namespace cl::sycl;


template<typename T>
class SyCLVNodeTest : public ::testing::Test {
public:
	static constexpr size_t num_float_elem() { return 1024; }
	static constexpr size_t num_cmpx_elem() { return num_float_elem()/2; }
	static constexpr T N = T::value();
	cpu_selector my_cpu;
	queue MyQueue;
	buffer<float,1> f_buf;
	SyCLVNodeTest() : f_buf{range<1>{num_float_elem()}}, MyQueue{my_cpu} {}

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

#ifdef TESTING_MG_VECTYPE_A
 using test_type_params = ::testing::Types< std::integral_constant<size_t,1>,
		 	 	 	 	 	 	 	 	    std::integral_constant<size_t,2>,
											std::integral_constant<size_t,4>,
											std::integral_constant<size_t,8>,
											std::integral_constant<size_t,16> >;
#else
 using test_type_params = ::testing::Types< std::integral_constant<std::size_t,1>,
		 	 	 	 	 	 	 	 	    std::integral_constant<std::size_t,2>,
											std::integral_constant<std::size_t,4>,
											std::integral_constant<std::size_t,8> >;
#endif

TYPED_TEST_SUITE(SyCLVNodeTest, test_type_params);

TEST_F(SyCLVNodeTest, Compile)
{
	VNode<MGComplex<float>,N> vn1;
}
#if 0
TEST_F(SyCLVNodeTest, CheckSquareId1)
{
	using T = SIMDComplexSyCL<float,8>;
	using VN = VNode<T,N>
	// All Vec load/stores need multi-ptr
	// Which are only in kernel scope.
	{
		MyQueue.submit([&](handler& cgh) {
			auto vecbuf = f_buf.get_access<access::mode::read_write>(cgh);
			cgh.single_task<class vec_test_load>([=](){
				// Read the first vector. (We know what this is)
				T fc;
				Load(fc,0,vecbuf.get_pointer());

				auto tmp = VN::permute<X_DIR>(fc);
				Store(1,vecbuf.get_pointer(),tmp);
				tmp = VN::permute<Y_DIR>(fc);
				Store(2,vecbuf.get_pointer(),tmp);
				tmp = VN::permute<Z_DIR>(fc);
				Store(3,vecbuf.get_pointer(),tmp);
				tmp = VN::permute<T_DIR>(fc);
				Store(4,vecbuf.get_pointer(),tmp);
			});
		});
	}
	auto h_f = f_buf.get_access<access::mode::read>();

	std::array<MGComplex<float>, N> orig,permX,permY,permZ,permT;
	// For each 'Lane'
	for(size_t i=0; i < N ; ++i) {
		// Load them up
		orig[i]=LoadLane<float,N>(i,0,h_f);
		permX[i]=LoadLane<float,N>(i,1,h_f);
		permY[i]=LoadLane<float,N>(i,2,h_f);
		permZ[i]=LoadLane<float,N>(i,3,h_f);
		permT[i]=LoadLane<float,N>(i,4,h_f);
	}

}
#endif
