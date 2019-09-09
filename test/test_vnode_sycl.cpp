#include "gtest/gtest.h"
#include <type_traits>
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_vnode.h"
#include "lattice/constants.h"

#include <CL/sycl.hpp>

using namespace MG;
using namespace cl::sycl;

// Dummy class name so that the prefill is not overloaded
// It will be disambiguated by the type Q which is going to be just T
template<class T> class prefill {};

// It is templated but really what we will do here is
// pass integral_constant<int,Literal>
// which allows us to template on the Vector Length

template<typename T>
class SyCLVNodeTest : public ::testing::Test {
public:
	static constexpr size_t num_float_elem() { return 1024; }
	static constexpr size_t num_cmpx_elem() { return num_float_elem()/2; }

	// This is where we grab the 'N' out of the T
	static const size_t N = static_cast<size_t>(T::value);

	cpu_selector my_cpu;
	queue MyQueue;
	buffer<float,1> f_buf;
	buffer<float,1>& getBuf() { return f_buf; }
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
				cgh.parallel_for(N_vecs, [=](id<1> vec_id) {
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

 // (RRRRR)(IIIII) storage. For systems which support Vec Len of 16
 //  This uses notionally two registers per SIMD Complex tho how this
 // is mapped to hardware resources is up to the compiler. We define tests for
 // Up to veclen 16
#if 0
 using test_type_params = ::testing::Types< std::integral_constant<int,1>,
		 	 	 	 	 	 	 	 	    std::integral_constant<int,2>,
											std::integral_constant<int,4>,
											std::integral_constant<int,8>,
											std::integral_constant<int,16> >;
#else
 using test_type_params = ::testing::Types< std::integral_constant<int,1>>;
#endif
#else
 // (RIRIRIRIRI... ) storage. For systems that support a vec len of 16
 // This uses notionally 1 register of length N, to store N/2 Complex numbers.
 // although how this is mapped to registers is up to the Compiler. For
 // systems supporting AVX512 we define up to vector length 8
#if 0 // PENDING FIX TO PR620 only one type can be used here Why?
 using test_type_params = ::testing::Types< std::integral_constant<int,1>,
		 	 	 	 	 	 	 	 	    std::integral_constant<int,2>,
											std::integral_constant<int,4>,
											std::integral_constant<int,8> >;
#else

 using test_type_params = ::testing::Types< std::integral_constant<int,1> >;

#endif

#endif


#if 1
 // This macro instantiates all the test casess
TYPED_TEST_CASE(SyCLVNodeTest,test_type_params);
#endif

#if 1
// This is a typed test, so it will be instantiated for
// all the types in the test_type_params, so all vector lengths.
// as such it needs only one vector element declared
TYPED_TEST(SyCLVNodeTest, Compile)
{

	VNode<MGComplex<float>,TestFixture::N> vn1;
}
#endif

// This next auxiliary struct assumes we have original
// arrays of { {0,20},{1,21},{2,22},...,{N, 20+N} }
// and then works out the expected X, Y, Z and T permutations
// which we can check against in the tests. Since the check
// for the test loads up from either complex storage format into
// a std::array< MGComplex<>,N > we don't need to worry about whether
// it is vectype a) or b) modulo at initialization. Even that could probably
// be written storage independently I suspect. We only need to worry about
// the vector lenght here

template<int N>
struct PermuteArrays;

template<>
struct PermuteArrays<1> {
	// These aliases are to save our fingers
	using C = MGComplex<float>;
	using A = std::array<C,1>;

	// The expected permutes are constexpr functions
	// N=1 permutations do nothing
	static constexpr A expect_X(){ return A{ C{0,20} }; }
	static constexpr A expect_Y(){ return A{ C{0,20} }; }
	static constexpr A expect_Z(){ return A{ C{0,20} }; }
	static constexpr A expect_T(){ return A{ C{0,20} }; }
};

template<>
struct PermuteArrays<2> {
	using C = MGComplex<float>;
	using A = std::array<C,2>;

	// N=2 T-permute is nontrivial
	static constexpr A expect_X() { return A{ C{0,20}, C{1,21} }; }
	static constexpr A expect_Y() { return A{ C{0,20}, C{1,21} }; }
	static constexpr A expect_Z() { return A{ C{0,20}, C{1,21} }; }
	static constexpr A expect_T() { return A{ C{1,21}, C{0,20} }; }
};

template<>
struct PermuteArrays<4> {
	using C = MGComplex<float>;
	using A = std::array<C,4>;

	// N=4, Z and T permutes are nontrivial
	static constexpr A expect_X(){ return A{ C{0,20}, C{1,21}, C{2,22}, C{3,23} }; }
	static constexpr A expect_Y(){ return A{ C{0,20}, C{1,21}, C{2,22}, C{3,23} }; }
	static constexpr A expect_Z(){ return A{ C{1,21}, C{0,20}, C{3,23}, C{2,22} }; }
	static constexpr A expect_T(){ return A{ C{2,22}, C{3,23}, C{0,20}, C{1,21} }; }
};



template<>
struct PermuteArrays<8> {
	using C = MGComplex<float>;
	using A = std::array<C,8>;

	// N=8, Y, Z and T permutes are nontrivial
	static constexpr A expect_X(){ return A{ C{0,20}, C{1,21}, C{2,22}, C{3,23},
		C{4,24}, C{5,25}, C{6,26}, C{7,27}}; }
	static constexpr A expect_Y(){ return A{ C{1,21}, C{0,20}, C{3,23}, C{2,22},
		C{5,25}, C{4,24}, C{7,27}, C{6,26}}; }
	static constexpr A expect_Z(){ return A{ C{2,22}, C{3,23}, C{0,20}, C{1,21},
		C{6,26}, C{7,27}, C{4,24}, C{5,25}}; }
	static constexpr A expect_T(){ return A{ C{4,24}, C{5,25}, C{6,26}, C{7,27},
		C{0,20}, C{1,21}, C{2,22}, C{3,23}}; }


};
template<>
struct PermuteArrays<16> {
	using C = MGComplex<float>;
	using A = std::array<C,16>;

	// T=16 all permutes are nontrivial
	static constexpr A expect_X(){ return A{
		C{1,21},    C{0,20}, C{3,23}, C{2,22},
		C{5,25},	C{4,24}, C{7,27}, C{6,26},
		C{9,29},	C{8,28}, C{11,31}, C{10,30},
		C{13,33},	C{12,32}, C{15,35}, C{14,34}
	}; }

	static constexpr A expect_Y(){ return A{
		C{2,22}, C{3,23}, C{0,20}, C{1,21},
		C{6,26}, C{7,27}, C{4,24}, C{5,25},
		C{10,30}, C{11,31},	C{8,28}, C{9,29},
		C{12,32}, C{13,33},	C{14,34}, C{15,35},
	}; }
	static constexpr A expect_Z(){ return A{
			C{4,24}, C{5,25}, C{6,26}, C{7,27},
			C{0,20}, C{1,21}, C{2,22}, C{3,23},
			C{12,32}, C{13,33}, C{14,34}, C{15,35},
			C{8,28}, C{9,29}, C{10,30}, C{11,31},
	}; }
	static constexpr A expect_T(){ return A{
			C{8,28}, C{9,29}, C{10,30}, C{11,31},
			C{12,32}, C{13,33}, C{14,34}, C{15,35},
			C{0,20}, C{1,21}, C{2,22}, C{3,23},
			C{4,24}, C{5,25}, C{6,26}, C{7,27}
	}; }
};

#if 1
// And now for the test case
template<typename Q> class vec_load {};
TYPED_TEST(SyCLVNodeTest, CheckPerms)
{
	// Types, and values etc in the Fixture now need TestFixture:: qualification
	using T  = SIMDComplexSyCL<float,TestFixture::N>;
	using VN = VNode<float, TestFixture::N>;

	{
		auto h_f = (this->f_buf).template get_access<access::mode::write>();
		// This could be cleaned up with StoreLane counterparts to LoadLane below
#ifdef MG_TESTING_VECTYPE_A
		// Vectype a, real and imaginary parts are separated by VECLEN
		// numbers
		for(int i=0; i < TestFixture::N; ++i) {
			// Reals
			h_f[i] = static_cast<float>(i);

			// Imaginaries
			h_f[i+TestFixture::N] = static_cast<float>(20+i);
		}
#else
		for(int i=0; i < TestFixture::N; i++) {

			// Vectype B: real and imaginary numbers follow each other
			// like in Fortran

			// Reals
			h_f[2*i] = static_cast<float>(i);

			// Imaginaries
			h_f[2*i+1] = static_cast<float>(20+i);
		}
#endif
	}

	// All Vec load/stores need multi-ptr
	// Which are only in kernel scope.
	{
		this->MyQueue.submit([&](handler& cgh) {
			auto vecbuf = (this->f_buf).template get_access<access::mode::read_write>(cgh);
			cgh.single_task<vec_load<TypeParam>>([=](){

				// Load up the original
				T fc;
				Load(fc,0,vecbuf.get_pointer());

				// Permute it each way
				auto tmp = VN::permuteX(fc);
				Store(1,vecbuf.get_pointer(),tmp);
				tmp = VN::permuteY(fc);
				Store(2,vecbuf.get_pointer(),tmp);
				tmp = VN::permuteZ(fc);
				Store(3,vecbuf.get_pointer(),tmp);
				tmp = VN::permuteT(fc);
				Store(4,vecbuf.get_pointer(),tmp);
			});
		});
	}

	// On the host load up the N-results into an array of non-vectorized
	// complexes
	auto h_f = (this->f_buf).template get_access<access::mode::read>();
	std::array<MGComplex<float>, TestFixture::N> orig,permX,permY,permZ,permT;
	// For each 'Lane'

	for(size_t i=0; i < TestFixture::N ; ++i) {
		// Load them up
		orig[i]=LoadLane<float,TestFixture::N>(i,0,h_f);
		permX[i]=LoadLane<float,TestFixture::N>(i,1,h_f);
		permY[i]=LoadLane<float,TestFixture::N>(i,2,h_f);
		permZ[i]=LoadLane<float,TestFixture::N>(i,3,h_f);
		permT[i]=LoadLane<float,TestFixture::N>(i,4,h_f);
	}

	// Check against the permute array.
	for(int i=0; i < TestFixture::N; ++i) {
		ASSERT_FLOAT_EQ( permX[i].real(), PermuteArrays<TestFixture::N>::expect_X()[i].real());
		ASSERT_FLOAT_EQ( permX[i].imag(), PermuteArrays<TestFixture::N>::expect_X()[i].imag());

		ASSERT_FLOAT_EQ( permY[i].real(), PermuteArrays<TestFixture::N>::expect_Y()[i].real());
		ASSERT_FLOAT_EQ( permY[i].imag(), PermuteArrays<TestFixture::N>::expect_Y()[i].imag());

		ASSERT_FLOAT_EQ( permZ[i].real(), PermuteArrays<TestFixture::N>::expect_Z()[i].real());
		ASSERT_FLOAT_EQ( permZ[i].imag(), PermuteArrays<TestFixture::N>::expect_Z()[i].imag());

		ASSERT_FLOAT_EQ( permT[i].real(), PermuteArrays<TestFixture::N>::expect_T()[i].real());
		ASSERT_FLOAT_EQ( permT[i].imag(), PermuteArrays<TestFixture::N>::expect_T()[i].imag());
	}

}
#endif
