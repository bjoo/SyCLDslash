#include <type_traits>
#include "gtest/gtest.h"
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
template<typename Q> class prefill {};

template<typename T>
class SyCLVNodeTest : public ::testing::Test {
public:
	static constexpr size_t num_float_elem() { return 1024; }
	static constexpr size_t num_cmpx_elem() { return num_float_elem()/2; }
	static constexpr typename T::value_type N = T::value;
	cpu_selector my_cpu;
	queue MyQueue;
	buffer<float,1> f_buf;
	buffer<float,1>& getBuf() { return f_buf; }
	SyCLVNodeTest() : f_buf{range<1>{num_float_elem()}}, MyQueue{my_cpu} {}
	//static constexpr typename T::value_type getN() { return N; }


protected:
	void SetUp() override
	{

		{

			std::cout << "Filling" << std::endl;
			range<1> N_vecs{num_cmpx_elem()/N};

			// Fill the buffers
			MyQueue.submit([&](handler& cgh) {
				auto write_fbuf = f_buf.get_access<access::mode::write>(cgh);

				cgh.parallel_for<prefill<T>>(N_vecs, [=](id<1> vec_id) {
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
 using test_type_params = ::testing::Types< std::integral_constant<int,1>,
		 	 	 	 	 	 	 	 	    std::integral_constant<int,2>,
											std::integral_constant<int,4>,
											std::integral_constant<int,8>,
											std::integral_constant<int,16> >;
#else

 using test_type_params = ::testing::Types< std::integral_constant<int,1>,
		 	 	 	 	 	 	 	 	    std::integral_constant<int,2>,
											std::integral_constant<int,4>,
											std::integral_constant<int,8> >;

#endif

TYPED_TEST_CASE(SyCLVNodeTest,test_type_params);

TYPED_TEST(SyCLVNodeTest, Compile)
{

	VNode<MGComplex<float>,TestFixture::N> vn1;
}

template<int N>
struct PermuteArrays;

template<>
struct PermuteArrays<1> {
	using C = MGComplex<float>;
	using A = std::array<C,1>;

	static constexpr A expect_X(){ return A{ C{0,20} }; }
	static constexpr A expect_Y(){ return A{ C{0,20} }; }
	static constexpr A expect_Z(){ return A{ C{0,20} }; }
	static constexpr A expect_T(){ return A{ C{0,20} }; }
};

template<>
struct PermuteArrays<2> {
	using C = MGComplex<float>;
	using A = std::array<C,2>;

	static constexpr A expect_X() { return A{ C{0,20}, C{1,21} }; }
	static constexpr A expect_Y() { return A{ C{0,20}, C{1,21} }; }
	static constexpr A expect_Z() { return A{ C{0,20}, C{1,21} }; }
	static constexpr A expect_T() { return A{ C{1,21}, C{0,20} }; }
};

template<>
struct PermuteArrays<4> {
	using C = MGComplex<float>;
	using A = std::array<C,4>;

	static constexpr A expect_X(){ return A{ C{0,20}, C{1,21}, C{2,22}, C{3,23} }; }
	static constexpr A expect_Y(){ return A{ C{0,20}, C{1,21}, C{2,22}, C{3,23} }; }
	static constexpr A expect_Z(){ return A{ C{1,21}, C{0,20}, C{3,23}, C{2,22} }; }
	static constexpr A expect_T(){ return A{ C{2,22}, C{3,23}, C{0,20}, C{1,21} }; }
};



template<>
struct PermuteArrays<8> {
	using C = MGComplex<float>;
	using A = std::array<C,8>;
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

template<typename Q> class vec_load {};
TYPED_TEST(SyCLVNodeTest, CheckPerms)
{
	using T  = SIMDComplexSyCL<float,TestFixture::N>;
//	using VT = typename VectorTraits<float,TestFixture::N,SIMDComplexSyCL>::VecType;
	using VN = VNode<typename VectorTraits<float,TestFixture::N,SIMDComplexSyCL>::BaseType, TestFixture::N>;

	{
		auto h_f = (this->f_buf).template get_access<access::mode::write>();
#ifdef MG_TESTING_VECTYPE_A
		for(int i=0; i < TestFixture::N; ++i) {
			// Reals
			h_f[i] = static_cast<float>(i);

			// Imaginaries
			h_f[i+TestFixture::N] = static_cast<float>(20+i);
		}
#else
		for(int i=0; i < TestFixture::N; i++) {
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

				T fc;
				Load(fc,0,vecbuf.get_pointer());

				auto tmp = VN::template permute<X_DIR>(fc);
				Store(1,vecbuf.get_pointer(),tmp);
				tmp = VN::template permute<Y_DIR>(fc);
				Store(2,vecbuf.get_pointer(),tmp);
				tmp = VN::template permute<Z_DIR>(fc);
				Store(3,vecbuf.get_pointer(),tmp);
				tmp = VN::template permute<T_DIR>(fc);
				Store(4,vecbuf.get_pointer(),tmp);
			});
		});
	}

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

