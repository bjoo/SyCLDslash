/*
 * test_vectype_tests.h
 *
 *  Created on: Jul 1, 2019
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"

#include "dslash/dslash_vectype_sycl.h"
#include <CL/sycl.hpp>
using namespace MG;
using namespace cl::sycl;


class SyCLVecTypeTest : public ::testing::Test {
public:
	static constexpr size_t num_float_elem() { return 1024; }
	static constexpr size_t num_cmpx_elem() { return num_float_elem()/2; }
	static constexpr size_t N=4;

	queue MyQueue;
	buffer<float,1> f_buf;
	SyCLVecTypeTest() : f_buf{range<1>{num_float_elem()}}  {}

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



TEST(SyCLVectorType, Construct)
{
	SIMDComplexSyCL<float,1>  cf1;
	SIMDComplexSyCL<float,2>  cf2;
	SIMDComplexSyCL<float,4>  cf4;
	SIMDComplexSyCL<float,8>  cf8;
	SIMDComplexSyCL<float,16>  cf16;


	SIMDComplexSyCL<double,1>  cd1;
	SIMDComplexSyCL<double,2>  cd2;
	SIMDComplexSyCL<double,4>  cd4;
	SIMDComplexSyCL<double,8>  cd8;
	SIMDComplexSyCL<double,16>  cd16;
}


#ifndef MG_FORTRANLIKE_COMPLEX
TEST(SyCLVectorType, CheckLen)
{
	SIMDComplexSyCL<float,1>  cf1;
	ASSERT_EQ(len(cf1), 1);
	ASSERT_EQ(num_fp(cf1), 2);

	SIMDComplexSyCL<float,2>  cf2;
	ASSERT_EQ(len(cf2), 2);
	ASSERT_EQ(num_fp(cf2), 4);


	SIMDComplexSyCL<float,4>  cf4;
	ASSERT_EQ(len(cf4), 4);
	ASSERT_EQ(num_fp(cf4), 8);

	SIMDComplexSyCL<float,8>  cf8;
	ASSERT_EQ(len(cf8), 8);
	ASSERT_EQ(num_fp(cf8), 16);

	SIMDComplexSyCL<float,16>  cf16;
	ASSERT_EQ(len(cf16), 16);
	ASSERT_EQ(num_fp(cf16), 32);

	SIMDComplexSyCL<double,1>  cd1;
	ASSERT_EQ(len(cd1), 1);
	ASSERT_EQ(num_fp(cd1), 2);

	SIMDComplexSyCL<double,2>  cd2;
	ASSERT_EQ(len(cd2), 2);
	ASSERT_EQ(num_fp(cd2), 4);

	SIMDComplexSyCL<double,4>  cd4;
	ASSERT_EQ(len(cd4), 4);
	ASSERT_EQ(num_fp(cd4), 8);

	SIMDComplexSyCL<double,8>  cd8;
	ASSERT_EQ(len(cd8), 8);
	ASSERT_EQ(num_fp(cd8), 16);

	SIMDComplexSyCL<double,16>  cd16;
	ASSERT_EQ(len(cd16), 16);
	ASSERT_EQ(num_fp(cd16), 32);

}
#else
TEST(SyCLVectorType, CheckLen)
{
	SIMDComplexSyCL<float,1>  cf1;
	ASSERT_EQ(cf1.len(), 1);

	SIMDComplexSyCL<float,2>  cf2;
	ASSERT_EQ(cf2.len(), 2);

	SIMDComplexSyCL<float,4>  cf4;
	ASSERT_EQ(cf4.len(), 4);

	SIMDComplexSyCL<float,8>  cf8;
	ASSERT_EQ(cf8.len(), 8);

	SIMDComplexSyCL<double,1>  cd1;
	ASSERT_EQ(cd1.len(), 1);

	SIMDComplexSyCL<double,2>  cd2;
	ASSERT_EQ(cd2.len(), 2);

	SIMDComplexSyCL<double,4>  cd4;
	ASSERT_EQ(cd4.len(), 4);

	SIMDComplexSyCL<double,8>  cd8;
	ASSERT_EQ(cd8.len(), 8);
}
#endif

// Verify that the TestFixture sets up the f_buf and d_buf arrays
// correctly and that we can reinterpret it as arrays of Complexes
TEST_F(SyCLVecTypeTest, CorrectSetUp)
{


	auto host_access_f = f_buf.get_access<access::mode::read>();

	for(size_t vec=0; vec < num_cmpx_elem()/N; ++vec) {
		for(size_t i=0; i < N ; ++i) {
			size_t j=vec*N + i;  // j-th complex number

			MGComplex<float> f=LoadLane<float,N>(i,vec,host_access_f);
			ASSERT_FLOAT_EQ( f.real(), static_cast<float>(2*j) );
			ASSERT_FLOAT_EQ( f.imag(), static_cast<float>(2*j+1));
		}

	}


}

// Use Complex Zero to zero out a vector
// Write it to position 0
TEST_F(SyCLVecTypeTest, TestComplexLoad)
{
	// All Vec load/stores need multi-ptr
	// Which are only in kernel scope.
	{

		using T = SIMDComplexSyCL<float,N>;


		MyQueue.submit([&](handler& cgh) {
			auto vecbuf = f_buf.get_access<access::mode::read_write>(cgh);


			cgh.single_task<class vec_test_load>([=](){
				// Read the first vector. (We know what this is)
				T fc;
				Load(fc,0,vecbuf.get_pointer());

				// Store it into the second vector
				Store(1,vecbuf.get_pointer(),fc);
			});
		});
	}
	auto h_f = f_buf.get_access<access::mode::read>();

	// For each 'Lane'
	for(size_t i=0; i < N ; ++i) {


		// This is what we expect the values are for the lane
		// for the first vector
		float expect_real = 2*i;
		float expect_imag = 2*i+1;

		// Load them up
		MGComplex<float> orig=LoadLane<float,N>(i,0,h_f);
		MGComplex<float> res=LoadLane<float,N>(i,1,h_f);

		// Check written is same as orig
		ASSERT_FLOAT_EQ( orig.real(), res.real() );
		ASSERT_FLOAT_EQ( orig.imag(), res.imag() );

		// Check it is also correct.
		ASSERT_FLOAT_EQ( res.real(), expect_real );
		ASSERT_FLOAT_EQ( res.imag(), expect_imag );
	}
}



// Use Complex Zero to zero out a vector
// Write it to position 0
TEST_F(SyCLVecTypeTest, TestComplexZero)
{
	// All Vec load/stores need multi-ptr
	// Which are only in kernel scope.
	{

		using T = SIMDComplexSyCL<float,N>;
		MyQueue.submit([&](handler& cgh) {
			auto vecbuf_write = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class vecstore_test_complex_zero>([=](){
				T fc;
				ComplexZero(fc);

				Store(0,vecbuf_write.get_pointer(),fc);
			});
		});
		auto host_access_f = f_buf.get_access<access::mode::read>();
		for(size_t i=0; i < N; ++i ) {
			MGComplex<float> read_back=LoadLane<float,N>(i,0,host_access_f);
			ASSERT_FLOAT_EQ( read_back.real(), 0);
			ASSERT_FLOAT_EQ( read_back.imag(), 0);
		}
	}

}


// Use Complex Zero to zero out a vector
// Write it to position 0
TEST_F(SyCLVecTypeTest, TestComplexCopy)
{
	// All Vec load/stores need multi-ptr
	// Which are only in kernel scope.
	{

		using T = SIMDComplexSyCL<float,N>;
		MyQueue.submit([&](handler& cgh) {
			auto vecbuf_r = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class vec_test_complex_copy>([=](){

				T fc; Load(fc,0,vecbuf_r.get_pointer());
				T fc2; ComplexCopy(fc2, fc);
				Store(1,vecbuf_r.get_pointer(), fc2);
			});
		});
	}

	auto f = f_buf.get_access<access::mode::read>();

	for(size_t i=0; i < N; ++i ) {

		MGComplex<float> orig=LoadLane<float,N>(i,0,f);
		MGComplex<float> copy=LoadLane<float,N>(i,1,f);
		ASSERT_FLOAT_EQ( orig.real(), copy.real());
		ASSERT_FLOAT_EQ( orig.imag(), copy.imag());
	}
}



// Use Complex Zero to zero out a vector
// Write it to position 0
TEST_F(SyCLVecTypeTest, TestComplexPeq)
{
	using T = SIMDComplexSyCL<float,N>;
	{
		MyQueue.submit([&](handler& cgh) {
			auto vecbuf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class vec_test_peq>([=](){
				T fc;
				Load(fc,0,vecbuf.get_pointer());

				T fc2;
				Load(fc2,1,vecbuf.get_pointer());

				ComplexPeq(fc2,fc);

				Store(2, vecbuf.get_pointer(),fc2);
			});
		});
	} // Stuff completes

	auto h_f = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N ; ++i ) {
		MGComplex<float> fc=LoadLane<float,N>(i,0,h_f);
		MGComplex<float> fc2=LoadLane<float,N>(i,1,h_f);
		MGComplex<float> fc3=LoadLane<float,N>(i,2,h_f);

		std::cout << "i=" << i << " fc = " << fc << " fc2 = " << fc2 << " fc3 = " << fc3 << std::endl;
		ComplexPeq(fc2,fc);
		ASSERT_FLOAT_EQ( fc2.real(), fc3.real());
		ASSERT_FLOAT_EQ( fc2.imag(), fc3.imag());
	}
}


TEST_F(SyCLVecTypeTest,TestAAddSignB)
{
	using T = SIMDComplexSyCL<float,N>;
	{

		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_sign_b>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				float sign = -1;
				A_add_sign_B(c,a,sign,b);
				Store(3,buf.get_pointer(),c);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);
		float sign=-1;
		A_add_sign_B(c, a, sign, b); // Do scalar version
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );
	}
}

TEST_F(SyCLVecTypeTest,TestAAddB)
{
	using T = SIMDComplexSyCL<float,N>;
	{

		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_b>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				A_add_B(c,a,b);
				Store(3,buf.get_pointer(),c);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);

		A_add_sign_B<float,1>(c, a, b); // Do scalar version
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );
	}
}

TEST_F(SyCLVecTypeTest,TestASubB)
{
	using T = SIMDComplexSyCL<float,N>;
	{

		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_sub_b>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				A_sub_B(c,a,b);
				Store(3,buf.get_pointer(),c);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);

		A_add_sign_B<float,-1>(c, a, b); // Do scalar version
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );
	}
}

TEST_F(SyCLVecTypeTest,TestAPeqSignB)
{	using T = SIMDComplexSyCL<float,N>;
{
	MyQueue.submit([&](handler& cgh) {
		auto buf = f_buf.get_access<access::mode::read_write>(cgh);

		// Single task to zero a vector and store it, to start of buffer.
		cgh.single_task<class test_a_add_sign_b2>([=](){
			T a; Load(a,0,buf.get_pointer());
			T b; Load(b,1,buf.get_pointer());
			float sign = -1;
			A_peq_sign_B(b,sign,a);
			Store(2,buf.get_pointer(),b);

		});
	});
}

auto buf = f_buf.get_access<access::mode::read>();
for(size_t i=0; i < N; ++i ) {
	MGComplex<float> a=LoadLane<float,N>(i,0,buf);
	MGComplex<float> b=LoadLane<float,N>(i,1,buf);
	MGComplex<float> c=LoadLane<float,N>(i,2,buf);
	float sign=-1;
	A_peq_sign_B(b, sign, a);
	ASSERT_FLOAT_EQ( c.real(), b.real() );
	ASSERT_FLOAT_EQ( c.imag(), b.imag() );

}
}

TEST_F(SyCLVecTypeTest,TestAPeqB)
{	using T = SIMDComplexSyCL<float,N>;
{
	MyQueue.submit([&](handler& cgh) {
		auto buf = f_buf.get_access<access::mode::read_write>(cgh);

		// Single task to zero a vector and store it, to start of buffer.
		cgh.single_task<class test_a_peq_b2>([=](){
			T a; Load(a,0,buf.get_pointer());
			T b; Load(b,1,buf.get_pointer());
			A_peq_B(b,a);
			Store(2,buf.get_pointer(),b);

		});
	});
}

auto buf = f_buf.get_access<access::mode::read>();
for(size_t i=0; i < N; ++i ) {
	MGComplex<float> a=LoadLane<float,N>(i,0,buf);
	MGComplex<float> b=LoadLane<float,N>(i,1,buf);
	MGComplex<float> c=LoadLane<float,N>(i,2,buf);
	float sign=+1;
	A_peq_sign_B(b, sign, a);
	ASSERT_FLOAT_EQ( c.real(), b.real() );
	ASSERT_FLOAT_EQ( c.imag(), b.imag() );

}
}

TEST_F(SyCLVecTypeTest,TestAMeqB)
{	using T = SIMDComplexSyCL<float,N>;
{
	MyQueue.submit([&](handler& cgh) {
		auto buf = f_buf.get_access<access::mode::read_write>(cgh);

		// Single task to zero a vector and store it, to start of buffer.
		cgh.single_task<class test_a_meq_b2>([=](){
			T a; Load(a,0,buf.get_pointer());
			T b; Load(b,1,buf.get_pointer());
			A_meq_B(b,a);
			Store(2,buf.get_pointer(),b);

		});
	});
}

auto buf = f_buf.get_access<access::mode::read>();
for(size_t i=0; i < N; ++i ) {
	MGComplex<float> a=LoadLane<float,N>(i,0,buf);
	MGComplex<float> b=LoadLane<float,N>(i,1,buf);
	MGComplex<float> c=LoadLane<float,N>(i,2,buf);

	float sign=-1;
	A_peq_sign_B(b, sign, a);
	ASSERT_FLOAT_EQ( c.real(), b.real() );
	ASSERT_FLOAT_EQ( c.imag(), b.imag() );

}
}


TEST_F(SyCLVecTypeTest, ComplexCMaddSalar )
{
	MGComplex<float> a(-2.3,1.2);
	using T = SIMDComplexSyCL<float,N>;
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);
			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_cmadd_scalar>([=](){
				T b; Load(b,0,buf.get_pointer());
				T c; Load(c,1,buf.get_pointer());
				ComplexCMadd(c, a, b);
				Store(2,buf.get_pointer(),c);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> b=LoadLane<float,N>(i,0,buf);
		MGComplex<float> c=LoadLane<float,N>(i,1,buf);
		MGComplex<float> d=LoadLane<float,N>(i,2,buf);
		ComplexCMadd(c,a,b);
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest, ComplexConjMaddSalar )
{
	MGComplex<float> a(-2.3,1.2);
	using T = SIMDComplexSyCL<float,N>;
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);
			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_conjmadd_scalar>([=](){
				T b; Load(b,0,buf.get_pointer());
				T c; Load(c,1,buf.get_pointer());
				ComplexConjMadd(c, a, b);
				Store(2,buf.get_pointer(),c);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {

		MGComplex<float> b=LoadLane<float,N>(i,0,buf);
		MGComplex<float> c=LoadLane<float,N>(i,1,buf);
		MGComplex<float> d=LoadLane<float,N>(i,2,buf);
		ComplexConjMadd(c,a,b);
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest, ComplexCMadd)
{

	using T = SIMDComplexSyCL<float,N>;
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);
			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_cmadd>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				ComplexCMadd(c, a, b);
				Store(3,buf.get_pointer(),c);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);
		ComplexCMadd(c,a,b);
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest, ComplexConjMadd)
{

	using T = SIMDComplexSyCL<float,N>;
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);
			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_conjmadd>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				ComplexConjMadd(c, a, b);
				Store(3,buf.get_pointer(),c);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);
		ComplexConjMadd(c,a,b);
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}
TEST_F(SyCLVecTypeTest,TestAAddSigniB)
{
	using T = SIMDComplexSyCL<float,N>;
	{

		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_sign_ib>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				float sign = -1;
				A_add_sign_iB(c,a,sign,b);
				Store(3,buf.get_pointer(),c);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);
		float sign=-1;
		A_add_sign_iB(c, a, sign, b); // Do scalar version
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );
	}
}

TEST_F(SyCLVecTypeTest,TestAAddiB)
{
	using T = SIMDComplexSyCL<float,N>;
	{

		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_ib>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				A_add_iB(c,a,b);
				Store(3,buf.get_pointer(),c);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);
		A_add_sign_iB<float,1>(c, a, b); // Do scalar version
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );
	}
}

TEST_F(SyCLVecTypeTest,TestASubiB)
{
	using T = SIMDComplexSyCL<float,N>;
	{

		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_sub_ib>([=](){
				T a; Load(a,0,buf.get_pointer());
				T b; Load(b,1,buf.get_pointer());
				T c; Load(c,2,buf.get_pointer());
				A_sub_iB(c,a,b);
				Store(3,buf.get_pointer(),c);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < N; ++i ) {
		MGComplex<float> a=LoadLane<float,N>(i,0,buf);
		MGComplex<float> b=LoadLane<float,N>(i,1,buf);
		MGComplex<float> c=LoadLane<float,N>(i,2,buf);
		MGComplex<float> d=LoadLane<float,N>(i,3,buf);
		A_add_sign_iB<float,-1>(c, a, b); // Do scalar version
		ASSERT_FLOAT_EQ( c.real(), d.real() );
		ASSERT_FLOAT_EQ( c.imag(), d.imag() );
	}
}

TEST_F(SyCLVecTypeTest,TestAPeqSignMiB)
{	using T = SIMDComplexSyCL<float,N>;
{
	MyQueue.submit([&](handler& cgh) {
		auto buf = f_buf.get_access<access::mode::read_write>(cgh);

		// Single task to zero a vector and store it, to start of buffer.
		cgh.single_task<class test_a_add_sign_mib>([=](){
			T a; Load(a,0,buf.get_pointer());
			T b; Load(b,1,buf.get_pointer());
			float sign = -1;
			A_peq_sign_miB(b,sign,a);
			Store(2,buf.get_pointer(),b);

		});
	});
}

auto buf = f_buf.get_access<access::mode::read>();
for(size_t i=0; i < N; ++i ) {
	MGComplex<float> a=LoadLane<float,N>(i,0,buf);
	MGComplex<float> b=LoadLane<float,N>(i,1,buf);
	MGComplex<float> c=LoadLane<float,N>(i,2,buf);
	float sign=-1;
	A_peq_sign_miB(b, sign, a);
	ASSERT_FLOAT_EQ( c.real(), b.real() );
	ASSERT_FLOAT_EQ( c.imag(), b.imag() );

}
}

TEST_F(SyCLVecTypeTest,TestAPeqMiB)
{	using T = SIMDComplexSyCL<float,N>;
{
	MyQueue.submit([&](handler& cgh) {
		auto buf = f_buf.get_access<access::mode::read_write>(cgh);

		// Single task to zero a vector and store it, to start of buffer.
		cgh.single_task<class test_a_peq_mib>([=](){
			T a; Load(a,0,buf.get_pointer());
			T b; Load(b,1,buf.get_pointer());
			A_peq_miB(b,a);
			Store(2,buf.get_pointer(),b);

		});
	});
}

auto buf = f_buf.get_access<access::mode::read>();
for(size_t i=0; i < N; ++i ) {
	MGComplex<float> a=LoadLane<float,N>(i,0,buf);
	MGComplex<float> b=LoadLane<float,N>(i,1,buf);
	MGComplex<float> c=LoadLane<float,N>(i,2,buf);
	float sign=+1;
	A_peq_sign_miB(b, sign, a);
	ASSERT_FLOAT_EQ( c.real(), b.real() );
	ASSERT_FLOAT_EQ( c.imag(), b.imag() );

}
}

TEST_F(SyCLVecTypeTest,TestAMeqMiB)
{	using T = SIMDComplexSyCL<float,N>;
{
	MyQueue.submit([&](handler& cgh) {
		auto buf = f_buf.get_access<access::mode::read_write>(cgh);

		// Single task to zero a vector and store it, to start of buffer.
		cgh.single_task<class test_a_meq_mib>([=](){
			T a; Load(a,0,buf.get_pointer());
			T b; Load(b,1,buf.get_pointer());
			A_meq_miB(b,a);
			Store(2,buf.get_pointer(),b);

		});
	});
}

auto buf = f_buf.get_access<access::mode::read>();
for(size_t i=0; i < N; ++i ) {
	MGComplex<float> a=LoadLane<float,N>(i,0,buf);
	MGComplex<float> b=LoadLane<float,N>(i,1,buf);
	MGComplex<float> c=LoadLane<float,N>(i,2,buf);
	float sign=-1;
	A_peq_sign_miB(b, sign, a);
	ASSERT_FLOAT_EQ( c.real(), b.real() );
	ASSERT_FLOAT_EQ( c.imag(), b.imag() );

}
}


template<typename T>
class LaneOpsTester : public ::testing::Test{};

#ifdef MG_FORTRANLIKE_COMPLEX
using test_types = ::testing::Types<
    std::integral_constant<int,1>,
    std::integral_constant<int,2>,
	std::integral_constant<int,4>,
	std::integral_constant<int,8> >;
#else
using test_types = ::testing::Types<
    std::integral_constant<int,1>,
    std::integral_constant<int,2>,
	std::integral_constant<int,4>,
	std::integral_constant<int,8>,
	std::integral_constant<int,16>	>;

#endif
TYPED_TEST_CASE(LaneOpsTester, test_types);

TYPED_TEST(LaneOpsTester, TestLaneAccess)
{
	static constexpr int N = TypeParam::value;
	SIMDComplexSyCL<double,N> v;
	ComplexZero(v);
	std::array<MGComplex<double>,N> f;

	for(size_t i=0; i < N; ++i ) {
		f[i].real(i+1);
		f[i].imag(3*i + N);
		LaneOps<double,N>::insert(v,f[i],i);
	}

	for(size_t i=0; i < N; ++i ) {
		MGComplex<double> out( LaneOps<double,N>::extract(v,i) );
		ASSERT_FLOAT_EQ( out.real(), f[i].real());
		ASSERT_FLOAT_EQ( out.imag(), f[i].imag());
	}

}

