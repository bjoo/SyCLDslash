/*
 * test_vectype_sycl_b.cpp
 *
 *  Created on: Jun 28, 2019
 *      Author: bjoo
 */

#include "gtest/gtest.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_vectype_sycl_b.h"
#include "dslash/dslash_scalar_complex_ops.h"


#include <CL/sycl.hpp>
using namespace MG;
using namespace cl::sycl;

class SyCLVecTypeTest : public ::testing::Test {
public:
	static constexpr size_t num_float_elem() { return 1024; }
	static constexpr size_t num_cmpx_elem() { return num_float_elem()/2; }
	buffer<float,1> f_buf;
	buffer<double,1> d_buf;
	cpu_selector my_cpu;
	queue MyQueue;
	static constexpr int N=4;
	SyCLVecTypeTest() : f_buf{range<1>{num_float_elem()}}, d_buf{range<1>{num_float_elem()}}, MyQueue{my_cpu} {}

protected:
	void SetUp() override
	{

		{

			std::cout << "Filling" << std::endl;
			range<1> N_items{num_cmpx_elem()/N};
			std::cout << "SetUp with Veclen (N) =" << N << " Buffer size=" << num_cmpx_elem() / N  << " Complex vectors" << std::endl;
			// Fill the buffers
			MyQueue.submit([&](handler& cgh) {
				auto write_fbuf = f_buf.get_access<access::mode::write>(cgh);
				auto write_dbuf = d_buf.get_access<access::mode::write>(cgh);

				cgh.parallel_for<class prefill>(N_items, [=](id<1> vec_idx) {

					for(size_t i=0; i < N; ++i) {
						size_t j=vec_idx[0]*N + i;


						float f_r = static_cast<float>(2*j);
						float f_i = static_cast<float>(2*j+1);
						double d_r = static_cast<double>(2*j);
						double d_i = static_cast<double>(2*j+1);

						id<1> real_offset( N*2*vec_idx[0] +i );
						id<1> imag_offset( N*(2*vec_idx[0] + 1) + i);

						write_fbuf[real_offset]=f_r;
						write_fbuf[imag_offset]=f_i;

						write_dbuf[real_offset]=d_r;
						write_dbuf[imag_offset]=d_i;
					} // loop over i
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


// Verify that the TestFixture sets up the f_buf and d_buf arrays
// correctly and that we can reinterpret it as arrays of Complexes
TEST_F(SyCLVecTypeTest, CorrectSetUp)
{


	auto host_access_f = f_buf.get_access<access::mode::read>();
	auto host_access_d = d_buf.get_access<access::mode::read>();

	for(size_t vec=0; vec < num_cmpx_elem()/N; ++vec) {
		for(size_t i=0; i < N ; ++i) {
			size_t j=vec*N + i;  // j-th complex number

			id<1> r_idx( N*2*vec +i );
			id<1> i_idx( N*(2*vec + 1) + i);

			MGComplex<float> f( host_access_f[r_idx], host_access_f[i_idx]);
			MGComplex<double> d( host_access_d[r_idx], host_access_d[i_idx]);

			ASSERT_FLOAT_EQ( f.real(), static_cast<float>(2*j) );
			ASSERT_FLOAT_EQ( f.imag(), static_cast<float>(2*j+1));

			ASSERT_DOUBLE_EQ( d.real(), static_cast<double>(2*j) );
			ASSERT_DOUBLE_EQ( d.imag(), static_cast<double>(2*j+1));
		}

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
			MGComplex<float> read_back(host_access_f[i], host_access_f[N+i]);
			ASSERT_FLOAT_EQ( read_back.real(), 0);
			ASSERT_FLOAT_EQ( read_back.imag(), 0);
		}

		MGComplex<float> read_past(host_access_f[2*N], host_access_f[2*N+N]);
		ASSERT_FLOAT_EQ(read_past.real(),static_cast<float>(2*N));
		//ASSERT_FLOAT_EQ(read_past.imag(),static_cast<float>(N+1));
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
		std::cout << "i=" << i << " f[i]=" << f[i] << " f[i+N]=" << f[i+N]
					 << " f[2N+i]=" << f[2*N +i] << " f[3N+i]=" << f[3*N+i] <<std::endl;
	}
	for(size_t i=0; i < N; ++i ) {

			MGComplex<float> orig(f[i],f[i+N]);
			MGComplex<float> copy(f[2*N+i],f[3*N+i]);
			ASSERT_FLOAT_EQ( orig.real(), copy.real());
			ASSERT_FLOAT_EQ( orig.imag(), copy.imag());
	}


}

#if 0
// Use Complex Zero to zero out a vector
// Write it to position 0
TEST_F(SyCLVecTypeTest, TestComplexLoad)
{
	// All Vec load/stores need multi-ptr
	// Which are only in kernel scope.
	{

		using T = SIMDComplexSyCL<float,2>;
		MyQueue.submit([&](handler& cgh) {
			auto vecbuf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class vec_test_load>([=](){
				T fc;
				fc._data=vec<float,4>(1.1,1.2,1.3,1.4);
				Store(0,vecbuf.get_pointer(),fc);



				T fc2;
				//ComplexCopy(fc2,fc);
				Load(fc2,0,vecbuf.get_pointer());

				Store(1, vecbuf.get_pointer(),fc2);
			});
		});
		auto host_access_f = f_buf.reinterpret<MGComplex<float>,1>(num_cmpx_elem()).get_access<access::mode::read>();

		//  Fist store
		for(size_t i=0; i < T::len(); ++i ) {
			MGComplex<float> read_back(host_access_f[i]);
			ASSERT_FLOAT_EQ( read_back.real(), (1.0+0.1*(2*i+1)));
			ASSERT_FLOAT_EQ( read_back.imag(), (1.0+0.1*(2*i+2)));
		}

		// Store of the readback
		for(size_t i=0; i < T::len(); ++i ) {
			MGComplex<float> read_back(host_access_f[i+T::len()]);
			ASSERT_FLOAT_EQ( read_back.real(), (1.0+0.1*(2*i+1)));
			ASSERT_FLOAT_EQ( read_back.imag(), (1.0+0.1*(2*i+2)));
		}
		MGComplex<float> read_past(host_access_f[2*T::len()]);
		ASSERT_FLOAT_EQ(read_past.real(),static_cast<float>(4*T::len()));
		ASSERT_FLOAT_EQ(read_past.imag(),static_cast<float>(4*T::len()+1));

	}

}


// Use Complex Zero to zero out a vector
// Write it to position 0
TEST_F(SyCLVecTypeTest, TestComplexPeq)
{
	using T = SIMDComplexSyCL<float,2>;
	{
		MyQueue.submit([&](handler& cgh) {
			auto vecbuf = f_buf.get_access<access::mode::read_write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class vec_test_peq>([=](){
				T fc;
				fc._data=vec<float,4>(1,2,1,2);


				T fc2;
				fc2._data=vec<float,4>(3,4,3,4);

				ComplexPeq(fc2,fc);

				Store(0, vecbuf.get_pointer(),fc2);
			});
		});
	} // Stuff completss

	auto host_access_f = f_buf.reinterpret<MGComplex<float>,1>(num_cmpx_elem()).get_access<access::mode::read>();

	//  Fist store
	for(size_t i=0; i < T::len(); ++i ) {
		MGComplex<float> read_back(host_access_f[i]);
		ASSERT_FLOAT_EQ( read_back.real(), (4.0));
		ASSERT_FLOAT_EQ( read_back.imag(), (6.0));
	}

	MGComplex<float> read_past(host_access_f[T::len()]);
	ASSERT_FLOAT_EQ(read_past.real(),static_cast<float>(2*T::len()));
	ASSERT_FLOAT_EQ(read_past.imag(),static_cast<float>(2*T::len()+1));
}


TEST_F(SyCLVecTypeTest, TestSwizzleEO1)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class swizzletest1>([=](){
				vec<float,2> t(1,2);
				vec<float,2> t2 = SIMDComplexSyCL<float,1>::permute_evenodd(t);
				t.store(0,buf);
				t2.store(1,buf);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	ASSERT_FLOAT_EQ(buf[1],buf[2]);
	ASSERT_FLOAT_EQ(buf[0],buf[3]);
}

TEST_F(SyCLVecTypeTest, TestSwizzleEO2)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class swizzletest2>([=](){
				vec<float,4> t(1,2,3,4);
				vec<float,4> t2 = SIMDComplexSyCL<float,2>::permute_evenodd(t);
				t.store(0,buf);
				t2.store(1,buf);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	ASSERT_FLOAT_EQ(buf[1],buf[4]);
	ASSERT_FLOAT_EQ(buf[0],buf[5]);
	ASSERT_FLOAT_EQ(buf[3],buf[6]);
	ASSERT_FLOAT_EQ(buf[2],buf[7]);

}

TEST_F(SyCLVecTypeTest, TestSwizzleEO4)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class swizzletest4>([=](){
				vec<float,8> t(1,2,3,4,5,6,7,8);
				vec<float,8> t2 = SIMDComplexSyCL<float,4>::permute_evenodd(t);
				t.store(0,buf);
				t2.store(1,buf);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	ASSERT_FLOAT_EQ(buf[1],buf[8]);
	ASSERT_FLOAT_EQ(buf[0],buf[9]);
	ASSERT_FLOAT_EQ(buf[3],buf[10]);
	ASSERT_FLOAT_EQ(buf[2],buf[11]);
	ASSERT_FLOAT_EQ(buf[5],buf[12]);
	ASSERT_FLOAT_EQ(buf[4],buf[13]);
	ASSERT_FLOAT_EQ(buf[7],buf[14]);
	ASSERT_FLOAT_EQ(buf[6],buf[15]);

}

TEST_F(SyCLVecTypeTest, TestSwizzleEO8)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class swizzletest8>([=](){
				vec<float,16> t(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
				vec<float,16> t2 = SIMDComplexSyCL<float,8>::permute_evenodd(t);
				t.store(0,buf);
				t2.store(1,buf);
			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();

	ASSERT_FLOAT_EQ(buf[1],buf[16]);
	ASSERT_FLOAT_EQ(buf[0],buf[17]);
	ASSERT_FLOAT_EQ(buf[3],buf[18]);
	ASSERT_FLOAT_EQ(buf[2],buf[19]);
	ASSERT_FLOAT_EQ(buf[5],buf[20]);
	ASSERT_FLOAT_EQ(buf[4],buf[21]);
	ASSERT_FLOAT_EQ(buf[7],buf[22]);
	ASSERT_FLOAT_EQ(buf[6],buf[23]);
	ASSERT_FLOAT_EQ(buf[9],buf[24]);
	ASSERT_FLOAT_EQ(buf[8],buf[25]);
	ASSERT_FLOAT_EQ(buf[11],buf[26]);
	ASSERT_FLOAT_EQ(buf[10],buf[27]);
	ASSERT_FLOAT_EQ(buf[13],buf[28]);
	ASSERT_FLOAT_EQ(buf[12],buf[29]);
	ASSERT_FLOAT_EQ(buf[15],buf[30]);
	ASSERT_FLOAT_EQ(buf[14],buf[31]);

}
//SIMDComplex<double,4> v4a,v4b,v4c,v4d;
//	for(int l=0; l < v4a.len(); ++l) {
//		v4a.set(l, Kokkos::complex<double>(l,-l));
//		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
//		v4c.set(l, Kokkos::complex<double>(0.1*l,0.5*l));
//		v4d.set(l, v4c(l));
//	}
//	double sign = -1;
//
//	A_add_sign_B(v4c, v4a, sign, v4b);
//
//	for(int l=0; l < v4d.len(); ++l) {
//		Kokkos::complex<double> result = v4d(l);
//
//		A_add_sign_B( result, v4a(l), sign, v4b(l));
//		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
//		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
//	}
TEST_F(SyCLVecTypeTest,TestAAddSignB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_sign_b>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
				SIMDComplexSyCL<float,4> d;
				d._data = c._data;
				float sign = -1;

				A_add_sign_B(c,a,sign,b);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);
				d._data.store(3,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
		 MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);
		 float sign=-1;
		 A_add_sign_B(d, a, sign, b);
		 ASSERT_FLOAT_EQ( c.real(), d.real() );
		 ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAAddB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_b>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
				SIMDComplexSyCL<float,4> d;
				d._data = c._data;


				A_add_B(c,a,b);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);
				d._data.store(3,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
		 MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);
		 float sign=-1;
		 A_add_sign_B<float,+1>(d, a, b);
		 ASSERT_FLOAT_EQ( c.real(), d.real() );
		 ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}
TEST_F(SyCLVecTypeTest,TestASubB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_sub_b>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
				SIMDComplexSyCL<float,4> d;
				d._data = c._data;


				A_sub_B(c,a,b);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);
				d._data.store(3,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
		 MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);

		 A_add_sign_B<float,-1>(d, a, b);
		 ASSERT_FLOAT_EQ( c.real(), d.real() );
		 ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAPeqSignB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_sign_b2>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = b._data;
				float sign = -1;

				A_peq_sign_B(b,sign,a);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);

		 float sign=-1;
		 A_peq_sign_B(c, sign, a);
		 ASSERT_FLOAT_EQ( c.real(), b.real() );
		 ASSERT_FLOAT_EQ( c.imag(), b.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAPeqB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_peq_b>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = b._data;
				float sign = -1;

				A_peq_B(b,a);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);

		 float sign=-1;
		 A_peq_sign_B<float,1>(c,  a);
		 ASSERT_FLOAT_EQ( c.real(), b.real() );
		 ASSERT_FLOAT_EQ( c.imag(), b.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAMeqB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_meq_b2>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = b._data;


				A_meq_B(b,a);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);

		 float sign=-1;
		 A_peq_sign_B<float,-1>(c, a);
		 ASSERT_FLOAT_EQ( c.real(), b.real() );
		 ASSERT_FLOAT_EQ( c.imag(), b.imag() );

	}
}

TEST_F(SyCLVecTypeTest, ComplexCMaddSalar )
{
  MGComplex<float> a(-2.3,1.2);
  {
	  MyQueue.submit([&](handler& cgh) {
		  auto buf = f_buf.get_access<access::mode::write>(cgh);
		  // Single task to zero a vector and store it, to start of buffer.
		  cgh.single_task<class test_cmadd_scalar>([=](){
			  SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
			  SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
			  SIMDComplexSyCL<float,4> d; d._data = c._data;

			  ComplexCMadd(c, a, b);

			  b._data.store(1,buf);
			  c._data.store(2,buf);
			  d._data.store(3,buf);
		  });
	  });
  }

  auto buf = f_buf.get_access<access::mode::read>();
  for(size_t i=0; i < 4; ++i ) {
	  MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
	  MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
	  MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);

	  ComplexCMadd(d, a,b);
	  ASSERT_FLOAT_EQ( c.real(), d.real() );
	  ASSERT_FLOAT_EQ( c.imag(), d.imag() );

  	}

}

TEST_F(SyCLVecTypeTest, ComplexConjMaddSalar )
{
  MGComplex<float> a(-2.3,1.2);
  {
	  MyQueue.submit([&](handler& cgh) {
		  auto buf = f_buf.get_access<access::mode::write>(cgh);
		  // Single task to zero a vector and store it, to start of buffer.
		  cgh.single_task<class test_conjmadd_scalar>([=](){
			  SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
			  SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
			  SIMDComplexSyCL<float,4> d; d._data = c._data;

			  ComplexConjMadd(c, a, b);

			  b._data.store(1,buf);
			  c._data.store(2,buf);
			  d._data.store(3,buf);
		  });
	  });
  }

  auto buf = f_buf.get_access<access::mode::read>();
  for(size_t i=0; i < 4; ++i ) {
	  MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
	  MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
	  MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);

	  ComplexConjMadd(d, a,b);
	  ASSERT_FLOAT_EQ( c.real(), d.real() );
	  ASSERT_FLOAT_EQ( c.imag(), d.imag() );

  	}

}

TEST_F(SyCLVecTypeTest, ComplexCMadd )
{

  {
	  MyQueue.submit([&](handler& cgh) {
		  auto buf = f_buf.get_access<access::mode::write>(cgh);
		  // Single task to zero a vector and store it, to start of buffer.
		  cgh.single_task<class test_cmadd_vector>([=](){
			  SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
			  SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
			  SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
			  SIMDComplexSyCL<float,4> d; d._data = c._data;

			  ComplexCMadd(c, a, b);
			  a._data.store(0,buf);
			  b._data.store(1,buf);
			  c._data.store(2,buf);
			  d._data.store(3,buf);
		  });
	  });
  }

  auto buf = f_buf.get_access<access::mode::read>();
  for(size_t i=0; i < 4; ++i ) {
	  MGComplex<float> a(buf[2*i], buf[2*i+1]);
 	  MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
	  MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
	  MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);

	  ComplexCMadd(d, a,b);
	  ASSERT_FLOAT_EQ( c.real(), d.real() );
	  ASSERT_FLOAT_EQ( c.imag(), d.imag() );

  	}

}

TEST_F(SyCLVecTypeTest, ComplexConjMadd )
{
  {
	  MyQueue.submit([&](handler& cgh) {
		  auto buf = f_buf.get_access<access::mode::write>(cgh);
		  // Single task to zero a vector and store it, to start of buffer.
		  cgh.single_task<class test_conjmadd_vector>([=](){
			  SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
			  SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
			  SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
			  SIMDComplexSyCL<float,4> d; d._data = c._data;

			  ComplexConjMadd(c, a, b);
			  a._data.store(0,buf);
			  b._data.store(1,buf);
			  c._data.store(2,buf);
			  d._data.store(3,buf);
		  });
	  });
  }

  auto buf = f_buf.get_access<access::mode::read>();
  for(size_t i=0; i < 4; ++i ) {
	  MGComplex<float> a(buf[2*i], buf[2*i+1]);
	  MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
	  MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
	  MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);

	  ComplexConjMadd(d, a,b);
	  ASSERT_FLOAT_EQ( c.real(), d.real() );
	  ASSERT_FLOAT_EQ( c.imag(), d.imag() );

  	}

}

TEST_F(SyCLVecTypeTest,TestAAddSigniB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_sign_ib>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
				SIMDComplexSyCL<float,4> d;
				d._data = c._data;
				float sign = -1;

				A_add_sign_iB(c,a,sign,b);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);
				d._data.store(3,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
		 MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);
		 float sign=-1;
		 A_add_sign_iB(d, a, sign, b);
		 ASSERT_FLOAT_EQ( c.real(), d.real() );
		 ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAAddiB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_add_ib>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
				SIMDComplexSyCL<float,4> d;
				d._data = c._data;
				float sign = -1;

				A_add_iB(c,a,b);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);
				d._data.store(3,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
		 MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);
		 float sign=-1;
		 A_add_sign_iB<float,1>(d, a, b);
		 ASSERT_FLOAT_EQ( c.real(), d.real() );
		 ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestASubiB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_sub_ib>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = {0,0,0.1,0.5,0.1*2,0.5*2,0.1*3,0.5*3};
				SIMDComplexSyCL<float,4> d;
				d._data = c._data;
				float sign = -1;

				A_sub_iB(c,a,b);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);
				d._data.store(3,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);
		 MGComplex<float> d(buf[24+2*i],buf[24+2*i+1]);
		 float sign=-1;
		 A_add_sign_iB<float,-1>(d, a, b);
		 ASSERT_FLOAT_EQ( c.real(), d.real() );
		 ASSERT_FLOAT_EQ( c.imag(), d.imag() );

	}
}
TEST_F(SyCLVecTypeTest,TestAPeqSignMiB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_peq_sign_mib>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = b._data;
				float sign = -1;

				A_peq_sign_miB(b,sign,a);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);

		 float sign=-1;
		 A_peq_sign_miB(c, sign, a);
		 ASSERT_FLOAT_EQ( c.real(), b.real() );
		 ASSERT_FLOAT_EQ( c.imag(), b.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAPeqMiB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_peq_mib>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = b._data;
				float sign = -1;

				A_peq_miB(b,a);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);

		 float sign=-1;
		 A_peq_sign_miB<float,1>(c,  a);
		 ASSERT_FLOAT_EQ( c.real(), b.real() );
		 ASSERT_FLOAT_EQ( c.imag(), b.imag() );

	}
}

TEST_F(SyCLVecTypeTest,TestAMeqMiB)
{
	{
		MyQueue.submit([&](handler& cgh) {
			auto buf = f_buf.get_access<access::mode::write>(cgh);

			// Single task to zero a vector and store it, to start of buffer.
			cgh.single_task<class test_a_meq_mib>([=](){
				SIMDComplexSyCL<float,4> a; a._data = {0,0,1,-1,2,-2,3,-3};
				SIMDComplexSyCL<float,4> b; b._data = {0,0,1.4,-0.3,1.4*2,-0.3*2,1.4*3,-0.3*3};
				SIMDComplexSyCL<float,4> c; c._data = b._data;


				A_meq_miB(b,a);
				a._data.store(0,buf);
				b._data.store(1,buf);
				c._data.store(2,buf);

			});
		});
	}

	auto buf = f_buf.get_access<access::mode::read>();
	for(size_t i=0; i < 4; ++i ) {
		 MGComplex<float> a(buf[2*i],buf[2*i+1]);
		 MGComplex<float> b(buf[8+2*i],buf[8+2*i+1]);
		 MGComplex<float> c(buf[16+2*i],buf[16+2*i+1]);

		 float sign=-1;
		 A_peq_sign_miB<float,-1>(c, a);
		 ASSERT_FLOAT_EQ( c.real(), b.real() );
		 ASSERT_FLOAT_EQ( c.imag(), b.imag() );

	}
}
#endif




