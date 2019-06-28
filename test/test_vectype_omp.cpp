/*
 * test_vectype_omp.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: bjoo
 */
#include "gtest/gtest.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_omp.h"
#include "dslash/dslash_vector_complex_ops_omp.h"
using namespace MG;

TEST(TestVectype, VectypeCreateD4)
{
        SIMDComplex<double,4> v2;
}

TEST(TestVectype, VectypeLenD4)
{
        SIMDComplex<double,4> v4;
        ASSERT_EQ( v4.len(), 4);
}

TEST(TestVectype, TestLaneAccessorsD4)
{
        SIMDComplex<double,4> v4;
        for(int i=0; i < v4.len(); ++i)
                v4.set(i,std::complex<double>(i,-i));

        for(int i=0; i < v4.len(); ++i) {
                double re = v4(i).real();
                double im = v4(i).imag();
                ASSERT_DOUBLE_EQ( re, static_cast<double>(i) );
                ASSERT_DOUBLE_EQ( im, static_cast<double>(-i) );

        }

}

TEST(TestVectype, VectypeCopyD4)
{
	SIMDComplex<double,4> v4;
	for(int i=0; i < v4.len(); ++i)
		v4.set(i,std::complex<double>(i,-i));

	SIMDComplex<double,4> v4_copy;

	ComplexCopy(v4_copy,v4);

	for(int i=0; i < v4.len(); ++i) {
		ASSERT_DOUBLE_EQ(  v4_copy(i).real(), v4(i).real());
		ASSERT_DOUBLE_EQ(  v4_copy(i).imag(), v4(i).imag());
	}
}
TEST(TestVectype, VectypeLoadStore)
{

        SIMDComplex<double,4> v4;
        for(int i=0; i < v4.len(); ++i)
                v4.set(i,std::complex<double>(i,-i));

        SIMDComplex<double,4> v4_3;
        {
                SIMDComplex<double,4> v4_2;

                Store(v4_2, v4);
                Load(v4_3, v4_2);
        }
        for(int i=0; i < v4.len(); ++i) {
                ASSERT_DOUBLE_EQ(  v4(i).real(), v4_3(i).real());
                ASSERT_DOUBLE_EQ(  v4(i).imag(), v4_3(i).imag());
        }
 }

TEST(TestVectype, VectypeZeroD4)
{


        SIMDComplex<double,4> v4;
        ComplexZero(v4);
        for(int i=0; i < v4.len(); ++i) {
                ASSERT_DOUBLE_EQ(  v4(i).real(),0);
                ASSERT_DOUBLE_EQ(  v4(i).imag(),0);
        }

}


TEST(TestVectype, VectypeComplexPeqD4 )
{

        SIMDComplex<double,4> v4a,v4b,v4c;
        for(int i=0; i < v4a.len(); ++i) {
                v4a.set(i,std::complex<double>(i,-i));
                v4b.set(i,std::complex<double>(1.4*i,-0.3*i));
                v4c.set(i, v4b(i));
        }

        ComplexPeq(v4b, v4a);

        for(int i=0; i < v4c.len(); ++i) {
                std::complex<double> result = v4c(i);
                ComplexPeq(result, v4a(i));
                ASSERT_DOUBLE_EQ( result.real(), v4b(i).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4b(i).imag());
        }
}

TEST(TestVectype, VectypeComplexCMaddSalarD4 )
{

        std::complex<double> a=std::complex<double>(-2.3,1.2);
        SIMDComplex<double,4> v4b,v4c,v4d;
        for(int l=0; l < v4b.len(); ++l) {
                v4b.set(l,std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l,std::complex<double>(0.1*l,0.5*l));
                v4d.set(l,v4c(l));
        }

        ComplexCMadd(v4c, a, v4b);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4d(l);
                ComplexCMadd( result, a, v4b(l));
                ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
        }


}

TEST(TestVectype, VectypeComplexConjMaddSalarD4 )
{
        std::complex<double> a=std::complex<double>(-2.3,1.2);
        SIMDComplex<double,4> v4b,v4c,v4d;
        for(int l=0; l < v4b.len(); ++l) {
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, std::complex<double>(0.1*l,0.5*l));
                v4d.set(l, v4c(l));
        }

        ComplexConjMadd(v4c, a, v4b);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4d(l);
                ComplexConjMadd( result, a, v4b(l));
                ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
        }

}

TEST(TestVectype, VectypeComplexCMaddD4 )
{
        SIMDComplex<double,4> v4a,v4b,v4c,v4d;
        for(int l=0; l < v4a.len(); ++l) {
                v4a.set(l, std::complex<double>(l,-l));
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, std::complex<double>(0.1*l,0.5*l));
                v4d.set(l, v4c(l));
        }

        ComplexCMadd(v4c, v4a, v4b);


        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4d(l);
                ComplexCMadd( result, v4a(l), v4b(l));
                ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
        }
}

TEST(TestVectype, VectypeComplexConjMaddD4 )
{
        SIMDComplex<double,4> v4a,v4b,v4c,v4d;
        for(int l=0; l < v4a.len(); ++l) {
                v4a.set(l, std::complex<double>(l,-l));
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, std::complex<double>(0.1*l,0.5*l));
                v4d.set(l, v4c(l));
        }

        ComplexConjMadd(v4c, v4a, v4b);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4d(l);
                ComplexConjMadd( result, v4a(l), v4b(l));
                ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
        }
}

TEST(TestVectype, Test_A_add_sign_B_D4 )
{
        SIMDComplex<double,4> v4a,v4b,v4c,v4d;
        for(int l=0; l < v4a.len(); ++l) {
                v4a.set(l, std::complex<double>(l,-l));
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, std::complex<double>(0.1*l,0.5*l));
                v4d.set(l, v4c(l));
        }
        double sign = -1;

        A_add_sign_B(v4c, v4a, sign, v4b);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4d(l);

                A_add_sign_B( result, v4a(l), sign, v4b(l));
                ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
        }
}

TEST(TestVectype, Test_A_add_sign_iB_D4 )
{

        SIMDComplex<double,4> v4a,v4b,v4c,v4d;
        for(int l=0; l < v4a.len(); ++l) {
                v4a.set(l, std::complex<double>(l,-l));
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, std::complex<double>(0.1*l,0.5*l));
                v4d.set(l, v4c(l));
        }
        double sign = -1.0;

        A_add_sign_iB(v4c, v4a, sign, v4b);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4d(l);
                A_add_sign_iB( result, v4a(l), sign, v4b(l));
                ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
        }


}

TEST(TestVectype, Test_A_peq_sign_miB_D4 )
{
        SIMDComplex<double,4> v4a,v4b,v4c,v4d;
        for(int l=0; l < v4a.len(); ++l) {
                v4a.set(l, std::complex<double>(l,-l));
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, v4b(l));
        }
        double sign = -1.0;

        A_peq_sign_miB(v4b, sign, v4a);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4c(l);
                A_peq_sign_miB( result, sign, v4a(l));
                ASSERT_DOUBLE_EQ( result.real(), v4b(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4b(l).imag());
        }
}

TEST(TestVectype, Test_A_peq_sign_B_D4 )
{
        SIMDComplex<double,4> v4a,v4b,v4c,v4d;
        for(int l=0; l < v4a.len(); ++l) {
                v4a.set(l, std::complex<double>(l,-l));
                v4b.set(l, std::complex<double>(1.4*l,-0.3*l));
                v4c.set(l, v4b(l));
        }
        double sign = -1.0;

        A_peq_sign_B(v4b, sign, v4a);

        for(int l=0; l < v4d.len(); ++l) {
                std::complex<double> result = v4c(l);

                A_peq_sign_B( result, sign, v4a(l));
                ASSERT_DOUBLE_EQ( result.real(), v4b(l).real());
                ASSERT_DOUBLE_EQ( result.imag(), v4b(l).imag());
        }
}




