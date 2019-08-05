/*
 * test_vspinproj_sycl.cpp
 *
 *  Created on: Aug 1, 2019
 *      Author: bjoo
 */

#include "sycl_dslash_config.h"
#include "gtest/gtest.h"
#include "test_env.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "utils/print_utils.h"

#include "sycl_dslash_config.h"   // Build options
#include "dslash/dslash_defaults.h" // Default layouts
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"     // Vector type s
#include "dslash/sycl_qdp_vutils.h" // Utils
#include "dslash/dslash_vectype_sycl.h"

#include "sycl_lattice_spin_proj_utils.h"
using namespace MG;
using namespace MGTesting;
using namespace QDP;

template<typename T>
class TestVSpinProj :  public ::testing::Test{};

#ifdef MG_FORTRANLIKE_COMPLEX
#if 0
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8> >;
#else
using test_types = ::testing::Types<
                std::integral_constant<int,1> >;
#endif

#else

#if 0
using test_types = ::testing::Types<
		std::integral_constant<int,1>,
		std::integral_constant<int,2>,
		std::integral_constant<int,4>,
		std::integral_constant<int,8>,
		std::integral_constant<int,16>	>;
#else
using test_types = ::testing::Types<
                std::integral_constant<int,1> >;
#endif
#endif

TYPED_TEST_CASE(TestVSpinProj, test_types);

TYPED_TEST(TestVSpinProj,TestVSpinor)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());


	LatticeFermion qdp_in=zero;
	LatticeHalfFermion qdp_out=zero;
	LatticeHalfFermion sycl_out=zero;

	using VN = VNode<MGComplex<REAL>,VectorLength>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<REAL>,VN,4>;
	using HalfSpinorType = SyCLCBFineVSpinor<MGComplex<REAL>,VN,2>;

	gaussian(qdp_in);
	SpinorType sycl_in(info,EVEN);
	HalfSpinorType sycl_hspinor_out(hinfo,EVEN);

	QDPLatticeFermionToSyCLCBVSpinor(qdp_in, sycl_in);

	{
		// sign = -1 dir = 0
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,-1);
		qdp_out[rb[0]] = spinProjectDir0Minus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>,VN, 0,-1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		// sign = -1 dir = 1
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,-1);
		qdp_out[rb[0]] = spinProjectDir1Minus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>, VN, 1,-1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = -1 dir = 2
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,-1);
		qdp_out[rb[0]] = spinProjectDir2Minus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>, VN, 2,-1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = -1 dir = 3
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,-1);
		qdp_out[rb[0]] = spinProjectDir3Minus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>, VN, 3,-1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}


	{
		// sign = 1 dir = 0
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",0,1);
		qdp_out[rb[0]] = spinProjectDir0Plus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>,VN,0,1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
		// sign = 1 dir = 1
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",1,1);
		qdp_out[rb[0]] = spinProjectDir1Plus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>, VN, 1,1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = 1 dir = 2
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",2,1);
		qdp_out[rb[0]] = spinProjectDir2Plus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>, VN, 2, 1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	{
		// sign = 1 dir = 3
		MasterLog(INFO,"SpinProjectTest: dir=%d sign=%d",3,1);
		qdp_out[rb[0]] = spinProjectDir3Plus(qdp_in);
		qdp_out[rb[1]] = zero;
		sycl_out = zero;

		SyCLVProjectLattice<MGComplex<float>, VN, 3,1>(sycl_in,sycl_hspinor_out);
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		qdp_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(qdp_out, rb[0])));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}


}

TYPED_TEST(TestVSpinProj, TestSpinRecons)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());

	LatticeHalfFermion qdp_in;
	LatticeFermion     qdp_out;
	LatticeFermion     sycl_out;

	using VN = VNode<MGComplex<REAL>,VectorLength>;
	using SpinorType = SyCLCBFineVSpinor<MGComplex<REAL>,VN,4>;
	using HalfSpinorType = SyCLCBFineVSpinor<MGComplex<REAL>,VN,2>;

	gaussian(qdp_in);

	HalfSpinorType sycl_hspinor_in(hinfo,EVEN);
	SpinorType sycl_spinor_out(info,EVEN);


	QDPLatticeHalfFermionToSyCLCBVSpinor2(qdp_in, sycl_hspinor_in);

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 0, -1);
	  qdp_out[rb[0]] = spinReconstructDir0Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,0,-1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 1, -1);
	  qdp_out[rb[0]] = spinReconstructDir1Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,1,-1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 2, -1);
	  qdp_out[rb[0]] = spinReconstructDir2Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,2,-1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 3, -1);
	  qdp_out[rb[0]] = spinReconstructDir3Minus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,3,-1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}


	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 0, +1);
	  qdp_out[rb[0]] = spinReconstructDir0Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,0,1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 1, +1);
	  qdp_out[rb[0]] = spinReconstructDir1Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,1,1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 2, +1);
	  qdp_out[rb[0]] = spinReconstructDir2Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,2,1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

	{
	  MasterLog(INFO, "Spin Recons Test: dir = %d sign = %d", 3, +1);
	  qdp_out[rb[0]] = spinReconstructDir3Plus(qdp_in);
	  qdp_out[rb[1]] = zero;

	  SyCLVReconsLattice<MGComplex<REAL>,VN,3,1>(sycl_hspinor_in,sycl_spinor_out);
	  SyCLCBVSpinorToQDPLatticeFermion(sycl_spinor_out,sycl_out);

	  qdp_out[rb[0]] -= sycl_out;

	  double norm_diff = toDouble(sqrt(norm2(qdp_out)));
	  MasterLog(INFO, "norm_diff = %lf", norm_diff);
	  ASSERT_LT( norm_diff, 1.0e-5);
	}

}

TYPED_TEST(TestVSpinProj, TestMatrixMV)
{
	static constexpr int VectorLength = TypeParam::value;
	using VN = VNode<MGComplex<REAL>,VectorLength>;
	using GT = MGComplex<REAL>;
	using ST = MGComplex<REAL>;
	using HalfSpinorType = SyCLCBFineVSpinor<ST,VN,2>;
	using GaugeFieldType = SyCLCBFineVGaugeField<GT,VN>;

	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	multi1d<LatticeColorMatrix> gauge_in(n_dim);
	for(int mu=0; mu < n_dim; ++mu) {
		gaussian(gauge_in[mu]);
	}

	LatticeHalfFermion psi_in;
	gaussian(psi_in);

	LatticeInfo info(latdims,4,3,NodeInfo());
	LatticeInfo hinfo(latdims,2,3,NodeInfo());


	HalfSpinorType sycl_hspinor_in(hinfo,EVEN);
	HalfSpinorType sycl_hspinor_out(hinfo,EVEN);
	GaugeFieldType  sycl_gauge_e(info, EVEN);


	// Import Gauge Field
	QDPGaugeFieldToSyCLCBVGaugeField(gauge_in, sycl_gauge_e);

	// Import input halfspinor
	QDPLatticeHalfFermionToSyCLCBVSpinor2(psi_in, sycl_hspinor_in);

	LatticeHalfFermion psi_out, sycl_out;
	MasterLog(INFO, "Testing MV");
	{
		psi_out[rb[0]] = gauge_in[0]*psi_in;
		psi_out[rb[1]] = zero;

		SyCLMVLattice<GT,VN,ST>(sycl_gauge_e, sycl_hspinor_in, 0, sycl_hspinor_out);

		// Export result HalfFermion
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		psi_out[rb[0]] -= sycl_out;
		double norm_diff = toDouble(sqrt(norm2(psi_out)));

		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}

	// ********* TEST ADJOINT ****************
	{

		psi_out[rb[0]] = adj(gauge_in[0])*psi_in;
		psi_out[rb[1]] = zero;

		SyCLHVLattice<GT,VN,ST>(sycl_gauge_e, sycl_hspinor_in, 0, sycl_hspinor_out);

		// Export result HalfFermion
		SyCLCBVSpinor2ToQDPLatticeHalfFermion(sycl_hspinor_out,sycl_out);
		psi_out[rb[0]] -= sycl_out;

		double norm_diff = toDouble(sqrt(norm2(psi_out)));
		MasterLog(INFO, "norm_diff = %lf", norm_diff);
		ASSERT_LT( norm_diff, 1.0e-5);

	}
}
