/*
 * test_qdp_utils.cpp
 *
 *  Created on: Jul 31, 2019
 *      Author: bjoo
 */

#include "sycl_dslash_config.h"
#include "gtest/gtest.h"
#include "test_env.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"

#include "sycl_dslash_config.h"   // Build options
#include "dslash/dslash_defaults.h" // Default layouts
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vtypes.h"     // Vector type s
#include "dslash/sycl_qdp_vutils.h" // Utils


using namespace MG;
using namespace MGTesting;
using namespace QDP;

template<typename T>
class TestVNode :  public ::testing::Test{};

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

TYPED_TEST_CASE(TestVNode, test_types);

TYPED_TEST(TestVNode,TestVSpinor)
{
	static constexpr int VectorLength = TypeParam::value;
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;
	LatticeInfo info(latdims,4,3,NodeInfo());

	using VN = VNode<float,VectorLength>;
	using SpinorType = CBFineVSpinor<MGComplex<float>,VN,4>;

	// 4 spins
	SpinorType vnode_spinor(info, MG::EVEN);

	const LatticeInfo& c_info = vnode_spinor.GetInfo();
	IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
	for(int mu=0; mu < 4;++mu) {
		ASSERT_EQ( c_info.GetLatticeDimensions()[mu],
				info.GetLatticeDimensions()[mu]/VNDims[mu] );
	}

	bool same_global_vectype = std::is_same< typename SpinorType::VecType, SIMDComplexSyCL<float,VectorLength> >::value;
	ASSERT_EQ( same_global_vectype, true);
}

