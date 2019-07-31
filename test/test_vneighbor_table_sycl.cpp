/*
 * test_vneighbor_table.cpp
 *
 *  Created on: Jul 30, 2019
 *      Author: bjoo
 */


#include <type_traits>
#include "gtest/gtest.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_vnode.h"
#include "dslash/sycl_vneighbor_table.h"
#include "lattice/constants.h"


#include "dslash/sycl_view.h"

#include <CL/sycl.hpp>

using namespace MG;
using namespace cl::sycl;

TEST(TestVNeihborTable, Instantiate)
{
	IndexArray cbdims({2,4,6,8});
	SiteTable tab(2,4,6,8);

	IndexType n_idx; bool doit;


	printf("Host\n");
	{
		SiteTableAccess t(tab);


		t.NeighborTMinus(0,0,0,0, n_idx, doit);
		IndexArray coords = LayoutLeft::coords(n_idx,cbdims);

		// Check permute is true
		ASSERT_TRUE( doit );

		// Check wrap in t
		ASSERT_EQ( coords[0],0 ); ASSERT_EQ(coords[1],0); ASSERT_EQ(coords[2],0); ASSERT_EQ(coords[3],7);

		// SiteTableAccess t should disappear here.
	}
	printf("Host done\n");


	printf("In the queue\n");
	cl::sycl::buffer<IndexType,1> nidx_buf(range<1>(1));
	cl::sycl::buffer<bool,1> doit_buf(range<1>(1));

	cl::sycl::queue q;
	{
		q.submit([&](cl::sycl::handler& cgh) {

		// FIXME: This gives me an error when MG_USE_NEIGHBOR TABLE IS ENABLED
		SiteTableAccess t2(tab);


		auto nidx_save = nidx_buf.get_access<cl::sycl::access::mode::write>(cgh);
		auto doit_save = doit_buf.get_access<cl::sycl::access::mode::write>(cgh);

		cgh.single_task<class lookup>([=]() {

			IndexType n_idx2=0;
			bool doit2=true;

			t2.NeighborTMinus(0,0,0,1, n_idx2, doit2);

			nidx_save[0] = n_idx2;
			doit_save[0] = doit2;


		}); // single task

	}); // queue.submit
	}
	printf("queue done\n");


	printf("Reading on host\n");
	auto nidx_out = nidx_buf.template get_access<cl::sycl::access::mode::read>();
	auto doit_out = doit_buf.template get_access<cl::sycl::access::mode::read>();

	ASSERT_FALSE( doit_out[0] );

	IndexArray coords2 = LayoutLeft::coords(nidx_out[0], cbdims);
	ASSERT_EQ( coords2[0],0 ); ASSERT_EQ(coords2[1],0); ASSERT_EQ(coords2[2],0); ASSERT_EQ(coords2[3],0);
}

// Need more tests here I think.
