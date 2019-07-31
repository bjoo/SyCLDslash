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

#define MG_USE_NEIGHBOR_TABLE
#include "dslash/sycl_vneighbor_table_broken.h"
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
		SiteTableAccess<SiteTable::host_accessor> t(tab.get_access());

		t.NeighborTMinus(LayoutLeft::index({0,0,0,0},cbdims),1,n_idx,doit);

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

	cl::sycl::cpu_selector cpu_sel;
	cl::sycl::queue MyQueue(cpu_sel);
	{
	MyQueue.submit([&](cl::sycl::handler& cgh) {

		// FIXME: This gives me an error when MG_USE_NEIGHBOR TABLE IS ENABLED
		SiteTableAccess<SiteTable::global_accessor> t2(tab.get_access(cgh));

		auto nidx_save = nidx_buf.get_access<cl::sycl::access::mode::write>(cgh);
		auto doit_save = doit_buf.get_access<cl::sycl::access::mode::write>(cgh);

		cgh.single_task<class lookup>([=]() {

			IndexType n_idx2=0;
			bool doit2=true;

			t2.NeighborTMinus(LayoutLeft::index({0,0,0,1},cbdims),0,n_idx2,doit2);

			nidx_save[0] = n_idx2;
			doit_save[0] = doit2;


		}); // single task

	}); // queue.submit
	} // queue finishes
	printf("queue done\n");


	printf("Reading on host\n");
	auto nidx_out = nidx_buf.template get_access<cl::sycl::access::mode::read>();
	auto doit_out = doit_buf.template get_access<cl::sycl::access::mode::read>();

	ASSERT_FALSE( doit_out[0] );

	IndexArray coords2 = LayoutLeft::coords(nidx_out[0], cbdims);
	ASSERT_EQ( coords2[0],0 ); ASSERT_EQ(coords2[1],0); ASSERT_EQ(coords2[2],0); ASSERT_EQ(coords2[3],0);
}
