/*
 * test_view_sycl.cpp
 *
 *  Created on: Jul 29, 2019
 *      Author: bjoo
 */


#include <type_traits>
#include "gtest/gtest.h"
#include "dslash/dslash_complex.h"
#include "dslash/dslash_scalar_complex_ops.h"
#include "dslash/dslash_vectype_sycl.h"
#include "dslash/dslash_vnode.h"
#include "lattice/constants.h"


#include "dslash/sycl_view.h"

#include <CL/sycl.hpp>

using namespace MG;
using namespace cl::sycl;


TEST(TestView, TestViewCreate )
{
	View<float,1,LayoutLeft> a("a",{10});

	ASSERT_STREQ( a.getName().data(), "a");
	ASSERT_EQ( a.getNumDims(), 1);
	ASSERT_EQ( a.getDims()[0],10);

	View<float,2,LayoutLeft> b("b",{10,20});

	ASSERT_STREQ(b.getName().data(), "b");
	ASSERT_EQ( b.getNumDims(), 2);
	ASSERT_EQ( b.getDims()[0],10);
	ASSERT_EQ( b.getDims()[1],20);

	View<float,3,LayoutLeft> c("c",{10,20,30});

	ASSERT_STREQ( c.getName().data(), "c");
	ASSERT_EQ( c.getNumDims(), 3);
	ASSERT_EQ( c.getDims()[0],10);
	ASSERT_EQ( c.getDims()[1],20);
	ASSERT_EQ( c.getDims()[2],30);

	View<float,4,LayoutLeft> d("d",{10,20,30,40});

	ASSERT_STREQ( d.getName().data(), "d");
	ASSERT_EQ( d.getNumDims(), 4);
	ASSERT_EQ( d.getDims()[0],10);
	ASSERT_EQ( d.getDims()[1],20);
	ASSERT_EQ( d.getDims()[2],30);
	ASSERT_EQ( d.getDims()[3],40);

}

TEST( TestLayout, TestLayout1D)
{
	std::array<size_t,1> dims({3});

	for(size_t x=0; x < dims[0]; ++x ) {
		auto x_new = LayoutLeft::coords(LayoutLeft::index({x}, dims),dims);
		ASSERT_EQ(x_new[0],x);

		auto x_new2 = LayoutRight::coords(LayoutRight::index({x}, dims),dims);
		ASSERT_EQ(x_new2[0],x);
	}
}

TEST( TestLayout, TestLayout2D)
{
	std::array<size_t,2> dims({3,5});
	for(size_t y=0; y < dims[1]; ++y ) {
		for(size_t x=0; x < dims[0]; ++x ) {
			auto x_new = LayoutLeft::coords(LayoutLeft::index({x,y}, dims),dims);
			ASSERT_EQ(x_new[0],x);
			ASSERT_EQ(x_new[1],y);

			auto x_new2 = LayoutRight::coords(LayoutRight::index({x,y}, dims),dims);
			ASSERT_EQ(x_new2[0],x);
			ASSERT_EQ(x_new2[1],y);
		}
	}
}

TEST( TestLayout, TestLayout3D)
{
	std::array<size_t,3> dims({3,5,7});
	for(size_t z=0; z < dims[2]; ++z ) {
		for(size_t y=0; y < dims[1]; ++y ) {
			for(size_t x=0; x < dims[0]; ++x ) {
				auto x_new = LayoutLeft::coords(LayoutLeft::index({x,y,z}, dims),dims);
				ASSERT_EQ(x_new[0],x);
				ASSERT_EQ(x_new[1],y);
				ASSERT_EQ(x_new[2],z);

				auto x_new2 = LayoutRight::coords(LayoutRight::index({x,y,z}, dims),dims);
				ASSERT_EQ(x_new2[0],x);
				ASSERT_EQ(x_new2[1],y);
				ASSERT_EQ(x_new2[2],z);
			}
		}
	}
}

TEST( TestLayout, TestLayout4D)
{
	std::array<size_t,4> dims({3,5,7,9});
	for(size_t t=0; t < dims[3]; ++t ) {
		for(size_t z=0; z < dims[2]; ++z ) {
			for(size_t y=0; y < dims[1]; ++y ) {
				for(size_t x=0; x < dims[0]; ++x ) {
					auto x_new = LayoutLeft::coords(LayoutLeft::index({x,y,z,t}, dims),dims);
					ASSERT_EQ(x_new[0],x);
					ASSERT_EQ(x_new[1],y);
					ASSERT_EQ(x_new[2],z);
					ASSERT_EQ(x_new[3],t);

					auto x_new2 = LayoutRight::coords(LayoutRight::index({x,y,z,t}, dims),dims);
					ASSERT_EQ(x_new2[0],x);
					ASSERT_EQ(x_new2[1],y);
					ASSERT_EQ(x_new2[2],z);
					ASSERT_EQ(x_new2[3],t);

				}
			}
		}
	}
}

TEST(TestView, TestViewAccessors )
{
	View<size_t,4,LayoutLeft> w("view", {2,4,6,8});
	const auto dims = w.getDims();

	// Host accessor
	{
		auto viewAccess = w.get_access<cl::sycl::access::mode::write>();


		for(size_t t=0; t < dims[3]; ++t) {
			for(size_t z=0; z < dims[2]; ++z) {
				for(size_t y=0; y < dims[1]; ++y) {
					for(size_t x=0; x < dims[0]; ++x) {
						viewAccess(x,y,z,t) = LayoutLeft::index({x,y,z,t},dims);
					}
				}
			}
		}
	}

	queue MyQueue;
	{

		MyQueue.submit( [&](handler& cgh) {
			// Command Group view
			auto viewAccess = w.get_access<cl::sycl::access::mode::read_write>(cgh);

			cgh.parallel_for<class doubleIt>( cl::sycl::range<1>{BodySize::bodySize(dims)}, [=](id<1> vec_id) {
				std::array<size_t,4> c_vals = LayoutLeft::coords( vec_id[0], dims);

				// Read write into the view
				viewAccess( c_vals[0], c_vals[1], c_vals[2], c_vals[3] ) *= 2;
			});
		});

	} // wait for operations to complete


	{ // Host read

		auto viewAccess = w.get_access<cl::sycl::access::mode::read>();
		const auto dims = w.getDims();

		for(size_t t=0; t < dims[3]; ++t) {
			for(size_t z=0; z < dims[2]; ++z) {
				for(size_t y=0; y < dims[1]; ++y) {
					for(size_t x=0; x < dims[0]; ++x) {
						size_t i = viewAccess(x,y,z,t);
						ASSERT_EQ( i, 2*LayoutLeft::index({x,y,z,t},dims));
					}
				}
			}
		}
	}

}

