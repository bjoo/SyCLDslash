/*
 * test_subblock_launch.cpp
 *
 *  Created on: Oct 22, 2019
 *      Author: bjoo
 */

#include "sycl_dslash_config.h"
#include "gtest/gtest.h"
#include "test_env.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
using namespace MGTesting;

using namespace  std::chrono;
using namespace cl;

using test_types = ::testing::Types<
		std::integral_constant<int,8>,
		std::integral_constant<int,16>>;

TYPED_TEST_CASE(TimeVDslash, test_types);

static constexpr size_t n_sites = 100;


template<typename T>
class TimeVDslash :  public ::testing::Test {
public:
	TimeVDslash() : _q(nullptr){
		auto dlist = sycl::device::get_devices();
		devices.clear();
		devices.insert(devices.end(),dlist.begin(),dlist.end());

		int choice = TestEnv::getChosenDevice();
		if ( choice == -1 ) {
			_q.reset( new sycl::queue );
		}
		else {
			_q.reset( new sycl::queue( devices[choice]));
		}
	}

	sycl::queue& getQueue() const {
		return (*_q);
	}
private:
	std::vector<sycl::device> devices;
	std::unique_ptr<sycl::queue> _q;
};

template< int V>
class fred;

constexpr std::array<size_t,16> shuffle_mask={0,2,1,3,4,6,5,7,8,10,12,14,9,11,13,15};

[[cl::intel_reqd_sub_group_size(8)]]
void force_sub_group_size8(){}

[[cl::intel_reqd_sub_group_size(16)]]
void force_sub_group_size16(){}



template<typename T,const int V>
struct Functor {
	sycl::accessor<T,1,sycl::access::mode::read,sycl::access::target::global_buffer> x;
	sycl::accessor<T,1,sycl::access::mode::write,sycl::access::target::global_buffer> z;


	void  operator()(sycl::nd_item<1> nd_idx)  {

		// Set Subgroup Size
		if constexpr (V == 8)
            force_sub_group_size8();
		else if constexpr (V ==16)
            force_sub_group_size16();

		// Identify the offset of the work group.
		size_t g_offset = nd_idx.get_group(0)* nd_idx.get_local_range(0);

		// Now we need to traverse the work group in steps of subgroup.
		// Each 'thread' is already part of a sub-group so let's get its info

		sycl::intel::sub_group sg = nd_idx.get_sub_group();

		size_t simd_offset = g_offset + sg.get_group_id()[0]*sg.get_max_local_range()[0];


			T simd_x = sg.load(x.get_pointer() + simd_offset);
			T shuffled_x = sg.shuffle(simd_x,sycl::id<1>(shuffle_mask[sg.get_local_id()[0]%16]));
			sg.store( z.get_pointer() + simd_offset,shuffled_x);
			sg.barrier();
	}
};


TYPED_TEST(TimeVDslash, SubBlockLaunch)
{
	static constexpr int V = TypeParam::value;

	using T = float;

	sycl::buffer<T,1> xbuf(V*n_sites);
	sycl::buffer<T,1> zbuf(V*n_sites);
	sycl::buffer<T,1> z2buf(V*n_sites);

	{
		auto x = xbuf.template get_access<sycl::access::mode::read_write>();

		auto z = zbuf.template get_access<sycl::access::mode::write>();
		auto z2 = z2buf.template get_access<sycl::access::mode::write>();


		for(int i=0; i < n_sites; ++i) {

			for(int v=0; v < V; ++v) {


				sycl::id<1> idx(v+V*i);
				x[idx] = (v%16);
				z[idx] = 0;
				z2[idx]= shuffle_mask[v%16];
			}
		}
	}


	{


		auto q = this->getQueue();

		q.submit([&](sycl::handler& cgh){
			auto x = xbuf.template get_access<sycl::access::mode::read>(cgh);
			auto z = zbuf.template get_access<sycl::access::mode::write>(cgh);
			Functor<T,V> f{x,z};

			cgh.parallel_for<fred<V>>(sycl::nd_range<1>( sycl::range<1>(V*n_sites), sycl::range<1>(V)),f);
		});
		q.wait_and_throw();

	}

	auto z2 = z2buf.template get_access<sycl::access::mode::read>();
	auto z = zbuf.template get_access<sycl::access::mode::read>();
	for(int i=0; i < n_sites; ++i) {


		for(int v=0; v < V; ++v) {
			sycl::id<1> idx(v+i*V);
			printf("i=%d z=%lf\n",i,z[idx]);
			ASSERT_FLOAT_EQ( z[idx], z2[idx]);
		}
	}

}

