/*
 * dslash_policies.h
 *
 *  Created on: Jun 24, 2019
 *      Author: bjoo
 */

#pragma once
#include <complex>
#include <CL/sycl.hpp>

namespace MG
{
template<typename T>
using MGComplex = std::complex<T>;

template<typename T>
struct BaseType ;


template<typename T>
struct BaseType<MGComplex<T>> {
	using Type = T;
};

// Std C++ float types
template<>
struct BaseType<float> {
	using Type = float;
};

template<>
struct BaseType<double> {
	using Type = double;
};

template<>
struct BaseType<unsigned long> {
	using Type = unsigned long;
};

#if 0
template<>
struct BaseType<
	std::enable_if< ! std::is_same<float,cl::sycl::cl_float>::value,
						    cl::sycl::cl_float>::type_t > {
	using Type = cl::sycl::cl_float;
};

template<>
struct BaseType< std::enable_if< ! std::is_same<double,cl::sycl::cl_double>::value, cl::sycl::cl_double>::type_t > {
	using Type = cl::sycl::cl_double;
};
#endif


template<>
struct BaseType<cl::sycl::cl_half> {
	using Type = cl::sycl::cl_half;
};


}
