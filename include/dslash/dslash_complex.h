/*
 * dslash_policies.h
 *
 *  Created on: Jun 24, 2019
 *      Author: bjoo
 */

#pragma once
#include <complex>

namespace MG
{
template<typename T>
using MGComplex = std::complex<T>;

template<typename T>
struct BaseType {
};

template<typename T>
struct BaseType<MGComplex<T>> {
	using Type = T;
};


}
