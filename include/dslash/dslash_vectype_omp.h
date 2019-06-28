/*
 * kokkos_ops.h
 *
 *  Created on: Jul 26, 2017
 *      Author: bjoo
 */
#pragma once

#include <dslash/dslash_complex.h>
#include <complex>
#include <array>


namespace MG
{

// General struct: An Array of Complexes
template<typename T, int N>
  struct  SIMDComplex {

  std::array<MGComplex<T>,N> _data;

  constexpr static int len()  { return N; }
  
  void set(int l, const MGComplex<T>& value)
  {
    _data[l] = value;
  }
  
  const MGComplex<T>& operator()(int i) const
  {
    return _data[i];
  }
  
  MGComplex<T>& operator()(int i) {
    return _data[i];
  }
};

}
