//  Include guard: Include this file only once
#ifndef TEST_QDPXX_REUNIT_H
#define TEST_QDPXX_REUNIT_H

#include <qdp.h>

using namespace QDP;
namespace MGTesting {

// Reunitarize a Lattice Color Matrix
void reunit(LatticeColorMatrixF& a);
void reunit(LatticeColorMatrixD& a);

};


// End of Include guard
#endif
