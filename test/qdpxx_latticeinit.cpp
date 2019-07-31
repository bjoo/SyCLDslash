/*
 * qdpxx_latticeinit.cpp
 *
 *  Created on: Mar 15, 2017
 *      Author: bjoo
 */

#include "qdpxx_latticeinit.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"
#include "qdp.h"

using namespace QDP;


namespace MGTesting {
void initQDPXXLattice(const IndexArray& latdims )
{
	NodeInfo node;
	LatticeInfo tmp_info(latdims,4,3,node);
	IndexArray gdims;
	tmp_info.LocalDimsToGlobalDims(gdims,latdims);

	multi1d<int> nrow(n_dim);
	for(int i=0; i < n_dim; ++i) nrow[i] =gdims[i];
	Layout::setLattSize(nrow);
	Layout::create();
}
}
