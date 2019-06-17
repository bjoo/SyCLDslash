/*
 * nodeinfo_single.cpp
 *
 * The constructor for a node-info on a single node
 * with no communications harness
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */

#include "lattice/nodeinfo.h"
#include <vector>



namespace MG {

	NodeInfo::NodeInfo(void)
	{


		_num_nodes = 1;
		_node_id = 0;
		for(IndexType mu=0; mu < n_dim; ++mu) {
			_node_dims[mu] = 1;
			_node_coords[mu] = 0;
			_neighbor_ids[mu][MG_BACKWARD] = 0;
			_neighbor_ids[mu][MG_FORWARD] = 0;
		}
	}


}


