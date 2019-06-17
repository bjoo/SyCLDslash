/*
 * nodeinfo_qmp.cpp
 *
 * Construct a NodeInfo using QMP
 *
 *  Created on: Sep 25, 2015
 *      Author: bjoo
 */

#include "lattice/nodeinfo.h"
#include "lattice/constants.h"
#include "utils/print_utils.h"
#include <vector>
#include <cstdlib>
#include <iostream>
#include "qmp.h"

using namespace MG;

namespace MG {

	NodeInfo::NodeInfo(void)
	{

		if ( QMP_is_initialized() == QMP_TRUE ) {
			// Let us not use the multiple communicator feature
			_num_nodes = QMP_get_number_of_nodes();
			_node_id = QMP_get_node_number();
			if( QMP_logical_topology_is_declared() == QMP_TRUE ) {
				int qmp_dims = QMP_get_logical_number_of_dimensions();

				/* Check that the QMP dimensions are compatible with n_dim */
				if( qmp_dims != n_dim ) {
					MasterLog(ERROR, "QMP Logical Topo dimensionst %d is different from n_dim(=%d)",
							qmp_dims, n_dim);
				}

				// Get the coordinates and dimensions of the PE Grid
				{
					const int *qmp_dims = QMP_get_logical_dimensions();
					const int *qmp_coords = QMP_get_logical_coordinates();

					for(IndexType mu=0; mu < n_dim; ++mu) {
						_node_dims[mu] = qmp_dims[mu];
						_node_coords[mu] = qmp_coords[mu];
					}
				}

				// Find the coordinates and node_id's of the neighbors
				{
					// Find neighbor in mu dim
					for(IndexType mu=0; mu < n_dim; ++mu) {

					  for(auto dir = MG_BACKWARD; dir <= MG_FORWARD; ++dir) {

						 // QMP Expects ints rather than unsigned ints
						 int neigh_coords[n_dim] = { static_cast<int>(_node_coords[0]),
													 static_cast<int>(_node_coords[1]),
													 static_cast<int>(_node_coords[2]),
													 static_cast<int>(_node_coords[3]) };


					   	// Implement wraparounds
						if( dir == MG_BACKWARD ) {
							neigh_coords[mu]--;
							if( neigh_coords[mu] < 0 ) neigh_coords[mu] = _node_dims[mu]-1;
						}

						if( dir  == MG_FORWARD ) {
							neigh_coords[mu]++;
							if( neigh_coords[mu] >= static_cast<int>(_node_dims[mu])) neigh_coords[mu] = 0;
						}

						_neighbor_ids[mu][dir] =static_cast<IndexType>(QMP_get_node_number_from(neigh_coords));

					  }

					}

				}

			}
			else {
				// Logical topo is not declared
				MasterLog(ERROR, "QMP local topology is not declared");
			} // Logical Topo declared
		}
		else {
			// Cannot use Logging Interface here because
			// QMP is not initialized
			std::cout << "FATAL: QMP IS NOT INITIALIZED" << std::endl;
			std::_Exit(EXIT_FAILURE);
		} // QMP is initialized




	} // Constructor

} // Namespace


