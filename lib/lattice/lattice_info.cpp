
#include "sycl_dslash_config.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"
#include "utils/print_utils.h"

#ifdef MG_USE_OPENMP
#include <omp.h>
#endif

/* These are for sorting */
#include <algorithm>
#include <functional>
using namespace MG;
using namespace std;

namespace MG { 

LatticeInfo::LatticeInfo(
		    const IndexArray& lat_origin, // Global Origin
			const IndexArray& lat_dims,
			 IndexType n_spin,
			 IndexType n_color,
			const NodeInfo& node) : _lat_origin(lat_origin), _lat_dims(lat_dims), _n_color{n_color},
									_n_spin{n_spin}, _node_info{node}
{

	/* Sanity check the volume. Right now I have the requirement that
	 * at least 2 dimensions be even for checkerboarding with equal
	 * sites in the forward and backward faces. Further I require local
	 * X to be even a la the chroma conventions
	 */

#pragma omp master
	{

		// Check minimum length of X for X checkerboarding.
		// Check X is divisible by 2
		if( (lat_dims[0] < 2) || (lat_dims[0] % 2 != 0 ) ) {
			MasterLog(ERROR, "X-dimension must be divisible by 2. It is %u\n", lat_dims[0]);
		}

		// Check that other dimensions are nonzero
		// Count other even dimensions
		unsigned int even_others = 0;
		for(IndexType mu=1; mu  < n_dim; ++mu ) {
			if ( lat_dims[mu] == 0 ) {
				MasterLog(ERROR, "Dimension %u has zero length\n", lat_dims[mu]);
			}

			// Count even dims
			if( lat_dims[mu] % 2 == 0 ) even_others++;
		}

		if( even_others == 0 ) {
			MasterLog(ERROR, "Need at least one more dimension beside X being even to have the same number of face sites for each checkerboard");
		}
	} // omp master
#pragma omp barrier


	/* Count the number of sites */
	_n_sites = _lat_dims[0];
	for(IndexType mu=1; mu < n_dim; ++mu) {
		_n_sites *= _lat_dims[mu];
	}


	/* Compute the origin of the lattice */
	_sum_orig_coords = 0;
	for(IndexType mu=0; mu < n_dim; ++mu) {
		_sum_orig_coords += _lat_origin[mu];
	}
	_orig_cb = _sum_orig_coords & 1;

	/* Count sites of various checkerboards */
	_n_cb_sites = _n_sites/2;

	_cb_lat_dims = lat_dims;
	_cb_lat_dims[0]/=2;

	/* number of surface sites */
	/* If my minimum requirements are met */
	_num_cb_surface_sites = {{ (_lat_dims[1] * _lat_dims[2] * _lat_dims[3])/2, // X-face: YZT
						(_lat_dims[0] * _lat_dims[2] * _lat_dims[3])/2,	// Y-face: XZT
						(_lat_dims[0] * _lat_dims[1] * _lat_dims[3])/2, // Z-face: XYT
						(_lat_dims[0] * _lat_dims[1] * _lat_dims[2])/2  // T-face: XYZ
									}};



}


LatticeInfo::LatticeInfo(const IndexArray& lat_dims,
			const IndexType n_spin,
			const IndexType n_color,
			const NodeInfo& node) : LatticeInfo::LatticeInfo( ComputeOriginCoords(lat_dims,node), lat_dims, n_spin, n_color, node) {}


LatticeInfo::LatticeInfo(const IndexArray& lat_dims) :
		LatticeInfo::LatticeInfo(lat_dims, 4, 3, NodeInfo()) {}


/** Destructor for the lattice class
 */
LatticeInfo::~LatticeInfo() {}

};
