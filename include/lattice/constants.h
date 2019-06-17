#pragma once
#include <array>

namespace MG { 

	using IndexType = unsigned long;
	const IndexType n_dim = 4; // Our lattices are four dimensional
	using IndexArray = std::array< IndexType, n_dim>;

	/* NB: I wanted to make these Enum-s but iterating over those cleanly is not easy */
	const IndexType X_DIR = 0;
	const IndexType Y_DIR = 1;
	const IndexType Z_DIR = 2;
	const IndexType T_DIR = 3;

	const IndexType n_forw_back = 2; 	/*!< There are two dirs: one forward one back */
	const IndexType MG_BACKWARD=0;
	const IndexType MG_FORWARD=1;

	const IndexType LINOP_OP=0;
	const IndexType LINOP_DAGGER = 1;

	// Will need these if we use non native complex numbers.
	const IndexType	n_complex=2;         /*!<  Number of complex numbers */
	const IndexType	RE = 0;              /*!<  Index for real part */
	const IndexType IM = 1;              /*!<  Index for complex part */

	const IndexType n_checkerboard = 2;  /*!<  Number of checkerboards */
	const IndexType	EVEN = 0;            /*!<  Even checkerboar index */
	const IndexType	ODD = 1;   		/*!	< Odd checkerboard index */

	enum HaloType { COARSE_SPINOR, COARSE_GAUGE };
}
