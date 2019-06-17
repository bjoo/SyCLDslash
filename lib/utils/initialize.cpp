/*
 * initialize.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#include "sycl_dslash_config.h"
#include "utils/print_utils.h"
#include <string>
#include <cstdlib>

#ifdef HAVE_QDPXX
#include <qdp.h>
#endif

namespace MG
{

	namespace {
		static bool isInitializedP = false;
	}

	void initialize(int *argc, char ***argv)
	{
		if ( ! isInitializedP ) {
			// Process args

			int i=0;
			int my_argc = (*argc);

			/* Process args here -- first step is to get the processor geomerty */
			/* No args to process just now */

#ifdef HAVE_QDPXX
			MasterLog(INFO, "Initializing QDP++");
			QDP::QDP_initialize(argc,argv);
			MasterLog(INFO, "QDP++ Initialized");
#endif
			isInitializedP = true;
		} // if (! isInitiealizedP )
	}

	void finalize(void)
	{
		if ( isInitializedP ) {

#if defined(HAVE_QDPXX)
		MasterLog(INFO, "Finalizing QDP++");
		QDP::QDP_finalize();
#endif
		isInitializedP = false;
		}
	}

	bool isInitialized(void)
	{
		return isInitializedP;
	}

	void abort(void)
	{
		if( isInitializedP ) {
#ifdef HAVE_QDPXX
			QDP::QDP_abort(1);
#else
			std::abort();
#endif
		}
	}


}

