#include "utils/print_utils.h"
#include "sycl_dslash_config.h"
#include "utils/initialize.h"
#include <string>
#include <cstdlib>
#include <cstdarg>

#ifdef MG_QMP_COMMS
#include "qmp.h"
#endif

namespace MG {

	/* Current Log Level */
	static volatile LogLevel current_log_level = MG_DEFAULT_LOGLEVEL;

	/* An array holding strings corresponding to log levels */
	static std::string log_string_array[] = {"ERROR", "INFO", "DEBUG", "DEBUG2", "DEBUG3"};


	/**
	 * SetLogLevel -- set the log level
	 *
	 * \param level  -- The LogLevel to set. LogMessages with levels <= level will be printed
	 *
	 * NB: This function may be called in several MPI processes, in which case it needs to
	 * be called collectively. Being called in one MPI process and not in another is considered
	 * a programming error. Likewise the function may be concurrently called from several
	 * OpenMP threads. If called from many threads potentially. While it would be weird for
	 * all threads to set different log levels, the safe thing to do is to guard the write with
	 * an OMP Criticla section
	 *
	 */
	void SetLogLevel(LogLevel level)
	{
#pragma omp master
		{
			current_log_level = level;
		}
#pragma omp barrier

	}

	/**
	 * SetLogLevel - get the log level
	 *
	 * \returns  The current log level
	 *
	 * NB: The design is for the loglevel to be kept on each MPI process. This function only
	 * reads the loglevel value, so no races can occur.
	 */
	LogLevel GetLogLevel(void) {
		return current_log_level;
	}


	/**
	 * 	LocalLog - Local process performs logging
	 * 	\param level -- if the level is <= the current log level, a message will be printed
	 * 	\param format_string
	 * 	\param variable list of arguments
	 *
	 * 	Current definition is that only the master thread on each nodes logs
	 */
	void LocalLog(LogLevel level, const char*  format,...)
	{
		va_list args;
		va_start(args,format);
#pragma omp master
		{
			if( level <= current_log_level ) {
#ifdef MG_QMP_COMMS
				int size = QMP_get_number_of_nodes();
				int rank = QMP_get_node_number();
#else
				int size = 1;
				int rank = 0;
#endif
				printf("%s: Rank %d of %d: ", log_string_array[level].c_str(), rank, size);

				vprintf(format, args);

				printf("\n");
			}	/* end If */
		 	va_end(args);
			/* If the level is error than we should abort */
			if( level == ERROR ) {
				MG::abort();
			} /* if level == ERROR */
		}
	}


	/**
	 * 	MasterlLog - Master  process performs logging
	 * 	\param level -- if the level is <= the current log level, a message will be printed
	 * 	\param format_string
	 * 	\param variable list of arguments
	 *
	 * 	Current definition is that only the master thread on each nodes logs
	 */
    void MasterLog(LogLevel level, const char *format, ...)
    {
			va_list args;
			va_start(args,format);
#pragma omp master
    	{

#ifdef MG_QMP_COMMS
    		if ( QMP_is_primary_node() )  {
#endif
    			if( level <= current_log_level ) {

    				printf("%s: ", log_string_array[level].c_str());

    				vprintf(format, args);

    				printf("\n");
    			}	/* en	d If */

    				/* If the level is error than we should abort */
    			if( level == ERROR ) {
    				MG::abort();
    			} /* if level == ERROR */

#ifdef MG_QMP_COMMS
    		} /* if ( QMP_is_primary_node())  */
#endif
    	} /* End OMP MASTER REGION */
    	va_end(args);
    }


} /* End Namespace MGUtils */
