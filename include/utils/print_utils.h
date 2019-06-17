#pragma once

namespace MG {
	     /*! some log levels */
         enum LogLevel { ERROR=0, INFO, DEBUG, DEBUG2, DEBUG3 };

        /*! Only Master Process Performs Logging */
         void MasterLog(LogLevel level, const char *, ...);

         /*! All Nodes Perform Logging */
         void LocalLog(LogLevel level, const char *, ...);

         /*! Set the log level */
         void SetLogLevel(LogLevel level);

         /*! Get the current log level */
         LogLevel GetLogLevel(void);
}
