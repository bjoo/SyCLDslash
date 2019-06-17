/*
 * initialize.h
 *
 *  Created on: Oct 13, 2015
 *      Author: bjoo
 */

#pragma once

namespace MG {
	bool isInitialized(void);
	void initialize(int *argc, char ***argv);  // Initialize our system
	void finalize();                        // Finalize our system
	void abort();                            // Abort the system
};
