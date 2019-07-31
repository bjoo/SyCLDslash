/*
 * dslash_defaults.h
 *
 *  Created on: Jul 29, 2019
 *      Author: bjoo
 */

#pragma once

#include "sycl_dslash_config.h"
#include "dslash/sycl_view.h"

namespace MG {

#ifdef MG_USE_LAYOUT_LEFT
using DefaultSpinorLayout = LayoutLeft;
using DefaultGaugeLayout = LayoutLeft;
using DefaultNeighborTableLayout = LayoutLeft;
#else
using DefaultSpinorLayout = LayoutRight;
using DefaultGaugeLayout = LayoutRight;
using DefaultNeighborTableLayout = LayoutRight;
#endif

};
