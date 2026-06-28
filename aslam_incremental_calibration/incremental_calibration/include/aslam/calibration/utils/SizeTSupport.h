/******************************************************************************
 * Copyright (C) 2013 by Jerome Maye                                          *
 * jerome.maye@gmail.com                                                      *
 *                                                                            *
 * This program is free software; you can redistribute it and/or modify       *
 * it under the terms of the Lesser GNU General Public License as published by*
 * the Free Software Foundation; either version 3 of the License, or          *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * Lesser GNU General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the Lesser GNU General Public License   *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 ******************************************************************************/

/** \file SizeTSupport.h
    \brief This file defines the Eigen support for size_t type.
  */

#ifndef ASLAM_CALIBRATION_UTILS_SIZETSUPPORT_H
#define ASLAM_CALIBRATION_UTILS_SIZETSUPPORT_H

#include <cstdlib>

#include <Eigen/Core>

// Modern Eigen already provides NumTraits for the unsigned integer type that
// size_t aliases on supported platforms. The original Kalibr compatibility
// specialization can be instantiated too late with newer Eigen/Clang, so this
// header is intentionally a no-op in this standalone build.

#endif // ASLAM_CALIBRATION_UTILS_SIZETSUPPORT_H
