/*
 * capi_tools.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 * Uses CImg library for debugging if needed: see http://cimg.eu
 */

#ifndef ALECTRNN_COMMON_CAPI_TOOLS_H_
#define ALECTRNN_COMMON_CAPI_TOOLS_H_

#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include "arrayobject.h"
// #include "CImg.h"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array);
// cimg_library::CImg<std::uint8_t> ConvertGrayFrameToImg(
//   const std::vector<std::uint8_t> &frame,
//   std::size_t frame_width, std::size_t frame_height);
// void SaveGrayFrameToPNG(const std::vector<std::uint8_t> &frame,
//   std::size_t frame_width, std::size_t frame_height,
//   const std::string &filename);

}



#endif /* ALECTRNN_COMMON_CAPI_TOOLS_H_ */
