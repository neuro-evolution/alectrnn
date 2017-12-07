/*
 * capi_tools.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 * Uses CImg library for debugging if needed: see http://cimg.eu
 */

#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "arrayobject.h"
// #include "CImg.h"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array) {
  return (float *) py_array->data;
}

// cimg_library::CImg<std::uint8_t> ConvertGrayFrameToImg(
//     const std::vector<std::uint8_t> &frame,
//     std::size_t frame_width, std::size_t frame_height) {
//   cimg_library::CImg<std::uint8_t> image(frame_width, frame_height, 1, 1, 0);
//   for (std::size_t iii = 0; iii < frame_width; iii++) {
//     for (std::size_t jjj = 0; jjj < frame_height; jjj++) {
//       image(iii, jjj) = frame[jjj * frame_width + iii];
//     }
//   }

//   return image;
// }

// void SaveGrayFrameToPNG(
//     const std::vector<std::uint8_t> &frame,
//     std::size_t frame_width, std::size_t frame_height,
//     const std::string &filename) {
//   cimg_library::CImg<std::uint8_t> image(ConvertGrayFrameToImg(frame, 
//                                          frame_width, frame_height));
//   image.save((filename + ".png").c_str());
// }

}
