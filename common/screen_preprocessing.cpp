/*
 * screen_preprocessing.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility function for use by the Agents to resize the ALE atari game
 * screen. ResizeGrayScreen currently just downsamples the image, but
 * GrayscaleAverage can be added easily to smooth it if need.
 *
 * (to-do: add color resizing function)
 *
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include "screen_preprocessing.h"

namespace alectrnn {

std::uint8_t GrayscaleAverage(std::uint8_t grayscale_value1,
                              std::uint8_t grayscale_value2) {
  return (std::uint8_t)( ((grayscale_value1) + (grayscale_value2)) >> 1);
}

void ResizeGrayScreen(std::size_t src_width, std::size_t src_height,
                      std::size_t tar_width, std::size_t tar_height,
                      const std::vector<std::uint8_t>& src_screen,
                      std::vector<std::uint8_t>& tar_screen) {
  tar_screen.resize(tar_width * tar_height);
  double height_ratio = src_height / (double) tar_height;
  double width_ratio = src_width / (double) tar_width;
  for (std::size_t iii = 0; iii < tar_height; iii++) {
    for (std::size_t jjj = 0; jjj < tar_width; jjj++) {
      tar_screen[iii * tar_height + jjj] = 
          src_screen[(std::size_t) (floor(iii * height_ratio) * src_height +
          floor(jjj * width_ratio) * src_width)];
    }
  }
}

}
