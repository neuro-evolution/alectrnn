/*
 * screen_preprocessing.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include "screen_preprocessing.h"

namespace alectrnn {

void Apply3x3BoxFilterGray(std::vector<std::uint8_t>& src_screen,
    std::vector<std::uint8_t>& buffer_screen, 
    std::size_t screen_width, std::size_t screen_height) {

  //Normalized element of box 3x3 filter. No need to use array since all values
  // are the same.
  static const double box_filter = 0.33333333333;

  // Accumulate along Columns
  for (std::size_t row = 0; row < screen_height; row++) {
    double cumulative_sum = 0.0;
    // Initiate rolling sum (boundary condition uses screen edge values)
    cumulative_sum += 2 * box_filter * src_screen[row * screen_width] \
                    + box_filter * src_screen[row * screen_width + 1];
    buffer_screen[row * screen_width] = \
        static_cast<std::uint8_t>(cumulative_sum);
    cumulative_sum += box_filter * src_screen[row * screen_width + 2] \
                    - box_filter * src_screen[row * screen_width];
    buffer_screen[row * screen_width + 1] = \
        static_cast<std::uint8_t>(cumulative_sum);

    // Initiate rolling sum (end before boundary conflict)
    for (std::size_t col = 2; col < screen_width-1; col++) {
      cumulative_sum += box_filter * src_screen[row * screen_width + col + 1] \
                      - box_filter * src_screen[row * screen_width + col - 2];
      buffer_screen[row * screen_width + col] = \
          static_cast<std::uint8_t>(cumulative_sum);
    }

    // Initiate end boundary handling
    cumulative_sum += box_filter*src_screen[row*screen_width + screen_width-1]\
                    - box_filter*src_screen[row*screen_width + screen_width-4];
    buffer_screen[row * screen_width + screen_width - 1] = \
        static_cast<std::uint8_t>(cumulative_sum);
  }

  // Accumulate along Rows
  for (std::size_t col = 0; col < screen_width; col++) {
    double cumulative_sum = 0.0;
    // Initiate rolling sum (boundary condition uses screen edge values)
    cumulative_sum += 2 * box_filter * buffer_screen[col] \
                    + box_filter * buffer_screen[1 * screen_width + col];
    src_screen[col] = \
        static_cast<std::uint8_t>(cumulative_sum);
    cumulative_sum += box_filter * buffer_screen[2*screen_width + col] \
                    - box_filter * buffer_screen[col];
    src_screen[1 * screen_width + col] = \
        static_cast<std::uint8_t>(cumulative_sum);

    // Initiate rolling sum (end before boundary conflict)
    for (std::size_t row = 2; row < screen_height-1; row++) {
      cumulative_sum += box_filter * buffer_screen[(row+1)*screen_width +col] \
                      - box_filter * buffer_screen[(row-2)*screen_width +col];
      src_screen[row * screen_width + col] = \
          static_cast<std::uint8_t>(cumulative_sum);
    }

    // Initiate end boundary handling
    cumulative_sum += box_filter*buffer_screen[(screen_height-1)*screen_width+col]\
                    - box_filter*buffer_screen[(screen_height-4)*screen_width+col];
    src_screen[(screen_height-1) * screen_width + col] = \
        static_cast<std::uint8_t>(cumulative_sum);
  }
}

void SubsampleGrayScreen(std::size_t src_width, std::size_t src_height,
                         std::size_t tar_width, std::size_t tar_height,
                         const std::vector<std::uint8_t>& src_screen,
                         std::vector<std::uint8_t>& tar_screen) {
  tar_screen.resize(tar_width * tar_height);
  double height_ratio = src_height / (double) tar_height;
  double width_ratio = src_width / (double) tar_width;
  for (std::size_t iii = 0; iii < tar_height; iii++) {
    for (std::size_t jjj = 0; jjj < tar_width; jjj++) {
      tar_screen[iii * tar_width + jjj] = 
          src_screen[(std::size_t) (floor(iii * height_ratio) * src_width 
            + floor(jjj * width_ratio))];
    }
  }
}

std::uint8_t GrayscaleAverage(std::uint8_t grayscale_value1,
                              std::uint8_t grayscale_value2) {
  return (std::uint8_t)( ((grayscale_value1) + (grayscale_value2)) >> 1);
}

void ResizeGrayScreen(std::size_t src_width, std::size_t src_height,
                      std::size_t tar_width, std::size_t tar_height,
                      std::vector<std::uint8_t>& src_screen,
                      std::vector<std::uint8_t>& tar_screen,
                      std::vector<std::uint8_t>& buffer_screen) {
  Apply3x3BoxFilterGray(src_screen, buffer_screen, src_width, src_height);
  SubsampleGrayScreen(src_width, src_height, tar_width, tar_height,
                      src_screen, tar_screen);
}

}
