// take ref greyscale sized vector, take ref downsized vector, take ale object
// greyscale sized vector 210x160 = 33600
// downsized vector 84x84 = 7056
// call ale's screen using greyscale vector
// make algorithm that will cut (or avg see #define AVERAGE(a, b)   (PIXEL)( ((a) + (b)) >> 1 ))
// the proper pixels
// void (row order first (160) where row is height, col is width (210))

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
