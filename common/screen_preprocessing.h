#ifndef ALECTRNN_COMMON_SCREEN_PREPROCESSING_H_
#define ALECTRNN_COMMON_SCREEN_PREPROCESSING_H_

#include <cstdint>
#include <cstddef>
#include <vector>

namespace alectrnn {

std::uint8_t GrayscaleAverage(std::uint8_t grayscale_value1,
                              std::uint8_t grayscale_value2);
void ResizeGrayScreen(std::size_t src_width, std::size_t src_height,
                      std::size_t tar_width, std::size_t tar_height,
                      const std::vector<std::uint8_t>& src_screen,
                      std::vector<std::uint8_t>& tar_screen);
}

#endif /* ALECTRNN_COMMON_SCREEN_PREPROCESSING_H_ */
