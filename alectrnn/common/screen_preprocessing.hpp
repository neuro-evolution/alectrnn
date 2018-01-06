/*
 * screen_preprocessing.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility function for use by the Agents to resize the ALE atari game
 * screen. ResizeGrayScreen currently just downsamples the image, but
 * GrayscaleAverage can be added easily to smooth it if need.
 *
 * (to-do: add color resizing function)
 * (to-do: add gaussian or box filter before downsizing)
 * * Note: ALE screen 1D contiguous arrays ordered by consecutive rows and so
 * can be accessed by [Y * WIDTH + X] where X is moves along columns and y 
 * along rows
 *
 */

#ifndef ALECTRNN_COMMON_SCREEN_PREPROCESSING_H_
#define ALECTRNN_COMMON_SCREEN_PREPROCESSING_H_

#include <cstdint>
#include <cstddef>
#include <vector>

namespace alectrnn {

void Apply3x3BoxFilterGray(std::vector<std::uint8_t>& src_screen,
                           std::vector<std::uint8_t>& buffer_screen,
                           std::size_t screen_width, std::size_t screen_height);
void SubsampleGrayScreen(std::size_t src_width, std::size_t src_height,
                         std::size_t tar_width, std::size_t tar_height,
                         const std::vector<std::uint8_t>& src_screen,
                         std::vector<std::uint8_t>& tar_screen);
std::uint8_t GrayscaleAverage(std::uint8_t grayscale_value1,
                              std::uint8_t grayscale_value2);
void ResizeGrayScreen(std::size_t src_width, std::size_t src_height,
                      std::size_t tar_width, std::size_t tar_height,
                      std::vector<std::uint8_t>& src_screen,
                      std::vector<std::uint8_t>& tar_screen,
                      std::vector<std::uint8_t>& buffer_screen);

void Apply3x3BoxFilterGray(const std::vector<std::uint8_t>& src_screen,
                           std::vector<float>& buffer_screen1,
                           std::vector<float>& buffer_screen2,
                           std::size_t screen_width, std::size_t screen_height);
void SubsampleGrayScreen(std::size_t src_width, std::size_t src_height,
                         std::size_t tar_width, std::size_t tar_height,
                         const std::vector<float>& src_screen,
                         std::vector<float>& tar_screen);
void ResizeGrayScreen(std::size_t src_width, std::size_t src_height,
                      std::size_t tar_width, std::size_t tar_height,
                      const std::vector<std::uint8_t>& src_screen,
                      std::vector<float>& tar_screen,
                      std::vector<float>& buffer_screen1,
                      std::vector<float>& buffer_screen2);

}

#endif /* ALECTRNN_COMMON_SCREEN_PREPROCESSING_H_ */
