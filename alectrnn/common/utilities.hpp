/*
 * General purpose helper functions
 */

#ifndef NN_UTILITIES_H_
#define NN_UTILITIES_H_

#include <limits>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace utilities {

typedef int Integer; // Can be signed or unsigned

template <typename TReal>
bool OverBound(TReal x) {
  return (x > std::numeric_limits<TReal>::max()) ? true : false;
}

template <typename TReal>
bool UnderBound(TReal x) {
  return (x < std::numeric_limits<TReal>::lowest()) ? true : false;
}

template <typename TReal>
TReal BoundState(TReal x) {
  /*
   * Return a value bounded by machine float
   * If NaN, return largest float (this is the equality check, nan fails
   * comparison checks).
   */
  return OverBound(x)  ?  std::numeric_limits<TReal>::max() :
         UnderBound(x) ?  std::numeric_limits<TReal>::lowest() :
         (x == x)      ?  x :
                          std::numeric_limits<TReal>::max();
}

template <typename TReal>
TReal sigmoid(TReal x) {
  return 1 / (1 + std::exp(-x));
}

/*
 * Takes a reference to some iterable and replaces the elements with their
 * corresponding softmax values
 */
template <typename IterIn, typename TReal>
void SoftMax(IterIn begin, IterIn end, TReal temperature) {

  using VType = typename std::iterator_traits<IterIn>::value_type;
  VType vtemp = static_cast<VType>(temperature);

  auto const max_element { *std::max_element(begin, end) };

  VType normalization_factor = 0;
  std::transform(begin, end, begin, [&](VType x){
    auto ex = std::exp((x - max_element) / vtemp);
    normalization_factor+=ex;
    return ex;});

  std::transform(begin, end, begin, std::bind2nd(std::divides<VType>(),
                                             normalization_factor));
}

/*
 * Modified from caffe's implementation @
 * https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
 */
inline bool is_a_ge_zero_and_a_lt_b(Integer a, Integer b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/*
 * Modified from caffe's implementation @
 * https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
 * Uses NCHW memory layout
 */
template <typename Dtype>
void Im2Col(const Dtype* data_im, const Integer channels,
                const Integer height, const Integer width,
                const Integer kernel_h, const Integer kernel_w,
                const Integer pad_h, const Integer pad_w,
                const Integer stride_h, const Integer stride_w,
                const Integer dilation_h, const Integer dilation_w,
                Dtype* data_col) {
  const Integer output_h = (height + 2 * pad_h -
                        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const Integer output_w = (width + 2 * pad_w -
                        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const Integer channel_size = height * width;
  for (Integer channel = channels; channel--; data_im += channel_size) {
    for (Integer kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (Integer kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        Integer input_row = -pad_h + kernel_row * dilation_h;
        for (Integer output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (Integer output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            Integer input_col = -pad_w + kernel_col * dilation_w;
            for (Integer output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

} // End utilities namespace

#endif /* NN_UTILITIES_H_ */