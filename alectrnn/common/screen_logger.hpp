//
// Created by nathaniel on 3/25/18.
//

#ifndef ALECTRNN_SCREEN_LOGGER_HPP
#define ALECTRNN_SCREEN_LOGGER_HPP

#include <cstddef>
#include <vector>
#include <stdexcept>
#include "multi_array.hpp"

namespace alectrnn {

template<typename TReal, typename Time=std::size_t>
class ScreenLogger {
  public:
    typedef std::size_t Index;

    ScreenLogger() {}

    ScreenLogger(const std::vector<Index> screen_shape, Index num_iter)
                : screen_shape_(screen_shape), time_stamps_(num_iter) {
      fill_count_ = 0;
      // Allocate space for the screens
      history_->resize(num_iter);

      // Allocate space for states
      for (Index iii = 0; iii < history_.size(); ++iii) {
          history_[iii] = multi_array::Tensor<TReal>(screen_shape_);
      }
    }

    ScreenLogger(const std::vector<Index> screen_shape)
                : StateLogger(screen_shape, 0) {
    }

    ~ScreenLogger()=default;

    void copy_screen(const std::vector<unsigned char>& buffer,
                     multi_array::Tensor<TReal>& screen_to_fill) {
      if (buffer.size() != screen_to_fill.size()) {
        throw std::out_of_range("buff is different size from log screen");
      }
      for (Index iii = 0; iii < buffer.size(); ++iii) {
        screen_to_fill[iii] = static_cast<TReal>(buffer[iii]);
      }
    }

    void operator()(const std::vector<unsigned char>& output_buffer) {
      // If no time-stamp is specified, then the next available time is used.
      // This time corresponds with the access index of the state.
      (*this)(output_buffer, fill_count_);
    }

    void operator()(const std::vector<unsigned char>& output_buffer, Time time_stamp) {
      if (fill_count_ >= time_stamps_.size()) {
        time_stamps_.push_back(time_stamp);
        history_.push_back(multi_array::Tensor<TReal>(screen_shape_));
        copy_screen(output_buffer, history_[history_.size()-1]);
      }
      else {
        time_stamps_[fill_count_] = time_stamp;
        copy_screen(output_buffer, history_[fill_count_]);
      }
      ++fill_count_;
    }

    const std::vector<multi_array::Tensor<TReal>>& GetHistory() const {
      return history_;
    }

    std::vector<multi_array::Tensor<TReal>>& GetHistory() {
      return history_;
    }

    const std::vector<Time>& GetTimes() const {
      return time_stamps_;
    }

    std::vector<Time>& GetTimes() {
      return time_stamps_;
    }

  protected:
    std::vector<Index> screen_shape_;
    std::vector<multi_array::Tensor<TReal>> history_;
    std::vector<Time> time_stamps_;
    Index fill_count_;
};

} // End alectrnn namespace

#endif //ALECTRNN_SCREEN_LOGGER_HPP
