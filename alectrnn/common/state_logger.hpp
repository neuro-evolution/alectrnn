/*
 * state_logger.h
 *
 *  Created on: Jan 20, 2018
 *      Author: Nathaniel Rodriguez
 *
 * Builds a logger from a nervous system object and can be called with that
 * object to add states to it.
 */

#ifndef STATE_LOGGER_H_
#define STATE_LOGGER_H_

#include <cstddef>
#include <vector>
#include "nervous_system.hpp"
#include "multi_array.hpp"

namespace nervous_system {

template<typename TReal, typename Time=std::size_t>
class StateLogger {
  public:
    typedef std::size_t Index;

    StateLogger(const NervousSystem<TReal>& neural_net, Index num_iter) 
        : time_stamps_(num_iter) {
      fill_count_ = 0;
      // Allocate space for the vectors
      history_.resize(neural_net.size());
      for (auto layer_iter = history_.begin(); layer_iter != history_.end(); ++layer_iter) {
        layer_iter->resize(num_iter);
      }
      // Allocate space for the states
      for (Index iii = 0; iii < history_.size(); ++iii) {
        for (Index jjj = 0; jjj < num_iter; ++jjj) {
          history_[iii][jjj] = multi_array::Tensor<TReal>(neural_net[iii].state().shape());
        }
      }
    }

    StateLogger(const NervousSystem<TReal>& neural_net) 
        : StateLogger(neural_net, num_iter=0) {
    }

    ~StateLogger()=default;

    void operator()(const NervousSystem<TReal>& neural_net) {
      // If no time-stamp is specified, then the next available time is used.
      // This time corresponds with the access index of the state.
      (*this)(neural_net, fill_count_);
    }

    void operator()(const NervousSystem<TReal>& neural_net, Time time_stamp) {
      if (fill_count_ >= time_stamps_.size()) {
        time_stamps_.push_back(time_stamp);
        for (Index iii = 0; iii < history_.size(); ++iii) {
          history_[iii].push_back(multi_array::Tensor<TReal>(neural_net[iii].state()));
        }
      }
      else {
        time_stamps_[fill_count_] = time_stamp;
        for (Index iii = 0; iii < history_.size(); ++iii) {
          history_[iii][fill_count_].Fill(neural_net[iii].state());
        }
      }
      ++fill_count_;
    }

    const std::vector<multi_array::Tensor<TReal>>& GetLayerHistory(Index layer) const {
      return history_[layer];
    }

    std::vector<multi_array::Tensor<TReal>>& GetLayerHistory(Index layer) {
      return history_[layer];
    }

    const std::vector<Time>& GetTimes() const {
      return time_stamps_;
    }

    std::vector<Time>& GetTimes() {
      return time_stamps_;
    }

  protected:
  std::vector<std::vector<multi_array::Tensor<TReal>>> history_;
  std::vector<Time> time_stamps_;
  Index fill_count_;
};

} // End nervous_system namespace

#endif /* STATE_LOGGER_H_ */