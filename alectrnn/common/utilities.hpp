/*
 * General purpose helper functions
 */

#ifndef NN_UTILITIES_H_
#define NN_UTILITIES_H_

#include <limits>
#include <cmath>

namespace utilities {

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

} // End utilities namespace

#endif /* NN_UTILITIES_H_ */