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

} // End utilities namespace

#endif /* NN_UTILITIES_H_ */