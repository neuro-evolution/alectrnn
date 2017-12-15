#ifndef MULTI_ARRAY_H_
#define MULTI_ARRAY_H_

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <functional>
#include <algorithm>

namespace multi_array {

// Forward declare
template<typename T, std::size_t NumElem>
class Array;

template<typename T>
class ArrayViewBase;

template<typename T, std::size_t NumDim>
class ArrayView;

template<typename T, std::size_t NumDim>
class MultiArray;
// End Forward declare

template<typename T, std::size_t NumElem>
class Array {
  public:
    typedef std::size_t Index;
    typedef T* TPtr;
    typedef T* iterator;
    typedef const T* const_iterator;

    // Base constructor that allocates the array, used in most other constructors
    Array() {
      data_ = new T[NumElem];
    }

    template<typename TPtr>
    Array(const TPtr array) : Array() {
      for (Index iii = 0; iii < NumElem; iii++) {
        data_[iii] = array[iii];
      }
    }
    
    template< template<typename, typename...> class Container, typename... Args>
    Array(const Container<T, Args...> &list) : Array() {
      std::copy(list.begin(), list.end(), this->begin());
    }

    Array(const std::initializer_list<T> &list) : Array() {
      std::copy(list.begin(), list.end(), this->begin());
    }

    Array(const Array<T,NumElem> &other) : Array() {
      for (Index iii = 0; iii < NumElem; iii++) {
        data_[iii] = other[iii];
      } 
    }

    Array(Array<T,NumElem> &&other) : data_( other.data_ ) {
      other.data_ = nullptr;
    }

    Array<T, NumElem>& operator=(const Array<T, NumElem> &other) {
      delete[] data_;
      data_ = new T[NumElem]; 
      for (Index iii = 0; iii < NumElem; iii++) {
        data_[iii] = other[iii];
      }
      return *this;
    }

    Array<T, NumElem>& operator=(Array<T, NumElem> &&other) {
      delete[] data_;
      data_ = other.data_;
      other.data_ = nullptr;
      return *this;
    }    
    
    ~Array() {
      delete[] data_;
    }

    TPtr data() {
      return data_;
    }

    const TPtr data() const {
      return data_;
    }

    std::size_t size() const {
      return NumElem;
    }

    const T& operator[](Index index) const {
      if (index >= NumElem || index < 0) {
        throw std::invalid_argument( "index out of Array bounds" );
      }
      return data_[index];
    }
    
    T& operator[](Index index) {
      if (index >= NumElem || index < 0) {
        throw std::invalid_argument( "index out of Array bounds" );
      }
      return data_[index];
    }

    iterator begin() {
      return &data_[0];
    }

    iterator end() {
      return &data_[NumElem];
    }

    const_iterator begin() const {
      return &data_[0];
    }

    const_iterator end() const {
      return &data_[NumElem];
    }
 
  private:
    TPtr data_;
};

template<typename T>
class ArrayViewBase {
  public:
    typedef std::size_t Index;
    typedef T* TPtr;
    typedef const std::size_t* DimPtr;

    ArrayViewBase(TPtr base, DimPtr strides) : base_(base), strides_(strides) {
    }

    ~ArrayViewBase() {};

    template< template<typename, typename...> class Container, typename... Args>
    T& operator()(const Container<T, Args...>& indices) {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    T& operator()(const std::initializer_list<T>& indices) {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    template< template<typename, typename...> class Container, typename... Args>
    const T& operator()(const Container<T, Args...>& indices) const {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    const T& operator()(const std::initializer_list<T>& indices) const {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

  private:
    TPtr base_;
    DimPtr strides_;
};

template<typename T, std::size_t NumDim>
class ArrayView : private ArrayViewBase<T> {
    typedef ArrayViewBase<T> super_type;
  public:
    typedef typename super_type::Index Index;
    typedef typename super_type::TPtr TPtr;
    typedef typename super_type::DimPtr DimPtr;

    ///TODO Fix bloody constructor, needs to set the pointer location according to ArrDims - NumDim

    ArrayView(TPtr base, DimPtr strides) : super_type(base, strides) {
    }

    template<std::size_t ArrDims>
    ArrayView(MultiArray<T, ArrDims>& array) 
        : super_type(array.data(), array.strides().data()) {
    }

    ArrayView<T, NumDim-1>& operator[](Index index) {
      return ArrayView<T, NumDim-1>(ArrayViewBase<T>::base_ + 
        index * super_type::strides_[0], super_type::strides_+1);
    }

    const ArrayView<T, NumDim-1>& operator[](Index index) const {
      return ArrayView<T, NumDim-1>(ArrayViewBase<T>::base_ + 
        index * super_type::strides_[0], super_type::strides_+1);
    }
};

template<typename T>
class ArrayView<T,1> : private ArrayViewBase<T> {
    typedef ArrayViewBase<T> super_type;
  public:
    typedef typename super_type::Index Index;
    typedef typename super_type::TPtr TPtr;
    typedef typename super_type::DimPtr DimPtr;



    T& operator[](Index index) {
      return *(super_type::base_ + index * super_type::strides_[0]);
    }

    const T& operator[](Index index) const {
      return *(super_type::base_ + index * super_type::strides_[0]);
    }
};

template<typename T, std::size_t NumDim>
class MultiArray {
  public:
    typedef T* TPtr;
    typedef std::size_t Index;

    /*
     * Build 'empty' MultiArray 
     */ 
    template< template<typename, typename...> class Container, typename... Args>
    MultiArray(const Container<Index, Args...> &shape) : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = new T[size_];
      CalculateStrides();
    }

    MultiArray(const std::initializer_list<Index> &shape) 
        : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = new T[size_];
      CalculateStrides();
    }

    /*
     * Copy (copies data as well)
     */
    MultiArray(const MultiArray<T,NumDim> &other) : shape_(other.shape_),
        strides_(other.strides_), size_(other.size_) {
      data_ = new T[size_];

      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = other.data_[iii];
      }
    }

    MultiArray(MultiArray<T,NumDim> &&other) : shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)), size_(other.size_), 
        data_(std::move(other.data_)) {
    }

    ~MultiArray() {
      delete[] data_;
    };

    void CalculateStrides() {
      for (Index iii = 0; iii < NumDim; iii++) {
        strides_[iii] = 1;
        for (Index jjj = iii + 1; jjj < NumDim; jjj++) {
          strides_[iii] *= shape_[jjj];
        }
      }
    }

    TPtr data() {
      return data_;
    }

    std::size_t size() const {
      return size_;
    }

    const Array<Index, NumDim>& shape() const {
      return shape_;
    }

    const Array<Index, NumDim>& strides() const {
      return strides_;
    }

    T& operator[](Index index) {
      return data_[index]; 
    }

    const T& operator[](Index index) const {
      return data_[index];
    }

    MultiArray<T,NumDim>& operator=(const MultiArray<T,NumDim>& other) {
      delete[] data_;
      shape_ = other.shape_;
      strides_ = other.strides_;
      size_ = other.size_;

      data_ = new T[size_];
      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = other.data_[iii];
      }
    }

    MultiArray<T,NumDim>& operator=(MultiArray<T,NumDim>&& other) {
      data_ = std::move(other.data_);
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      size_ = other.size_;

      return *this;
    }

  private:
    TPtr data_;
    Array<Index, NumDim> shape_;
    Array<Index, NumDim> strides_;
    std::size_t size_;
};

} // End namespace multi_array

#endif /* MULTI_ARRAY_H_ */