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
    
    // Lazy std container support. Breaks if container iterator doesn't return T
    // TODO: Add some constraint that requires Container iter to have T*
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

    TPtr GetData() {
      return data_;
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

template<typename T, std::size_t NumDim>
class ArrayView {
  public:
    typedef std::size_t Index;
    typedef std::shared_ptr<T> TSharedptr;

};

template<typename T>
class ArrayView {
  
};

template<typename T, std::size_t NumDim>
class MultiArray {
  public:
    typedef std::size_t Index;
    typedef std::shared_ptr<T> TSharedptr;

    /*
     * Build 'empty' MultiArray 
     */ 
    template< template<typename, typename...> class Container, Args...>
    MultiArray(const Container<T, Args...> &shape) : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      std::partial_sum(shape_ std::multiplies<>());
      std::reverse(strides_.begin(), strides_.end());
    }

    MultiArray(const std::initializer_list<T> &shape) 
        : shape_(shape) {

    }

    /*
     * Build MultiArray with data from pointer -> assumes contiguous
     * data and correct length
     */ 
    template< template<typename, typename...> class Container, Args...>
    MultiArray(TSharedptr data, const Container<T, Args...> &shape)
        : data_(data), shape_(shape) {

    }

    MultiArray(TSharedptr data, const std::initializer_list<T> &shape)
        : data_(data), shape_(shape) {

    }

    ~MultiArray() { };

  private:
    TSharedptr data_;
    Array<T, NumDim> shape_;
    Array<T, NumDim> strides_;
    std::size_t size_;
};

} // End namespace multi_array

#endif /* MULTI_ARRAY_H_ */