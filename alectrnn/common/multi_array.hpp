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
    typedef T* TPtr;
    typedef std::size_t* DimPtr;

    ArrayView(TPtr base, DimPtr strides) : base_(base), strides_(strides) {
    }

    ~ArrayView() {};

    T& operator(IndexList index)() {

    }

    const T& operator(IndexList index)() const {

    }

    ArrayView<T, NumDim-1>& operator[](Index index) {
      return ArrayView(base + index * strides_[0], strides_+1);
    }

    const ArrayView<T, NumDim-1>& operator[](Index index) const {
      return ArrayView(base + index * strides_[0], strides_+1);
    }

  private:
    TPtr base_;
    DimPtr strides_;
};

template<typename T>
class ArrayView<T,1> {
  public:
    typedef std::size_t Index;
    typedef T* TPtr;
    typedef std::size_t* DimPtr;

    ArrayView(TPtr base, DimPtr strides) : base_(base), strides_(strides) {
    }

    ~ArrayView() {};

    T& operator(IndexList index)() {

    }

    const T& operator(IndexList index)() const {

    }

    T& operator[](Index index) {
      return *(base_ + index * strides_[0]);
    }

    const T& operator[](Index index) const {
      return *(base_ + index * strides_[0]);
    }

  private:
    TPtr base_;
    DimPtr strides_;
};

template<typename T, std::size_t NumDim>
class MultiArray {
  public:
    typedef T* TPtr;
    typedef std::size_t Index;
    typedef std::shared_ptr<T> TSharedptr;

    /*
     * Build 'empty' MultiArray 
     */ 
    template< template<typename, typename...> class Container, Args...>
    MultiArray(const Container<Index, Args...> &shape) : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = std::shared_ptr<Index>(new Index[size_], std::default_delete<Index[]>());
      CalculateStrides();
    }

    MultiArray(const std::initializer_list<Index> &shape) 
        : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = std::shared_ptr<Index>(new Index[size_], std::default_delete<Index[]>());
      CalculateStrides();
    }

    /*
     * Build MultiArray with data from pointer -> assumes contiguous
     * data and correct length
     */ 
    template< template<typename, typename...> class Container, Args...>
    MultiArray(TSharedptr data, const Container<Index, Args...> &shape)
        : data_(data), shape_(shape) {
      CalculateStrides();
    }

    MultiArray(TSharedptr data, const std::initializer_list<Index> &shape)
        : data_(data), shape_(shape) {
      CalculateStrides();
    }

    ~MultiArray() {};

    void CalculateStrides() {
      std::partial_sum(shape_.begin(), shape_.end(), strides_, std::multiplies<>());
      Index major(shape_[NumDim-1]);
      std::for_each(strides_.begin(), strides_.end(), [major](Index num_elem) -> Index {
        return num_elem / major;
      });
      std::reverse(strides_.begin(), strides_.end());
    }

    TPtr GetData() {
      return data_.get();
    }

  private:
    TSharedptr data_;
    Array<Index, NumDim> shape_;
    Array<Index, NumDim> strides_;
    std::size_t size_;
};

} // End namespace multi_array

#endif /* MULTI_ARRAY_H_ */