#include <cstddef>
#include <vector>
#include <stdexcept>
#include <initializer_list>

namespace multi_array {


template<typename T, std::size_t NumElem>
class Array {
  public:
    typedef std::size_t Index;
    typedef T* TPointer;

    Array() {
      data_ = new T[NumElem];
    }
    
    template<typename TPointer>
    Array(const TPointer array) : Array() {
      for (Index iii = 0; iii < NumElem; iii++) {
        data_[iii] = array[iii];
      }
    }
    
    template<typename Container<T> >
    Array(const Container<T> &list) : Array() {
      for (Index iii = 0; iii < NumElem; iii++) {
        data_[iii] = list[iii];
      }
    }

    Array(const std::initializer_list<T> &list) : Array() {
      for (Index iii = 0; iii < NumElem; iii++) {
        data_[iii] = list[iii];
      } 
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

    T* GetData() {
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
 
  private:
    T* data_;
}

template<typename T, std::size_t NumDim>
class ArrayView {
  
}

template<typename T>
class ArrayView {
  
}

template<typename T, std::size_t NumDim>
class MultiArray {
  public:
    MultiArray(T* data, std::vector<std::size_t> shape) 
        : data_(data), shape_(shape) {
      
      for (std::size_t iii = 0; iii < NumDim; iii++) {
        
      }
    }
    ~MultiArray();

  private:
    T* data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
}

} // End namespace multi_array