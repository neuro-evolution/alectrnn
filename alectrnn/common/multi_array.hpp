#include <cstddef>
#include <vector>
#include <stdexcept>

namespace multi_array {

typedef std::size_t Index;

template<typename T, std::size_t NumElem>
class Array {
  public:
    Array() {
      data_ = new T[NumElem];
    };
    
    typedef typename T* Pointer;
    template<typename Pointer>
    Array(Pointer array) : Array() {
      for (std::size_t iii = 0; iii < NumElem; iii++) {
        data_[iii] = array[iii];
      }
    };
    
    template<typename Container<T> >
    Array(Container<T> list) : Array() {
      for (std::size_t iii = 0; iii < NumElem; iii++) {
        data_[iii] = list[iii];
      }
    }
    
    ~Array() {
      delete[] data_;
    };

    const T& operator[](Index index) const {
      if (index >= NumElem) {
        throw std::invalid_argument( "index out of Array bounds" );
      }
      return data_[index];
    }
    
    T& operator[](Index index) {
      if (index >= NumElem) {
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