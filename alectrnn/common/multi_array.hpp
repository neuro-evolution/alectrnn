/*
 * Contains a series of classes for creating and accessing items from a 
 * multi-dimensional array.
 * 
 * The Array class is a simple 1D contiguous array with forward iterator and
 * index access using [].
 *
 * The ArrayView class is a view into some contiguous data, namely a MultiArray.
 * It owns no data itself, but access the stride and data arrays from the
 * MultiArray it views. It can use (IndexSequence) to access an element from
 * the N-dimensional data, or it can used chained [] to access elements.
 *
 * The MultiArray class is just a container for a contiguous array, which will
 * be represented as a multi-dimensional array. It requires a shape to construct
 *
 * The ArraySlice class is a 1D slice along an axis of some data. It recieves a
 * pointer to the data and requires a start, size, and stride to determine which 
 * elements to access. 
 *
 * The Slice class is also a 1D slice, however it uses start/stop/stride to 
 * deterine which elements to access.
 * 
 * accessors are available for Tensor and MultiArray and SharedMultiArray.
 * accessors return an ArrayView. Importantly, if the above containers are const
 * the accessors are guaranteed to be const as well, so that the underlying
 * data can not be changed. This is helpful in situations where containers need
 * to be passed as const reference.
 *
 * TODO: Implement at() for everyone which will check bounds
 */

#ifndef MULTI_ARRAY_H_
#define MULTI_ARRAY_H_

#include <cstddef>
#include <cassert>
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

template<typename T, std::size_t NumDim>
class SharedMultiArray;

template<typename T>
class ArraySlice;

template<typename T>
class ConstArraySlice;

template<typename T>
class Slice;

template<typename T>
class ConstSlice;

template<typename T>
class Tensor;
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
 
  protected:
    TPtr data_;
};

template<typename T>
class ArrayViewBase {
  public:
    typedef std::size_t Index;
    typedef T* TPtr;
    typedef const std::size_t* DimPtr;

    ArrayViewBase() {
      base_ = nullptr;
      strides_ = nullptr;
    }

    ArrayViewBase(TPtr base, DimPtr strides) : base_(base), strides_(strides) {
    }

    ArrayViewBase(const ArrayViewBase<T>& other) : base_(other.base_), 
        strides_(other.strides_) {
    }

    ArrayViewBase(ArrayViewBase<T>&& other) {
      base_ = other.base_;
      other.base_ = nullptr;
      strides_ = other.strides_;
      other.strides_ = nullptr;
    }

    virtual ~ArrayViewBase()=default;

    template< template<typename, typename...> class Container, typename... Args>
    T& operator()(const Container<T, Args...>& indices) {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    template< template<typename, typename...> class Container, typename... Args>
    const T& operator()(const Container<T, Args...>& indices) const {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    T& operator()(const std::initializer_list<T>& indices) {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    const T& operator()(const std::initializer_list<T>& indices) const {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_, 0);
      return base_[flat_index];
    }

    ArrayViewBase& operator=(const ArrayViewBase& other) {
      base_ = other.base_;
      strides_ = other.strides_;
      return *this;
    }

    ArrayViewBase& operator=(ArrayViewBase&& other) {
      base_ = other.base_;
      other.base_ = nullptr;
      strides_ = other.strides_;
      other.strides_ = nullptr;
      return *this;
    }

  protected:
    TPtr base_;
    DimPtr strides_;
};

template<typename T, std::size_t NumDim>
class ArrayView : public ArrayViewBase<T> {
    typedef ArrayViewBase<T> super_type;
  public:
    typedef typename super_type::Index Index;
    typedef typename super_type::TPtr TPtr;
    typedef typename super_type::DimPtr DimPtr;

    ArrayView() : super_type() {
      shape_ = nullptr;
    }

    ArrayView(TPtr base, DimPtr strides, DimPtr shape) 
        : super_type(base, strides), shape_(shape) {
    }

    ArrayView(MultiArray<T, NumDim>& array)
        : super_type(array.data(), array.strides().data()), 
          shape_(array.shape().data()) {
    }

    ArrayView(SharedMultiArray<T, NumDim>& array) 
        : super_type(array.data(), array.strides().data()), 
          shape_(array.shape().data()) {
    }

    ArrayView(const ArrayView<T, NumDim>& view) : super_type(view),
      shape_(view.shape_) {
    }

    ArrayView(ArrayView<T, NumDim>&& view) : super_type(std::move(view)) {
      shape_ = view.shape_;
      view.shape_ = nullptr;
    }

    ~ArrayView() {}

    ArrayView<T, NumDim>& operator=(const ArrayView<T,NumDim>& view) {
      super_type::operator=(view);
      shape_ = view.shape_;
      return *this;
    }

    ArrayView<T, NumDim>& operator=(ArrayView<T,NumDim>&& view) {
      super_type::operator=(std::move(view));
      shape_ = view.shape_;
      view.shape_ = nullptr;
      return *this;
    }

    ArrayView<T, NumDim-1> operator[](Index index) {
      return ArrayView<T, NumDim-1>(super_type::base_ + 
        index * super_type::strides_[0], super_type::strides_+1, shape_+1);
    }

    const ArrayView<T, NumDim-1> operator[](Index index) const {
      return ArrayView<T, NumDim-1>(super_type::base_ + 
        index * super_type::strides_[0], super_type::strides_+1, shape_+1);
    }

    Index extent(Index dimension) const {
      return shape_[dimension];
    }

  protected:
    DimPtr shape_;
};

template<typename T>
class ArrayView<T,1> : public ArrayViewBase<T> {
    typedef ArrayViewBase<T> super_type;
  public:
    typedef typename super_type::Index Index;
    typedef typename super_type::TPtr TPtr;
    typedef typename super_type::DimPtr DimPtr;

    ArrayView() : super_type() {
      shape_ = nullptr;
    }

    ArrayView(TPtr base, DimPtr strides, DimPtr shape) 
        : super_type(base, strides), shape_(shape) {
    }

    ArrayView(MultiArray<T, 1>& array) 
        : super_type(array.data(), array.strides().data()), 
          shape_(array.shape().data()) {
    }

    ArrayView(SharedMultiArray<T, 1>& array) 
        : super_type(array.data(), array.strides().data()), 
          shape_(array.shape().data()) {
    }

    ArrayView(const ArrayView<T, 1>& view) : super_type(view),
      shape_(view.shape_) {
    }

    ArrayView(ArrayView<T, 1>&& view) : super_type(std::move(view)) {
      shape_ = view.shape_;
      view.shape_ = nullptr;
    }

    ~ArrayView() {}

    ArrayView<T, 1>& operator=(const ArrayView<T,1>& view) {
      super_type::operator=(view);
      shape_ = view.shape_;
      return *this;
    }

    ArrayView<T, 1>& operator=(ArrayView<T,1>&& view) {
      super_type::operator=(std::move(view));
      shape_ = view.shape_;
      view.shape_ = nullptr;
      return *this;
    }

    T& operator[](Index index) {
      return *(super_type::base_ + index * super_type::strides_[0]);
    }

    const T& operator[](Index index) const {
      return *(super_type::base_ + index * super_type::strides_[0]);
    }

    Index extent(Index dimension) const {
      return shape_[dimension];
    }

    Index extent() const {
      return *shape_;
    }

  protected:
    DimPtr shape_;
};

template<typename T, std::size_t NumDim>
class MultiArray {
  public:
    typedef T* TPtr;
    typedef std::size_t Index;

    MultiArray() {
      data_ = nullptr;
      size_ = 0;
    }
    /*
     * Build 'empty' MultiArray 
     */ 
    MultiArray(const Array<Index, NumDim> &shape) : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = new T[size_];
      CalculateStrides();
    }

    MultiArray(const std::vector<Index> &shape) : shape_(shape) {
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
     * Generates array from existing data and takes ownership of data
     */
    MultiArray(TPtr data, const std::vector<Index> &shape) 
        : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = data;
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
        strides_(std::move(other.strides_)), size_(std::move(other.size_)) {
      data_ = other.data_;
      other.data_ = nullptr;
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

    const TPtr data() const {
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

      return *this;
    }

    MultiArray<T,NumDim>& operator=(MultiArray<T,NumDim>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      size_ = std::move(other.size_);

      return *this;
    }

    ArrayView<T, NumDim> accessor() {
      return ArrayView<T, NumDim>(this->data_, this->strides_.data(), 
        this->shape_.data());
    }

    const ArrayView<T, NumDim> accessor() const {
      return ArrayView<T, NumDim>(this->data_, this->strides_.data(),
        this->shape_.data());
    }

    void Fill(T value) {
      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = value;
      }
    }

  protected:
    TPtr data_;
    Array<Index, NumDim> shape_;
    Array<Index, NumDim> strides_;
    std::size_t size_;
};

/*
 * This array shares data and does not take ownership. Think of it as the 
 * base for a view, as the view also does not take ownership, but it requires
 * pointers to stride and shape objects.
 */
template<typename T, std::size_t NumDim>
class SharedMultiArray {
  public:
    typedef T* TPtr;
    typedef std::size_t Index;

    SharedMultiArray() {
      data_ = nullptr;
      size_ = 0;
    }

    SharedMultiArray(TPtr data, const Array<Index, NumDim> &shape)
        : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = data;
      CalculateStrides();
    }

    SharedMultiArray(TPtr data, const std::vector<Index> &shape) 
        : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = data;
      CalculateStrides();
    }

    SharedMultiArray(TPtr data, const std::initializer_list<Index> &shape) 
        : shape_(shape) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = data;
      CalculateStrides();
    }

    /*
     * Copy (doesn't copy data)
     */
    SharedMultiArray(const SharedMultiArray<T,NumDim> &other) 
        : data_(other.data_), shape_(other.shape_), strides_(other.strides_), 
        size_(other.size_) {
    }

    SharedMultiArray(SharedMultiArray<T,NumDim> &&other) : shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)), size_(std::move(other.size_)) {
      data_ = other.data_;
      other.data_ = nullptr;
    }

    ~SharedMultiArray()=default;

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

    const TPtr data() const {
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

    SharedMultiArray<T,NumDim>& operator=(const SharedMultiArray<T,NumDim>& other) {
      data_ = other.data_;
      shape_ = other.shape_;
      strides_ = other.strides_;
      size_ = other.size_;
      return *this;
    }

    SharedMultiArray<T,NumDim>& operator=(SharedMultiArray<T,NumDim>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      size_ = std::move(other.size_);

      return *this;
    }

    ArrayView<T, NumDim> accessor() {
      return ArrayView<T, NumDim>(this->data_, this->strides_.data(), 
        this->shape_.data());
    }

    const ArrayView<T, NumDim> accessor() const {
      return ArrayView<T, NumDim>(this->data_, this->strides_.data(),
        this->shape_.data());
    }

    void Fill(T value) {
      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = value;
      }
    }

  protected:
    TPtr data_;
    Array<Index, NumDim> shape_;
    Array<Index, NumDim> strides_;
    std::size_t size_;
};

/*
 * This is a Size based slice, where the shape of an array is used to specify
 * the stop point.
 */
template<typename T>
class ArraySlice {
  public:
    typedef std::size_t Index;

    ArraySlice() {
      data_ = nullptr;
      start_ = 0;
      size_ = 0;
      stride_ = 0;
    }

    ArraySlice(T* data, Index start, Index size, Index stride=1) : data_(data),
        start_(start), size_(size), stride_(stride) {
    }

    ArraySlice(const ArraySlice<T>& other) : data_(other.data_), 
        start_(other.start_), size_(other.size_), stride_(other.stride_) {
    }

    ArraySlice(ArraySlice<T>&& other) : start_(std::move(other.start_)),
        size_(std::move(other.size_)), stride_(std::move(other.stride_)) {
      data_ = other.data_;
      other.data_ = nullptr;
    }

    ArraySlice<T>& operator=(const ArraySlice<T>& other) {
      data_ = other.data_;
      start_ = other.start_;
      size_ = other.size_;
      stride_ = other.stride_;
      return *this; 
    }

    ArraySlice<T>& operator=(ArraySlice<T>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      start_ = std::move(other.start_);
      size_ = std::move(other.size_);
      stride_ = std::move(other.stride_);
      return *this; 
    }

    ~ArraySlice() {}

    std::size_t size() const {
      return size_;
    }

    const T& operator[](Index index) const {
      return data_[start_ + index * stride_];
    }

    T& operator[](Index index) {
      return data_[start_ + index * stride_];
    }

    T* data() {
      return data_;
    }

    const T* data() const {
      return data_;
    }

    void data(T* new_data) {
      data_ = new_data;
    }

    Index stride() const {
      return stride_;
    }

    Index start() const {
      return start_;
    }

  protected:
    T* data_;
    Index start_;
    Index size_;
    Index stride_;
};

/*
 * Const version of ArraySlice
 */
template<typename T>
class ConstArraySlice {
  public:
    typedef std::size_t Index;

    ConstArraySlice() {
      data_ = nullptr;
      start_ = 0;
      size_ = 0;
      stride_ = 0;
    }

    ConstArraySlice(const T* data, Index start, Index size, Index stride=1) : 
        data_(data), start_(start), size_(size), stride_(stride) {
    }

    ConstArraySlice(const ConstArraySlice<T>& other) : data_(other.data_), 
        start_(other.start_), size_(other.size_), stride_(other.stride_) {
    }

    ConstArraySlice(ConstArraySlice<T>&& other) : start_(std::move(other.start_)),
        size_(std::move(other.size_)), stride_(std::move(other.stride_)) {
      data_ = other.data_;
      other.data_ = nullptr;
    }

    ConstArraySlice<T>& operator=(const ConstArraySlice<T>& other) {
      data_ = other.data_;
      start_ = other.start_;
      size_ = other.size_;
      stride_ = other.stride_;
      return *this; 
    }

    ConstArraySlice<T>& operator=(ConstArraySlice<T>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      start_ = std::move(other.start_);
      size_ = std::move(other.size_);
      stride_ = std::move(other.stride_);
      return *this; 
    }

    ~ConstArraySlice() {}

    std::size_t size() const {
      return size_;
    }

    const T& operator[](Index index) const {
      return data_[start_ + index * stride_];
    }

    const T* data() const {
      return data_;
    }

    void data(const T* new_data) {
      data_ = new_data;
    }

    Index stride() const {
      return stride_;
    }

    Index start() const {
      return start_;
    }

  protected:
    const T* data_;
    Index start_;
    Index size_;
    Index stride_;
};

/*
 * This is a Stop based slice, where the stopping index is used to specify
 * the size. More like a Numpy slice.
 */
template<typename T>
class Slice {
  public:
    typedef std::size_t Index;

    Slice() {
      data_ = nullptr;
      start_ = 0;
      stop_ = 0;
      stride_ = 0;
      size_ = 0;
    }

    Slice(T* data, Index start, Index stop, Index stride=1) : data_(data),
        start_(start), stop_(stop), stride_(stride) {
      size_ = (stop_ - start_) / stride_;
    }

    Slice(const Slice<T>& other) : data_(other.data_), start_(other.start_), 
        stop_(other.stop_), stride_(other.stride_),
        size_(other.size_) {
    }

    Slice(Slice<T>&& other) : start_(std::move(other.start_)), 
        stop_(std::move(other.stop_)), stride_(std::move(other.stride_)),
        size_(std::move(other.size_)) {
      data_ = other.data_;
      other.data_ = nullptr;
    }

    Slice<T>& operator=(const Slice<T>& other) {
      data_ = other.data_;
      start_ = std::move(other.start_);
      stop_ = std::move(other.stop_);
      stride_ = std::move(other.stride_);
      size_ = std::move(other.size_);
      return *this; 
    }    

    Slice<T>& operator=(Slice<T>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      start_ = std::move(other.start_);
      stop_ = std::move(other.stop_);
      stride_ = std::move(other.stride_);
      size_ = std::move(other.size_);
      return *this; 
    }

    ~Slice() {}

    const T& operator[](Index index) const {
      return data_[start_ + index * stride_];
    }

    T& operator[](Index index) {
      return data_[start_ + index * stride_];
    }

    std::size_t size() const {
      return size_;
    }

    T* data() {
      return data_;
    }

    void data(T* new_data) {
      data_ = new_data;
    }

    Index stride() const {
      return stride_;
    }

    Index start() const {
      return start_;
    }

    Index stop() const {
      return stop_;
    }

  protected:
    T* data_;
    Index start_;
    Index stop_;
    Index stride_;
    Index size_;
};

/*
 * A constant version of slice
 */
template<typename T>
class ConstSlice {
  public:
    typedef std::size_t Index;

    ConstSlice() {
      data_ = nullptr;
      start_ = 0;
      stop_ = 0;
      stride_ = 0;
      size_ = 0;
    }

    ConstSlice(const T* data, Index start, Index stop, Index stride=1) : data_(data),
        start_(start), stop_(stop), stride_(stride) {
      size_ = (stop_ - start_) / stride_;
    }

    ConstSlice(const ConstSlice<T>& other) :  data_(other.data_), 
        start_(other.start_), stop_(other.stop_), stride_(other.stride_),
        size_(other.size_) {
    }

    ConstSlice(ConstSlice<T>&& other) : start_(std::move(other.start_)), 
        stop_(std::move(other.stop_)), stride_(std::move(other.stride_)),
        size_(std::move(other.size_)) {
      data_ = other.data_;
      other.data_ = nullptr;
    }

    ConstSlice<T>& operator=(const ConstSlice<T>& other) {
      data_ = other.data_;
      start_ = other.start_;
      stop_ = other.stop_;
      stride_ = other.stride_;
      size_ = other.size_;
      return *this;      
    }

    ConstSlice<T>& operator=(ConstSlice<T>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      start_ = std::move(other.start_);
      stop_ = std::move(other.stop_);
      stride_ = std::move(other.stride_);
      size_ = std::move(other.size_);
      return *this; 
    }

    ~ConstSlice() {}

    const T& operator[](Index index) const {
      return data_[start_ + index * stride_];
    }

    std::size_t size() const {
      return size_;
    }

    const T* data() const {
      return data_;
    }

    void data(T* new_data) {
      data_ = new_data;
    }

    Index stride() const {
      return stride_;
    }

    Index start() const {
      return start_;
    }

    Index stop() const {
      return stop_;
    }

  protected:
    const T* data_;
    Index start_;
    Index stop_;
    Index stride_;
    Index size_;
};

/*
 * Tensor is a MultiArray with untyped dimension. An accessor is created
 * which will check if the casted view dimensions match the tensor dimensions.
 */
template<typename T>
class Tensor {
  public:
    typedef T* TPtr;
    typedef std::size_t Index;

    // Default constructor
    Tensor() {
      ndims_ = 0;
      size_ = 0;
      data_ = nullptr;
    }

    /*
     * Build 'empty' Tensor.
     */
    template<typename Index, std::size_t NumDim>
    Tensor(const Array<Index, NumDim> &shape) : shape_(shape.begin(), shape.end()),
        ndims_(shape.size()) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = new T[size_];
      strides_.resize(ndims_);
      CalculateStrides();
    }

    Tensor(const std::vector<Index> &shape) : shape_(shape), 
        ndims_(shape.size()) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = new T[size_];
      strides_.resize(ndims_);
      CalculateStrides();
    }

    Tensor(const std::initializer_list<Index> &shape) 
        : shape_(shape), ndims_(shape.size()) {
      size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
      data_ = new T[size_];
      strides_.resize(ndims_);
      CalculateStrides();
    }

    /*
     * Copy (copies data as well)
     */
    Tensor(const Tensor<T> &other) : shape_(other.shape_),
        strides_(other.strides_), ndims_(other.ndims_), size_(other.size_) {
      data_ = new T[size_];

      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = other.data_[iii];
      }
    }

    Tensor(Tensor<T> &&other) : shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)), ndims_(std::move(other.ndims_)),
        size_(std::move(other.size_)) {
      data_ = other.data_;
      other.data_ = nullptr;
    }

    ~Tensor() {
      delete[] data_;
    };

    void CalculateStrides() {
      for (Index iii = 0; iii < ndims_; iii++) {
        strides_[iii] = 1;
        for (Index jjj = iii + 1; jjj < ndims_; jjj++) {
          strides_[iii] *= shape_[jjj];
        }
      }
    }

    TPtr data() {
      return data_;
    }

    const TPtr data() const {
      return data_;
    }

    std::size_t size() const {
      return size_;
    }

    std::size_t ndimensions() const {
      return ndims_;
    }

    const std::vector<Index>& shape() const {
      return shape_;
    }

    const std::vector<Index>& strides() const {
      return strides_;
    }

    T& operator[](Index index) {
      return data_[index]; 
    }

    const T& operator[](Index index) const {
      return data_[index];
    }

    T& operator()(const std::initializer_list<T>& indices) {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_.begin(), 0);
      return data_[flat_index];
    }

    const T& operator()(const std::initializer_list<T>& indices) const {
      Index flat_index = std::inner_product(indices.begin(), indices.end(), strides_.begin(), 0);
      return data_[flat_index];
    }

    Tensor<T>& operator=(const Tensor<T>& other) {
      delete[] data_;
      shape_ = other.shape_;
      strides_ = other.strides_;
      size_ = other.size_;
      ndims_ = other.ndims_;

      data_ = new T[size_];
      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = other.data_[iii];
      }

      return *this;
    }

    Tensor<T>& operator=(Tensor<T>&& other) {
      data_ = other.data_;
      other.data_ = nullptr;
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      size_ = std::move(other.size_);
      ndims_ = std::move(other.ndims_);

      return *this;
    }

    template<std::size_t NumDim>
    ArrayView<T, NumDim> accessor() {
      assert(NumDim == ndims_);
      return ArrayView<T, NumDim>(this->data_, this->strides_.data(), 
        this->shape_.data());
    }

    template<std::size_t NumDim>
    const ArrayView<T, NumDim> accessor() const {
      assert(NumDim == ndims_);
      return ArrayView<T, NumDim>(this->data_, this->strides_.data(),
        this->shape_.data());
    }

    void Fill(T value) {
      for (Index iii = 0; iii < size_; iii++) {
        data_[iii] = value;
      }
    }

  protected:
    TPtr data_;
    std::vector<Index> shape_;
    std::vector<Index> strides_;
    std::size_t ndims_;
    std::size_t size_;
};

void CalculateStrides(const std::size_t* shape, std::size_t* strides, std::size_t ndims) {
  for (std::size_t iii = 0; iii < ndims; iii++) {
    strides[iii] = 1;
    for (std::size_t jjj = iii + 1; jjj < ndims; jjj++) {
      strides[iii] *= shape[jjj];
    }
  }
}

} // End namespace multi_array

#endif /* MULTI_ARRAY_H_ */