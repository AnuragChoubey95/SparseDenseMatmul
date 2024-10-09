#include<vector>
#include<stdexcept>
#include <iostream>
#include <omp.h>

class Tensor{
public:

  //Regular data for the regular original matrix
  std::vector<float> data;
  std::vector<size_t> dims;

  //sparse matrix data 
  std::vector<float> csr_data;
  std::vector<size_t> csr_columns;
  std::vector<size_t> csr_row_pointers;
  
  Tensor(std::vector<size_t> dims) : dims(dims){
    size_t len = 1;
    for(auto d : dims)
      len *= d;
    data.resize(len);
  }

  Tensor(std::vector<size_t> dims,std::vector<std::vector<size_t>> idx,std::vector<float> val) : dims(dims){
    size_t len = 1;
    for(auto d : dims)
      len *= d;
    data.resize(len);////non zero matrix back to a matrix with zeros

    if(idx.size() != val.size())
      throw std::runtime_error("Mismatched idx and val size");
    for(size_t i = 0;i < idx.size();++i){
      data[index(idx[i])] = val[i];
    }
  }

  static Tensor ones(std::vector<size_t> dims){
    Tensor ret(dims);
    for(size_t i = 0;i < ret.data.size();++i)
      ret.data[i] = 1;
    return ret;
  }

  inline size_t index(std::vector<size_t> x){
    if(x.size() != dims.size())
      throw std::runtime_error("Mismatched dims in index");
    size_t ret = 0;
    size_t prod = 1;
    for(int i = dims.size() - 1;i >= 0;--i){
      if(x[i] >= dims[i])
        throw std::runtime_error("Index out of bound");
      ret += x[i] * prod;
      prod *= dims[i];
    } 
    return ret;
  }

  Tensor reshape(std::vector<size_t> new_dims){
    size_t len = 1;
    for(auto d : new_dims)
      len *= d;
    if(len != data.size())
      throw std::runtime_error("Mismatched dims in reshape");
    Tensor ret(new_dims);
    ret.data = data;
    return ret;
  }

  Tensor transpose() {
    if (dims.size() == 2) {
        Tensor ret({dims[1], dims[0]});
        #pragma omp parallel for collapse(2) num_threads(8)
        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                ret.data[j * dims[0] + i] = data[i * dims[1] + j];
            }
        }
        return ret;
    } else if (dims.size() == 3) {
        Tensor ret({dims[0], dims[2], dims[1]});
        #pragma omp parallel for collapse(2) num_threads(8)
        for (size_t b = 0; b < dims[0]; ++b) {
            for (size_t i = 0; i < dims[1]; ++i) {
                for (size_t j = 0; j < dims[2]; ++j) {
                    ret.data[b * dims[1] * dims[2] + j * dims[1] + i] = data[b * dims[1] * dims[2] + i * dims[2] + j];
                }
            }
        }
        return ret;
    } else {
        throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
    }
  }


  Tensor neg(){
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = -data[i];
    return ret;
  }
  
  Tensor reciprocal(){
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = 1.0 / data[i];
    return ret;
  }

  Tensor add(Tensor x){
    if(dims != x.dims)
      throw std::runtime_error("Mismatched shape in add");
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] + x.data[i];
    return ret;
  }
  
  Tensor subtract(Tensor x){
    if(dims != x.dims)
      throw std::runtime_error("Mismatched shape in subtract");
    return add(x.neg());
  }

  Tensor mult(float x){
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] * x;
    return ret;
  }
  
  Tensor elementwise_mult(Tensor x){
    if(dims != x.dims)
      throw std::runtime_error("Mismatched shape in elementwise_mult");
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] * x.data[i];
    return ret;
  }
  
  Tensor pow(float x){
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = std::pow(data[i],x);
    return ret;
  }
  
  Tensor relu(){
    Tensor ret(dims);
    #pragma omp parallel for num_threads(8)
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] > 0 ? data[i] : 0;
    return ret;
  }

  Tensor binarilize(){
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = data[i] > 0 ? 1 : 0;
    return ret;
  }

  Tensor exp(){
    Tensor ret(dims);
    for(size_t i = 0;i < data.size();++i)
      ret.data[i] = std::exp(data[i]);
    return ret;

  }

  template <typename T> 
  void printVector(const std::vector<T>& V, const char* msg) {
      std::cout << msg << "[ ";
      for (const T& element : V) {
          std::cout << element << " ";
      }
      std::cout << "]" << std::endl;
  }

  void sparsify() 
  {
    csr_data.clear();
    csr_columns.clear();
    csr_row_pointers.clear();

    int NNZ = 0;
    csr_row_pointers.push_back(NNZ);

    for(size_t i = 0; i < dims[0]; ++i){
      for(size_t j = 0; j < dims[1]; ++j){
        if(data[i*dims[1] + j] != 0){
          csr_data.push_back(data[i*dims[1] + j]);
          csr_columns.push_back(j);
          NNZ++;
        }
      }
      csr_row_pointers.push_back(NNZ);
    }
  }

  Tensor matmul(Tensor x) {
    if (x.dims.size() != 2) {
        throw std::runtime_error("The right operand of matmul must be 2D tensors");
    }
    if (dims.size() != 2 && dims.size() != 3) {
        throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    }
    if (dims[dims.size() - 1] != x.dims[0]) {
        throw std::runtime_error("Mismatched matmul matrix dimensions");
    }

    this->sparsify();

    size_t num_csr_rows = csr_row_pointers.size() - 1;
    size_t num_dense_columns = x.dims[1];
    Tensor ret({dims[0], x.dims[1]});

    #pragma omp parallel for schedule(dynamic) num_threads(8)
    for (size_t i = 0; i < num_csr_rows; ++i) {
        size_t ret_row_index = csr_row_pointers[i];
        size_t ret_next_row_index = csr_row_pointers[i + 1];

        for (size_t j = ret_row_index; j < ret_next_row_index; ++j) {
            size_t csr_column = csr_columns[j];
            float csr_value = csr_data[j];

            for (size_t k = 0; k < num_dense_columns; ++k) {
                float dense_value = x.data[csr_column * num_dense_columns + k];
                ret.data[i * num_dense_columns + k] += csr_value * dense_value;
            }
        }
    }
    return ret;
}

  void print(){
    for(auto x : data)
      printf("%s\n",std::to_string(x).c_str());
  }

  std::vector<float> get_data(){
    return data;
  }

  std::vector<size_t> get_dims(){
    return dims;
  }
  
};
