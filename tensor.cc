#include<stdio.h>
#include<string.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include"tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(hw3tensor,m){
  py::class_<Tensor>(m,"Tensor")
    .def(py::init<std::vector<size_t>>())
    .def(py::init<std::vector<size_t>,std::vector<std::vector<size_t>>,std::vector<float>>())
    .def_static("ones",&Tensor::ones)
    .def("index",&Tensor::index)
    .def("reshape",&Tensor::reshape)
    .def("transpose",&Tensor::transpose)
    .def("neg",&Tensor::neg)
    .def("reciprocal",&Tensor::reciprocal)
    .def("add",&Tensor::add)
    .def("subtract",&Tensor::subtract)
    .def("mult",&Tensor::mult)
    .def("elementwise_mult",&Tensor::elementwise_mult)
    .def("pow",&Tensor::pow)
    .def("relu",&Tensor::relu)
    .def("binarilize",&Tensor::binarilize)
    .def("exp",&Tensor::exp)
    .def("matmul",&Tensor::matmul)
    .def("print",&Tensor::print)
    .def("get_data",&Tensor::get_data)
    .def("get_dims",&Tensor::get_dims);
}
