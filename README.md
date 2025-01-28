# CSR-Based Sparse-Dense Matrix Multiplication

This project implements a robust framework for performing **Sparse-Dense Matrix Multiplication** using the **Compressed Sparse Row (CSR)** format. It is designed for efficient matrix computations with a focus on optimizing performance and memory usage.

## Features

- **Compressed Sparse Row (CSR) Format**:
  - Efficient representation of sparse matrices.
  - Reduces memory consumption by storing only non-zero elements and their indices.

- **Dense Tensor Operations**:
  - Support for basic tensor manipulations such as addition, subtraction, scalar multiplication, and reshaping.
  - Ability to handle multi-dimensional tensor data.

- **Sparse-Dense Matrix Multiplication**:
  - Sparse matrix (in CSR format) multiplied with dense matrices.
  - Optimized for parallel computation using OpenMP.

- **Advanced Functionalities**:
  - Element-wise operations like negation, reciprocal, exponential, and ReLU.
  - Tensor transformations such as transpose and reshape.
  - Binarization and power operations for advanced data manipulation.

## File Structure

### Core Code

- **`tensor.h`**:
  - Contains the main `Tensor` class for tensor operations.
  - Implements functions for:
    - Tensor initialization (dense and sparse).
    - Converting dense tensors to CSR format.
    - Sparse-Dense matrix multiplication.
    - Utility functions like transpose, add, subtract, reshape, and element-wise operations.

- **`tensor.cc`**:
  - Entry point for testing the `Tensor` class.
  - Demonstrates tensor creation, sparsification, and matrix multiplication.

### Python Scripts

- **`sample1.py`, `sample2.py`, `sample3.py`**:
  - Provide utility functions or test cases for the C++ implementation.
  - Use Python for data preprocessing, visualization, or validating results.

## Usage

### Compilation

The project uses a C++ compiler with OpenMP support. Compile the project with the following command:

```bash
g++ -std=c++17 -fopenmp tensor.cc -o tensor
