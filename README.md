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

Understanding the Compressed Sparse Row (CSR) Format
----------------------------------------------------

### What is a Sparse Matrix?

In many real-world applications---such as natural language processing (NLP), recommender systems, and graph algorithms---matrices often contain a large number of zero elements. These matrices are known as **sparse matrices**. When a matrix is extremely large and mostly composed of zeros, storing it in a standard dense format can be:

-   **Memory-inefficient**, since we allocate space for all elements (including zeros).
-   **Computationally wasteful**, as standard matrix multiplication algorithms will still iterate over zero elements.

### Key Idea Behind CSR

The **Compressed Sparse Row (CSR)** format is one of the most commonly used representations for sparse matrices because it balances ease of indexing with memory savings. Instead of storing every element, CSR focuses on storing **only non-zero elements** and the necessary indexing metadata.

A matrix AA in CSR format is typically represented with three arrays:

1.  **Values (`val`)**:\
    A 1D array that holds all the non-zero values in the matrix, ordered row by row.

2.  **Column Indices (`col_idx`)**:\
    A 1D array that stores the column index of each non-zero value in `val`. The `i`-th entry of `col_idx` corresponds to the column of the `i`-th entry in `val`.

3.  **Row Offsets (`row_ptr`)**:\
    A 1D array that indicates where each row starts and ends in the `val` and `col_idx` arrays.

    -   `row_ptr[r]` gives the index in `val`/`col_idx` where the **r-th** row begins.
    -   `row_ptr[r+1] - row_ptr[r]` gives the number of non-zero elements in the **r-th** row.

For example, consider a small 4×4 matrix:

0030\
5000\
0600\
0007

-   **Values (`val`)** = [3,5,6,7][3,5,6,7]
-   **Column Indices (`col_idx`)** = [2,0,1,3][2,0,1,3]
-   **Row Offsets (`row_ptr`)** = [0,1,2,3,4][0,1,2,3,4]

### Why CSR is More Efficient than a Standard Dense MatMul

1.  **Reduced Memory Footprint**

    -   In a dense representation, you must store every element---including zeros. For an m×nm×n matrix, that requires m×nm×n space.
    -   In CSR, you only store the non-zero elements and their row/column indices, saving a significant amount of memory when the matrix is large and sparse.
2.  **Faster Iteration Over Non-Zero Elements**

    -   Standard dense algorithms process every entry, including zeros.
    -   With CSR, computations skip the zero elements automatically since only non-zero entries and their indexes are stored.
3.  **Efficient Row Access**

    -   CSR compresses by rows. For row rr, you simply iterate from `row_ptr[r]` to `row_ptr[r+1]`.
    -   This organization can improve cache locality and thus speed up computations.
4.  **Parallelization-Friendly**

    -   Many sparse matrix operations can be parallelized on a row-by-row basis.
    -   OpenMP or other parallel frameworks can efficiently split the workload across threads.
5.  **Accelerated Matrix Multiplication**

    -   Standard dense matrix multiplication has a complexity of O(m×n×p)O(m×n×p) for a (m×n)(m×n)matrix multiplied by a (n×p)(n×p) matrix.
    -   CSR-based multiplication depends on the number of non-zero elements, nnznnz. When nnz≪(m×n)nnz≪(m×n), the performance gains can be substantial.

* * * * *

Conclusion
----------

The **CSR format** is pivotal for optimizing sparse matrix storage and computation. By focusing on non-zero values, CSR greatly reduces memory usage and accelerates key operations like sparse-dense matrix multiplication, especially when the sparse matrix is large and has a relatively small number of non-zero elements.

**Key Takeaways**:

-   CSR stores only non-zero elements and minimal indexing metadata.
-   It enables rapid row-wise access and efficient parallelization.
-   Sparse-dense multiplication benefits significantly in terms of speed and memory footprint compared to a standard dense approach.

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
