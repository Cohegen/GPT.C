#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

// A basic Tensor structure
typedef struct {
    float* data;    // Pointer to the data
    int* shape;     // Array representing the dimensions of the tensor
    int n_dims;     // Number of dimensions
    int size;       // Total number of elements
} Tensor;

// Function prototypes for tensor operations
Tensor* create_tensor(const int* shape, int n_dims);
void free_tensor(Tensor* tensor);
float get_tensor_value(const Tensor* tensor, const int* indices);
void set_tensor_value(Tensor* tensor, const int* indices, float value);
void print_tensor(const Tensor* tensor);
float rand_float();

// Mathematical operations
Tensor* matmul(const Tensor* a, const Tensor* b);
Tensor* add(const Tensor* a, const Tensor* b);
void softmax(Tensor* tensor, int dim);
Tensor* transpose(const Tensor* a);
void scale(Tensor* tensor, float scalar);
Tensor* concatenate(const Tensor** tensors, int n_tensors, int dim);

#endif // TENSOR_H
