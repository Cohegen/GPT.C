#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

// A simple Linear layer structure
typedef struct {
    Tensor* weights;
    Tensor* bias;
} Linear;

// Function prototypes for the Linear layer
Linear* create_linear_layer(int in_features, int out_features);
void free_linear_layer(Linear* layer);
Tensor* linear_forward(const Linear* layer, const Tensor* input);

#endif // LINEAR_H
