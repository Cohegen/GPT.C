#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"

// A simple Layer Normalization layer
typedef struct {
    Tensor* gamma; // Scale parameter
    Tensor* beta;  // Shift parameter
    float epsilon; // Small value to avoid division by zero
} LayerNorm;

// Function prototypes
LayerNorm* create_layer_norm(int normalized_shape);
void free_layer_norm(LayerNorm* ln);
Tensor* layer_norm_forward(LayerNorm* ln, const Tensor* input);

#endif // LAYER_NORM_H
