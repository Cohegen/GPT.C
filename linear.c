#include "linear.h"
#include <stdlib.h>

// Function to create a new Linear layer
Linear* create_linear_layer(int in_features, int out_features) {
    Linear* layer = (Linear*)malloc(sizeof(Linear));

    int weights_shape[] = {in_features, out_features};
    layer->weights = create_tensor(weights_shape, 2);

    int bias_shape[] = {out_features};
    layer->bias = create_tensor(bias_shape, 1);

    // Initialize weights and biases (e.g., with random values)
    // For simplicity, we'll initialize with small constant values for now
    for (int i = 0; i < layer->weights->size; ++i) {
        layer->weights->data[i] = 0.1f;
    }
    for (int i = 0; i < layer->bias->size; ++i) {
        layer->bias->data[i] = 0.01f;
    }

    return layer;
}

// Function to free a Linear layer
void free_linear_layer(Linear* layer) {
    free_tensor(layer->weights);
    free_tensor(layer->bias);
    free(layer);
}

// Function to perform the forward pass of the Linear layer
Tensor* linear_forward(const Linear* layer, const Tensor* input) {
    Tensor* output = matmul(input, layer->weights);
    if (output) {
        // Add bias
        for (int i = 0; i < output->shape[0]; ++i) {
            for (int j = 0; j < output->shape[1]; ++j) {
                int indices[] = {i, j};
                float current_value = get_tensor_value(output, indices);
                set_tensor_value(output, indices, current_value + layer->bias->data[j]);
            }
        }
    }
    return output;
}
