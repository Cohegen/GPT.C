#include "layer_norm.h"
#include <math.h>
#include <stdio.h>

// Function to create a new LayerNorm layer
LayerNorm* create_layer_norm(int normalized_shape) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    int shape[] = {normalized_shape};
    ln->gamma = create_tensor(shape, 1);
    ln->beta = create_tensor(shape, 1);
    ln->epsilon = 1e-5; // Default epsilon

    // Initialize gamma to 1s and beta to 0s
    for (int i = 0; i < ln->gamma->size; ++i) {
        ln->gamma->data[i] = 1.0f;
        ln->beta->data[i] = 0.0f;
    }
    return ln;
}

// Function to free a LayerNorm layer
void free_layer_norm(LayerNorm* ln) {
    free_tensor(ln->gamma);
    free_tensor(ln->beta);
    free(ln);
}

// Forward pass for Layer Normalization
Tensor* layer_norm_forward(LayerNorm* ln, const Tensor* input) {
    // Assuming input is 2D (batch_size, features)
    if (input->n_dims != 2) {
        fprintf(stderr, "LayerNorm currently only supports 2D tensors.\n");
        return NULL;
    }

    int batch_size = input->shape[0];
    int features = input->shape[1];

    Tensor* output = create_tensor(input->shape, input->n_dims);

    for (int i = 0; i < batch_size; ++i) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int j = 0; j < features; ++j) {
            int indices[] = {i, j};
            float val = get_tensor_value(input, indices);
            sum += val;
            sum_sq += val * val;
        }

        float mean = sum / features;
        float variance = (sum_sq / features) - (mean * mean);
        float std_dev = sqrtf(variance + ln->epsilon);

        for (int j = 0; j < features; ++j) {
            int indices[] = {i, j};
            float val = get_tensor_value(input, indices);
            float normalized_val = (val - mean) / std_dev;
            float scaled_shifted_val = normalized_val * ln->gamma->data[j] + ln->beta->data[j];
            set_tensor_value(output, indices, scaled_shifted_val);
        }
    }
    return output;
}
