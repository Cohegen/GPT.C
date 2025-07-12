#include "tensor.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// Function to create a new tensor
Tensor* create_tensor(const int* shape, int n_dims) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->n_dims = n_dims;
    tensor->shape = (int*)malloc(n_dims * sizeof(int));
    memcpy(tensor->shape, shape, n_dims * sizeof(int));

    tensor->size = 1;
    for (int i = 0; i < n_dims; ++i) {
        tensor->size *= shape[i];
    }

    tensor->data = (float*)calloc(tensor->size, sizeof(float));
    return tensor;
}

// Function to free a tensor
void free_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor->shape);
    free(tensor);
}

// Helper function to get the index in the flat data array
int get_data_index(const Tensor* tensor, const int* indices) {
    int index = 0;
    int stride = 1;
    for (int i = tensor->n_dims - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }
    return index;
}

// Function to get a value from the tensor
float get_tensor_value(const Tensor* tensor, const int* indices) {
    return tensor->data[get_data_index(tensor, indices)];
}

// Function to set a value in the tensor
void set_tensor_value(Tensor* tensor, const int* indices, float value) {
    tensor->data[get_data_index(tensor, indices)] = value;
}

// Function to print the tensor (for debugging)
void print_tensor(const Tensor* tensor) {
    printf("Tensor (shape: [ ");
    for (int i = 0; i < tensor->n_dims; ++i) {
        printf("%d ", tensor->shape[i]);
    }
    printf("]):\n");

    // This is a simplified print for 1D/2D tensors
    if (tensor->n_dims == 1) {
        for (int i = 0; i < tensor->shape[0]; ++i) {
            printf("%.4f ", tensor->data[i]);
        }
        printf("\n");
    } else if (tensor->n_dims == 2) {
        for (int i = 0; i < tensor->shape[0]; ++i) {
            for (int j = 0; j < tensor->shape[1]; ++j) {
                int indices[] = {i, j};
                printf("%.4f ", get_tensor_value(tensor, indices));
            }
            printf("\n");
        }
    }
}

// Function to generate a random float between 0 and 1
float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

// Function for matrix multiplication (for 2D tensors)
Tensor* matmul(const Tensor* a, const Tensor* b) {
    if (a->n_dims != 2 || b->n_dims != 2) {        fprintf(stderr, "Matrix multiplication is only implemented for 2D tensors.\n");        return NULL;    }    if (a->shape[1] != b->shape[0]) {        fprintf(stderr, "Matrix dimensions are not compatible for multiplication.\n");        return NULL;    }    int new_shape[] = {a->shape[0], b->shape[1]};    Tensor* result = create_tensor(new_shape, 2);    for (int i = 0; i < a->shape[0]; ++i) {        for (int j = 0; j < b->shape[1]; ++j) {            float sum = 0.0f;            for (int k = 0; k < a->shape[1]; ++k) {                int a_indices[] = {i, k};                int b_indices[] = {k, j};                sum += get_tensor_value(a, a_indices) * get_tensor_value(b, b_indices);            }            int result_indices[] = {i, j};            set_tensor_value(result, result_indices, sum);        }    }    return result;}

// Function to add two tensors
Tensor* add(const Tensor* a, const Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Tensors must have the same size for addition\n");
        return NULL;
    }
    Tensor* result = create_tensor(a->shape, a->n_dims);
    for (int i = 0; i < a->size; ++i) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

// Function to apply softmax to a tensor
// Function to apply softmax to a tensor along a specific dimension
void softmax(Tensor* tensor, int dim) {
    if (dim < 0 || dim >= tensor->n_dims) {
        fprintf(stderr, "Invalid dimension for softmax\n");
        return;
    }

    // This is a simplified softmax for a specific dimension (e.g., the last one)
    // A more general implementation would be needed for arbitrary dimensions
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= tensor->shape[i];
    }
    int inner_size = tensor->shape[dim];

    for (int i = 0; i < outer_size; ++i) {
        float max_val = -INFINITY;
        int offset = i * inner_size;

        for (int j = 0; j < inner_size; ++j) {
            if (tensor->data[offset + j] > max_val) {
                max_val = tensor->data[offset + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < inner_size; ++j) {
            tensor->data[offset + j] = expf(tensor->data[offset + j] - max_val);
            sum += tensor->data[offset + j];
        }

        for (int j = 0; j < inner_size; ++j) {
            tensor->data[offset + j] /= sum;
        }
    }
}

// Function to transpose a 2D tensor
Tensor* transpose(const Tensor* a) {
    if (a->n_dims != 2) {
        fprintf(stderr, "Transpose is only implemented for 2D tensors.\n");
        return NULL;
    }
    int new_shape[] = {a->shape[1], a->shape[0]};
    Tensor* result = create_tensor(new_shape, 2);
    for (int i = 0; i < a->shape[0]; ++i) {
        for (int j = 0; j < a->shape[1]; ++j) {
            int old_indices[] = {i, j};
            int new_indices[] = {j, i};
            set_tensor_value(result, new_indices, get_tensor_value(a, old_indices));
        }
    }
    return result;
}

// Function to scale a tensor by a scalar value
void scale(Tensor* tensor, float scalar) {
    for (int i = 0; i < tensor->size; ++i) {
        tensor->data[i] *= scalar;
    }
}

// Function to concatenate tensors along a specific dimension
Tensor* concatenate(const Tensor** tensors, int n_tensors, int dim) {
    // For simplicity, this implementation will concatenate along the last dimension
    if (dim != tensors[0]->n_dims - 1) {
        fprintf(stderr, "Concatenation is only implemented for the last dimension.\n");
        return NULL;
    }

    int new_dim_size = 0;
    for (int i = 0; i < n_tensors; ++i) {
        new_dim_size += tensors[i]->shape[dim];
    }

    int new_shape[tensors[0]->n_dims];
    memcpy(new_shape, tensors[0]->shape, tensors[0]->n_dims * sizeof(int));
    new_shape[dim] = new_dim_size;

    Tensor* result = create_tensor(new_shape, tensors[0]->n_dims);

    int offset = 0;
    for (int i = 0; i < n_tensors; ++i) {
        // This is a simplified copy and assumes a 3D tensor (B, T, C)
        for (int b = 0; b < tensors[i]->shape[0]; ++b) {
            for (int t = 0; t < tensors[i]->shape[1]; ++t) {
                for (int c = 0; c < tensors[i]->shape[2]; ++c) {
                    int src_indices[] = {b, t, c};
                    int dest_indices[] = {b, t, offset + c};
                    set_tensor_value(result, dest_indices, get_tensor_value(tensors[i], src_indices));
                }
            }
        }
        offset += tensors[i]->shape[dim];
    }

    return result;
}
