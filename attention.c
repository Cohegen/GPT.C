#include "attention.h"
#include <math.h>

// Function to create a single attention head
Head* create_head(int n_embd, int head_size) {
    Head* head = (Head*)malloc(sizeof(Head));
    head->key = create_linear_layer(n_embd, head_size);
    head->query = create_linear_layer(n_embd, head_size);
    head->value = create_linear_layer(n_embd, head_size);

    int tril_shape[] = {1, 1, 1}; // Placeholder shape
    head->tril = create_tensor(tril_shape, 3);

    head->dropout = 0.0; // Dropout is not implemented
    return head;
}

// Function to free a single attention head
void free_head(Head* head) {
    free_linear_layer(head->key);
    free_linear_layer(head->query);
    free_linear_layer(head->value);
    free_tensor(head->tril);
    free(head);
}

// Forward pass for a single attention head
Tensor* head_forward(Head* head, const Tensor* x) {
    Tensor* k = linear_forward(head->key, x);
    Tensor* q = linear_forward(head->query, x);

    Tensor* k_transposed = transpose(k);
    Tensor* wei = matmul(q, k_transposed);
    free_tensor(k_transposed);

    scale(wei, 1.0f / sqrtf(k->shape[1]));

    // Masking is not fully implemented here, but this is where it would be applied
    // wei = masked_fill(wei, head->tril == 0, -INFINITY);

    softmax(wei, wei->n_dims - 1);

    Tensor* v = linear_forward(head->value, x);
    Tensor* out = matmul(wei, v);

    free_tensor(k);
    free_tensor(q);
    free_tensor(v);
    free_tensor(wei);

    return out;
}

// Function to create a multi-head attention module
MultiHeadAttention* create_multi_head_attention(int n_embd, int n_heads) {
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    mha->n_heads = n_heads;
    mha->heads = (Head**)malloc(n_heads * sizeof(Head*));
    int head_size = n_embd / n_heads;
    for (int i = 0; i < n_heads; ++i) {
        mha->heads[i] = create_head(n_embd, head_size);
    }
    mha->proj = create_linear_layer(n_embd, n_embd);
    return mha;
}

// Function to free a multi-head attention module
void free_multi_head_attention(MultiHeadAttention* mha) {
    for (int i = 0; i < mha->n_heads; ++i) {
        free_head(mha->heads[i]);
    }
    free(mha->heads);
    free_linear_layer(mha->proj);
    free(mha);
}

// Forward pass for multi-head attention
Tensor* multi_head_attention_forward(MultiHeadAttention* mha, const Tensor* x) {
    Tensor* head_outputs[mha->n_heads];
    for (int i = 0; i < mha->n_heads; ++i) {
        head_outputs[i] = head_forward(mha->heads[i], x);
    }

    Tensor* concatenated = concatenate((const Tensor**)head_outputs, mha->n_heads, x->n_dims - 1);

    for (int i = 0; i < mha->n_heads; ++i) {
        free_tensor(head_outputs[i]);
    }

    Tensor* out = linear_forward(mha->proj, concatenated);
    free_tensor(concatenated);

    return out;
}
