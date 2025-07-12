#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "linear.h"

// A single head of self-attention
typedef struct {
    Linear* key;
    Linear* query;
    Linear* value;
    Tensor* tril;
    float dropout; // Dropout is not implemented, but included for completeness
} Head;

// Multi-head attention module
typedef struct {
    Head** heads;
    int n_heads;
    Linear* proj;
} MultiHeadAttention;

// Function prototypes
Head* create_head(int n_embd, int head_size);
void free_head(Head* head);
Tensor* head_forward(Head* head, const Tensor* x);

MultiHeadAttention* create_multi_head_attention(int n_embd, int n_heads);
void free_multi_head_attention(MultiHeadAttention* mha);
Tensor* multi_head_attention_forward(MultiHeadAttention* mha, const Tensor* x);

#endif // ATTENTION_H
