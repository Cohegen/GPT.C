#ifndef BLOCK_H
#define BLOCK_H

#include "tensor.h"
#include "attention.h"
#include "feed_forward.h"
#include "layer_norm.h"

// A single Transformer Block
typedef struct {
    MultiHeadAttention* sa;
    FeedForward* ffwd;
    LayerNorm* ln1;
    LayerNorm* ln2;
} Block;

// Function prototypes
Block* create_block(int n_embd, int n_head);
void free_block(Block* block);
Tensor* block_forward(Block* block, const Tensor* x);

#endif // BLOCK_H
