#include "block.h"

// Function to create a new Transformer Block
Block* create_block(int n_embd, int n_head) {
    Block* block = (Block*)malloc(sizeof(Block));
    block->sa = create_multi_head_attention(n_embd, n_head);
    block->ffwd = create_feed_forward(n_embd);
    block->ln1 = create_layer_norm(n_embd);
    block->ln2 = create_layer_norm(n_embd);
    return block;
}

// Function to free a Transformer Block
void free_block(Block* block) {
    free_multi_head_attention(block->sa);
    free_feed_forward(block->ffwd);
    free_layer_norm(block->ln1);
    free_layer_norm(block->ln2);
    free(block);
}

// Forward pass for the Transformer Block
Tensor* block_forward(Block* block, const Tensor* x) {
    // Self-attention with residual connection and layer normalization
    Tensor* ln1_out = layer_norm_forward(block->ln1, x);
    Tensor* sa_out = multi_head_attention_forward(block->sa, ln1_out);
    Tensor* x1 = add(x, sa_out);
    free_tensor(ln1_out);
    free_tensor(sa_out);

    // Feed-forward with residual connection and layer normalization
    Tensor* ln2_out = layer_norm_forward(block->ln2, x1);
    Tensor* ffwd_out = feed_forward_forward(block->ffwd, ln2_out);
    Tensor* x2 = add(x1, ffwd_out);
    free_tensor(ln2_out);
    free_tensor(ffwd_out);
    free_tensor(x1);

    return x2;
}
