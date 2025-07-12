#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "tensor.h"
#include "linear.h"

// A simple FeedForward layer
typedef struct {
    Linear* layer1;
    Linear* layer2;
    // ReLU activation is applied in the forward pass
} FeedForward;

// Function prototypes
FeedForward* create_feed_forward(int n_embd);
void free_feed_forward(FeedForward* ffwd);
Tensor* feed_forward_forward(FeedForward* ffwd, const Tensor* input);

#endif // FEED_FORWARD_H
