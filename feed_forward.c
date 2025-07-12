#include "feed_forward.h"

// Function to create a new FeedForward layer
FeedForward* create_feed_forward(int n_embd) {
    FeedForward* ffwd = (FeedForward*)malloc(sizeof(FeedForward));
    ffwd->layer1 = create_linear_layer(n_embd, 4 * n_embd);
    ffwd->layer2 = create_linear_layer(4 * n_embd, n_embd);
    return ffwd;
}

// Function to free a FeedForward layer
void free_feed_forward(FeedForward* ffwd) {
    free_linear_layer(ffwd->layer1);
    free_linear_layer(ffwd->layer2);
    free(ffwd);
}

// Forward pass for the FeedForward layer
Tensor* feed_forward_forward(FeedForward* ffwd, const Tensor* input) {
    Tensor* hidden = linear_forward(ffwd->layer1, input);

    // Apply ReLU activation
    for (int i = 0; i < hidden->size; ++i) {
        if (hidden->data[i] < 0) {
            hidden->data[i] = 0;
        }
    }

    Tensor* output = linear_forward(ffwd->layer2, hidden);
    free_tensor(hidden);
    return output;
}
