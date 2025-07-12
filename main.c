#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "data.h"
#include "tensor.h"
#include "model.h"

// Parameters (matching Python script for conceptual consistency)
#define BATCH_SIZE 64
#define BLOCK_SIZE 128
#define MAX_ITERS 100 // Reduced for quick demonstration
#define EVAL_INTERVAL 10
#define N_EMBD 384
#define N_HEAD 6
#define N_LAYER 6

int main() {
    srand(time(NULL)); // Initialize random seed for generation

    // Read the content of the text file
    char* raw_text = read_file_content("pride_and_prejudice.txt");
    if (!raw_text) {
        return 1; // Exit if file reading failed
    }

    // Build the vocabulary
    Vocabulary* vocab = build_vocabulary(raw_text);
    int vocab_size = vocab->vocab_size;
    printf("Vocabulary Size: %d\n", vocab_size);

    // Encode the entire text dataset
    int* encoded_data = encode(raw_text, vocab);
    int data_len = strlen(raw_text);

    // Split data into train and validation sets (conceptually)
    int n = (int)(0.9 * data_len);
    int* train_data = encoded_data; // For simplicity, use full data as train
    int train_data_len = data_len;
    // int* val_data = encoded_data + n; // Not used in this placeholder
    // int val_data_len = data_len - n;

    // Create the model
    BigramLanguageModel* model = create_bigram_language_model(vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER, N_HEAD);

    printf("\nStarting placeholder training loop...\n");
    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        if (iter % EVAL_INTERVAL == 0) {
            // In a real scenario, you'd evaluate on validation set here
            printf("Step %d: (Placeholder) Loss calculation...\n", iter);
        }

        // Get a batch of data
        Tensor* xb = NULL;
        Tensor* yb = NULL;
        get_batch(train_data, train_data_len, BATCH_SIZE, BLOCK_SIZE, &xb, &yb);

        // Perform forward pass
        Tensor* logits = model_forward(model, xb);

        // Reshape yb for loss calculation (B*T, 1) -> (B*T)
        int target_flat_shape[] = {BATCH_SIZE * BLOCK_SIZE};
        Tensor* yb_flat = create_tensor(target_flat_shape, 1);
        for(int i=0; i < BATCH_SIZE * BLOCK_SIZE; ++i) {
            yb_flat->data[i] = yb->data[i];
        }

        // Reshape logits for loss calculation (B, T, vocab_size) -> (B*T, vocab_size)
        int logits_flat_shape[] = {BATCH_SIZE * BLOCK_SIZE, vocab_size};
        Tensor* logits_flat = create_tensor(logits_flat_shape, 2);
        memcpy(logits_flat->data, logits->data, logits->size * sizeof(float));

        // Calculate loss
        float loss = cross_entropy_loss(logits_flat, yb_flat);
        printf("  Loss: %.4f\n", loss);

        // In a real scenario, backpropagation and optimizer.step() would go here

        // Clean up batch tensors
        free_tensor(xb);
        free_tensor(yb);
        free_tensor(logits);
        free_tensor(yb_flat);
        free_tensor(logits_flat);
    }
    printf("Placeholder training loop finished.\n");

    // Generate text after (conceptual) training
    const char* start_text = "The ";
    int max_new_tokens = 100;
    char* generated_text = generate(model, vocab, start_text, max_new_tokens);
    printf("\nGenerated Text:\n%s\n", generated_text);

    // Clean up all allocated memory
    free(raw_text);
    free(encoded_data);
    free_vocabulary(vocab);
    free_bigram_language_model(model);
    free(generated_text);

    return 0;
}