#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "block.h"
#include "data.h"

// The main Bigram Language Model
typedef struct {
    Tensor* token_embedding_table;
    Tensor* position_embedding_table;
    Block** blocks;
    int n_layers;
    Linear* lm_head;
    LayerNorm* ln_final;
} BigramLanguageModel;

// Function prototypes
BigramLanguageModel* create_bigram_language_model(int vocab_size, int n_embd, int block_size, int n_layer, int n_head);
void free_bigram_language_model(BigramLanguageModel* model);
Tensor* model_forward(BigramLanguageModel* model, const Tensor* idx);
float cross_entropy_loss(const Tensor* logits, const Tensor* targets);
char* generate(BigramLanguageModel* model, Vocabulary* vocab, const char* start_text, int max_new_tokens);

#endif // MODEL_H
