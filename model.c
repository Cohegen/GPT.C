#include "model.h"

// Function to create the Bigram Language Model
BigramLanguageModel* create_bigram_language_model(int vocab_size, int n_embd, int block_size, int n_layer, int n_head) {
    BigramLanguageModel* model = (BigramLanguageModel*)malloc(sizeof(BigramLanguageModel));

    int token_emb_shape[] = {vocab_size, n_embd};
    model->token_embedding_table = create_tensor(token_emb_shape, 2);

    int pos_emb_shape[] = {block_size, n_embd};
    model->position_embedding_table = create_tensor(pos_emb_shape, 2);

    model->n_layers = n_layer;
    model->blocks = (Block**)malloc(n_layer * sizeof(Block*));
    for (int i = 0; i < n_layer; ++i) {
        model->blocks[i] = create_block(n_embd, n_head);
    }

    model->lm_head = create_linear_layer(n_embd, vocab_size);
    model->ln_final = create_layer_norm(n_embd);

    return model;
}

// Function to free the Bigram Language Model
void free_bigram_language_model(BigramLanguageModel* model) {
    free_tensor(model->token_embedding_table);
    free_tensor(model->position_embedding_table);
    for (int i = 0; i < model->n_layers; ++i) {
        free_block(model->blocks[i]);
    }
    free(model->blocks);
    free_linear_layer(model->lm_head);
    free_layer_norm(model->ln_final);
    free(model);
}

// Forward pass for the Bigram Language Model
Tensor* model_forward(BigramLanguageModel* model, const Tensor* idx) {
    // For simplicity, this forward pass will handle a single input sequence
    int B = idx->shape[0];
    int T = idx->shape[1];

    int tok_emb_shape[] = {B, T, model->token_embedding_table->shape[1]};
    Tensor* tok_emb = create_tensor(tok_emb_shape, 3);

    // Manual embedding lookup
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int indices[] = {b, t};
            int token_index = (int)get_tensor_value(idx, indices);
            for (int c = 0; c < tok_emb_shape[2]; ++c) {
                int src_indices[] = {token_index, c};
                int dest_indices[] = {b, t, c};
                set_tensor_value(tok_emb, dest_indices, get_tensor_value(model->token_embedding_table, src_indices));
            }
        }
    }

    // Positional embedding
    int pos_emb_shape[] = {T, model->position_embedding_table->shape[1]};
    Tensor* pos_emb = create_tensor(pos_emb_shape, 2);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < pos_emb_shape[1]; ++c) {
            int src_indices[] = {t, c};
            int dest_indices[] = {t, c};
            set_tensor_value(pos_emb, dest_indices, get_tensor_value(model->position_embedding_table, src_indices));
        }
    }

    Tensor* x = add(tok_emb, pos_emb);
    free_tensor(tok_emb);
    free_tensor(pos_emb);

    for (int i = 0; i < model->n_layers; ++i) {
        Tensor* next_x = block_forward(model->blocks[i], x);
        free_tensor(x);
        x = next_x;
    }

    // Final layer normalization
    Tensor* ln_final_out = layer_norm_forward(model->ln_final, x);
    free_tensor(x);
    x = ln_final_out;

    Tensor* logits = linear_forward(model->lm_head, x);
    free_tensor(x);

    return logits;
}

// Function to calculate cross-entropy loss
float cross_entropy_loss(const Tensor* logits, const Tensor* targets) {
    // Assuming logits are (B*T, vocab_size) and targets are (B*T)
    if (logits->n_dims != 2 || targets->n_dims != 1) {
        fprintf(stderr, "Invalid tensor dimensions for cross_entropy_loss.\n");
        return -1.0f;
    }
    if (logits->shape[0] != targets->shape[0]) {
        fprintf(stderr, "Batch sizes do not match for cross_entropy_loss.\n");
        return -1.0f;
    }

    float total_loss = 0.0f;
    int num_elements = logits->shape[0];
    int vocab_size = logits->shape[1];

    for (int i = 0; i < num_elements; ++i) {
        int target_idx = (int)targets->data[i];
        float log_softmax_val = -INFINITY;

        // Compute log_softmax for the current row of logits
        float max_logit = -INFINITY;
        for (int j = 0; j < vocab_size; ++j) {
            if (logits->data[i * vocab_size + j] > max_logit) {
                max_logit = logits->data[i * vocab_size + j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; ++j) {
            sum_exp += expf(logits->data[i * vocab_size + j] - max_logit);
        }

        log_softmax_val = (logits->data[i * vocab_size + target_idx] - max_logit) - logf(sum_exp);
        total_loss += -log_softmax_val;
    }

    return total_loss / num_elements;
}

// Function to generate new text
char* generate(BigramLanguageModel* model, Vocabulary* vocab, const char* start_text, int max_new_tokens) {
    int current_len = strlen(start_text);
    int* encoded_start = encode(start_text, vocab);

    // Create a dynamic array to hold the generated sequence
    int* generated_sequence = (int*)malloc((current_len + max_new_tokens) * sizeof(int));
    memcpy(generated_sequence, encoded_start, current_len * sizeof(int));
    free(encoded_start);

    for (int i = 0; i < max_new_tokens; ++i) {
        // Crop the sequence to the last block_size tokens
        int start_idx = (current_len > model->position_embedding_table->shape[0]) ? 
                         (current_len - model->position_embedding_table->shape[0]) : 0;
        int context_len = current_len - start_idx;

        int* context_data = (int*)malloc(context_len * sizeof(int));
        memcpy(context_data, generated_sequence + start_idx, context_len * sizeof(int));

        int context_shape[] = {1, context_len};
        Tensor* context_tensor = create_tensor(context_shape, 2);
        for(int j=0; j<context_len; ++j) {
            context_tensor->data[j] = (float)context_data[j];
        }
        free(context_data);

        Tensor* logits = model_forward(model, context_tensor);
        free_tensor(context_tensor);

        // Get the logits for the last token
        int last_token_logits_offset = (logits->shape[0] - 1) * logits->shape[1];
        Tensor* last_logits = create_tensor((int[]){1, logits->shape[1]}, 2);
        for(int j=0; j<logits->shape[1]; ++j) {
            last_logits->data[j] = logits->data[last_token_logits_offset + j];
        }
        free_tensor(logits);

        softmax(last_logits, 1);

        // Sample the next token
        float r = rand_float();
        int next_token = 0;
        float cumulative_prob = 0.0f;
        for (int j = 0; j < last_logits->shape[1]; ++j) {
            cumulative_prob += last_logits->data[j];
            if (r <= cumulative_prob) {
                next_token = j;
                break;
            }
        }
        free_tensor(last_logits);

        generated_sequence[current_len++] = next_token;
    }

    char* decoded_text = decode(generated_sequence, current_len, vocab);
    free(generated_sequence);
    return decoded_text;
}
