#include "data.h"
#include <stdlib.h>
#include <string.h>

// Function to read the entire content of a file into a string
char* read_file_content(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = (char*)malloc(length + 1);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate memory for file content\n");
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, length, file);
    buffer[length] = '\0';

    fclose(file);
    return buffer;
}

// Function to build the vocabulary from the text
Vocabulary* build_vocabulary(const char* text) {
    int text_len = strlen(text);
    char* unique_chars = (char*)malloc(text_len + 1);
    int unique_count = 0;

    for (int i = 0; i < text_len; ++i) {
        if (strchr(unique_chars, text[i]) == NULL) {
            unique_chars[unique_count++] = text[i];
        }
    }
    unique_chars[unique_count] = '\0';

    // Sort the unique characters
    for (int i = 0; i < unique_count - 1; ++i) {
        for (int j = i + 1; j < unique_count; ++j) {
            if (unique_chars[i] > unique_chars[j]) {
                char temp = unique_chars[i];
                unique_chars[i] = unique_chars[j];
                unique_chars[j] = temp;
            }
        }
    }

    Vocabulary* vocab = (Vocabulary*)malloc(sizeof(Vocabulary));
    vocab->vocab_size = unique_count;
    vocab->chars = (char**)malloc(unique_count * sizeof(char*));
    for (int i = 0; i < unique_count; ++i) {
        vocab->chars[i] = (char*)malloc(2 * sizeof(char));
        vocab->chars[i][0] = unique_chars[i];
        vocab->chars[i][1] = '\0';
    }

    free(unique_chars);
    return vocab;
}

// Function to free the vocabulary
void free_vocabulary(Vocabulary* vocab) {
    for (int i = 0; i < vocab->vocab_size; ++i) {
        free(vocab->chars[i]);
    }
    free(vocab->chars);
    free(vocab);
}

// Function to encode the text
int* encode(const char* text, Vocabulary* vocab) {
    int text_len = strlen(text);
    int* encoded = (int*)malloc(text_len * sizeof(int));
    for (int i = 0; i < text_len; ++i) {
        for (int j = 0; j < vocab->vocab_size; ++j) {
            if (text[i] == vocab->chars[j][0]) {
                encoded[i] = j;
                break;
            }
        }
    }
    return encoded;
}

// Function to decode the text
char* decode(const int* encoded_text, int len, Vocabulary* vocab) {
    char* decoded = (char*)malloc(len + 1);
    for (int i = 0; i < len; ++i) {
        decoded[i] = vocab->chars[encoded_text[i]][0];
    }
    decoded[len] = '\0';
    return decoded;
}

// Function to get a batch of data for training
void get_batch(const int* data, int data_len, int batch_size, int block_size, Tensor** x, Tensor** y) {
    int x_shape[] = {batch_size, block_size};
    int y_shape[] = {batch_size, block_size};

    *x = create_tensor(x_shape, 2);
    *y = create_tensor(y_shape, 2);

    for (int b = 0; b < batch_size; ++b) {
        int ix = rand() % (data_len - block_size);
        for (int t = 0; t < block_size; ++t) {
            int x_indices[] = {b, t};
            int y_indices[] = {b, t};
            set_tensor_value(*x, x_indices, (float)data[ix + t]);
            set_tensor_value(*y, y_indices, (float)data[ix + t + 1]);
        }
    }
}
