#ifndef DATA_H
#define DATA_H

#include <stdio.h>

// Structure to hold the vocabulary and its size
typedef struct {
    char** chars;
    int vocab_size;
} Vocabulary;

// Structure for the tokenizer
typedef struct {
    Vocabulary* vocab;
    // Add mappings if necessary, though for char-level it's simple
} Tokenizer;

// Function prototypes
Vocabulary* build_vocabulary(const char* text);
void free_vocabulary(Vocabulary* vocab);
int* encode(const char* text, Vocabulary* vocab);
char* decode(const int* encoded_text, int len, Vocabulary* vocab);
char* read_file_content(const char* filepath);

// Data batching function
void get_batch(const int* data, int data_len, int batch_size, int block_size, Tensor** x, Tensor** y);

#endif // DATA_H
