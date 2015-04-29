#include <stdio.h>
#include "minunit.h"
#include "../src/nlk_text.h"
#include "../src/nlk_vocabulary.h"
#include "../src/nlk_window.h"
 
int tests_run = 0;
int tests_passed = 0;


/**
 * Test writing an array to a text file and loading from it
 */
static char *
test_context_pvdbow()
{
    size_t max_word_size = NLK_LM_MAX_WORD_SIZE;
    size_t max_line_size = NLK_LM_MAX_LINE_SIZE;
    size_t zz;
    size_t par_id;
    size_t window = 10;
    size_t ctx_size = window * 2 + 1; // pvdbow
    size_t n_subsampled = 0;
    size_t line_len = 0;
    size_t n_examples = 0;

    /* allocate memory for reading from the input file */
    char **text_line = (char **) calloc(max_line_size, sizeof(char *));
    mu_assert("alloc fail", text_line != NULL);

    for(zz = 0; zz < max_line_size; zz++) {
        text_line[zz] = calloc(max_word_size, sizeof(char));
        mu_assert("alloc fail", text_line[zz] != NULL);
    }

    /* for converting to a vocabularized representation of text */
    struct nlk_vocab_t *vectorized[max_line_size];

    /* for converting a sentence to a series of training contexts */
    struct nlk_context_t **contexts = (struct nlk_context_t **) 
        malloc(max_line_size * sizeof(struct nlk_context_t *));
    mu_assert("alloc fail", contexts != NULL);

    for(zz = 0; zz < max_line_size; zz++) {
        contexts[zz] = nlk_context_create(ctx_size);
        mu_assert("alloc fail", contexts[zz] != NULL);
    }

    /* load vocab */
    struct nlk_vocab_t *vocab;
    vocab = nlk_vocab_load("data/cat.vocab", NLK_LM_MAX_WORD_SIZE);
    nlk_vocab_sort(&vocab);
    nlk_vocab_encode_huffman(&vocab);

    /* context opts */
    struct nlk_context_opts_t ctx_opts;
    nlk_context_model_opts(NLK_PVDBOW, window, &vocab, &ctx_opts);
    ctx_opts.random_windows = false;


    /* open */
    FILE *train = fopen("data/cat.txt", "rb");

    /* read line */
    nlk_read_number_line(train, text_line, &par_id, max_word_size, 
                         max_line_size);


    /* vectorize */
    line_len = nlk_vocab_vocabularize(&vocab, 0, text_line, 
                                      0, NULL, false, vectorized, 
                                      &n_subsampled); 


     /* contextify */
    n_examples = nlk_context_window(vectorized, line_len, par_id, 
                                            &ctx_opts, contexts);
    mu_assert("Wrong number of contexts", n_examples == 6);

    for(zz = 0; zz < n_examples; zz++) {
        mu_assert("Bad Context Size", contexts[zz]->size == 7);
        mu_assert("Bad Target", contexts[zz]->target->index == zz + 1);
        mu_assert("Bad Context Paragraph", contexts[zz]->window[6] == 1);
        mu_assert("Bad Context Paragraph", contexts[zz]->window[0] == 1);
        mu_assert("Bad Context Type", contexts[zz]->is_paragraph[6] == true);
        mu_assert("Bad Context Type", contexts[zz]->is_paragraph[0] == true);
        for(size_t jj = 1; jj < 6; jj++) {
            mu_assert("Bad Context Type", 
                      contexts[zz]->is_paragraph[jj] == false);
        }
    }
    
    
    for(zz = 0; zz < n_examples; zz++) {
        nlk_context_print(contexts[zz], &vocab);
    }
    


    /* compare headers */
    mu_assert("hello", 1 == 1);

    /* close shop */
    fclose(train);

    return 0;
}

/**
 * Function that runs all tests
 */
static char *
all_tests() {
    mu_run_test(test_context_pvdbow);
    return 0;
}
 
int main(int argc, char **argv) {
    printf("---------------------------------------------------------\n");
    printf("Serialization Tests\n");
    printf("---------------------------------------------------------\n");
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests Passed: %d/%d\n", tests_run, tests_passed);
 
    return result != 0;
}
