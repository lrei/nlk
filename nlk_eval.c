/******************************************************************************
* NLK - Neural Language Kit
*
* Copyright (c) 2014 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to 
* deal in the Software without restriction, including without limitation the 
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*****************************************************************************/


/** @file nlk_eval.c
* Evaluation functions
*/


#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "nlk_text.h"
#include "nlk_vocabulary.h"
#include "nlk_array.h"
#include "nlk_err.h"
#include "nlk_eval.h"


/** 
 * Parses a line of a word relation test set
 *
 * @param vocab         the vocabulary
 * @param lower_words   convert words in test set to lower case
 * @param line          the line string to parse
 * @param test          the parsed test from the line - result if return = true
 *
 * @return false on failure, true on success
 */
bool
__nlk_read_question_line(struct nlk_vocab_t **vocab, bool lower_words, 
                         char *line, struct nlk_analogy_test_t *test)
{
    char *word;
    struct nlk_vocab_t *vi;
    size_t ii = 0;

    if(vocab == NULL) {
        return false;
    }

    do {
        /* tokenize string */
        if(ii == 0) {
            word = strtok(line, " ");
        } else if(ii < 3) {
            word = strtok(NULL, " ");
        } else {
            word = strtok(NULL, "\n");
        }
        if(word == NULL) {
            return false;   /* something is wrong, fail */
        }
        
        /* to lower case if necessary */
        if(lower_words) {
            nlk_text_lower(word, NULL); 
        }
        
        /* find word in voculary */
        vi = nlk_vocab_find(vocab, word); 
        if(vi == NULL) {
            return false;   /* if any word is not in vocabulary, fail */
        }
        
        /* assign value to test structure */
        if(ii < 3) {
            test->question[ii] = vi; 
        } else {
            test->answer = vi;
        }

        ii++;
    } while(ii < 4);
    
    return true;
}

/** 
 * Read/Parse a word analogy test file
 *
 * @param filepath      file path of the test file
 * @param vocab         the vocabulary
 * @param lower_words   convert words in test set to lower case
 * @param total_tests   will be overwritten with the total number of tests read    
 *
 * @return an array of word analogy tests
 */
struct nlk_analogy_test_t *
__nlk_read_analogy_test_file(const char *filepath, struct nlk_vocab_t **vocab,
                             const bool lower_words, size_t *total_tests)
{
    char line[NLK_WORD_REL_MAX_LINE_SIZE];
    char *fr;
    struct nlk_analogy_test_t *tests;    /* array that will contain all test cases */
    void *re_tests;
    size_t alloc_size = 0;      /* allocated size of the tests array  */
    size_t test_number;         /* number of tests read into array */

    FILE *in = fopen(filepath, "rb");
    if (in == NULL) {
        NLK_ERROR_NULL(strerror(errno), errno);
        /* unreachable */
    }

    /* alloc */
    alloc_size = NLK_WORD_REL_DEFAULT_SIZE;
    tests = (struct nlk_analogy_test_t *) calloc(alloc_size, 
                                            sizeof(struct nlk_analogy_test_t));
    if(tests == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for the test set", 
                       NLK_ENOMEM);
        /* unreachable */
    }

    /* 
     * read the entire file into memory first 
     */
    *total_tests = 0;
    test_number = 0;
    while(!feof(in)) {
        /* read line */
        fr = fgets(line, NLK_WORD_REL_MAX_LINE_SIZE, in);
        if(fr == NULL) {
            break;
        }
        if(line[0] == ':') { /* 'header' line */
            continue;
        }

        /* parse a single test case, if successful add to tests array */
        if(__nlk_read_question_line(vocab, lower_words, line,
                                    &tests[test_number])) {
            test_number++;
        }

        /* check if we need more memory for tests */
        if(test_number == alloc_size) {
            re_tests = realloc(tests, 2 * alloc_size);
            if(re_tests == NULL) {
                free(tests);
                NLK_ERROR_NULL("failed to allocate memory for the test set", 
                               NLK_ENOMEM);
                /* unreachable */
            }
        } /* memory reallocated if it was necessary */
    }

    fclose(in);
    *total_tests = test_number;
    return tests;
}


/**
 * Uses a word relation test set to evaluate the quality of word vectors.
 *
 * The test is described in
 *
 * "Efficient Estimation of Word Representations in Vector Space. 
 * Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
 * In Proceedings of Workshop at ICLR, 2013. 
 *
 * And a test set is available at https://code.google.com/p/word2vec/
 *
 * @param filepath      file path of the test file
 * @param vocab         the vocabulary
 * @param weights       the weights matrix containing the representation of the
 *                      words in the vocabulary
 * @param limit         limit for the number of words in the vocabulary
 * @param lower_words   convert words in test set to lower case
 * @param accuracy      the total accuracy (return value)
 *
 * @return NLK_SUCCESS or NLK_FAILURE
 *
 * @note
 * This function ignores tests with OOV words. It is mostly meant to be 
 * used to monitor the progress of training a word representation.
 * @endnote
 */
int
nlk_eval_on_questions(const char *filepath, struct nlk_vocab_t **vocab,
                      const NLK_ARRAY *weights, const size_t limit, 
                      const bool lower_words, nlk_real *accuracy)
{
    struct nlk_analogy_test_t *tests; /* will contain all test cases */
    size_t total_tests;

    size_t _limit;
    NLK_ARRAY *weights_norm;    /* for the normalized copy of the weights */
    size_t correct = 0;
    size_t executed = 0;

    /* read file */
    tests = __nlk_read_analogy_test_file(filepath, vocab, lower_words, 
                                         &total_tests);
    if(tests == NULL) {
        return NLK_FAILURE;
    }

    /* normalize weights to make distance calculations easier */
    weights_norm = nlk_array_create_copy(weights, 0);
    nlk_array_normalize_row_vectors(weights_norm);


    *accuracy = 0;
    if(limit == 0) {
        _limit = weights_norm->rows;
    } else {
        _limit = limit;
    }

    /*
     * perform the tests 
     */
#pragma omp parallel reduction(+ : correct) reduction(+ : executed)
{
    NLK_ARRAY *predicted = nlk_array_create(1, weights->cols);
    NLK_ARRAY *sub = nlk_array_create(1, weights->cols);
    NLK_ARRAY *add = nlk_array_create(1, weights->cols);
    NLK_ARRAY *word_vector = nlk_array_create(1, weights->cols);
    struct nlk_analogy_test_t *test;     /* iteration test case*/
    

#pragma omp for 
    for(size_t test_number = 0; test_number < total_tests; test_number++) {
        nlk_real similarity = 0;
        nlk_real best_similarity = 0;
        size_t most_similar = 0;

        test = &tests[test_number];

        /* if any of the words is not in the limited vocab, skip */
        if(test->answer->index > _limit
           || test->question[0]->index > _limit
           || test->question[1]->index > _limit
           || test->question[2]->index > _limit) {
            continue;
        }

        /* vector for the second word in test: word_vector2 */
        nlk_array_copy_row(predicted, 0, weights_norm, 
                           test->question[1]->index);

        /* word_vector1 (vector for first word)  */
        nlk_array_copy_row(sub, 0, weights_norm, test->question[0]->index);
        /* word_vector2 - word_vector1 */
        nlk_array_scale(-1.0, sub);
        nlk_array_add(sub, predicted);
        
        /* word_vector3 (vector for the third word) */
        nlk_array_copy_row(add, 0, weights_norm, test->question[2]->index);
        /* word_vector2 - word_vector1  + word_vector3 */
        nlk_array_add(add, predicted);
        
        /* find the closest vector to *predicted* in the weights matrix */
       for(size_t word_index = 0; word_index < _limit; word_index++) {
            /* ignore words in the test */
            if(word_index == test->question[0]->index
               || word_index == test->question[1]->index 
               || word_index == test->question[2]->index) {
                continue;
            }
            /* compute similarity */
            nlk_array_copy_row(word_vector, 0, weights_norm, word_index);
            similarity = nlk_array_dot(word_vector, predicted, 1);
            /* check if it is better than previous best */
            if(similarity > best_similarity) {
                best_similarity = similarity;
                most_similar = word_index;
            }
        }   /* end of find closes word */
        /* check to see of the closest word found is the target word */
       if(test->answer->index == most_similar) {
            correct++;
        }
        executed++;
    }

    /* cleanup */
    nlk_array_free(word_vector);
    nlk_array_free(predicted);
    nlk_array_free(sub);
    nlk_array_free(add);
} /* END OF PARALLEL BLOCk */
    free(tests);
    nlk_array_free(weights_norm);

    /* result */
    *accuracy = correct / (nlk_real) executed;

    return NLK_SUCCESS;
}

int
nlk_eval_on_paraphrases(const char *test_file, struct nlk_vocab_t **vocab, 
                        const NLK_ARRAY *weights,  const bool lower_words, 
                        nlk_real *accuracy)
{
    /** @section Allocation and Initialization
     */
    *accuracy = 0;
    int correct = 0;
    int total = 0;
    size_t ii;
    size_t max_line_size = NLK_LM_MAX_LINE_SIZE;
    size_t max_word_size = NLK_LM_MAX_WORD_SIZE;

    NLK_ARRAY *weights_norm;    /* for the normalized copy of the weights */
    /* normalize weights to make distance calculations easier */
    weights_norm = nlk_array_create_copy(weights, 0);
    nlk_array_normalize_row_vectors(weights_norm);
    size_t limit = weights_norm->rows;

    FILE *fin = fopen(test_file, "rb");
    if(fin == NULL) {
        NLK_ERROR(strerror(errno), errno);
    }
    size_t num_lines = nlk_text_count_lines(fin);

    struct nlk_vocab_t **par1 = (struct nlk_vocab_t **) malloc(num_lines / 2 * 
                                             sizeof(struct nlk_vocab_t *));
    struct nlk_vocab_t **par2 = (struct nlk_vocab_t **) malloc(num_lines / 2 *
                                             sizeof(struct nlk_vocab_t *));

    if(par1 == NULL || par2 == NULL) {
        NLK_ERROR("failed to allocate memory for test items",  NLK_ENOMEM);
    }

    char **line1 = (char **) malloc(max_line_size * sizeof(char *));
    char **line2 = (char **) malloc(max_line_size * sizeof(char *));
    if(line1 == NULL || line2 == NULL) {
        NLK_ERROR("failed to allocate memory for a line",  NLK_ENOMEM);
        /* unreachable */
    }
    for(ii = 0; ii < max_line_size; ii++) {
        line1[ii] = (char *) malloc(max_word_size * sizeof(char));
        line2[ii] = (char *) malloc(max_word_size * sizeof(char));
        if(line1[ii] == NULL || line2[ii]) {
            NLK_ERROR("failed to allocate memory for a line",  NLK_ENOMEM);
            /* unreachable */
        }
    }

    wchar_t *low_tmp = NULL;
    if(lower_words) {
        low_tmp = (wchar_t *) malloc(max_word_size * sizeof(wchar_t));
    }

    /* find the index of the first paragraph vector */
    /* @TODO change to generate PVs */
    size_t start_pos = 0;

    /* read file into memory */
    char *word = (char *) malloc(max_word_size * sizeof(char));
    char *tmp = (char *) malloc((max_word_size * max_line_size 
                                 + max_line_size) * sizeof(char));

    for(ii = 0; ii < num_lines / 2; ii++) {
        nlk_read_line(fin, line1, low_tmp, max_word_size, max_line_size);
        nlk_read_line(fin, line2, low_tmp, max_word_size, max_line_size);

        nlk_text_concat_hash(line1, tmp, word);
        par1[ii] = nlk_vocab_find(vocab, word);

        nlk_text_concat_hash(line2, tmp, word);
        par2[ii] = nlk_vocab_find(vocab, word);
    }

    /* free the memory for the file reading */
    for(ii = 0; ii < max_line_size; ii++) {
        free(line1[ii]);
        free(line2[ii]);
    }
    free(line1);
    free(line2);
    free(word);
    if(low_tmp != NULL) {
        free(low_tmp);
    }
    free(tmp);


    /** @section Eval Loop
     */
#pragma omp parallel reduction(+ : correct) reduction(+ : total)
{
#pragma omp for
    for(size_t tt = 0; tt < num_lines / 2; tt++) {
        nlk_real best_similarity = 0;
        size_t most_similar = 0;
        nlk_real sim = 0;
        size_t index = start_pos;
        size_t pos1;
        if(par1[tt] == NULL || par2[tt] == NULL) {
            continue;
        }
        pos1 = par1[tt]->index;
        
        for(; index < limit; index++) {
            sim = nlk_array_row_dot(weights_norm, pos1, weights_norm, index);
            if(sim > best_similarity) {
                best_similarity = sim;
                most_similar = index;
            }
        }
        if(most_similar == par2[tt]->index) {
            correct++;
        }
        total += 1;
    }
    
} /* end of parallel for */
    *accuracy = (nlk_real) correct / (nlk_real) total;

    free(par1);
    free(par2);
    nlk_array_free(weights_norm);
 
    return NLK_SUCCESS;
}

