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


/**
 * @file nlk_vocabulary.c - create and use vocabulary structures
 */

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>

#include "uthash.h"
#include "nlk_err.h"
#include "nlk_text.h"
#include "nlk_vocabulary.h"


/** @fn __nlk_vocab_add_item(nlk_Vocab **vocab, const char *word, 
 *                           const size_t id, const uint64_t count)
 * Adds an item (word) to a vocabulary
 *
 * @param vocab     the vocabulary
 * @param word      the word to add
 * @param id        the id of the word
 * @param count     the count associated with the word
 *
 * @return the vocabulary item
 */
nlk_Vocab *
__nlk_vocab_add_item(nlk_Vocab **vocab, const char *word, const size_t id,
                     const uint64_t count)
{
    nlk_Vocab *vocab_word;
    size_t length = strlen(word);
    
    vocab_word = (nlk_Vocab *) calloc(1, sizeof(nlk_Vocab));
    if(vocab_word == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for vocabulary item struct", 
                  NLK_ENOMEM);
        /* unreachable */
    }
    
    /* handle the main field - word */
    vocab_word->word = (char *) calloc((length + 1), sizeof(char));
    if(vocab_word->word == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for vocabulary string", 
                       NLK_ENOMEM);
        /* unreachable */
    }
    strcpy(vocab_word->word, word);

    /* set id, index and count */
    vocab_word->id = id;
    vocab_word->index = id;
    vocab_word->count = count;

    /* zero the other stuff */
    memset(vocab_word->point, 0, NLK_MAX_CODE * sizeof(size_t));
    memset(vocab_word->code, 0, NLK_MAX_CODE * sizeof(uint8_t));
    vocab_word->code_length = 0;

    HASH_ADD_STR(*vocab, word, vocab_word); /* hash by word */

    return vocab_word;
}

/** @fn nlk_Vocab *nlk_vocab_create(char *filename, const size_t max_word_size)
 * Build a vocabulary from a file
 *
 * @param filepath        the path of the file to read from
 * @param max_word_size   maximum size of a word (including null terminator)
 *
 * @return a pointer to the first item in the vocabulary use &vocab to pass it
 *         to other functions (it acts as a pointer to the entire vocabulary)
 *
 * @note
 * This file should have sentences separated by a newline and words separated 
 * by spaces or tabs.
 * @endnote
 */
nlk_Vocab *
nlk_vocab_create(char *filepath, const size_t max_word_size) {
    FILE *file;
    nlk_Vocab *vocab_word;
    nlk_Vocab *end_symbol;
    nlk_Vocab *vocab = NULL;
    char *word;
    int terminator;
    size_t length;
    size_t wid = 0;

    file = fopen(filepath, "rb");
    if(file == NULL) {
        NLK_ERROR_NULL(strerror(errno), errno);
        /* unreachable */
    }
 

    word = (char *) malloc(max_word_size * sizeof(char));
    if(word == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for string", 
                       NLK_ENOMEM);
    }

    /* word id 0 is reserved for end symbol </s> */
    end_symbol = __nlk_vocab_add_item(&vocab, NLK_END_SENT_SYMBOL, wid, 0);
    if(end_symbol == NULL) {
        return NULL;
    }
    wid++;

    /*
     * Read file cycle
     */
    do {
        terminator = nlk_read_word(file, word, max_word_size);

        if(strlen(word) == 0) {
            if(terminator == EOF) {
                break;
            } else {
                continue;
            }
        }

        HASH_FIND_STR(vocab, word, vocab_word);
        if(vocab_word == NULL) { /* word is not in vocabulary */
            vocab_word = __nlk_vocab_add_item(&vocab, word, wid, 1);
            if(vocab_word == NULL) {
                return NULL;
            }
            wid++;
        }
        else { /* word is in vocabulary */
            vocab_word->count = vocab_word->count + 1;
        }

        /* all sentences must end with </s> */
        if(terminator == '\n' || terminator == EOF) {
            end_symbol->count = end_symbol->count + 1;
        }

    } while(terminator != EOF);

    /* free temporary memory and close file */
    free(word);
    fclose(file);
    
    return vocab;
}

/** @fn void nlk_vocab_free(nlk_Vocab *vocab)
 * Free all memory associated with the vocabulary
 *
 * @param vocab the vocabulary structure
 */
void
nlk_vocab_free(nlk_Vocab **vocab)
{
    nlk_Vocab *vocab_word;
    nlk_Vocab *tmp;

    HASH_ITER(hh, *vocab, vocab_word, tmp) {
        /* free structure contents */
        if(vocab_word->word != NULL) {
            free(vocab_word->word);
        }
        /* delete from hashmap and free structure **/
        HASH_DEL(*vocab, vocab_word);
        free(vocab_word);
    }
}

/** @fn void nlk_vocab_size(nlk_Vocab *vocab)
 * Count of unique words in vocabulary
 *
 * @param vocab the vocabulary structure
 */
size_t
nlk_vocab_size(nlk_Vocab **vocab)
{
    return HASH_COUNT(*vocab);
}

/** @fn void nlk_vocab_total(nlk_Vocab *vocab)
 * Total of word counts in vocabulary
 *
 * @param vocab the vocabulary structure
 */
uint64_t
nlk_vocab_total(nlk_Vocab **vocab)
{
    uint64_t total = 0;
    nlk_Vocab *vocab_word;
    nlk_Vocab *tmp;

    HASH_ITER(hh, *vocab, vocab_word, tmp) {
        total += vocab_word->count;
    }
    return total;
}

/** @fn void nlk_vocab_reduce(nlk_Vocab *vocab, const size_t min_count)
 * Remove words with count smaller (<) than min_count. Calls sort after.
 *
 * @param vocab     the vocabulary structure
 * @param min_count the minimum number of occurrences a word
 *
 * @note
 * Sort is called to update the position index which after the reduce call
 * has "holes".
 * @endnote
 */
void
nlk_vocab_reduce(nlk_Vocab **vocab, const size_t min_count)
{
    nlk_Vocab *vi;
    nlk_Vocab *tmp;

    HASH_ITER(hh, *vocab, vi, tmp) {
        /* always protect end symbol vi->index = 0 */
        if(vi->count < min_count && vi->index != 0) {
            /* free structure contents */
            if(vi->word != NULL) {
                free(vi->word);
                vi->word = NULL;
            }
            /* delete from hashmap and free structure **/
            HASH_DEL(*vocab, vi);
            /* free(vi); */
        }
    }

    /* call sort to update the index */
    nlk_vocab_sort(vocab);
}

/** @fn void nlk_vocab_reduce_replace(nlk_Vocab *vocab, const size_t min_count)
 * Remove words with count smaller (<) than min_count and replace them with 
 * a special token/symbol. This special token will have the sum of the counts 
 * of the words it replaced. Call sort.
 *
 * @param vocab     the vocabulary structure
 * @param min_count the minimum number of occurrences a word
 *
 * @note
 * Sort is called to update the position index which after the reduce call
 * has "holes".
 * @endnote
 */
void
nlk_vocab_reduce_replace(nlk_Vocab **vocab, const size_t min_count)
{
    nlk_Vocab *vi;
    nlk_Vocab *tmp;
    nlk_Vocab *unk_symbol;
    uint64_t unk_count = 0;
    size_t first_id = -1;

    HASH_FIND_STR(*vocab, NLK_UNK_SYMBOL, unk_symbol);

    HASH_ITER(hh, *vocab, vi, tmp) {
        if(vi->count < min_count && vi->index != 0) {

            if(unk_symbol == NULL || vi != unk_symbol) {
                unk_count += vi->count;
                if(vi->id < first_id) {
                    first_id = vi->id;
                }
                /* free structure contents */
                if(vi->word != NULL) {
                    free(vi->word);
                    vi->word = NULL;
                }
                /* delete from hashmap and free structure **/
                HASH_DEL(*vocab, vi);
                /* free(vi); */
            }
        }
    }

    /* 
     * does the symbol already exist? i.e. not first call to this function 
     */

    if(unk_symbol == NULL) { /* nope, lets create it */
        unk_symbol = __nlk_vocab_add_item(vocab, NLK_UNK_SYMBOL,
                                          first_id + 1, unk_count);
    }
    else { /* just an update */
        unk_symbol->count = unk_symbol->count + unk_count;
    }

    /* call sort to update the index */
    nlk_vocab_sort(vocab);
}

/** @fn void __nlk_vocab_item_comparator(nlk_Vocab *a, nlk_Vocab *b)
 * Vocab item comparator - used for sorting with most frequent words first
 *
 * @param a vocab item
 * @param b another vocab item
 *
 * @returns positive if b.count > a.count, negative if b.count < a.count
 */
int 
__nlk_vocab_item_comparator(nlk_Vocab *a, nlk_Vocab *b)
{
    if(a->index == 0) { 
        return -1; 
    } else if(b->index == 0) {
        return 1;
    }
    return (b->count - a->count);
}

/** @fn void __nlk_vocab_item_comparator_reverse(nlk_Vocab *a, nlk_Vocab *b)
 * Vocab item comparator - used for sorting with the least frequent words first
 *
 * @param a vocab item
 * @param b another vocab item
 *
 * @returns positive if b.count < a.count, negative if b.count > a.count
 */
int 
__nlk_vocab_item_comparator_reverse(nlk_Vocab *a, nlk_Vocab *b)
{
    if(a->index == 0) { 
        return 1; 
    } else if(b->index == 0) {
        return 0;
    }
    return (a->count - b->count);
}


/** @fn void nlk_vocab_sort(nlk_Vocab **vocab)
 * Sort the vocabulary by word count, most frequent first i.e. desc by count.
 * Also updates the *index* property.
 *
 * @param vocab     the vocabulary structure
 */
void nlk_vocab_sort(nlk_Vocab **vocab)
{
    nlk_Vocab *vi;
    size_t ii = 1;  /* 0 is the end symbol */

    HASH_SORT(*vocab, __nlk_vocab_item_comparator);
    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi->index == 0) {
            continue;   /* keep end symbol at position 0 */
        }
        vi->index = ii;
        ii++;
    }
}

/* @fn nlk_vocab_encode_huffman(nlk_Vocab **vocab)
 * Create Huffman binary tree for hierarchical softmax (HS).
 * Adds *code* (huffman encoded representation) and HS *point* fields to 
 * vocabulary items.
 * 
 * @params vocab    the vocabulary
 *
 * @note
 * thanks to word2vec: https://code.google.com/p/word2vec/
 * See 
 * "Hierarchical probabilistic neural network language model"
 * Frederic Morin & Yoshua Bengio, AISTATS 2005 
 * 
 * and
 *
 * “A Scalable Hierarchical Distributed Language Model”
 * Andrew Mnih & Geoffrey Hinton, NIPS 2008
 *
 * and
 *
 * "Distributed Representations of Words and Phrases and their 
 * Compositionality"
 * Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 
 * NIPS, 2013
 * @endnote
 */
void
nlk_vocab_encode_huffman(nlk_Vocab **vocab)
{
    nlk_Vocab        *vi;                   /* vocabulary iterator */
    size_t            vsize;                /* the vocabulary size */ 
    size_t            nn;                   /* for indexing over nodes */
    size_t            min1;                 /*  the minimum index */
    size_t            min2;                 /* the second minimum index */
    int64_t           pos1;                 /* position in queue1 */
    int64_t           pos2;                 /* position in queue2 */
    size_t            code_length;          /* for holding code lengths */
    uint8_t           code[NLK_MAX_CODE];   /* temporay storage for a code */
    size_t            point[NLK_MAX_CODE];  /* temporay storage for a point */
    size_t            ii;

    /* sort the vocabulary */
    nlk_vocab_sort(vocab);

    /* 
     * The vocabulary is sorted so we can use the fast version O(n) of the 
     * huffman tree building algorithm.
     * First we allocate space for the nodes and create the queues 
     */
    vsize = nlk_vocab_size(vocab);
    uint64_t *count = (uint64_t *) calloc(vsize * 2 + 1, sizeof(uint64_t));
    uint8_t *binary = (uint8_t *)  calloc(vsize * 2 + 1, sizeof(uint64_t));
    size_t *parent = (size_t *)    calloc(vsize * 2 + 1, sizeof(size_t));
    if(count == NULL || binary == NULL || parent == NULL) {
        NLK_ERROR_VOID("failed to allocate memory for huffman tree", 
                       NLK_ENOMEM);
    }

    /* 
     * (1) create a (leaf) node for each vocab item 
     * (2) Enqueue all leaf nodes into the first queue by probability (count) 
     *     in increasing order so that the least likely item (lowest count) is 
     *     in the head of the queue
     *  Here *count* is basically two queues in a single array,
     *  queue1 from [0, vsize) and queue2 from [vsize, pos2]
     */
    nn = 0;
    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        count[nn] = vi->count;
        nn++;
    }
    for(; nn < vsize * 2; nn++) {
        count[nn] = -1;
    }
    
    pos1 = vsize - 1;   /* second lowest count */
    pos2 = vsize;       /* lowest count */
    
    /* (3) while there's more than one node */
    for(nn = 0; nn < vsize - 1; nn++) {
        /* 
         * (3.1) - get the two nodes with the lowest weight by examining the
         * fronts of both queues.
         */
        /* pick min1 - lowest count (prob) node */
        if (pos1 >= 0) { /* if queue 1 is not empty */
            if (count[pos1] < count[pos2]) {
                /* pos1 has lowest count => pick from head of queue1 */
                min1 = pos1;
                pos1--;
            } else {
                /* pos2 has lowest count => pick from head of queue2 */
                min1 = pos2;
                pos2++;
            }
        } else {
            /* queue1 is empty so just pick from queue2 */
            min1 = pos2;
            pos2++;
        }
        /* pick min2 - 2nd lowest count (lowest since min1 was removed) */
        if (pos1 >= 0) { /* if queue 1 is not empty */
            if (count[pos1] < count[pos2]) {
                /* pos1 has lowest count => pick from head of queue1 */
                min2 = pos1;
                pos1--;
            } else {
                /* pos2 has lowest count => pick from head of queue2 */
                min2 = pos2;
                pos2++;
            }
        } else {
            /* queue1 is empty so just pick from queue2 */
            min2 = pos2;
            pos2++;
        }

        /* 
         * (3.2) - create a new internal node, with the two just-removed nodes 
         * as children and the sum of their counts as the count for the new 
         * node. Right node gets a '1', left node gets a '0' (calloc).
         */
        count[vsize + nn] = count[min1] + count[min2];
        parent[min1] = vsize + nn;
        parent[min2] = vsize + nn;
        binary[min2] = 1;
    }
    /* 
     * (4) - the tree has been generated, assign the codes 
     * parent[vsize * 2 - 2] countains the root node
     */

    /* iterate over all the vocabulary */
    size_t maxpoint = 0;
    nn = 0;
    for(vi = *vocab; vi != NULL; vi = (nlk_Vocab *)(vi->hh.next)) {
      /* now, traverse the tree and assign the code to this word */
        code_length = 0;
        /* traverse the tree from current node to the root */
        for(ii = nn; ii != (vsize * 2) - 2; ii = parent[ii]) { 
            /* code and point in reverse order */
            code[code_length] = binary[ii];
            point[code_length] = ii; 
            code_length++;
        }
        /* assign code and point to vocabulary item in correct order */
        vi->code_length = code_length;
        vi->point[0] = vsize - 2;
        for(ii = 0; ii < code_length; ii++) {
            vi->code[code_length - ii - 1] = code[ii];
            vi->point[code_length - ii] = point[ii] - vsize;
        }
        nn++;
    }
    free(count);
    free(parent);
    free(binary);
}

/** @fn size_t vocab_max_code_length(nlk_Vocab **vocab)
 * Return the maximum code length of any word in the vocabulary
 *
 * @param vocab the vocabulary
 *
 * @return the maximum code length of any word in the vocabulary
 */
size_t 
vocab_max_code_length(nlk_Vocab **vocab)
{
    nlk_Vocab *vi;
    size_t code_length = 0;

    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi->code_length > code_length) {
            code_length = vi->code_length;
        }
    }
    return code_length;
}


/** @fn int nlk_vocab_save(const char *filepath, nlk_Vocab **vocab)
 * Save the vocabulary - only strings and counts
 * 
 * @param filepath  the path of the file to which the vocabulary will be saved
 * @param vocab     the vocabulary structure
 *
 * @return NLK_SUCCESS or ernno
 * 
 * @note
 * There is no escaping if the word strings somehow contains tabs and newlines.
 * The format is one vocabulary item per line, tab separated values. The order 
 * of the items is the same as specified in the structure definition
 * (e.g. sort order).
 * @endnote
 */
int
nlk_vocab_save(const char *filepath, nlk_Vocab **vocab)
{
    nlk_Vocab *vi;
    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }

    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        fprintf(out, "%s %zu\n", vi->word, vi->count);
    }
    fclose(out);
    return NLK_SUCCESS;
}

/** @fn int nlk_vocab_load(const char *filepath, const size_t max_word_size, 
 *                         nlk_Vocab **vocab)
 * Load the (simple) vocabulary structure from disk, saved via *nlk_save_vocab*
 *
 * @param filepath          the path of the file to which will be read
 * @param max_word_size     maximum string size
 * @param vocab             the vocabulary structure
 *
 * @returns NLK_SUCCESS or errno
 */
int
nlk_vocab_load(const char *filepath, const size_t max_word_size, 
               nlk_Vocab **vocab)
{
    size_t count;
    size_t wid = 0;
    char *word = (char *) calloc(max_word_size, sizeof(char));
    FILE *in = fopen(filepath, "rb");
    if(in == NULL) {
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }

    while(!feof(in)) {
        nlk_read_word(in, word, max_word_size);
        if(fscanf(in, "%zu", &count) != 1) {
            NLK_ERROR("Parsing error", NLK_FAILURE);
        }
        __nlk_vocab_add_item(vocab, word, wid, count);
        wid++;
    }

    free(word);
    fclose(in);
    return NLK_SUCCESS;
}

/** @fn int nlk_vocab_save_full(const char *filepath, nlk_Vocab **vocab)
 * Save the vocabulary structure to disk.
 *
 * @param filepath  the path of the file to which the vocabulary will be saved
 * @param vocab     the vocabulary structure
 *
 * @note
 * There is no escaping if the word strings somehow contains tabs and newlines.
 * The format is one vocabulary item per line, tab separated values. The order 
 * of the items is the same as specified in the structure definition
 * (e.g. sort order).
 * @endnote
 *
 * @returns 0 or errno
 */
int
nlk_vocab_save_full(const char *filepath, nlk_Vocab **vocab)
{
    nlk_Vocab *vi;
    FILE *file;
    size_t ii;

    file = fopen(filepath, "wb");
    if (file == NULL) {
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }

    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        fprintf(file, "%s\t%zu\t%"PRIu64"\t%zu\t", 
                 vi->word, vi->index, vi->count, vi->code_length);
        /* code */
        for(ii = 0; ii < vi->code_length - 1; ii++) {
            fprintf(file, "%"PRIu8" ", vi->code[ii]);
        }
        fprintf(file, "%"PRIu8"\t", vi->code[vi->code_length - 1]);
        /* point */
        for(ii = 0; ii < vi->code_length - 1; ii++) {
            fprintf(file, "%zu ", vi->point[ii]);
        }
        fprintf(file, "%zu\n", vi->point[vi->code_length - 1]);
    }

    fclose(file);
}

/* @fn size_t nlk_vocab_vectorize_paragraph(
 *              nlk_Vocab **vocab, char **paragraph, bool replace_missing,
 *              size_t replacement, size_t *vector)
 * Vectorizes a series of words: for each word, the output vector will contain
 * the respective index of that word.
 *
 * @param vocab                 the vocabulary used for vectorization
 * @param paragraph             an array of strings (char arrays)
 * @param replace_missing       replace words that are not in the vocabulary?
 * @param replacement           index to use for words not in the vocabulary
 * @param vector                the index vector to be written
 *
 * @returns number of words vectorized (size of vector)
 *
 * @note
 * The paragraph is expected to be terminated by a null word (word[0] = 0)
 * as generated by nlk_read_line()
 *
 * If replace_missing is false, the word will simply be ignored, i.e. treated 
 * as if was not there and the returned size will be small than the size of the
 * paragraph.
 * @endnote
 */
size_t
nlk_vectorize(nlk_Vocab **vocab, char **paragraph, bool replace_missing, 
              size_t replacement, size_t *vector) 
              
{
    nlk_Vocab *vocab_word;  /* vocabulary item that corresponds to word */
    size_t par_idx;         /* position in paragraph */
    size_t vec_idx = 0;     /* position in vector */

    for(par_idx = 0; paragraph[par_idx] != '\0'; par_idx++) {
        HASH_FIND_STR(*vocab, paragraph[par_idx], vocab_word);

        if(vocab_word == NULL && replace_missing) { 
            /* word NOT in vocabulary but will be replaced*/ 
            vector[vec_idx] = replacement;
            vec_idx++;
        }  else if(vocab_word != NULL) {
            /* the word is in the vocabulary */
            vector[vec_idx] = vocab_word->index;
            vec_idx++;
        } 
        /* 
         * if word is not in vocab and replace_missing is false, nothing needs 
         * to be done in this loop 
        */
    }
    
    return vec_idx + 1; /* +1: we return a count not the zero-based index */
}

/* @fn size_t nlk_vocab_vocabularize(nlk_Vocab **vocab, 
 *                                   char **paragraph, 
 *                                   bool replace_missing,
 *                                   nlk_Vocab *replacement,
 *                                   nlk_Vocab **vector)
 * "Vocabularizes" a series of words: for each word, the output vector will 
 * contain a pointer to the vocabulary item of that word.
 *
 * @param vocab                 the vocabulary used for vectorization
 * @param paragraph             an array of strings (char arrays)
 * @param replacement           vocab to use for words not in the vocabulary
 *                              - NULL means do not replace
 * @param end_symbol            if true, add the end symbol at the end
 * @param vector                the vocabulary item vector to be written
 *
 * @returns number of words vocabularized (size of vector)
 *
 * @note
 * The paragraph is expected to be terminated by a null word (word[0] = 0)
 * as generated by nlk_read_line(). This will be replaced by the end sentence 
 * token.
 *
 * If replacement is NULL, the word will simply be ignored, i.e. treated 
 * as if it was not there and the returned size will be small than the size of 
 * the paragraph.
 * @endnote
 */
size_t
nlk_vocab_vocabularize(nlk_Vocab **vocab, char *paragraph[], 
                       nlk_Vocab *replacement, bool end_symbol,
                       nlk_Vocab **vector) 
              
{
    nlk_Vocab *vocab_word;  /* vocabulary item that corresponds to word */
    size_t par_idx;         /* position in paragraph */
    size_t vec_idx = 0;     /* position in vector */

    for(par_idx = 0; paragraph[par_idx] != '\0'; par_idx++) {
        HASH_FIND_STR(*vocab, paragraph[par_idx], vocab_word);

        if(vocab_word == NULL && replacement != NULL) { 
            /* word NOT in vocabulary but will be replaced*/ 
            vector[vec_idx] = replacement;
            vec_idx++;
        }  else if(vocab_word != NULL) {
            /* the word is in the vocabulary */
            vector[vec_idx] = vocab_word;
            vec_idx++;
        } 
    }
    if(end_symbol) {
        HASH_FIND_STR(*vocab, NLK_END_SENT_SYMBOL, vocab_word);
        vector[vec_idx] = vocab_word;
        vec_idx++;
    }
    
    return vec_idx;
}

/** @fn nlk_vocab *nlk_vocab_find(nlk_Vocab **vocab, char *word)
 * Find a word (string) in the vocabulary
 *
 * @param vocab     the vocabulary
 * @param word      the word
 *
 * @return the vocabulary item corresponding to the word or NULL if not found
 */
nlk_Vocab *
nlk_vocab_find(nlk_Vocab **vocab, char *word)
{
    nlk_Vocab *vocab_word;

    HASH_FIND_STR(*vocab, word, vocab_word);
    
    return vocab_word;
}

