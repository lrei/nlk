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


/** @file nlk_vocabulary.h
 * Defines the vocabulary handling structures and functions
 */

#ifndef __NLK_VOCABULARY_H__
#define __NLK_VOCABULARY_H__


#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "uthash.h"


#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif
__BEGIN_DECLS


#define NLK_END_SENT_SYMBOL "</s>"
#define NLK_UNK_SYMBOL "<UNK>"
#define NLK_MAX_CODE 40


/** @struct nlk_vocab
 * Vocabulary structure - a hashmap from words to their huffman code and count
 *
 * @note
 * Each nlk_Vocab is a vocabulary item (e.g. a word). However this uses 
 * uthash to create a hashtable, and thus a single **nlk_vocab pointer 
 * points to the entire vocabulary (e.g. all words) stored in a hastable 
 * with a (possibly sorted) doubly linked list and a bloom filter for quick
 * misses.
 * @endnote
 */
struct nlk_vocab {
    char            *word;                  /**< the word string (key) */
    size_t           id;                    /**< word id */
    size_t           index;                 /**< sorted index position */
    uint64_t         count;                 /**< word count */
    size_t           code_length;           /**< length of *code array */
    uint8_t          code[NLK_MAX_CODE];    /**< huffman code */
    size_t           point[NLK_MAX_CODE];   /**< hierarchical softmax nodes */
    UT_hash_handle   hh;                    /**< handle for hash table */
};
typedef struct nlk_vocab nlk_Vocab;



nlk_Vocab *nlk_vocab_create(char *, const size_t); 
void nlk_vocab_sort(nlk_Vocab **);
void nlk_vocab_free(nlk_Vocab **);
size_t nlk_vocab_size(nlk_Vocab **);
size_t vocab_max_code_length(nlk_Vocab **);
uint64_t nlk_vocab_total(nlk_Vocab **);
void nlk_vocab_reduce(nlk_Vocab **, const size_t);
void nlk_vocab_reduce_replace(nlk_Vocab **, const size_t);
void nlk_vocab_encode_huffman(nlk_Vocab **);
int nlk_vocab_save(const char *, nlk_Vocab **);
int nlk_vocab_load(const char *, const size_t, nlk_Vocab **);
int nlk_vocab_save_full(const char *, nlk_Vocab **);
size_t nlk_vocab_vocabularize(nlk_Vocab **, char **, nlk_Vocab *, bool,
                              nlk_Vocab **);
nlk_Vocab *nlk_vocab_find(nlk_Vocab **, char *);



__END_DECLS
#endif /* __NLK_VOCABULARY_H__ */
