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

#include "nlk_array.h"

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


#define NLK_START_SYMBOL "</s>"
#define NLK_UNK_SYMBOL "<UNK>"
#define NLK_MAX_CODE 40
#define NLK_VOCAB_MAX_THREADS 512             /**< maximum number of threads */
#define NLK_VOCAB_MIN_SIZE_THREADED (long)1e4 /**< min lines to use threads */
#define NLK_NEG_TABLE_SIZE (size_t)1e8
#define NLK_NEG_TABLE_POW (double)0.75


/** @enum NLK_VOCAB_TYPE
 * The type of the vocabulary item
 */
enum nlk_vocab_type_t { 
    NLK_VOCAB_WORD      = 0, 
    NLK_VOCAB_SPECIAL   = 1,
    NLK_VOCAB_CHAR      = 2,
    NLK_VOCAB_LABEL     = 3
};
typedef enum nlk_vocab_type_t NLK_VOCAB_TYPE;


/** @struct nlk_vocab_t
 * Vocabulary structure - a hashmap from words to their huffman code and count
 *
 * @note
 * Each struct nlk_vocab_t is a vocabulary item (e.g. a word). However this uses 
 * uthash to create a hashtable, and thus a single **nlk_vocab pointer 
 * points to the entire vocabulary (e.g. all words) stored in a hastable 
 * with a (possibly sorted) doubly linked list and a bloom filter for quick
 * misses.
 * @endnote
 */
struct nlk_vocab_t {
    char                 *word;                  /**< the word string (key) */
    enum nlk_vocab_type_t type;                  /**< vocab item type */
    size_t                index;                 /**< sorted index position */
    uint64_t              count;                 /**< word count */
    size_t                code_length;           /**< length of *code array */
    uint8_t               code[NLK_MAX_CODE];    /**< huffman code */
    size_t                point[NLK_MAX_CODE];   /**< HS Softmax nodes */
    UT_hash_handle        hh;                    /**< handle for hash table */
};
typedef struct nlk_vocab_t NLK_VOCAB;

/** @struct nlk_vocab_opts_t
 * Options for creating a vocabulary and vocabularizing lines
 */
struct nlk_vocab_opts_t {
    bool    replace;
    size_t  total_words;
    size_t  total_paragraphs;
};

/** @struct nlk_vocab_line_t
 * A vocabularized line of text and associated options
 * @TODO Move here from train_opts
 */
struct nlk_line_t {
    size_t                line_id;          /**< line id read from file */
    size_t                len;              /**< vocabularized array length */
    struct nlk_vocab_t  **varray;           /**< vocabularized line read */
};

/* creation */
struct nlk_vocab_t   *nlk_vocab_create(const char *, const uint64_t, 
                                       const bool, const bool); 
void                  nlk_vocab_extend(struct nlk_vocab_t **, char *); 
void                  nlk_vocab_add_vocab(struct nlk_vocab_t **dest, 
                                          struct nlk_vocab_t **source);
struct nlk_vocab_t   *nlk_vocab_add(struct nlk_vocab_t **, char *,
                                    const NLK_VOCAB_TYPE); 
void                  nlk_vocab_free(struct nlk_vocab_t **);

/* reduce */
void                  nlk_vocab_reduce(struct nlk_vocab_t **, const uint64_t);
void                  nlk_vocab_reduce_replace(struct nlk_vocab_t **, 
                                               const size_t);
/* stats */
size_t       nlk_vocab_size(struct nlk_vocab_t **);
size_t       nlk_vocab_words_size(struct nlk_vocab_t **);
uint64_t     nlk_vocab_total(struct nlk_vocab_t **);
size_t       vocab_max_code_length(struct nlk_vocab_t **);

/* sorting */
void        nlk_vocab_sort(struct nlk_vocab_t **);
void        nlk_vocab_encode_huffman(struct nlk_vocab_t **);

/* save & load */
int                   nlk_vocab_export(const char *, struct nlk_vocab_t **);
struct nlk_vocab_t   *nlk_vocab_import(const char *, const size_t);
void                  nlk_vocab_save(struct nlk_vocab_t **, FILE *);
struct nlk_vocab_t   *nlk_vocab_load(FILE *);

/* vocabularize */
void nlk_vocab_line_subsample(const struct nlk_line_t *, 
                              const uint64_t, const float, 
                              struct nlk_line_t *);

size_t  nlk_vocab_vocabularize(struct nlk_vocab_t **, char **,
                               struct nlk_vocab_t *, struct nlk_vocab_t **);
void    nlk_vocab_read_vocabularize(int, struct nlk_vocab_t **, 
                                    struct nlk_vocab_t *, char **, 
                                    struct nlk_line_t *, char *);

void         nlk_vocab_print_line(struct nlk_vocab_t **, size_t, bool);

/* NEG table */
size_t      *nlk_vocab_neg_table_create(struct nlk_vocab_t **, const size_t, 
                                        double);

/* find */
struct nlk_vocab_t   *nlk_vocab_find(struct nlk_vocab_t **, char *);
struct nlk_vocab_t   *nlk_vocab_at_index(struct nlk_vocab_t **, size_t);
/*size_t       nlk_vocab_last_id(struct nlk_vocab_t **); */
size_t                nlk_vocab_last_index(struct nlk_vocab_t **);
struct nlk_vocab_t   *nlk_vocab_get_start_symbol(struct nlk_vocab_t **);

uint64_t nlk_vocab_count_words(struct nlk_vocab_t **, const char *, 
                               const size_t);

/* line */
struct nlk_line_t  *nlk_line_create(const size_t);
void                nlk_line_ids(struct nlk_line_t *, size_t *);
void                nlk_line_free(struct nlk_line_t *);



__END_DECLS
#endif /* __NLK_VOCABULARY_H__ */
