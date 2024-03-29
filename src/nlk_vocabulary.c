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
#include <math.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

#include "uthash.h"

#include "nlk_err.h"
#include "nlk_text.h"
#include "nlk_array.h"
#include "nlk_random.h"
#include "nlk_tic.h"
#include "nlk_util.h"
#include "nlk.h"

#include "nlk_vocabulary.h"


/**
 * Displays the progress stats while building a vocabulary from a file
 *
 * @param start         the clocks used just before starting to read the file
 */
static void
nlk_vocab_display_progress(const size_t line_counter, const size_t total_lines, 
                           const clock_t start)
{
    double progress;
    double speed;
    char display_str[256];

    clock_t now = clock();

    progress = (line_counter / (double) total_lines) * 100;
    speed = line_counter / ((double)(now - start + 1) / 
            (double)CLOCKS_PER_SEC * 1000),

    snprintf(display_str, 256, 
            "Vocabulary Progress: %.2f%% Lines/Thread/sec: %.2fK Threads: %d", 
            progress, speed, omp_get_num_threads());

    nlk_tic(display_str, false);
}


/**  
 * Adds an item (word) to a vocabulary
 *
 * @param vocab     the vocabulary
 * @param word      the word to add
 * @param count     the count associated with the word
 * @param type      the item's type
 *
 * @return the vocabulary item
 */
static inline struct nlk_vocab_t *
nlk_vocab_add_item(struct nlk_vocab_t **vocab, const char *word,
                     const uint64_t count, const NLK_VOCAB_TYPE type)
{
    struct nlk_vocab_t *vocab_word;
    size_t length = strlen(word);
    
    vocab_word = (struct nlk_vocab_t *) calloc(1, sizeof(struct nlk_vocab_t));
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

    /* set index and count */
    vocab_word->index = 0;
    vocab_word->count = count;
    vocab_word->hc = NULL;
    vocab_word->type = type;

    HASH_ADD_STR(*vocab, word, vocab_word); /* hash by word */

    return vocab_word;
}


/**
 * Add or increment item to vocabulary
 * @Note: This function is convenient but slower than necessary. It was
 * implemented for adding labels in small labelled datasets
 *
 * @param vocab     the vocabulary
 * @param word      the word (item)
 * @param type      the item's type
 *
 * @return the vocabulary item corresponding to the word or NULL if not found
 */
struct nlk_vocab_t *
nlk_vocab_add(struct nlk_vocab_t **vocab, char *word, 
              const NLK_VOCAB_TYPE type)
{
    struct nlk_vocab_t *vocab_word = NULL;

    if(*vocab != NULL) {
        HASH_FIND_STR(*vocab, word, vocab_word);
    }

    if(vocab_word == NULL) { /* add */
        vocab_word = nlk_vocab_add_item(vocab, word, 1, type);
        vocab_word->index = nlk_vocab_last_index(vocab) + 1;
    } else { /* increment */
        vocab_word->count += 1;
    }

    return vocab_word;
}


/** 
 * Returns an initialized vocabulary (i.e. with a start symbol)
 */
static struct nlk_vocab_t *
nlk_vocab_init()
{
    struct nlk_vocab_t *vocab = NULL;

   /* word 0 is reserved for start symbol </s> */
    struct nlk_vocab_t *start_symbol = nlk_vocab_add_item(&vocab, 
                                                          NLK_START_SYMBOL, 
                                                          0, NLK_VOCAB_SPECIAL);
    if(start_symbol == NULL) {
        NLK_ERROR_NULL("unable to add to vocabulary", NLK_EINVAL);
        /* unreachable */
    }
    return vocab;
}

static int
nlk_vocab_read_add(struct nlk_vocab_t **vocabulary, const char *filepath,
                   const bool line_has_id, const bool verbose) 
{
    /** @section Shared Initializations
     */
    size_t total_lines = 0;
    size_t line_counter = 0;
    size_t updated = 0;
    clock_t start = clock();

    /* open file */
    total_lines = nlk_text_count_lines(filepath);
    
    /* Limit the number of threads */
    int num_threads = nlk_get_num_threads();
    if(num_threads > NLK_VOCAB_MAX_THREADS) {
        num_threads = NLK_VOCAB_MAX_THREADS;
    }
    /* make sure it's an even number, simplifies things */
    if(num_threads % 2 != 0 && num_threads > 1) {
        num_threads--;
    }
    /* check if file is too small to make sense to parallel stuff */
    if(total_lines < NLK_VOCAB_MIN_SIZE_THREADED) {
        num_threads = 1;
    }

    /* create and initialize vocabularies */
    struct nlk_vocab_t *vocabs[NLK_VOCAB_MAX_THREADS];
    for(int vv = 0; vv < num_threads; vv++) {
        vocabs[vv] = nlk_vocab_init();
    }

    /** @section Parallel Allocations and Initializations
     */
#pragma omp parallel shared(line_counter, updated)
{
    size_t zz;
    size_t cur_line;
    size_t end_line;
    size_t par_id;

    /* vocabulary */
    struct nlk_vocab_t *vocab_word;

    /* word */
    char *word = NULL;
    int ret = 0;

    /* open file */
    int fd = nlk_open(filepath);
    if(fd < 0) {
        NLK_ERROR_ABORT(strerror(errno), errno);
        /* unreachable */
    }

    /* allocate memory for reading from the input file */
    char **text_line = nlk_text_line_create();
    char *buffer = (char *) malloc(sizeof(char) * NLK_BUFFER_SIZE);
    if(buffer == NULL) {
        NLK_ERROR_ABORT("failed to  allocate buffer for reading", NLK_ENOMEM);
        /* unreachable */
    }


    /** @section Parallel Creation of Vocabularies (Map)
     * Each thread reads from a different part of the file.
     */
#pragma omp for
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        /* set train file part position */

        cur_line = nlk_text_get_split_start_line(total_lines, num_threads, 
                                                  thread_id);
        nlk_text_goto_line(fd, cur_line);
        end_line = nlk_text_get_split_end_line(total_lines, num_threads, 
                                                  thread_id);
        
        struct nlk_vocab_t *vocab = vocabs[thread_id];
        struct nlk_vocab_t *start_symbol;
        HASH_FIND_STR(vocab, NLK_START_SYMBOL, start_symbol);

        while(1) {

            /** @subsection Progress display 
             */
            if(line_counter - updated > 1000) {
                updated = line_counter;
                if(verbose) {
                    nlk_vocab_display_progress(line_counter, total_lines,
                                                start);
                }
            }

            /* read from file */
            if(line_has_id) {
                ret = nlk_read_line(fd, text_line, &par_id, buffer);
            } else {
                ret = nlk_read_line(fd, text_line, NULL, buffer);
                par_id = cur_line;
            }
           
            line_counter++;
            cur_line++;

            /* all sentences must start with </s> except empty lines */
            if(text_line[0][0] != '\0') {
                    start_symbol->count += 1;
            }

            /* process each word in line */
            for(zz = 0; text_line[zz][0] != '\0'; zz++) {
                word = text_line[zz];

                if(strlen(word) == 0) {
                    continue;
                }

                /** @subsection Increment Count or Add
                 */
                HASH_FIND_STR(vocab, word, vocab_word);
                if(vocab_word == NULL) { /* word is not in vocabulary */
                    vocab_word = nlk_vocab_add_item(&vocab, word, 1, 
                                                      NLK_VOCAB_WORD);
                    if(vocab_word == NULL) {
                        NLK_ERROR_ABORT("adding to vocabulary failed", 
                                        NLK_FAILURE);
                    }
                }
                else { /* word is in vocabulary */
                    vocab_word->count = vocab_word->count + 1;
                }
            }   /* end of words in line */

            /* check for end of thread part */
            if(ret == EOF) {
                break;
            } else if(cur_line > end_line) {
                break;
            }
        } /* file is over (end of while) */
    } /* end of for() thread */

    if(verbose) {
        nlk_vocab_display_progress(line_counter, total_lines, start);
    }


    /** @section Free Memory and Close File
     */
    nlk_text_line_free(text_line);
    free(buffer); 
    buffer = NULL;
    close(fd); 
    fd = 0;

} /* end of pragma omp parallel */

    /** @section Parallel reduce of vocabularies
     */
    if(num_threads > 1 && verbose) {
        printf("\n");
        nlk_tic("vocabulary: merging parallel vocabularies", true);
    }
    int vs = num_threads / 2;
    while(vs > 1) {
#pragma omp parallel for
        for(int v = 0; v < vs; v++) {
            nlk_vocab_add_vocab(&vocabs[v], &vocabs[vs + v]);
            nlk_vocab_free(&vocabs[vs +v]);
        }
        vs /= 2;
    }

    nlk_vocab_add_vocab(vocabulary, &vocabs[0]);
    nlk_vocab_free(&vocabs[0]);

    if(num_threads > 1) {
        nlk_vocab_add_vocab(vocabulary, &vocabs[1]);
        nlk_vocab_free(&vocabs[0]);
    }

    if(num_threads > 1 && verbose) {
        nlk_tic("vocabulary: merging finished.", true);
    }

    return NLK_SUCCESS;
}

/**
 * Add a vocabulary to another vocabulary. 
 * If an item exists in dest, it's count will be updated by adding the source's
 * count to it.
 * If the item does not exits in dest, it will be created with the count from 
 * the source. 
 *
 * @param dest      the destination vocabulary (will be updated)
 * @param source    the source vocabulary (will not be changed)
 */
void
nlk_vocab_add_vocab(struct nlk_vocab_t **dest, struct nlk_vocab_t **source)
{
    struct nlk_vocab_t *si;
    struct nlk_vocab_t *di;

    for(si = *source; si != NULL; si = si->hh.next) {
        HASH_FIND_STR(*dest, si->word, di);
        if(di == NULL) {
            nlk_vocab_add_item(dest, si->word, si->count, si->type);
        } else { /* just update counts */
            di->count += si->count;
        }
    }

}

/**
 * Build a vocabulary from a file
 *
 * @param filepath        the path of the file to read from
 * @param par_id          true if each line starts with an id
 * @param min_count       minimum word frequency (count)
 * @param replace         replace tokens below min_count
 * @param verbose         display progress
 *
 * @return a pointer to the first item in the vocabulary use &vocab to pass it
 *         to other functions (it acts as a pointer to the entire vocabulary)
 *
 * @note
 * This file should have sentences separated by a newline and words separated 
 * by spaces.
 * @endnote
 *
 * @note
 * Vocabulary is sorted and huffman encoding is created
 * @endnote
 */
struct nlk_vocab_t *
nlk_vocab_create(const char *filepath, const bool par_id, 
                 const uint64_t min_count, 
                 const bool replace, const bool verbose) {

    struct nlk_vocab_t *vocab = nlk_vocab_init();

    /* create vocabulary */
    if(verbose) {
        nlk_tic(NULL, false);
    }

    /* create */
    nlk_vocab_read_add(&vocab, filepath, par_id, verbose);

    /* reduce to min_count - also sorts and encodes */
    if(replace) {
        if(verbose) {
            nlk_tic("vocabulary: replacing < min_count and sorting", true);
        }
        nlk_vocab_reduce_replace(&vocab, min_count);
    } else {
         if(verbose) {
            nlk_tic("vocabulary: removing < min_count and sorting", true);
        }
        nlk_vocab_reduce(&vocab, min_count);
    }

    if(verbose) {
        nlk_tic_reset();
        printf("vocabulary: words: %zu (total count: %"PRIu64")\n", 
                nlk_vocab_size(&vocab), nlk_vocab_total(&vocab));
    }

    return vocab;
}

/** 
 * Extend a vocabulary
 */
void
nlk_vocab_extend(struct nlk_vocab_t **vocab, const char *filepath, 
                 const bool line_id) {

    nlk_vocab_read_add(vocab, filepath, line_id, false);
}

/** @fn void nlk_vocab_free(struct nlk_vocab_t *vocab)
 * Free all memory associated with the vocabulary
 *
 * @param vocab the vocabulary structure
 */
void
nlk_vocab_free(struct nlk_vocab_t **vocab)
{
    struct nlk_vocab_t *vocab_word;
    struct nlk_vocab_t *tmp;

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

/**
 * Count of unique elements in vocabulary
 *
 * @param vocab the vocabulary structure
 */
size_t
nlk_vocab_size(struct nlk_vocab_t **vocab)
{
    return HASH_COUNT(*vocab);
}

/**
 * Count of unique words in vocabulary
 *
 * @param vocab the vocabulary structure
 *
 * @return number of unique words in dictionary
 */
size_t
nlk_vocab_words_size(struct nlk_vocab_t **vocab)
{
    struct nlk_vocab_t *vocab_word;
    struct nlk_vocab_t *tmp;
    enum nlk_vocab_type_t vt;
    size_t n = 0;

    HASH_ITER(hh, *vocab, vocab_word, tmp) {
        if(vocab_word->word != NULL) {
            vt = vocab_word->type;
            if(vt == NLK_VOCAB_WORD || vt == NLK_VOCAB_SPECIAL) {
                n++;
            }
        }
    }
    return n;
}

/**
 * Highest index in vocabulary (requires sorted vocab)
 *
 * @param vocab the vocabulary structure
 * 
 * @return the highest index in the vocabulary
 */
size_t
nlk_vocab_last_index(struct nlk_vocab_t **vocab)
{
    struct nlk_vocab_t *vocab_word;
    struct nlk_vocab_t *tmp;
    size_t n = 0;

    /* @TODO replace this with reverse iter => last = highest index... */
    HASH_ITER(hh, *vocab, vocab_word, tmp) {
        if(vocab_word->word != NULL) {
            if(vocab_word->index > n) {
                n = vocab_word->index;
            }
        }
    }
    return n;
}

/**
 * Total of word counts in vocabulary
 *
 * @param vocab the vocabulary structure
 */
uint64_t
nlk_vocab_total(struct nlk_vocab_t **vocab)
{
    uint64_t total = 0;
    struct nlk_vocab_t *vocab_word;
    struct nlk_vocab_t *tmp;

    HASH_ITER(hh, *vocab, vocab_word, tmp) {
        total += vocab_word->count;
    }
    return total;
}

/**
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
nlk_vocab_reduce(struct nlk_vocab_t **vocab, const uint64_t min_count)
{
    struct nlk_vocab_t *vi;
    struct nlk_vocab_t *tmp;
    struct nlk_vocab_t *start_symbol;

    HASH_FIND_STR(*vocab, NLK_START_SYMBOL, start_symbol);

    HASH_ITER(hh, *vocab, vi, tmp) {
        /* always protect end symbol and paragraphs */
        if(vi->count < min_count && vi->type == NLK_VOCAB_WORD) {
            /* free structure contents */
            if(vi->word != NULL) {
                free(vi->word);
                vi->word = NULL;
            }
            /* delete from hashmap and free structure **/
            HASH_DEL(*vocab, vi);
            free(vi);
        }
    }

    /* call sort to update the index */
    nlk_vocab_sort(vocab);

}

/**
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
nlk_vocab_reduce_replace(struct nlk_vocab_t **vocab, const size_t min_count)
{
    struct nlk_vocab_t *vi;
    struct nlk_vocab_t *tmp;
    struct nlk_vocab_t *unk_symbol;
    struct nlk_vocab_t *start_symbol;
    uint64_t unk_count = 0;

    HASH_FIND_STR(*vocab, NLK_UNK_SYMBOL, unk_symbol);
    HASH_FIND_STR(*vocab, NLK_START_SYMBOL, start_symbol);

    HASH_ITER(hh, *vocab, vi, tmp) {
        if(vi->count < min_count) {
            /* ignore special symbols */
            if(vi->type != NLK_VOCAB_WORD) {
                continue;
            }

            unk_count += vi->count;

            /* free structure contents */
            if(vi->word != NULL) {
                free(vi->word);
                vi->word = NULL;
            }
            /* delete from hashmap and free structure **/
            HASH_DEL(*vocab, vi);
            free(vi);
        }
    }

    /* 
     * does the symbol already exist? i.e. not first call to this function 
     */
    if(unk_symbol == NULL) { /* nope, lets create it */
        unk_symbol = nlk_vocab_add_item(vocab, NLK_UNK_SYMBOL, 
                                        unk_count, NLK_VOCAB_SPECIAL);
    }
    else { /* just an update */
        unk_symbol->count = unk_symbol->count + unk_count;
    }

    /* call sort to update the index */
    nlk_vocab_sort(vocab);
}

/** 
 * Vocab item comparator - used for sorting with most frequent words first
 *
 * @param a vocab item
 * @param b another vocab item
 *
 * @returns positive if b.count > a.count, negative if b.count < a.count
 */
static int 
nlk_vocab_item_comparator(struct nlk_vocab_t *a, struct nlk_vocab_t *b)
{
    return (b->count - a->count);
}

/**
 * Vocab item comparator - used for sorting with the least frequent words first
 *
 * @param a vocab item
 * @param b another vocab item
 *
 * @returns positive if b.count < a.count, negative if b.count > a.count
 */
int 
nlk_vocab_item_comparator_reverse(struct nlk_vocab_t *a, struct nlk_vocab_t *b)
{
    return (a->count - b->count);
}


/**
 * Sort the vocabulary by word count, most frequent first i.e. desc by count.
 * Also updates the *index* property, the start symbol's index is always 0.
 *
 * @param vocab     the vocabulary structure
 */
void nlk_vocab_sort(struct nlk_vocab_t **vocab)
{
    struct nlk_vocab_t *vi;
    struct nlk_vocab_t *start_symbol;
    size_t ii = 1;  /* 0 is the start symbol */

    HASH_FIND_STR(*vocab, NLK_START_SYMBOL, start_symbol);
    start_symbol->index = 0;

    HASH_SORT(*vocab, nlk_vocab_item_comparator);
    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi == start_symbol) {
            continue;
        }
        vi->index = ii;
        ii++;
    }
}


/**
 * Creates (alloc) a code structure for holding huffman coding data
 */
struct nlk_vocab_code_t *
nlk_vocab_code_create(uint8_t code_length)
{
    struct nlk_vocab_code_t *hc = NULL;

    hc = (struct nlk_vocab_code_t *) malloc(sizeof(struct nlk_vocab_code_t));
    nlk_assert(hc != NULL, "failed to allocate memory for code struct");

    hc->length = code_length;

    /*
    hc->code = (uint8_t *) malloc(sizeof(uint8_t) * code_length);
    nlk_assert(hc->code != NULL, "failed to allocate memory for code array");

    hc->point = (uint32_t *) malloc(sizeof(uint32_t) * code_length);
    nlk_assert(hc->point != NULL, "failed to allocate memory for code point");
    */

    return hc;

error:
    NLK_ERROR_ABORT("", NLK_ENOMEM);
}


void
nlk_vocab_code_free(struct nlk_vocab_code_t *hc)
{
    if(hc != NULL) {
        /*
        if(hc->code != NULL) {
            free(hc->code);
            hc->code = NULL;
        }
        if(hc->point != NULL) {
            free(hc->point);
            hc->point = NULL;
        }
        */
        free(hc);
        hc = NULL;
    }
}


/**
 * Create Huffman binary tree for hierarchical softmax (HS).
 * Adds *code* (huffman encoded representation) and HS *point* fields to 
 * vocabulary items.
 * 
 * @params vocab    the vocabulary
 *
 * @warning
 * Requires vocabulary to be sorted.
 * @endwarning
 *
 * @note
 * implementation owes thanks to word2vec https://code.google.com/p/word2vec/
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
nlk_vocab_encode_huffman(struct nlk_vocab_t **vocab)
{
    uint32_t          vsize;                /* the vocabulary size */ 
    size_t            nn;                   /* for indexing over nodes */
    size_t            min1;                 /*  the minimum index */
    size_t            min2;                 /* the second minimum index */
    int64_t           pos1;                 /* position in queue1 */
    int64_t           pos2;                 /* position in queue2 */
    size_t            code_length;          /* for holding code lengths */
    uint8_t           code[NLK_MAX_CODE];   /* temporay storage for a code */
    size_t            point[NLK_MAX_CODE];  /* temporay storage for a point */

    size_t                     ii;
    struct nlk_vocab_t        *vi;          /* vocabulary iterator */

    /**
     * The vocabulary is sorted so we can use the fast version O(n) of the 
     * huffman tree building algorithm.
     * First we allocate space for the nodes and create the queues 
     */
    vsize = nlk_vocab_words_size(vocab);
    uint64_t *count = (uint64_t *) calloc(vsize * 2 + 1, sizeof(uint64_t));
    uint8_t *binary = (uint8_t *)  calloc(vsize * 2 + 1, sizeof(uint8_t));
    uint32_t *parent = (uint32_t *)    calloc(vsize * 2 + 1, sizeof(uint32_t));
    if(count == NULL || binary == NULL || parent == NULL) {
        NLK_ERROR_VOID("failed to allocate memory for huffman tree", 
                       NLK_ENOMEM);
    }

    /**
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
    nn = 0;
    for(vi = *vocab; vi != NULL; vi = (struct nlk_vocab_t *)(vi->hh.next)) {
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
        vi->hc = nlk_vocab_code_create(code_length);
        vi->hc->point[0] = vsize - 2;
        for(ii = 0; ii < code_length; ii++) {
            vi->hc->code[code_length - ii - 1] = code[ii];
            vi->hc->point[code_length - ii] = point[ii] - vsize;
        }
        nn++;
    }
    free(count);
    free(parent);
    free(binary);
}

/**
 * Return the maximum code length of any word in the vocabulary
 *
 * @param vocab the vocabulary
 *
 * @return the maximum code length of any word in the vocabulary
 */
size_t 
vocab_max_code_length(struct nlk_vocab_t **vocab)
{
    struct nlk_vocab_t *vi = *vocab;
    size_t code_length = 0;

    if(vi->hc == NULL) {
        return 0;
    }

    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi->hc->length > code_length) {
            code_length = vi->hc->length;
        }
    }
    return code_length;
}


/**
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
nlk_vocab_export(const char *filepath, struct nlk_vocab_t **vocab)
{
    struct nlk_vocab_t *vi;
    struct nlk_vocab_t *vstart;
    FILE *out = fopen(filepath, "wb");
    if(out == NULL) {
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }

    HASH_FIND_STR(*vocab, NLK_START_SYMBOL, vstart);
    fprintf(out, "%s %"PRIu64"\n", vstart->word, vstart->count);

    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi == vstart) {
            continue;
        }
        fprintf(out, "%s %"PRIu64"\n", vi->word, vi->count);
    }
    fclose(out);
    out = NULL;
    return NLK_SUCCESS;
}

/**
 * Load the (simple) vocabulary structure from disk
 * E.g. saved via *nlk_save_vocab*
 *
 * @param filepath          the path of the file to which will be read
 * @param max_word_size     maximum string size
 * @param counts            contains counts 
 *
 * @return the loaded vocabulary
 */
struct nlk_vocab_t *
nlk_vocab_import(const char *filepath, const size_t max_word_size, 
                 const bool counts)
{
    struct nlk_vocab_t *vocab = nlk_vocab_init();
    struct nlk_vocab_t *vocab_item = NULL;
    struct nlk_vocab_t *start;
    size_t index = 0;

    uint64_t count = 0;
    char *word = (char *) calloc(max_word_size, sizeof(char));
    FILE *in = fopen(filepath, "rb");
    if(in == NULL) {
        NLK_ERROR_NULL(strerror(errno), errno);
        /* unreachable */
    }

    HASH_FIND_STR(vocab, NLK_START_SYMBOL, start);

    nlk_read_word(in, word, max_word_size);
    if(counts) {
        if(fscanf(in, "%"SCNu64"\n", &count) != 1) {
            NLK_ERROR_NULL("Parsing error", NLK_FAILURE);
        }
        start->count += count;
    }


    while(!feof(in)) {
        if(nlk_read_word(in, word, max_word_size) == EOF) {
            break;
        }
        if(counts) {
            if(fscanf(in, "%"SCNu64"\n", &count) != 1) {
                NLK_ERROR_NULL("Parsing error", NLK_FAILURE);
            }
        }

        if(strcmp(word, NLK_START_SYMBOL) == 0) {
            start->index = index;
        }

        vocab_item = nlk_vocab_add_item(&vocab, word, count, NLK_VOCAB_WORD);
        vocab_item->index = index;
        index++;
    }

    /* free/close */
    free(word);
    word = NULL;
    fclose(in);
    in = NULL;

    /* sort */
    if(counts) {
        nlk_vocab_sort(&vocab);
    }

    return vocab;
}

void
nlk_vocab_save_item(struct nlk_vocab_t *vi, FILE *file)
{
    size_t ii = 0;
    uint8_t code_length = 0;

    if(vi->hc != NULL) {
        code_length = vi->hc->length;
    }
        
    fprintf(file, "%s\t%d\t%zu\t%"PRIu64"\t%"PRIu8"\t", 
             vi->word, vi->type, vi->index, vi->count, code_length);

    if(code_length != 0) {
        /* code */
        for(ii = 0; ii < (uint8_t) (code_length - 1); ii++) {
            fprintf(file, "%"PRIu8" ", vi->hc->code[ii]);
        }
        fprintf(file, "%"PRIu8"\t", vi->hc->code[code_length - 1]);

        /* point */
        for(ii = 0; ii < (uint8_t) (code_length - 1); ii++) {
            fprintf(file, "%"PRIu32" ", vi->hc->point[ii]);
        }
        fprintf(file, "%"PRIu32"\n", vi->hc->point[code_length - 1]);
    } else {
        fprintf(file, "\n");
    }
}


/**
 * Save the vocabulary structure to disk.
 *
 * @param vocab             the vocabulary structure
 * @param file              the file to which the vocabulary will be saved
 *
 * @note
 * There is no escaping if the word strings somehow contains tabs and newlines.
 * The format is one vocabulary item per line, tab separated values. The order 
 * of the items is the same as specified in the structure definition
 * (e.g. sort order).
 * @endnote
 */
void
nlk_vocab_save(struct nlk_vocab_t **vocab, FILE *file)
{
    struct nlk_vocab_t *vi;
    struct nlk_vocab_t *vstart;

    /* save header */
    fprintf(file, "%zu\n", nlk_vocab_size(vocab));

    /* save start symbol */
    HASH_FIND_STR(*vocab, NLK_START_SYMBOL, vstart);
    nlk_vocab_save_item(vstart, file);

    /* save the rest */
    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi == vstart) {
            continue;
        }
        nlk_vocab_save_item(vi, file);
    }
}


/**
 * Loads a vocabulary structure from a file
 */
struct nlk_vocab_t *
nlk_vocab_load(FILE *file)
{
    struct nlk_vocab_t *vocab = NULL;
    struct nlk_vocab_t *vocab_word = NULL;
    size_t vocab_size = 0;
    size_t vv, ii;
    int ret = 0;
    char *word = NULL;
    int type = 0;
    size_t index = 0;
    uint64_t count = 0;
    uint8_t code_length = 0;

    /* load header */
    ret = fscanf(file, "%zu\n", &vocab_size);
    nlk_assert(ret > 0, "invalid header");

    /* allocate space for words */
    word = (char *) malloc(NLK_MAX_WORD_SIZE * sizeof(char));
    if(word == NULL) {
        NLK_ERROR_NULL("not enough memory for a word", NLK_ENOMEM);
        /* unreachable */
    }

    for(vv = 0; vv < vocab_size; vv++) {
        ret = fscanf(file, "%s\t%d\t%zu\t%"SCNu64"\t%"SCNu8"\t", 
                     word, &type, &index, &count, &code_length);
        nlk_assert(ret > 0, "invalid word");
        vocab_word = nlk_vocab_add_item(&vocab, word, count, type);
        if(vocab_word == NULL) {
            NLK_ERROR_NULL("unable to add to vocabulary", NLK_FAILURE);
            /* unreachable */
        }

        if(code_length != 0) {
            vocab_word->hc = nlk_vocab_code_create(code_length);
            vocab_word->hc->length = code_length;

            /* code */
            for(ii = 0; ii < (uint8_t)(code_length - 1); ii++) {
                ret = fscanf(file, "%"SCNu8" ", &vocab_word->hc->code[ii]);
                nlk_assert(ret > 0, "invalid code");
            }
            ret = fscanf(file, "%"SCNu8"\t", 
                         &vocab_word->hc->code[code_length - 1]);
            nlk_assert(ret > 0, "invalid code");

            /* point */
            for(ii = 0; ii < (uint8_t)(code_length - 1); ii++) {
                ret = fscanf(file, "%"SCNu32" ", &vocab_word->hc->point[ii]);
                nlk_assert(ret > 0, "invalid point");
            }
            ret = fscanf(file, "%"SCNu32"\n",
                         &vocab_word->hc->point[code_length - 1]);
            nlk_assert(ret > 0, "invalid point");


        } else {
            fscanf(file, "\n");
        }
    }

    nlk_vocab_sort(&vocab);
    return vocab;


error:
    NLK_ERROR_NULL("invalid vocabulary", NLK_FAILURE);
    /* unreachable */
}


/**
 * Find a word (string) in the vocabulary
 *
 * @param vocab     the vocabulary
 * @param word      the word
 *
 * @return the vocabulary item corresponding to the word or NULL if not found
 */
struct nlk_vocab_t *
nlk_vocab_find(struct nlk_vocab_t **vocab, char *word)
{
    struct nlk_vocab_t *vocab_word;

    HASH_FIND_STR(*vocab, word, vocab_word);
    
    return vocab_word;
}

/**
 * Find a word (string) in the vocabulary
 *
 * @param vocab     the vocabulary
 * @param index     the word index
 *
 * @return the vocabulary item corresponding to the word index or NULL
 */
struct nlk_vocab_t *
nlk_vocab_at_index(struct nlk_vocab_t **vocab, size_t index)
{
    struct nlk_vocab_t *vi;

    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        if(vi->index == index) {
            return vi;
        }
    }

    return NULL;
}

/** 
 * Print a vocabularized array
 */
void
nlk_vocab_print_line(struct nlk_vocab_t **varray, size_t length, bool indexes)
{
    for(size_t ii = 0; ii < length; ii++) {
        if(indexes == true) {
            printf("%s [%zu] ", varray[ii]->word, varray[ii]->index);
        } else {
            printf("%s ", varray[ii]->word);
        }
    }
    printf("\n");
}

/**
 * Save a NEG table to disk, used for debug purposes
 */
void
nlk_vocab_neg_table_save(struct nlk_vocab_t **vocab, size_t *neg_table, size_t size, 
                         char *filepath)
{
    FILE *fp;
    size_t target;
    size_t ii;
    size_t count;
    struct nlk_vocab_t *word;


    fp = fopen(filepath, "w");
    if(fp == NULL) {
        NLK_ERROR_VOID("unable to open neg table file", NLK_FAILURE);
        /* unreachable */
    }

    ii = 0;
    target = neg_table[ii];
    word = nlk_vocab_at_index(vocab, target);
    count = 0;

    while(ii < size) {
        if(target != neg_table[ii]) {
            fprintf(fp, "%s\t%zu\n", word->word, count);
            target = neg_table[ii];
            word = nlk_vocab_at_index(vocab, target);
            count = 1;
        } else {
            count++;
        }

        ii++;
    }
    fprintf(fp, "%s\t%zu\n", word->word, count);

    fclose(fp);
    fp = NULL;
}

/**
 * Create NEG table
 *
 * @param vocab the vocabulary
 * @param size  the NEG table size
 * @param power defaults (power=0) to 0.75
 */
size_t *
nlk_vocab_neg_table_create(struct nlk_vocab_t **vocab, const size_t size, 
                           double power)
{
    struct nlk_vocab_t *vi;
    size_t z = 0;
    size_t table_pos = 0;
    size_t index = 0;
    double d1 = 0;

    /* allocate */
    if(size == 0) {
        NLK_ERROR_NULL("attempt at allocation with 0 size", NLK_EINVAL);
    }
    size_t *neg_table = (size_t *) malloc(size * sizeof(size_t));
    if(neg_table == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for NEG table", NLK_ENOMEM);
        /* unreachable */
    }
    
    /* calculate "z" */
    for(vi = *vocab; vi != NULL; vi = vi->hh.next) {
        z += pow(vi->count, power);
    }

    /* initialize table */
    vi = *vocab;
    index = vi->index;
    d1 = pow(vi->count, power) / (double) z;

    for(table_pos = 0; table_pos < size; table_pos++) {
        neg_table[table_pos] = index;

        if(table_pos / (float)size > d1) {
            vi = vi->hh.next;
            if(vi == NULL) {
                vi = vi->hh.prev;
            }
            index = vi->index;
            d1 += pow(vi->count, power) / (double)z;
        }
    }

    /* nlk_vocab_neg_table_save(vocab, neg_table, size, "neg.debug"); */
    return neg_table;
}

/**
 * @brief Return the start symbol for the vocabulary
 *
 * @param vocab the vocabulary
 *
 * @return the start symbol vocabulary item (NULL if not found)
 *
 */
struct nlk_vocab_t *
nlk_vocab_get_start_symbol(struct nlk_vocab_t **vocab) 
{
    struct nlk_vocab_t *vi;

    HASH_FIND_STR(*vocab, NLK_START_SYMBOL, vi);

    return vi;
}



/**
 * @param sample                sample rate for subsampling frequent words
 *                              - if <= 0, no subsampling will happen
 */
void
nlk_vocab_line_subsample(const struct nlk_line_t *in, 
                         const uint64_t total_words, 
                         const float sample, struct nlk_line_t *out)
{
    struct nlk_vocab_t *vocab_word;
    float prob;                     /* probability of being sampled */
    float r;                        /* random number */
    size_t out_idx = 0;

    out->line_id = in->line_id;

    for(size_t ii = 0; ii < in->len; ii++) {
        vocab_word = in->varray[ii];
        if(sample <= 0) {
            /* simply copy */
            out->varray[out_idx] = vocab_word;
            out_idx++;
            continue;
        }
        /* calculate sampling probability */
        prob = sqrt((float)vocab_word->count / (sample * total_words)) + 1;
        prob *= (sample * total_words) / (float) vocab_word->count;

        /* "flip coin " */
        r = nlk_random_xs1024_float();
        if(prob < r) {
            continue;
            /* 
             * word was not sampled 
             * unreachable
             */
        } else {
            out->varray[out_idx] = vocab_word;
            out_idx++;
        }
    } /* end of input array */

    /* set length */
    out->len = out_idx;
}

/**
 * "Vocabularizes" a series of words: for each word, the output vector will 
 * contain a pointer to the vocabulary item of that word.
 *
 * @param vocab                 the vocabulary used for vectorization
 * @param paragraph             an array of strings (char arrays)
 * @param replacement           vocab to use for words not in the vocabulary
 *                              - NULL means do not replace
 * @param varray                the vocabulary item array to be written
 *
 * @returns number of words vocabularized (size of the array)
 *
 * @note
 * The paragraph is expected to be terminated by a null word (word[0] = 0)
 * as generated by nlk_read_line(). This will be replaced by the end sentence 
 * token.
 *
 * If replacement is NULL, the word will simply be ignored, i.e. treated 
 * as if it was not there and the returned size will be small than the size of 
 * the paragraph.
 *
 * @endnote
 */
size_t
nlk_vocab_vocabularize(struct nlk_vocab_t **vocab,  char **paragraph, 
                       struct nlk_vocab_t *replacement,
                       struct nlk_vocab_t **varray) 
              
{
    struct nlk_vocab_t *vocab_word; /* vocab item that corresponds to word */
    size_t par_idx;                 /* position in paragraph */
    size_t vec_idx = 0;             /* position in vectorized array */

    for(par_idx = 0; paragraph[par_idx][0] != '\0'; par_idx++) {
        HASH_FIND_STR(*vocab, paragraph[par_idx], vocab_word);
        if(vocab_word != NULL) {
            /* the word is in the vocabulary */
            varray[vec_idx] = vocab_word;
            vec_idx++;
        } else if(vocab_word == NULL && replacement != NULL) { 
            /* word NOT in vocabulary but will be replaced */ 
            varray[vec_idx] = replacement;
            vec_idx++;
        }
        /* word not in vocabulary, not replacing, do nothing */
    }
   
    return vec_idx; /* = the count, not index because of ++ */
}


uint64_t
nlk_vocab_count_words_worker(struct nlk_vocab_t **vocab, const char *file_path,
                             const bool line_ids, const size_t total_lines, 
                             const int thread_id, const int num_threads)
{
    uint64_t total_words = 0;
    size_t cur_line;
    size_t end_line;
    size_t line_len;
    size_t par_id;
    size_t *par_id_ptr = NULL;
    int ret = 0;
     

    /**@section Init
     */
    if(line_ids) {
        par_id_ptr = &par_id;
    } else {
        par_id_ptr = NULL;

    }
    /* allocate memory for reading from the input file */
    char **text_line = nlk_text_line_create();
    char *buffer = malloc(sizeof(char) * NLK_BUFFER_SIZE);

    /* for converting to a vocabularized representation of text */
    struct nlk_vocab_t *vectorized[NLK_MAX_LINE_SIZE];

    /* open file */
    errno = 0;
    int fd = nlk_open(file_path);
    if(fd < 0) {
        NLK_ERROR_ABORT(strerror(errno), errno);
        /* unreachable */
    }

    /* set train file part position */
    cur_line = nlk_text_get_split_start_line(total_lines, num_threads, 
                                              thread_id);
    nlk_text_goto_line(fd, cur_line);
    end_line = nlk_text_get_split_end_line(total_lines, num_threads, 
                                              thread_id);

    /**@section Count
     */
    while(ret != EOF && cur_line < end_line) {
        /* read line */
        ret = nlk_read_line(fd, text_line, par_id_ptr, buffer);
        
        /* vocabularize */
        line_len = nlk_vocab_vocabularize(vocab, text_line, NULL, 
                                          vectorized); 

        /* increment word and line counts */
        total_words += line_len;
        cur_line++;
    }

    /* end of file */
    close(fd);
    fd = 0;
    free(buffer);
    nlk_text_line_free(text_line);

    return total_words;
}   


/**
 * Count how many vocabulary words there are in the file
 */
uint64_t
nlk_vocab_count_words(struct nlk_vocab_t **vocab, const char *file_path,
                      const bool line_ids, const size_t total_lines)
{
    size_t total_words = 0;
    int num_threads = omp_get_num_threads();

    /** @section Parallel Count (Map)
     */
#pragma omp parallel for reduction(+ : total_words)
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        total_words = nlk_vocab_count_words_worker(vocab, file_path,
                                                   line_ids, total_lines, 
                                                   thread_id, num_threads);
    }
    return total_words;
}

/**
 * Read line from file and vocabularize it.
 *
 * @param fp        the file pointer to read from
 * @param vocab     the vocabulary
 * @param text_line temporary memory for text read from file
 * @param v         vocalularized line
 */
void
nlk_vocab_read_vocabularize(int fd, const bool line_ids,
                            struct nlk_vocab_t **vocab, 
                            struct nlk_vocab_t *replacement,
                            char **text_line, struct nlk_line_t *v, char *buf)
{
    int ret;
    size_t *line_id = NULL;
    if(line_ids) {
        line_id = &v->line_id;
    }

    /* read text line */
    ret = nlk_read_line(fd, text_line, line_id, buf);

    /* unexpected end of file (empty line) */
    if(ret == EOF && text_line[0][0] == '\0' ) {
        v->len = 0; 
        return;
    }

    /* vocabularize */
    v->len = nlk_vocab_vocabularize(vocab, text_line, replacement, v->varray); 


#ifndef NCHECKS 
    if(v->len > NLK_MAX_LINE_SIZE) {
        NLK_ERROR_VOID("Bad line length", NLK_EINVAL);
    }
#endif
}

/**
 * Create a line 
 *
 * @param size  size of the line
 *
 * @return the line
 */
struct nlk_line_t *
nlk_line_create(const size_t size) {
    struct nlk_line_t *line;

    if(size == 0) {
        NLK_ERROR_NULL("invalid size parameter", NLK_EINVAL);
        /* unreachable */
    }
    
    line = malloc(sizeof(struct nlk_line_t));
    if(line == NULL) {
        NLK_ERROR_NULL("insufficient memory for line", NLK_ENOMEM);
        /* unreachable */
    }

    line->varray = (struct nlk_vocab_t **) malloc(sizeof(struct nlk_vocab_t *) 
                                                  * size);
    if(line->varray == NULL) {
        NLK_ERROR_NULL("insufficient memory for line", NLK_ENOMEM);
        /* unreachable */
    }

    return line;
}


/**
 * Gets the id (index) of the words in the line
 *
 * @param line  the line
 * @param ids   an array overwritten with the indices of the words in the line
 */
void
nlk_line_ids(struct nlk_line_t *line, size_t *ids)
{
    for(size_t ii = 0; ii < line->len; ii++) {
        ids[ii] = line->varray[ii]->index;
    }
}


/**
 * Free a line
 *
 * @param line  the line to free
 */
void
nlk_line_free(struct nlk_line_t *line)
{
    if(line == NULL) {
        return;
    }

    if(line->varray != NULL) {
        free(line->varray);
        line->varray = NULL;
    }

    free(line);
}
