/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
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
 * @file nlk_corpus.c
 * Create and use corpus structures
 */

#include <omp.h>

#include "nlk_text.h"
#include "nlk_vocabulary.h"
#include "nlk_util.h"

#include "nlk_corpus.h"

/**
 *
 */
uint64_t
nlk_vocab_read_file_worker(char *file_path, const size_t total_lines, 
                           struct nlk_vocab_t **vocab, const int thread_id, 
                           const int num_threads, 
                           struct nlk_line_t *lines)
{
    uint64_t word_count = 0;

    /* open file and get start and end positions for thread */
    FILE *fp = fopen(file_path, "rb");
    const size_t line_start = nlk_text_get_split_start_line(total_lines, 
                                                            num_threads, 
                                                            thread_id);
    const size_t end_line = nlk_text_get_split_end_line(total_lines, 
                                                        num_threads, 
                                                        thread_id);

    /* allocate memory for a line of text */
    char **text_line = nlk_text_line_create();

    /* memory for vocabularizing */
    struct nlk_vocab_t *varray[NLK_LM_MAX_LINE_SIZE];
    struct nlk_line_t vline;
    vline.varray = varray;

    /* go to start position */
    size_t line_cur = line_start;
    nlk_text_goto_line(fp, line_cur);

    while(line_cur <= end_line) {
        nlk_vocab_read_vocabularize(fp, vocab, text_line, &vline);
     
        /* create */
        if(vline.len != 0) {
            lines[line_cur].varray = (struct nlk_vocab_t **) 
                                      malloc(sizeof(struct nlk_vocab_t *) * 
                                             vline.len);
        } else {
            lines[line_cur].varray = NULL;
        }
        /* copy */
        lines[line_cur].len = vline.len;
        lines[line_cur].line_id = vline.line_id;

        for(size_t ii = 0; ii < vline.len; ii++) {
            lines[line_cur].varray[ii] = varray[ii];
        }

        word_count += vline.len;
        line_cur++;
    }

    /* free memory */
    nlk_text_line_free(text_line);
    fclose(fp);
    fp = NULL;

    return word_count;
}

/**
 * Reads a corpus (in id-text line delimited format)
 *
 * @param file_path the path to the corpus
 * @param vocab     the vocabulary to use
 *
 * @return a corpus structure
 */
struct nlk_corpus_t *
nlk_corpus_read(char *file_path, struct nlk_vocab_t **vocab)
{
    struct nlk_corpus_t *corpus;
    size_t total_lines;
    uint64_t total_words = 0;

    const int num_threads = omp_get_num_procs();

    /* count lines */
    FILE *lc = fopen(file_path, "rb");
    if(lc == NULL) {
        NLK_ERROR_ABORT(strerror(errno), errno);
        /* unreachable */
    }
    total_lines = nlk_text_count_lines(lc);
    fclose(lc);
    lc = NULL;

    /* allocate corpus */
    corpus = (struct nlk_corpus_t *) malloc(sizeof(struct nlk_corpus_t));
    if(corpus == NULL) {
      NLK_ERROR_NULL("unable to allocate memory for the vocabularized file", 
                     NLK_ENOMEM);

    }
    corpus->len = total_lines;

    /* allocate memory for the file (the line array) */
    corpus->lines = (struct nlk_line_t *) 
                     malloc(total_lines * sizeof(struct nlk_line_t));

    if(corpus->lines == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for the vocabularized file", 
                        NLK_ENOMEM);
        /* unreachable */
    }

    /* thread it */
#pragma omp parallel for reduction(+ : total_words)
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        total_words = nlk_vocab_read_file_worker(file_path,  total_lines, 
                                                 vocab, thread_id, 
                                                 num_threads, corpus->lines);

    } /* end of parallel for */

    corpus->count = total_words;

    return corpus;
}

/**
 * Free a corpus structure
 *
 * @param corpus    the corpus structure to free
 */
void 
nlk_corpus_free(struct nlk_corpus_t *corpus)
{
    /* free individual lines */
    for(size_t ii = 0; ii < corpus->len; ii++) {
        if(corpus->lines[ii].varray != NULL) {
            free(corpus->lines[ii].varray);
            corpus->lines[ii].varray = NULL;
        }
    }

    /* free the lines array */
    free(corpus->lines);
    corpus->lines = NULL;

    /* free the corpus */
    free(corpus);
}


/**
 * Counts the number of word occurrences in the subset of a corpus
 *
 * @param corpus    the corpus
 * @param ids       the ids of the corpus that make up the subset
 * @param           n_ids   the size of the ids array
 *
 * @return the number of word occurrences in the corpus subset
 */
uint64_t
nlk_corpus_subset_count(const struct nlk_corpus_t *corpus, const size_t *ids, 
                        const size_t n_ids)
{
    uint64_t total = 0;
    for(size_t ii = 0; ii < corpus->len; ii++) {
        if(nlk_in(corpus->lines[ii].line_id, ids, n_ids)) {
            total += corpus->lines[ii].len;
        }
    }
    return total;
}
