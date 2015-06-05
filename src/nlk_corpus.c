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

#include <time.h>
#include <inttypes.h>

#include <omp.h>

#include "nlk_text.h"
#include "nlk_vocabulary.h"
#include "nlk_util.h"
#include "nlk_tic.h"

#include "nlk_corpus.h"


/**
 * Displays the progress stats while building a corpus from a file
 *
 * @param start         the clocks used just before starting to read the file
 */
static void
nlk_corpus_display_progress(const size_t line_counter, 
                            const size_t total_lines, const clock_t start)
{
    double progress;
    double speed;
    char display_str[256];

    clock_t now = clock();

    progress = (line_counter / (double) total_lines) * 100;
    speed = line_counter / ((double)(now - start + 1) / 
            (double)CLOCKS_PER_SEC * 1000),

    snprintf(display_str, 256, 
            "Corpus Progress: %.2f%% Lines/s: %.2f Threads: %d", 
            progress, speed, omp_get_num_threads());

    nlk_tic(display_str, false);
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
nlk_corpus_read(char *file_path, struct nlk_vocab_t **vocab, 
                const bool verbose)
{
    struct nlk_corpus_t *corpus;
    size_t total_lines;

    const int num_threads = omp_get_num_threads();

    /* count lines */
    if(verbose) {
        printf("Reading Corpus: %s\n", file_path);
        nlk_tic("counting lines...", true);
    }

    total_lines = nlk_text_count_lines(file_path);

    if(verbose) {
        nlk_tic("done: ", false);
        printf("%zu lines\n", total_lines);
    }

    /* allocate corpus */
    corpus = (struct nlk_corpus_t *) malloc(sizeof(struct nlk_corpus_t));
    if(corpus == NULL) {
      NLK_ERROR_NULL("unable to allocate memory for the vocabularized file", 
                     NLK_ENOMEM);

    }
    corpus->len = total_lines;

    /* allocate memory for the file (the line array) */
    corpus->lines = (struct nlk_line_t *) 
                     calloc(total_lines, sizeof(struct nlk_line_t));

    if(corpus->lines == NULL) {
        NLK_ERROR_NULL("unable to allocate memory for the vocabularized file", 
                        NLK_ENOMEM);
        /* unreachable */
    }
    struct nlk_line_t *lines = corpus->lines;

    uint64_t word_count = 0;
    size_t line_counter = 0; 
    size_t updated = 0;
    clock_t start = clock();

    /* thread it */
#pragma omp parallel reduction(+ : word_count) shared(line_counter, updated) 
{
    /* allocate memory for a line of text */
    char **text_line = nlk_text_line_create();

    /* memory for vocabularizing */
    struct nlk_vocab_t *varray[NLK_LM_MAX_LINE_SIZE];
    struct nlk_line_t vline;
    vline.varray = varray;


#pragma omp for
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        /** @subsection File Reading Position
         * open file and get start and end positions for thread 
         */
        FILE *fp = fopen(file_path, "rb");
        const size_t line_start = nlk_text_get_split_start_line(total_lines, 
                                                                num_threads, 
                                                                thread_id);
        const size_t end_line = nlk_text_get_split_end_line(total_lines, 
                                                            num_threads, 
                                                            thread_id);
        /* go to start position */
        size_t line_cur = line_start;
        nlk_text_goto_line(fp, line_cur);


        /** @subsection Read lines
         */
        while(line_cur <= end_line) {
            /* display */
            if(verbose) {
                if(line_counter - updated > 1000) {
                    updated = line_counter;
                    nlk_corpus_display_progress(line_counter, total_lines, 
                                                start);
                }
            } /* end of display */

            /* read */
            nlk_vocab_read_vocabularize(fp, vocab, text_line, &vline);
         
            /* check for errors */
            if(vline.len == 0) {
                lines[line_cur].varray = NULL;
                lines[line_cur].len = 0;
                lines[line_cur].line_id = (size_t)-1;
                /* nlk_debug("\nBad line: %zu\n", line_cur); */
                line_cur++;
                line_counter++;
                continue;
            }

            /* create */
            lines[line_cur].varray = (struct nlk_vocab_t **) 
                                        malloc(sizeof(struct nlk_vocab_t *) * 
                                               vline.len);
            if(lines[line_cur].varray == NULL) {
                NLK_ERROR_ABORT("unable to allocate memory for line", 
                                NLK_ENOMEM);
                /* unreachable */
            }

            /* copy */
            lines[line_cur].len = vline.len;
            lines[line_cur].line_id = vline.line_id;

            for(size_t ii = 0; ii < vline.len; ii++) {
                lines[line_cur].varray[ii] = varray[ii];
            }

            word_count += vline.len;
            line_cur++;
            line_counter++;

        } /* end of lines for thread */
        fclose(fp);
        fp = NULL;
    } /* end of for() threads */

    if(verbose) {
        nlk_corpus_display_progress(line_counter, total_lines, start);
    }

    /* free memory */
    nlk_text_line_free(text_line);
    
} /* end of parallel region */

    corpus->count = word_count;

    if(verbose) {
        printf("\n");
        nlk_tic("done reading corpus: ", false);
        printf("%"PRIu64" words\n", word_count);
    }

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
