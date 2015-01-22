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


/** @file nlk_window.c
 * Input/Output Window and Context
 */


#include <stdbool.h>

#include "tinymt32.h"

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_random.h"
#include "nlk_window.h"

/** @fn size_t nlk_context_window(nlk_Vocab **line_array,
 *                                const size_t line_length,
                                  const bool self, const size_t before, 
                                  const size_t after, nlk_Context **contexts)
 *
 *  @param varray           vocab items for the line/paragraph/document
 *  @param line_length      lengh of the line array
 *  @param self             include the word itself (center) in its window?
 *  @param size_before      how many words from before the current word to 
 *                          include in the context window
 *  @param size_after       how many words after the current word to include
 *                          in the context window
 *  @param random_windows   use word2vec style random_window in range
 *                          before=[1, before], after=[1, after]
 *  @param r                the random number / rng state
 *  @param vocab_par        if not NULL, the paragraph vocabulary item
 *                          will be included in the contexts
 *  @param center_par       if true, the paragraph will be the center of the
 *                          contexts, otherwise it will be the first element
 *  @param context          the context for each word in the line_array
 *
 *  @return number of elements in the *contexts* array (== line_length).
 *
 *  @note
 *  Memory for contexts and it's words array should be pre-allocated. 
 *  If _before == _after and random_windows is true, the random window sizes
 *  will continue to be equal.
 *  @endnote
 */
size_t
nlk_context_window(nlk_Vocab **varray, const size_t line_length,
                   const bool self, const size_t _before, const size_t _after,
                   const bool random_windows,
                   nlk_Vocab *vocab_par,  bool center_par,
                   nlk_Context **contexts)
{
    size_t line_pos         = 0;        /* position in line/par (input) */
    int window_pos          = 0;        /* position in window for line/par */
    int window_end          = 0;        /* end of window */
    size_t window_idx       = 0;        /* position in the current window */
    size_t context_idx      = 0;        /* position in the contexts array */
    nlk_Vocab *vocab_word   = NULL;     /* current center of the window */
    size_t random_window    = 0;        /* random window size */
    size_t before           = _before;  /* reduced window before current w */
    size_t after            = _after;   /* reduced window after current w */

    /* 
     * The position in the contexts (context_idx) will only be different from
     * the position in the line in one situation: PVDBOW (center_par == true).
     *
     * This is a sort of hack. In the Skipgram implementation, the input to 
     * lookup1 is the index of a context word while the center word is the
     * input to lookup2 (points if HS, indices otherwise).
     * However, in PVDBOW, the input to lookup1 is the paragraph index while
     * the input to lookup2 are the window words (points if HS, indices
     * otherwise).
     */

    for(line_pos = 0; line_pos < line_length; line_pos++) {
        /** @section Determine Window
         */
        /* random window */
        if(random_windows && _before == _after) {
            /* if after == before, keep it that way (word2vec style) */
            random_window = (nlk_random_xs1024() % _before) + 1;
            before = random_window;
            after = random_window;
        } else if(random_windows) {
            if(_before > 0) {
                random_window = (nlk_random_xs1024()  % _before) + 1;
                before = random_window;
            }
            if(_after > 0) {
                random_window = (nlk_random_xs1024()  % _after) + 1;
                after = random_window;
            }
        }

        /* determine where in *line* the window begins */
        if(line_pos < before) {
            window_pos = 0;
        } else {
            window_pos = line_pos - before;
        }
        /* determine where in *line* the window ends */
        if(line_pos + after >= line_length) {
            window_end = line_length;
        } else {
            window_end = line_pos + after + 1;
        }
        
        if(center_par == true && vocab_par != NULL) {
            /** @section PVDBOW
             */
            for(; window_pos < window_end; window_pos++) {
                contexts[context_idx]->size = 1;
                contexts[context_idx]->center = varray[window_pos];
                contexts[context_idx]->window[0] = vocab_par;
                context_idx++;
            }
        } else if(center_par == true && vocab_par == NULL) {
            /* force failure */
            NLK_ERROR("something went horribly wrong with context generation",
                      NLK_FAILURE);
        } else if(vocab_par != NULL) {
            /** @section PVDM specific
            */
            contexts[context_idx]->size = window_end - window_pos;
            /* 
             * paragraph is in the context items, but self is not:
             * contexts[context_idx]->size += 1 - 1
             */
            /* first context item is the paragraph */
            contexts[context_idx]->window[0] = vocab_par;
            window_idx = 1;

            for(; window_pos < window_end; window_pos++) {
                if(window_pos == line_pos) {
                    continue; /* skip the "target" */
                }
                vocab_word = varray[window_pos];
                contexts[context_idx]->window[window_idx] = vocab_word;
                window_idx++;
            }
            context_idx++;
        } else { 
            /** @section CBOW, Skipgram
             */
             /* self not in its own context window so -1 */
            contexts[context_idx]->size = window_end - window_pos - 1;
            window_idx = 0;

            for(; window_pos < window_end; window_pos++) {
                if(window_pos == line_pos) {
                    contexts[context_idx]->center = varray[window_pos];
                    continue;
                }
                contexts[context_idx]->window[window_idx] = varray[window_pos];;
                window_idx++;
            }
            context_idx++;
        }
    }

    return context_idx;
}

/** @fn nlk_Context *nlk_context_create(size_t max_context_size) 
 * Creates a context window of size max_context_size
 *
 * @param max_context_size  maximum size of the context window
 * 
 * @return the context or NULL
 */
nlk_Context *
nlk_context_create(size_t max_context_size) 
{
    nlk_Context *context;
    context = (nlk_Context *) malloc(sizeof(nlk_Context));
    if(context == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for context struct", 
                       NLK_ENOMEM);
        /* unreachable */
    }
    context->window = (nlk_Vocab **) malloc(max_context_size *
                                            sizeof(nlk_Vocab *));
    if(context->window == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for context", 
                       NLK_ENOMEM);
        /* unreachable */
    }

    return context;
}

/** @fn void nlk_context_free(nlk_Context *context)
 * Free context
 *
 * @param the context to free
 */
void
nlk_context_free(nlk_Context *context)
{
    free(context->window);
    free(context);
}

/** @fn void nlk_context_print(nlk_Context *context)
 * Print a context
 */
void nlk_context_print(nlk_Context *context)
{
    printf("%s (%zu): ", context->center->word, context->size);
    fflush(stdout);
    for(size_t ii = 0; ii < context->size; ii++) {
        printf("%s ", context->window[ii]->word);
        fflush(stdout);
    }
    printf("\n");

}
