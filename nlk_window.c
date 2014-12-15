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

#include "nlk_err.h"
#include "nlk_vocabulary.h"
#include "nlk_array.h"
#include "nlk_window.h"

/** @fn size_t nlk_context_window(nlk_Vocab **line_array,
 *                                const size_t line_length,
                                  const bool self, const size_t before, 
                                  const size_t after, nlk_Context **contexts)
 *
 *  @param line_array       vocab items for the line/paragraph/document
 *  @param line_length      lengh of the line array
 *  @param self             include the word itself (center) in its window?
 *  @param size_before      how many words from before the current word to 
 *                          include in the context window
 *  @param size_after       how many words after the current word to include
 *                          in the context window
 *  @param random_windows   use skipgram style random_window in range
 *                          before=[1, before], after=[1, after]
 *  @param context      the context for each word in the line_array
 *
 *  @return number of elements in the *contexts* array (== line_length).
 *
 *  @note
 *  Memory for contexts and it's words array should be pre-allocated 
 *  @endnote
 */
size_t
nlk_context_window(nlk_Vocab **line_array, const size_t line_length,
                   const bool self, const size_t _before, const size_t _after,
                   const bool random_windows, nlk_Context **contexts)
{
    size_t line_pos      = 0;           /* position in line (input) */
    int window_pos       = 0;           /* position in window */
    int window_end       = 0;           /* end of window */
    size_t word_index    = 0;           /* position in the current context */
    nlk_Vocab *vocab_word;              /* current center of the window */
    size_t random_window;
    size_t before = _before;
    size_t after = _after;

    for(line_pos = 0; line_pos < line_length; line_pos++) {
        /* random window */
        if(random_windows && _before == _after) {
            /* if after == before, keep it that way (skipgram style) */
            random_window = (nlk_random_uint() % _before) + 1;
            before = random_window;
            after = random_window;
        } else if(random_windows) {
            if(_before > 0) {
                random_window = (nlk_random_uint() % _before) + 1;
                before = random_window;
            }
            if(_after > 0) {
                random_window = (nlk_random_uint() % _after) + 1;
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
        if(line_pos + after + 1 >= line_length) {
            window_end = line_length;
        } else {
            window_end = line_pos + after + 1;
        }
        contexts[line_pos]->word = line_array[line_pos];
        contexts[line_pos]->size =  window_end - window_pos;
        if(self == false) {
            contexts[line_pos]->size -= 1;
        }

        /* create window */
        word_index = 0;
        for(window_pos; window_pos < window_end; window_pos++) {
            if(self == false && window_pos == line_pos) {
                continue; /* self not in its own context window */
            }
            vocab_word = line_array[window_pos];
            contexts[line_pos]->context_words[word_index] = vocab_word;
            word_index++;
        }
    }
    return line_length;
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
    context->context_words = (nlk_Vocab **) calloc(max_context_size,
                                             sizeof(nlk_Vocab *));
    if(context == NULL) {
        free(context);
        NLK_ERROR_NULL("failed to allocate memory for context", 
                       NLK_ENOMEM);
        /* unreachable */
    }

    return context;
}
