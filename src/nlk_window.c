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
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_neuralnet.h"
#include "nlk_random.h"
#include "nlk_window.h"


/**
 * Random Windows
 */
static void
nlk_window_random(const unsigned int _before, const unsigned int _after, 
                  const bool equal, unsigned int *before, unsigned int *after)
{
    unsigned int random_window;
    if(equal) {/* if after == before, keep it that way (word2vec style) */
        random_window = (nlk_random_xs1024() % (uint64_t) _before) + 1;
        *before = random_window;
        *after = random_window;
    } else {
        if(_before > 0) {
            random_window = (nlk_random_xs1024()  % (uint64_t) _before) + 1;
            *before = random_window;
        } else {
            *before = 0;
        }
        if(_after > 0) {
            random_window = (nlk_random_xs1024()  % (uint64_t) _after) + 1;
            *after = random_window;
        } else {
            *after = 0;
        }
    } 
}


/**
 * Context for a given position in the text
 */
static void
nlk_context_for_pos(struct nlk_vocab_t **varray, const size_t paragraph_id,
                    const bool paragraph, const unsigned int center_pos,
                    unsigned int window_pos, const unsigned int window_end,
                    const bool prepad_paragraph, struct nlk_context_t *context)
{
    unsigned int window_idx = 0;

    /* the target i.e. word to be predicted is the center of the window */
    context->target = varray[center_pos];

    /* the context size is the window except the target */
    context->size = (window_end - window_pos) - 1;

    /* if before < window and this is a paragraph model */
    if(prepad_paragraph && paragraph) {
        context->window[0] = paragraph_id;
        context->is_paragraph[0] = true;
        context->size = context->size + 1;
        window_idx++;
    }

    /* remaining context items are the window items except for the target */
    for(; window_pos < window_end; window_pos++) {
        if(window_pos == center_pos) {
            continue;
        }
        context->window[window_idx] = varray[window_pos]->index;
        context->is_paragraph[window_idx] = false;
        window_idx++;
    }

   /* if it's a paragraph model, 0 => paragraph_id */
    if(paragraph) {
        context->window[window_idx] = paragraph_id;
        context->is_paragraph[window_idx] = true;
        context->size = context->size + 1;
        window_idx++;
    }

}


/** 
 * Creates a context window from a vocabularized line/sentence/pararaph
 *
 *  @param varray           vocab items for the line/paragraph/document
 *  @param line_length      lengh of the line array
 *  @param paragraph_id     the paragraph id/index
 *  @param opts             context generaton options
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
nlk_context_window(struct nlk_vocab_t **varray, const size_t line_length,
                   const size_t paragraph_id,
                   struct nlk_context_opts_t *opts,
                   struct nlk_context_t **contexts)
{
    size_t center_pos       = 0;        /* position in line/par (input) */
    int window_pos          = 0;        /* position in window for line/par */
    int window_end          = 0;        /* end of window */
    unsigned int ctx_idx    = 0;        /* position in the contexts array */
    unsigned int before     = 0;        /* reduced window before current w */
    unsigned int after      = 0;        /* reduced window after current w */
    bool prepad_paragraph   = false;    /* prepad context with par id */


    /* go through the paragraph changing the center word */
    for(center_pos = 0; center_pos < line_length; center_pos++) {
        /* random window */
        if(opts->random_windows) {
            nlk_window_random(opts->before, opts->after, opts->b_equals_a, 
                              &before, &after);
        } else {
            before = opts->before;
            after = opts->after;
        }
        
        /* determine where in *line* the window begins */
        if(center_pos < before) {
            window_pos = 0;
            if(opts->model == NLK_PVDBOW) {
                prepad_paragraph = true;
            }
        } else {
            window_pos = center_pos - before;
            prepad_paragraph = false;
        }
        /* determine where in *line* the window ends */
        if(center_pos + after >= line_length) {
            window_end = line_length;
        } else {
            /* we always end at < window_end hence the +1 */
            window_end = center_pos + after + 1;
        }

        /* create the context for this window */
        nlk_context_for_pos(varray, paragraph_id, opts->paragraph, center_pos, 
                            window_pos, window_end, prepad_paragraph,
                            contexts[ctx_idx]);

#ifndef NCHECKS
        if(contexts[ctx_idx]->size > opts->before + opts->after + 1) {
            NLK_ERROR("Context size too large for allocated memory.", 
                      NLK_EBADLEN);
        }
#endif

        ctx_idx++;
    }

    return ctx_idx;
}

/**
 * Creates a context window of size max_context_size
 *
 * @param max_context_size  maximum size of the context window
 * 
 * @return the context or NULL
 */
struct nlk_context_t *
nlk_context_create(size_t max_context_size) 
{
    struct nlk_context_t *context;
    context = (struct nlk_context_t *) malloc(sizeof(struct nlk_context_t));
    if(context == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for context struct", 
                       NLK_ENOMEM);
        /* unreachable */
    }
    context->window = (size_t *) malloc(max_context_size * sizeof(size_t));
    if(context->window == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for context", 
                       NLK_ENOMEM);
        /* unreachable */
    }
    context->is_paragraph = (bool *) malloc(max_context_size * sizeof(bool));
    if(context->is_paragraph == NULL) {
        NLK_ERROR_NULL("failed to allocate memory for context", 
                       NLK_ENOMEM);
        /* unreachable */
    }


    /* make it easier to find attempt to access at non initialized location */
    context->target = NULL;
    context->size = (size_t) -1;
    for(size_t ii = 0; ii < max_context_size; ii++) {
        context->window[ii] = (size_t) -1;
        context->is_paragraph[ii] = false;
    }

    return context;
}

/**
 * Free context
 *
 * @param the context to free
 */
void
nlk_context_free(struct nlk_context_t *context)
{
    free(context->window);
    free(context->is_paragraph);
    free(context);
}

/**
 * @brief Print a context
 */
void nlk_context_print(struct nlk_context_t *context, 
                       struct nlk_vocab_t **vocab)
{
    printf("[len=%zu] target=%s (%zu), context: ", context->size, 
            context->target->word, context->target->index);
    fflush(stdout);
    for(size_t ii = 0; ii < context->size; ii++) {
        if(context->is_paragraph[ii]) {
            printf("*_%zu |%zu|", context->window[ii], context->window[ii]);

        } else {
            printf("%s (%zu) ", 
                    nlk_vocab_at_index(vocab, context->window[ii])->word, 
                    context->window[ii]);
        }
        fflush(stdout);
    }
    printf("\n");

}


void
nlk_context_model_opts(NLK_LM model, unsigned int window, 
                       struct nlk_vocab_t **vocab,
                       struct nlk_context_opts_t *opts)
{
    opts->model = model;

    /* defaults */
    opts->before = window;
    opts->after = window;
    opts->b_equals_a = true;
    opts->prepad = false;
    opts->postpad = false;

    /* random_window in range before=[1, before], after=[1, after] */
    opts->random_windows = true;

    switch(model) {
        case NLK_PVDM_CONCAT:
            /* fixed size windows */
            opts->random_windows = false;
            /* prepad if smaller */
            opts->prepad = true;
            /* FALL THROUGH: all other options are common to PVDM */
        case NLK_PVDM:
            /* predict the next word => after = 0 */
            opts->b_equals_a = false;
            opts->after = 0;
            opts->paragraph = true;
            break;
        case NLK_PVDBOW:
            opts->paragraph = true;
            break;
        case NLK_CBOW:
        case NLK_SKIPGRAM:
            opts->paragraph = false;
            /* options for CBOW/SKIPGRAM are already set */
            break;
        case NLK_MODEL_NULL:
        default:
            NLK_ERROR_VOID("Invalid model for context generation", NLK_EINVAL);
            /*unreachable */
    }
    opts->start = nlk_vocab_get_start_symbol(vocab)->index;

}
