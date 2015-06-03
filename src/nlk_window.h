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


/** @file nlk_window.h
 * Input/Output Window and Context
 */


#ifndef __NLK_WINDOW_H__
#define __NLK_WINDOW_H__

#include <stdbool.h>

#include "nlk_vocabulary.h"


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


/** @struct nlk_context_t
 * The context window for a given word
 */
struct nlk_context_t {
    size_t  size;                   /**< size of the context words array */
    struct  nlk_vocab_t *target;    /**< the item that has this context */
    size_t *window;                 /**< the context window */
    bool   *is_paragraph;           /**< the item in window is a paragraph? */
};
typedef struct nlk_context_t NLK_CONTEXT;

/** @struct nlk_context_opts_t
 * The model specific context options
 */
struct nlk_context_opts_t {
    unsigned int  max_size;        /**< maximum context size */
    bool          random_windows;  /**< word2vec style random windows */
    unsigned int  before;          /**< window before center */
    unsigned int  after;           /**< window after center */
    bool          b_equals_a;      /**< force before == after in rand */
    bool          paragraph;       /**< is a paragraph model */
    bool          prepad_paragraph;/**< just prepad paragraph once */
    bool          prepad;          /**< fixed size window: prepad */
    bool          postpad;         /**< fixed size window: postpad */
    size_t        start;           /**< index of the start symbol */
};
typedef struct nlk_context_opts_t NLK_CONTEXT_OPTS;



size_t  nlk_context_window(struct nlk_vocab_t **, const size_t, const size_t,
                           struct nlk_context_opts_t *,
                           struct nlk_context_t **);


struct nlk_context_t  *nlk_context_create(const size_t); 
struct nlk_context_t **nlk_context_create_array(const size_t);
void nlk_context_free(struct nlk_context_t *);
void nlk_context_free_array(struct nlk_context_t **);

void nlk_context_print(struct nlk_context_t *, struct nlk_vocab_t **);

__END_DECLS
#endif /* __NLK_WINDOW__ */
