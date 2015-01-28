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


/** @struct nlk_context
 * The context window for a given word
 */
struct nlk_context_t {
    size_t size;                /**< size of the context words array */
    struct nlk_vocab_t *center;          /**< the vocab item that has this context */
    struct nlk_vocab_t **window;        /**< the context window */
};
typedef struct nlk_context_t NLK_CONTEXT;

size_t          nlk_context_window(struct nlk_vocab_t **, const size_t, 
                                   const bool, const size_t, const size_t, 
                                   const bool, struct nlk_vocab_t *,  bool, 
                                   struct nlk_context_t **);

unsigned int    nlk_window_for_word(struct nlk_vocab_t **, const unsigned int,
                                    const unsigned int, bool,  
                                    unsigned int *, size_t *, unsigned int *, 
                                    unsigned int *, size_t *);
                           


struct nlk_context_t *nlk_context_create(size_t); 
void nlk_context_free(struct nlk_context_t *);
void nlk_context_print(struct nlk_context_t *context);

__END_DECLS
#endif /* __NLK_WINDOW__ */
