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


/** @file nlk_w2v.h
 * Word2Vec: CBOW, Skipgram, PVDM, PVDBOW definitions
 */

#ifndef __NLK_W2V_H__
#define __NLK_W2V_H__


#include "nlk.h"
#include "nlk_layer_lookup.h"
#include "nlk_corpus.h"
#include "nlk_window.h"


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

/* create */
struct nlk_neuralnet_t *nlk_w2v_create(struct nlk_nn_train_t, 
                                       const bool, const size_t, 
                                       struct nlk_vocab_t *, 
                                       const size_t, const bool);

/* train */
void     nlk_w2v_hs(NLK_LAYER_LOOKUP *, const bool, const NLK_ARRAY *, 
                    const nlk_real, const struct nlk_vocab_t *, NLK_ARRAY *);

void     nlk_w2v_neg(NLK_LAYER_LOOKUP *, const bool, const size_t *, 
                     const size_t, const size_t, const nlk_real, const size_t, 
                     const NLK_ARRAY *, NLK_ARRAY *);

void    nlk_pvdm(NLK_LAYER_LOOKUP *, const bool, NLK_LAYER_LOOKUP *, 
                 const bool, NLK_LAYER_LOOKUP *, const bool,
                 NLK_LAYER_LOOKUP *, const size_t, const size_t *, 
                 const bool, const size_t, const nlk_real, 
                 const struct nlk_context_t *, NLK_ARRAY *, NLK_ARRAY *);

void    nlk_pvdm_cc(NLK_LAYER_LOOKUP *, const bool, NLK_LAYER_LOOKUP *, 
                    const bool, NLK_LAYER_LOOKUP *, const bool, 
                    NLK_LAYER_LOOKUP *,  const size_t, const size_t *, 
                    const bool, const size_t, const nlk_real, 
                    const struct nlk_context_t *, NLK_ARRAY *, NLK_ARRAY *);

void    nlk_pvdbow(NLK_LAYER_LOOKUP *, const bool, NLK_LAYER_LOOKUP *, 
                   const bool, NLK_LAYER_LOOKUP *, const bool, 
                   NLK_LAYER_LOOKUP *, const size_t, const size_t *, 
                   const bool, const size_t, const nlk_real, 
                   const struct nlk_context_t *, NLK_ARRAY *, NLK_ARRAY *);

void     nlk_w2v(struct nlk_neuralnet_t *nn, const struct nlk_corpus_t *, 
                 const bool, const bool, const bool, int, const bool);

void     nlk_w2v_train(struct nlk_neuralnet_t *nn, const struct nlk_corpus_t *, 
                       int);

/* export */
void    nlk_w2v_export_word_vectors(NLK_ARRAY *, NLK_FILE_FORMAT, 
                                    struct nlk_vocab_t **, const char *);
void    nlk_w2v_export_paragraph_vectors(NLK_ARRAY *, NLK_FILE_FORMAT, 
                                         const char *);


__END_DECLS
#endif /* __NLK_W2V_H__ */
