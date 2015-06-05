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


/** @file nlk_pv.h
 * Paragraph Vector specific definitions
 */


#ifndef __NLK_PV_H__
#define __NLK_PV_H__


#include "nlk_corpus.h"
#include "nlk_class.h"


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


struct nlk_layer_lookup_t *
nlk_pv_gen(struct nlk_neuralnet_t *, const struct nlk_corpus_t *, 
           const unsigned int iter, const bool);

unsigned int *nlk_pv_classify(struct nlk_neuralnet_t *, 
                              struct nlk_layer_lookup_t *, size_t *, size_t,
                              const bool);

float nlk_pv_classifier(struct nlk_neuralnet_t *, struct nlk_dataset_t *,
                        const unsigned int, nlk_real, 
                        const nlk_real, const bool);


float nlk_pv_classify_test(struct nlk_neuralnet_t *, const char *, const bool);

__END_DECLS
#endif /* __NLK_PV_H__ */
