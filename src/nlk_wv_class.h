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


/** @file nlk_wv_class.h
 * Word Vector classifier definitions
 */


#ifndef __NLK_WV_CLASS_H__
#define __NLK_WV_CLASS_H__


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

void    nlk_wv_class_senna_train(struct nlk_neuralnet_t *, 
                                 struct nlk_supervised_corpus_t *,
                                 int);

struct nlk_neuralnet_t  *nlk_wv_class_create_senna(struct nlk_nn_train_t,
                                                   struct nlk_vocab_t *,
                                                   struct nlk_layer_lookup_t *, 
                                                   const size_t, 
                                                   const bool);

float   nlk_wv_class_senna_test_eval(struct nlk_neuralnet_t *,
                                     struct nlk_supervised_corpus_t *, 
                                     const int);
void    nlk_wv_class_senna_test_out(struct nlk_neuralnet_t *,
                                    struct nlk_supervised_corpus_t *, FILE *);




__END_DECLS
#endif /* __NLK_WV_CLASS_H__ */
