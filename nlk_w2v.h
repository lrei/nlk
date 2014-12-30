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
 * Word2Vec: CBOW & Skipgram model definitions
 */


#ifndef __NLK_W2V_H__
#define __NLK_W2V_H__



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

/** @enum NLK_LM_TYPE
 * The type of the language model
 */
enum NLK_LM_TYPE { 
    NLK_CBOW      = 0,  /**< CBOW and PVDM */
    NLK_SKIPGRAM  = 1   /**< Skipgram and PVDBOW */
};
typedef enum NLK_LM_TYPE nlk_Lm;

nlk_real nlk_word2vec(nlk_Lm, nlk_Layer_Lookup *, nlk_Layer_Lookup *,
                      bool, bool, char *, nlk_Vocab **, size_t, 
                      size_t, float, size_t, nlk_real, unsigned int, int, 
                      char *, char *, nlk_Format);

nlk_real nlk_skipgram_for_context(nlk_Layer_Lookup *, nlk_Layer_Lookup *, bool, 
                                  nlk_real, nlk_Table *, nlk_Context *, 
                                  nlk_Array *, nlk_Array *, nlk_Array *, 
                                  nlk_Array *, nlk_Array *);

nlk_real nlk_cbow_for_context(nlk_Layer_Lookup *, nlk_Layer_Lookup *, bool,
                              nlk_real, nlk_Table *, nlk_Context *, 
                              size_t *ctx_ids, nlk_Array *, nlk_Array *, 
                              nlk_Array *, nlk_Array *, nlk_Array *);

__END_DECLS
#endif /* __NLK_W2V_H__ */
