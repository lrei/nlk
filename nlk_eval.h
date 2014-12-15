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


/** @file nlk_eval.h
* Evaluation function definitions
*/

#ifndef __NLK_EVAL_H__
#define __NLK_EVAL_H__


#define NLK_WORD_REL_MAX_LINE_SIZE 512      /**< max number of chars in test */
#define NLK_WORD_REL_DEFAULT_SIZE 19558     /**< default tests array size */


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


/** @struct nlk_analogy_test
 * An individual word analogy test
 */
struct nlk_analogy_test {
    nlk_Vocab *question[3]; /**< the question as an array of 3 words */
    nlk_Vocab *answer;      /**< the answer to w1 - w2 + w3 */
};
typedef struct nlk_analogy_test nlk_Analogy_Test;

int nlk_eval_on_questions(const char *, nlk_Vocab **, const nlk_Array *, 
                          const size_t, const bool, const bool, 
                          nlk_real *accuracy);

__END_DECLS
#endif /* __NLK_EVAL_H__ */
