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


/** @file nlk_transfer.h
 * Neural Network Transfer/Activations
 */


#ifndef __NLK_TRANSFER_H__
#define __NLK_TRANSFER_H__


#include "nlk_array.h"


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


void nlk_transfer_concat_forward(const NLK_ARRAY *, NLK_ARRAY *);
void nlk_transfer_concat_backprop(const NLK_ARRAY *, NLK_ARRAY *);

void nlk_average(const NLK_ARRAY *, size_t, NLK_ARRAY *);

void nlk_sigmoid_forward_table (const nlk_real *, NLK_ARRAY *);
void nlk_sigmoid_forward(NLK_ARRAY *input);
int nlk_sigmoid_backprop(const NLK_ARRAY *, const NLK_ARRAY *, NLK_ARRAY *);


__END_DECLS
#endif /* __NLK_TRANSFER_H__ */
