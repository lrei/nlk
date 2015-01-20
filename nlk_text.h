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


/** @file nlk_text.h
 * Read text files
 */

#ifndef __NLK_TEXT_H__
#define __NLK_TEXT_H__


#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>


#define NLK_LM_MAX_WORD_SIZE    512
#define NLK_LM_MAX_LINE_SIZE    1024


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

void nlk_text_lower(char *, wchar_t *);
int nlk_read_word(FILE *, char *, wchar_t *, const size_t);
int nlk_read_line(FILE *, char **, wchar_t *, const size_t, const size_t);
void nlk_text_concat_hash(char **, char *, char *);

size_t nlk_text_count_words(FILE *);
size_t nlk_text_count_lines(FILE *);
size_t nlk_set_file_pos(FILE *, bool, size_t, size_t, int);
void nlk_text_print_line(char **);


__END_DECLS
#endif /* __NLK_TEXT_H__ */
