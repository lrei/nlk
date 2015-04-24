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

#include <sys/types.h>


#define NLK_LM_MAX_WORD_SIZE    128
#define NLK_LM_MAX_LINE_SIZE    50000


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


void    nlk_text_lower(char *, wchar_t *);
void    nlk_text_ascii_lower(char *st);
int     nlk_read_word(FILE *, char *, const size_t);
int     nlk_read_line(FILE *, char **, const size_t, const size_t);
int     nlk_read_number_line(FILE *, char **, size_t *, const size_t, 
                             const size_t);
size_t  nlk_text_line_size(char **line);
size_t  nlk_text_get_line(FILE *);
size_t  nlk_text_count_words(FILE *);
size_t  nlk_text_count_lines(FILE *);
size_t  nlk_set_file_pos(FILE *, bool, size_t, size_t, int);
void    nlk_text_print_line(char **);
void    nlk_text_print_numbered_line(char **, size_t, int);

/* Mem Mapped functions */
off_t   nlk_text_mem_open(char *, char **);
void    nlk_text_mem_close(caddr_t *, size_t);
size_t  nlk_text_mem_count_lines(char *, off_t);
int     nlk_text_mem_read_word(char *, off_t *, off_t, char *, wchar_t *, 
                               const size_t);
off_t   nlk_text_mem_get_line_pos(char *, size_t, size_t);
int     nlk_text_mem_read_line(char *, off_t *, off_t, char **, off_t *,
                               wchar_t *, const size_t, const size_t);


__END_DECLS
#endif /* __NLK_TEXT_H__ */
