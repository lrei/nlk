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


#define NLK_MAX_WORD_SIZE    256
#define NLK_MAX_LINE_SIZE    100000
#define NLK_MAX_CHARS   (NLK_MAX_LINE_SIZE * NLK_MAX_WORD_SIZE)

#define BUFFER_SIZE (16 * 1024)
#define NLK_BUFFER_SIZE (NLK_MAX_CHARS + BUFFER_SIZE)



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


/* create/free/size char **line */
char    **nlk_text_line_create();
void    nlk_text_line_free(char **);
size_t  nlk_text_line_size(char **line);

/* lower */
void    nlk_text_ascii_lower(char *st);

/* read */
int nlk_open(const char *);
int     nlk_read_line(int, char **, size_t *, char *);
void    nlk_text_line_read(char *, const ssize_t, char **);
int     nlk_read_word(FILE *, char *, const size_t);


/* lines */
size_t  nlk_text_count_lines(const char *);
size_t  nlk_text_count_empty_lines(const char *);
off_t   nlk_text_goto_line(int, const size_t);
void    nlk_text_goto_location(int, const off_t);

/** @TODO: move to util: */
size_t  nlk_text_get_split_start_line(size_t, unsigned int, unsigned int);
size_t  nlk_text_get_split_end_line(size_t,  unsigned int, unsigned int);

/* util */
size_t  nlk_text_line_size(char **line);

/* print */
void    nlk_text_print_line(char **);
void    nlk_text_print_numbered_line(char **, size_t, int);



__END_DECLS
#endif /* __NLK_TEXT_H__ */
