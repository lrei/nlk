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


/** @file nlk_text.c
 * Read text files
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <inttypes.h>

#include "MurmurHash3.h"

#include "nlk_err.h"
#include "nlk_text.h"


/** @fn int nlk_read_word(FILE *file, char *word, const size_t max_word_size)
Read a word from a file
 * @param file            FILE object to read from
 * @param word            char pointer where the word will be stored
 * @param max_word_size   maximum size of a word (including null terminator)
 * @param lower           convert uppercase chars to lowercase
 *
 * @return integer of the word separator (terminator), EOF or NLK_ETRUNC
 *
 * @note
 * This function is made for reading already pre-processed files, does not
 * do actual tokenization or handle weird stuff.
 *
 * Assumes words are separated by:
 *      * spaces (' ')
 *      * tabs ('\t')
 *      * newlines ('\n')
 *
 * @endnote
 */
int
nlk_read_word(FILE *file, char *word, const size_t max_word_size,
              const bool lower)
{
    int ch;
    size_t len = 0;

    while(!feof(file)) {
        ch = fgetc(file);
        
        /* end of word */
        if((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if(len == 0) {
                continue;   /* prevent empty words */
            }
            word[len] = 0;  /* null terminate the word */
            return ch;
        }

        word[len] = tolower(ch);
        len++;

        if(len == max_word_size - 2) {
            word[len] = 0;
            return NLK_ETRUNC;
        }
    }
    return EOF;
}

/** @fn Read a line from a file
 * @param file            FILE object to read from
 * @param line            array  where the words+1 will be stored
 * @param max_word_size   maximum size of a word (including null terminator)
 * @param max_line_size   maximum size of a line in number of words (including 
 *                        null terminator)
 * @param lower_words     conver uppercase chars to lowercase
 *
 * @return integer of the sentence separator '\n', EOF or NLK_ETRUNC
 *
 * @note
 * This function is made for reading already pre-processed files.
 * Lines are assumed to be separated by the newline character. 
 *
 * The purpose is is to read files where each line is a sentence 
 * (or similar, e.g. a tweet or a "paragraph").
 *
 * Calls read_word. Each word in a line is a char array and the line is a NULL 
 * terminated array of pointers to the individual words. Obviously, memory for
 * this should be allocated before calling the function.
 *
 * The line terminator is a null-word (a char array beggining with a NULL)
 * 
 * If necessary to truncate the sentence, it will be truncated at a word
 * boundary and thus NLK_ETRUNC does not guarantee that the array is of size
 * max_line_size.
 *
 * @endnote
 */
int
nlk_read_line(FILE *file, char **line, const size_t max_word_size, 
              const size_t max_line_size, const bool lower_words)
{
    int term;               /* word terminator, on return, line terminator */
    size_t word_idx = 0;    /* word in line index */

    while(!feof(file)) {
        term = nlk_read_word(file, line[word_idx], max_word_size, lower_words);
        
        word_idx++;

        if(term == '\n') {
            *line[word_idx] = '\0';
            return term;
        }

        if(word_idx == max_line_size - 1) {
            *line[word_idx] = '\0';
            return NLK_ETRUNC;
        }
    }
    return EOF;
}

/** @fn nlk_text_lower(char *word)
 * Convert string to lowercase in-place
 * 
 * @param word  the word to convert to lowercase
 *
 * @return no return, word i overwritten
 */
void
nlk_text_lower(char *word) {
    char *wp;
    for (wp = word ; *wp; ++wp) {
        *wp = tolower(*wp);
    }
}

/** @fn void nlk_text_concat_hash
 * Turns an arbitraly large sequence of strings into a single (128bit) hash 
 * stored as a string in key.
 *
 * @param line  the sequence of strings
 * @param tmp   tempory memory for the concatenated strings
 *              must be at least (size(string) + 1) * size(seq)
 * @param key   the resulting hash in string form
 */
void
nlk_text_concat_hash(const char **line, char *tmp, char *key)
{
    size_t ii;
    size_t jj;
    size_t pos = 0;
    uint64_t hash[2];

    /* loop through words */
    for(ii = 0; *line[ii] != '\0'; ii++) {  
        /* loop through chars */
        for(jj = 0; line[ii][jj] != '\0'; jj++) {
            tmp[pos] = line[ii][jj];
            pos++;
        }
        tmp[pos] = ' '; /* space separate words */
        pos++;
    }
    tmp[pos] = '\0'; /* replace final space with null terminator */

    /* hash and convert to hexadecimal string */
    MurmurHash3_x64_128(tmp, pos, 1, hash);
    sprintf(key, "0x%"PRIX64 "%"PRIX64, hash[0], hash[1]);
}


