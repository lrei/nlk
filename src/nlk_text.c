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
 * Read text files and basic text related operations
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <wctype.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "nlk_err.h"
#include "nlk_text.h"


/**
 * Create a line (allocate memory for a line)
 */
char **
nlk_text_line_create()
{
    char **text_line = (char **) calloc(NLK_LM_MAX_LINE_SIZE, sizeof(char *));
    if(text_line == NULL) {
        NLK_ERROR_ABORT("unable to allocate memory for text", NLK_ENOMEM);
        /* unreachable */
    }
    for(size_t zz = 0; zz < NLK_LM_MAX_LINE_SIZE; zz++) {
        text_line[zz] = calloc(NLK_LM_MAX_WORD_SIZE, sizeof(char));
        if(text_line[zz] == NULL) {
            NLK_ERROR_NULL("unable to allocate memory for text", NLK_ENOMEM);
            /* unreachable */
        }
    }

    return text_line;
}

void
nlk_text_line_free(char **text_line)
{
    if(text_line == NULL) {
        return;
    }
    /* free array elements (words) */
    for(size_t zz = 0; zz < NLK_LM_MAX_LINE_SIZE; zz++) {
        free(text_line[zz]);
        text_line[zz] = NULL;
    }

    /* free text_line */
    free(text_line);
    text_line = NULL;
}


/**
 * @brief Convert string to lowercase in-place
 * 
 * @param word      the word to convert to lowercase
 * @param tmp       temporary memory for widechar conversion
 */
void
nlk_text_lower(char *word, wchar_t *tmp) {
    size_t ii;
    size_t len;
    wchar_t *low_tmp;

    len = strlen(word) + 1;

    if(tmp == NULL) {
        low_tmp = malloc(len * sizeof(wchar_t));
    } else {
        low_tmp = tmp;
    }

    /* convert to wide representation */
    len = mbstowcs(low_tmp, word, len);
    if(len == (size_t) - 1) {
        NLK_ERROR_VOID("Invalid char conversion.", NLK_FAILURE);
        /* unreachable */
    }

    for(ii = 0; ii < len; ii++) {
        low_tmp[ii] = towlower(low_tmp[ii]);
    }

    /* convert back to multibyte representation */
    len = wcstombs(word, low_tmp, len + 1);
    if(len == (size_t) - 1) {
        NLK_ERROR_VOID("Invalid char conversion.", NLK_FAILURE);
        /* unreachable */
    }

    if(tmp == NULL) {
        free(low_tmp);
    }
}

/**
 * @brief ASCII in-place convertion to lower case
 *
 * @param st    the string (overwritten with lower case version) 
 */
void
nlk_text_ascii_lower(char *st)
{
    for(char *p = st; *p != '\0'; p++) {
        *p = tolower(*p);
    }
}

/** 
 * @brief Read a word from a file
 *
 * @param file            FILE object to read from
 * @param word            char pointer where the word will be stored
 * @param max_word_size   maximum size of a word (including null terminator)
 *
 * @return integer of the word separator (terminator), EOF or NLK_ETRUNC
 *
 * @note
 * This function is made for reading already pre-processed files, does not
 * do actual tokenization or handle weird stuff.
 *
 * Assumes words are separated by whitespaces (isspace() == true).
 * Ignores CRs.
 * @endnote
 */
int
nlk_read_word(FILE *file, char *word, const size_t max_word_size)
{
    int ch = EOF;
    size_t len = 0;

    while(!feof(file)) {
        ch = fgetc(file);

        /* space chars */
        if(isspace(ch) || ch == EOF) {
            /* 
             * Because of line checks tests we need to ignore CRs.
             * All other "space" chars are end of word chars 
             */
            if(ch == '\r') {
                continue;
            }

            /* prevent empty words */
            if(len == 0 && ch != EOF && ch != '\n') {
                continue;
            }             

            break; 
            /* goto EOW */
        }

        /* just another non-space char */
        word[len] = ch;
        len++;

        /* truncate at max_word_size */
        if(len == max_word_size - 2) {
            ch = NLK_ETRUNC;
            break;  /* goto EOW */
        }

    }
    /*
     * END-OF-WORD (EOW):
     */
    word[len] = '\0';

    return ch;
}

/** 
 * @brief Read a line from a file
 *
 * @param file            FILE object to read from
 * @param line            array  where the words+1 will be stored
 *                        null terminator)
 * @param max_word_size   maximum size of a word (including null terminator)
 * @param max_line_size   maximum size of a line in number of words (including 
 *
 * @return integer of the sentence separator '\n', EOF or NLK_ETRUNC
 *
 * @note
 * Lines are assumed to be separated by the NEWLINE character. 
 *
 * This function is made for reading already pre-processed files.
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
              const size_t max_line_size)
{
    int term;                   /* word terminator */
    size_t word_idx = 0;        /* word in line index */

    while(!feof(file)) {
        term = nlk_read_word(file, line[word_idx], max_word_size);
        word_idx++;

        if(term == '\n' || term == EOF) {
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


/**
 * Reads a number-line pair (separated by a whitespace).
 * Does not consider tabs as valid line separators.
 * See also nlk_read_line() for additional information
 *
 * @param file            FILE object to read from
 * @param line            array  where the words+1 will be stored - 
 *                        null terminated)
 * @param number          variable where line number will be stored
 *
 * @return integer of the sentence separator '\n' or EOF
 */
int
nlk_read_number_line(FILE *file, char **line, size_t *number)
{
    int term = -1;              /* word terminator */
    size_t word_idx = 0;        /* word in line index */
    int ret;


    /* inititialize some vars to hadle empty lines */
    *number = (size_t) -1;
    line[0][0] = '\0';

    /* read line number */
    if(!feof(file)) {
        ret = fscanf(file, "%zu", number);
        if(ret == EOF) {
            return EOF;
        } else if(ret <= 0) {
            NLK_ERROR("invalid line id", NLK_EINVAL);
            /* unreachable */
        }
    }

    /* read words */
    while(!feof(file)) {
        term = nlk_read_word(file, line[word_idx], NLK_LM_MAX_WORD_SIZE);
        /*
        if(term == NLK_ETRUNC) {
            nlk_debug("very large word truncated: %s", line[word_idx]);
        } */

        word_idx++;

        if(term == '\n' || term == EOF) {
            *line[word_idx] = '\0';
            return term;
        }
        

        if(word_idx == NLK_LM_MAX_LINE_SIZE - 1) {
            NLK_ERROR("Line length > max_line_size", NLK_ETRUNC);
            /* unreachable */
        }
    }

    return EOF;
}


/**
 * Determine the number of words in a line read with nlk_read_line
 *
 * @param line  the line
 * 
 * @return number of words in line
 */
size_t
nlk_text_line_size(char **line)
{
    size_t count = 0;

    while(*line[count] != '\0') {
        count++;
    }
    return count;
}

/**
 * Counts lines in a file
 * Lines are terminated by NEWLINE
 * Code heavily inpired by GNU coreutils/wc
 *
 * @return number of lines in file
 */
size_t
nlk_text_count_lines(const char *filepath)
{
    int fd;
    size_t lines = 0;
    char buf[BUFFER_SIZE];
    ssize_t bytes_read = 0;
    ssize_t lines_read;
    bool long_lines = false;

    /* open file */
    if((fd = open(filepath, O_RDONLY)) < 0) {
        nlk_log_err("%s", filepath);
        NLK_ERROR(strerror(errno), errno);
        /* unreachable */
    }

    /* access will be sequential */
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    /* read file loop */
    while((bytes_read = read(fd, buf, BUFFER_SIZE)) > 0) {
        char *p = buf;
        char *end = p + bytes_read;
        uint64_t plines = lines;

        /* count newlines */
        if(long_lines == false) {
            /* Avoid function call overhead for shorter lines.  */
            while(p != end) {
                lines += *p++ == '\n';
            }
        } else {
            /* memchr is more efficient with longer lines.  */
            while((p = memchr(p, '\n', end - p))) {
                ++p;
                ++lines;
            }
        }
        lines_read = lines - plines;

        /* determine which line counting mode to use */
        if(lines_read <= bytes_read / 15) {
            long_lines = true;
        } else {
            long_lines = false;
        }
    }

    if(bytes_read < 0) {
        lines = 0;
        NLK_ERROR(strerror(errno), errno);
    } 

    close(fd);
    return lines;
}

/**
 * Get current line number 
 *
 * @param fp    file pointer
 *
 * @return current line number
 */
size_t
nlk_text_get_line(FILE *fp)
{
    size_t line = 0;
    int buf = 0;
    long pos = 0;
    long pos_origin;

    /* save current position */
    pos_origin = ftell(fp);

    /* go to start */
    fseek(fp, 0, SEEK_SET);

    /* count until current */
    while(pos <= pos_origin) {
        buf = fgetc(fp);

        if(buf == '\n') {
            line++;
        }
        pos = ftell(fp);
    }

    /* restore position */
    fseek(fp, pos_origin, SEEK_SET);

    return line;
}


/**
 * Counts words remaining in a file. Does not change position of file.
 *
 * @param fp the FILE pointer
 *
 * @return number of words in file
 */
size_t
nlk_text_count_words(FILE *fp)
{
    size_t words = 0;
    int buf = 0;
    int last = 0;
    fpos_t pos;

    /* determine current position */
    fgetpos(fp, &pos);

    /* count words until the end */
    while((buf = fgetc(fp)) != EOF) {
        if(isspace(buf)) {
            words++;
        }
        last = buf;
    }
    if(!isspace(last)) {
        words++;
    }

    /* rewind file pointer so it can be used by caller */
    fsetpos(fp, &pos);

    return words;
}

/**
 * Set file position to the beginning of a given line
 *
 * @param fp    the file point
 * @param line  the line number to set the file pointer to
 *
 */
void
nlk_text_goto_line(FILE *fp, long line)
{
    int buf;
    long pos = 0;

    rewind(fp);
    
    while(pos < line) {
        while((buf = fgetc(fp)) != EOF) {
            if(buf == '\n') {
                break;
            }
        }
        pos++;
    }
}

/**
* Set file position to the start of the line
*
* @param fp    the file pointer
*
*/
static void
nlk_text_goto_line_start(FILE *fp)
{
    int buf;

    buf = fgetc(fp); 
    while(buf != '\n') {
        if(ftell(fp) < 2) {
            fseek(fp, 0, SEEK_SET);
            return;
        }
        fseek(fp, -2, SEEK_CUR);
        buf = fgetc(fp);
    }
}

/**
* Set file position to the start of the word
*
* @param fp    the file pointer
*
*/
static void
nlk_text_goto_word_start(FILE *fp)
{
    int buf;

    buf = fgetc(fp); 
    while(!isspace(buf)) {
        if(ftell(fp) < 2) {
            fseek(fp, 0, SEEK_SET);
            return;
        }
        fseek(fp, -2, SEEK_CUR);
        buf = fgetc(fp);
    }
}


/**
 * @brief Used to get the start position in a file for a given worker thread
 *
 * @param fp        the file where the position will be set
 * @param use_lines if true, goes to the start of a line instead of a byte
 * @param total     total number of bytes or lines in file
 * @param thread_id the "worker thread" id
 *
 * @return end position
 */
static inline size_t
nlk_get_file_pos(FILE *fp, bool use_lines, size_t total, size_t num_threads,
                 int thread_id)
{
    size_t start_pos;

    start_pos = (total / (double)num_threads) * (size_t)thread_id;
    fseek(fp, start_pos, SEEK_SET);  

    if(use_lines) {
        nlk_text_goto_line_start(fp);
    } else {
        nlk_text_goto_word_start(fp);
    }

    start_pos = ftell(fp);
    rewind(fp);

    return start_pos;
}


/**
 * @brief Used to set the start position in a file for a given worker thread
 *
 * @param fp        the file where the position will be set
 * @param use_lines if true, goes to the start of a line instead of a byte
 * @param total     total number of bytes or lines in file
 * @param thread_id the "worker thread" id
 *
 * @return end position
 */
size_t
nlk_set_file_pos(FILE *fp, bool use_lines, size_t total, int num_threads,
                 int thread_id)
{
    size_t start_pos;
    size_t end_pos;
    int next_thread;
    
    /* determine start position */
    start_pos = nlk_get_file_pos(fp, use_lines, total, num_threads, thread_id);

    /* determine end position i.e. start pos of next thread */
    next_thread = thread_id + 1;
    if(next_thread == num_threads) {
        /* this is the last thread */
        fseek(fp, 0, SEEK_END); 
        end_pos = ftell(fp);
    } else {
        end_pos = nlk_get_file_pos(fp, use_lines, total, num_threads,
                                   next_thread);
    }

    /* set position */
    fseek(fp, start_pos, SEEK_SET);

    return end_pos;
}


size_t
nlk_text_get_split_start_line(size_t total_lines, unsigned int splits, 
                              unsigned int split_id)
{
    size_t start = 0;
    start = (total_lines / (double)splits) * (size_t)split_id;
    return start;
}

size_t
nlk_text_get_split_end_line(size_t total_lines, unsigned int splits, 
                            unsigned int split_id)
{
    size_t end = 0;
    if(split_id == splits) {
        return total_lines - 1;
    }
    end = (total_lines / (double)splits) * (size_t)(split_id + 1) - 1;
    return end;
}


/**
 * @brief Print an NLK "line".
 *
 * @param line      the NLK line (array of (strings) array's of chars
 */
void
nlk_text_print_line(char **line)
{
    char buf[NLK_LM_MAX_LINE_SIZE * NLK_LM_MAX_WORD_SIZE];
    unsigned int num_chars = 0;
    size_t ii = 0;

    while(line[ii][0] != '\0') {
        num_chars += sprintf(&buf[num_chars], "%s ", line[ii]);
        ii++;
    }
    printf("%s\n", buf);
}

/**
 * @brief Print an NLK "line" with a number.
 *
 * @param line      the NLK line (array of (strings) array's of chars
 * @param line_num  the line number associated with the line
 * @param thread_id the id of the thread
 *
 * @TODO: fix this function and make it defined only when debug
 * @warning: not working at the moment, can't be bothered to fix
 *
 * @note
 * This function is used for debugging.
 * @endnote
 */
#ifdef DEBUG
void
nlk_text_debug_numbered_line(char **line, size_t line_num, int thread_id)
{
    char buf[NLK_LM_MAX_LINE_SIZE * NLK_LM_MAX_WORD_SIZE];
    unsigned int num_chars = 0;
    size_t ii = 0;

    while(line[ii][0] != '\0') {
        num_chars += sprintf(&buf[num_chars], "%s ", line[ii]);
        ii++;
    }
    nlk_debug("line: %zu\tthread: %d\t%s\n", line_num, thread_id, buf);

}
#else
#define nlk_text_debug_numbered_line(...)
#endif
