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
#include <limits.h>
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

#include "nlk.h"
#include "nlk_err.h"
#include "nlk_text.h"


/**
 * Create a line (allocate memory for a line)
 */
char **
nlk_text_line_create()
{
    char **text_line = (char **) calloc(NLK_MAX_LINE_SIZE, sizeof(char *));
    if(text_line == NULL) {
        NLK_ERROR_ABORT("unable to allocate memory for text", NLK_ENOMEM);
        /* unreachable */
    }
    for(size_t zz = 0; zz < NLK_MAX_LINE_SIZE; zz++) {
        text_line[zz] = calloc(NLK_MAX_WORD_SIZE, sizeof(char));
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
    for(size_t zz = 0; zz < NLK_MAX_LINE_SIZE; zz++) {
        free(text_line[zz]);
        text_line[zz] = NULL;
    }

    /* free text_line */
    free(text_line);
    text_line = NULL;
}


/**
 * @brief ASCII in-place convertion to lower case
 *
 * @param st    the string (overwritten with lower case version) 
 */
void
nlk_text_ascii_lower(register char *st)
{
    for(register char *p = st; *p != '\0'; p++) {
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
 * Open file for reading
 *
 * @param filepath  the path to the file to be opened
 *
 * @return the file descriptor
 */
int
nlk_open(const char *filepath)
{
    int fd;

    /* open file */
    if((fd = open(filepath, O_RDONLY)) < 0) {
        nlk_log_err("%s", filepath);
        NLK_ERROR(strerror(errno), NLK_FAILURE);
        /* unreachable */
    }

    /* access will be sequential */
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    return fd;
}


/**
 * Copies a buffer into a char ** with tokenized words
 * Called only from nlk_read_line
 */
inline static int
nlk_text_make_words(char *buf, const ssize_t len, char **line, size_t *number)
{
    register int idx = 0;
    register char *dest = &line[0][0];
    register char *s = buf;
    register char *p = buf;
    const char *end = &buf[len];

    /* ignore whitespaces before the line id (number) - i am a merciful god */
    while(p != end && isspace(*p)) { p++; }
    s = p;

    /** @section Buffered Read From File 
     * idx = 0 is the line number, hence the = in <= 
     * our index is offset by +1 from the word_index i.e. 
     * line[word_idx] = line[idx-1]
     * dest will point line[idx-1] after the first "word"
     * and thus point to the right array in **line
     */
    while(p != end && idx <= NLK_MAX_LINE_SIZE) {

        while( ! isspace(*p) && p != end) { 
            p++; 
        } /* p points to the token end, s to the token start */

        
        if(p - s > NLK_MAX_WORD_SIZE) { /* ignore large words */
            s = p;
            continue;
        }

        /* copy without the terminator (whitespace or null) */
        while(s != p) { 
            *dest = *s;  
            dest++;
            s++;
        }
        /* NULL terminate word */
        *dest = '\0'; 

        /* move forward skipping whitespaces */
        while(p != end && isspace(*p)) { p++; }
        s = p;

        /** @subsection Handle The Line Number (id)
         * first "word" => the line number 
         */
        if(idx == 0) { 
            char *endptr;
            errno = 0;
            unsigned long long int val = strtoull(line[0], &endptr, 10);

            /* error handling */
            if ((errno == ERANGE && (val == ULLONG_MAX))
                || (errno != 0 && val == 0)) {

                NLK_ERROR(strerror(errno), NLK_FAILURE);
               /* unreachable */
            }
            if(endptr == line[0]) {
               NLK_ERROR("No Line Number (id) Found", NLK_FAILURE);
               /* unreachable */
            } else if (*endptr != '\0') { 
                nlk_log_err("number parsed: %llu\ncharacters after number: %s", 
                            val, endptr);
                NLK_ERROR("file parsing issue", NLK_FAILURE);
               /* unreachable */
            }

            /**! SUCCESS: we have our number
             */
            *number = val;
        }


        /**@subsection Go to next word
         */
        idx++;
        dest = &line[idx-1][0];
        *dest = '\0';   /* null terminate it in case it is the last */ 

    } /* end of buffer */

    if(idx < NLK_MAX_LINE_SIZE) {
        return 0;
    }

    return NLK_ETRUNC;
}


/**
 * Reads a number-line pair (separated by a whitespace).
 *
 * @param fd    the file descriptor to read from
 * @param line  array where the words will be stored - will be null terminated)
 * @param number          variable where line number will be stored
 *
 * @return integer of the sentence separator '\n' or EOF
 */
int
nlk_read_line(int fd, char **line, size_t *number, char *buf)
{
    ssize_t  bytes_read = 0;            /**< bytes read by read(2) */
    ssize_t  len        = 0;            /**< current length of buffer */
    char    *b          = buf;          /**< current pos in buf */  
    int      term       = NLK_FAILURE;

    /* inititialize some vars to hadle empty lines */
    *number = (size_t) -1;
    line[0][0] = '\0';


    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);


    /* read */
    while( (bytes_read = read(fd, b, BUFFER_SIZE)) > 0 ) {
        len += bytes_read;
        /* find first newline */
        char *end = memchr(b, '\n', bytes_read);

        /* end: newline found */
        if(end != NULL) { /* found */
            term = '\n';
            /* return fd to rightful place: cur_pos - bytes_read - bytes_used */
            end++; /* move over the newline */
            off_t loc = lseek(fd, 0, SEEK_CUR) - (bytes_read - (end - b));
            lseek(fd, loc, SEEK_SET);
            len = end - buf;
            break;

        /* end: EOF */
        } else if(bytes_read < BUFFER_SIZE) { 
            len += bytes_read;
            term = EOF;
            break;
    
        /* continue reading file */
        } else if((len + bytes_read) < NLK_MAX_CHARS) {
            /* if we've haven't exceeded our buffer */
            b += bytes_read;    /* increment b and go for another read() */
            continue;

        /* buffer size exceeded */
        } else { /* line is too big, puppy is sad */
            nlk_log_err("line_len = %zu", (len + bytes_read));
            NLK_ERROR("Line length > max_line_size", NLK_ETRUNC);
            /* unreachable */
        }
    }

    /* if(len == 1) => empty newline */
    if(len > 1) {
        nlk_text_make_words(buf, len, line, number);
    }

    if(bytes_read == 0) {
        return EOF;
    } else if(bytes_read < 0) {
        NLK_ERROR(strerror(errno), NLK_FAILURE);
            /* unreachable */
    }

    return term;
}


/**
 * Create text_line from string
 */
void 
nlk_text_line_read(char *str, const ssize_t len, char **line) 
{
    register int idx = 0;
    register char *p = str;
    register char *dest = &line[0][0];
    register char *s;
    const char *end = &str[len];

    /* skip starting whitespaces whitespaces */
    while(p != end && isspace(*p)) { p++; }
    s = p;

    /* read word loop */
    while(p != end && idx <= NLK_MAX_LINE_SIZE) {

        while( ! isspace(*p) && p != end) { 
            p++; 
        } /* p points to the token end, s to the token start */

        
        if(p - s > NLK_MAX_WORD_SIZE) { /* ignore large words */
            s = p;
            continue;
        }

        /* copy without the terminator (whitespace or null) */
        while(s != p) { 
            *dest = *s;  
            dest++;
            s++;
        }
        /* NULL terminate word */
        *dest = '\0'; 

        /* skip whitespaces at the end */
        while(p != end && isspace(*p)) { p++; }
        s = p;

        /** go to next word */
        idx++;
        dest = &line[idx-1][0];
        *dest = '\0';   /* null terminate it in case it is the last */
    }
}


/**
 * Counts lines in a file
 * Lines are terminated by NEWLINE
 * Code inpired by GNU coreutils/wc
 *
 * @param filepath  path of the file to count lines for
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
    if((fd = nlk_open(filepath)) < 0) {
        return fd;
    }

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

    /* handle error */ 
    if(bytes_read < 0) {
        lines = 0;
        NLK_ERROR(strerror(errno), NLK_FAILURE);
    } 

    close(fd);
    return lines;
}


/**
 * Set file position to the beginning of a given line
 *
 * @param fp    the file point
 * @param line  the line number to set the file pointer to
 *
 */
off_t
nlk_text_goto_line(int fd, const size_t line)
{
    size_t lines = 0;
    char buf[BUFFER_SIZE];
    ssize_t bytes_read = 0;
    ssize_t lines_read = 0;
    size_t line_cur = 0;
    bool long_lines = false;
    off_t loc = 0;
    char *p = 0;
    char *end = 0;

    /* goto start */
    lseek(fd, 0, SEEK_SET);

    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    /* handle trivial case line = 0 */
    if(line == 0) {
        return 0;
    }

    /**@section Read File Loop: count newlines 
     */
    while((bytes_read = read(fd, buf, BUFFER_SIZE)) > 0) {
        p = buf;
        end = p + bytes_read;
        uint64_t plines = lines;


        /**@subsection Short Lines
         * Avoid function call overhead for shorter lines
         */
        if(long_lines == false) {
            while(p != end) {
                if(*p++ == '\n') {
                    ++lines;
                    ++line_cur;
                    if(line_cur == line) { 
                        /**! FOUND IT */
                        goto line_found;
                    }
                }
            }
        } else {
            /** @subsection Long Lines
             * memchr is more efficient with longer lines
             */
            while((p = memchr(p, '\n', end - p))) {
                ++p;
                ++lines;
                ++line_cur;
                if(line_cur == line) { 
                    /**! FOUND IT */
                    goto line_found;
                }
            }
        } /* end of long lines */

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
        NLK_ERROR(strerror(errno), NLK_FAILURE);
        /* unreachable */
    } 

    /* end of file: line not found */
    NLK_ERROR("line not in file", NLK_EBADLEN);
    /* unreachable */

line_found:
    /* (end - p) represents the bytes read after the line */
    loc = lseek(fd, 0, SEEK_CUR) - (end - p);
    lseek(fd, loc, SEEK_SET);
    return loc;
}


/**
 * Set file position at a given offset from the start
 * @note: simple wrapper for lseek
 */
void
nlk_text_goto_location(int fd, const off_t location_offset)
{
    lseek(fd, location_offset, SEEK_SET);
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
 * @brief Print an NLK "line".
 *
 * @param line      the NLK line (array of (strings) array's of chars
 */
void
nlk_text_print_line(char **line)
{
    char buf[NLK_MAX_LINE_SIZE * NLK_MAX_WORD_SIZE];
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
    char buf[NLK_MAX_LINE_SIZE * NLK_MAX_WORD_SIZE];
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
