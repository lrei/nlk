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

#include "MurmurHash3.h"

#include "nlk_err.h"
#include "nlk_text.h"


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
 * @param low_tmp         temporary storage for converting strings to lowercase
 *                        if NULL, no conversion happens
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
nlk_read_word(FILE *file, char *word, wchar_t *low_tmp, 
              const size_t max_word_size)
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
            if(len == 0 && ch != EOF) {
                continue;
            }
            
            break; /* goto EOW */
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
    if(low_tmp != NULL || len > 0) {
        nlk_text_lower(word, low_tmp);
    }

    return ch;
}

/** 
 * @brief Read a line from a file
 *
 * @param file            FILE object to read from
 * @param line            array  where the words+1 will be stored
 *                        null terminator)
 * @param low_tmp         temporary storage for converting strings to lowercase
 *                        if NULL, no conversion happens
 * @param max_word_size   maximum size of a word (including null terminator)
 * @param max_line_size   maximum size of a line in number of words (including 
 *
 * @return integer of the sentence separator '\n', EOF or NLK_ETRUNC
 *
 * @note
 * Lines are assumed to be separated by the NEWLINE or TAB character. 
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
nlk_read_line(FILE *file, char **line, wchar_t *low_tmp,
              const size_t max_word_size, const size_t max_line_size)
{
    int term;                   /* word terminator */
    size_t word_idx = 0;        /* word in line index */

    while(!feof(file)) {
        term = nlk_read_word(file, line[word_idx], low_tmp, max_word_size);
        word_idx++;

        if(term == '\n' || term == '\t' || term == EOF) {
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
 * Counts lines remaining in a file. Does not change position of file.
 * "Lines" are terminated by NEWLINE or TAB characters
 *
 * @param fp the FILE pointer
 *
 * @return number of lines in file
 */
size_t
nlk_text_count_lines(FILE *fp)
{
    size_t lines = 0;
    int buf = 0;
    int last = 0;
    fpos_t pos;

    /* determine current position */
    fgetpos(fp, &pos);

    /* count lines until the end */
    while((buf = fgetc(fp)) != EOF) {
        if(buf == '\n' || buf == '\t') {
            lines++;
        }
        last = buf;
    }
    /* handle files that do not terminate in newline */
    if(last != '\n' && last != '\t') {
        lines++;
    }

    /* rewind file pointer so it can be used by caller */
    fsetpos(fp, &pos);

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

        if(buf == '\n' || buf == '\t') {
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

    /* count lines until the end */
    while((buf = fgetc(fp)) != EOF) {
        if(buf == '\n' || buf == '\t' || buf != ' ') {
            words++;
        }
        last = buf;
    }
    /* handle files that do not terminate in newline */
    if(last != '\n' && last != '\t' && last != ' ') {
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
nlk_text_goto_line(FILE *fp, size_t line)
{
    int buf;
    long pos = 0;

    rewind(fp);
    
    while(pos < line) {
        while((buf = fgetc(fp)) != EOF) {
            if(buf == '\n' || buf == '\t') {
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
    while(buf != '\n' && buf != '\t') {
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
    while(buf != '\n' && buf != '\t' && buf != ' ') {
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
nlk_set_file_pos(FILE *fp, bool use_lines, size_t total, size_t num_threads,
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
 * @note
 * This function is used for debugging.
 * @endnote
 */
void
nlk_text_print_numbered_line(char **line, size_t line_num, int thread_id)
{
    char buf[NLK_LM_MAX_LINE_SIZE * NLK_LM_MAX_WORD_SIZE];
    unsigned int num_chars = 0;
    size_t ii = 0;

    while(line[ii][0] != '\0') {
        num_chars += sprintf(&buf[num_chars], "%s ", line[ii]);
        ii++;
    }
    printf("%zu\t%d\t%s\n", line_num, thread_id, buf);

}


/** @section Memory Mapped File Functions
 */

/**
 * @brief open memory mapped file (training file)
 */
off_t
nlk_text_mem_open(char *file_path, char **ptr)
{
    int fd;
    int page_size;
    struct stat st;

    /* normal fd open */
    errno = 0;
    fd = open(file_path, O_RDONLY);
    if(fd == -1) {
        NLK_ERROR("unable to open training file.", errno);
        /* unreachable */
    }

    /* get file size */
    errno = 0;
    if(stat(file_path, &st) == -1) {
        NLK_ERROR("unable to get file size (stat)", errno);
        /* unreachable */
    }

    /* get OS mem page size */
    page_size = getpagesize();

    /* memory map */
    errno = 0;
    *ptr = (char *) mmap((caddr_t)0, st.st_size, PROT_READ, MAP_SHARED, fd, 
                         page_size);
    if(*ptr == (char *) -1) {
        NLK_ERROR("unable to get file size (stat)", errno);
        /* unreachable */
    }

    /* tell OS to read ahead access /thread should be sequential ?*/
	/* madvise(ptr, st.st_size, MADV_SEQUENTIAL); */

    return st.st_size;
}

/**
 * @brief close memory mapped file
 */
void
nlk_text_mem_close(caddr_t *mem_ptr, size_t size) {
    errno = 0;
    if(munmap(mem_ptr, size) == -1) {
        NLK_ERROR_VOID("unable to close memory mapped file.", errno);
        /* unreachable */
    }
}

/**
 * brief Count lines in memory mapped file
 */
size_t
nlk_text_mem_count_lines(char *mem_ptr, off_t size) {
    size_t counter = 0;
    char last = 0;

    for(size_t ii = 0; ii < size; ii++) {
        last = mem_ptr[ii];
        if(last == '\0') {
            break;
        }
        if(last == '\n' || mem_ptr[ii] == '\t') {
            counter++;
        }
    }

    if(last != '\n' && last != '\t') {
        counter++;
    }

    return counter;
}   


/**
 * @brief Get the offset of a line in a memory mapped file
 *
 * @param mem_ptr         the mmaped pointer
 * @param line_mum        the line number of the line to read
 * @param size            the size of the file
 *
 * @return file offset off the start of the line
 */
off_t
nlk_text_mem_get_line_pos(char *mem_ptr, size_t line_num, size_t size)
{
    int term;                   /* word terminator */
    size_t word_idx = 0;        /* word in line index */
    size_t cur_ln = 0;          /* position in file (line number) */
    size_t start = 0;           /* start of the line to retrieve (offset) */


    for(off_t ii = 0; ii < size && cur_ln <= line_num; ii++) {
        /* check for end of line */
		if(mem_ptr[ii] != '\n' || mem_ptr[ii] != '\t') {
            continue;
        }

		if(++cur_ln == line_num) {
            start = ii + 1;
            break;
        }
    }

    return start;
}

/** 
 * @brief Read a word from a memory mapped file
 *
 * @param buf             file memory pointer
 * @param loc             current location in memory mapped file
 * @param end             file size
 * @param word            char pointer where the word will be stored
 * @param low_tmp         temporary storage for converting strings to lowercase
 *                        if NULL, no conversion happens
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
 *
 * @warning words (tokens) bigger than max_word_size are invalid
 */
int
nlk_text_mem_read_word(char *mem_ptr, off_t *loc, off_t size, char *word, 
                       wchar_t *low_tmp, const size_t max_word_size)
{
    int ch = EOF;
    size_t len = 0;
    bool invalid = false;

    while(*loc < size) {
        ch = mem_ptr[*loc];
        *loc = *loc + 1;

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
            if(len == 0 && ch != EOF) {
                continue;
            }
            
            break; /* goto EOW */
        }

        /* just another non-space char */
        word[len] = ch;
        len++;

        /* check max_word_size */
        if(len == max_word_size - 2) {
            invalid = true;
            len = 1;
            /* keep going until the end of the token */
        }
    }
    /*
     * END-OF-WORD (EOW):
     */
    word[len] = '\0';

    /* handle invalid tokens/words */
    if(invalid) {
        return NLK_ETRUNC;
    }
    
    if(low_tmp != NULL || len > 0) {
        nlk_text_lower(word, low_tmp);
    }

    return ch;
}

/**
 *
 * @warning No truncation
 */
int
nlk_text_mem_read_line(char *mem_ptr, off_t *pos, off_t size, char **line, 
                       off_t *next, wchar_t *low_tmp,
                       const size_t max_word_size, const size_t max_line_size)
{
    int term;                   /* word terminator */
    size_t word_idx = 0;        /* word in line index */

    while(*pos < size) {
        term = nlk_text_mem_read_word(mem_ptr, pos, size,  line[word_idx], 
                                      low_tmp, max_word_size);
        word_idx++;

        if(term == '\n' || term == '\t' || term == EOF) {
            *line[word_idx] = '\0';
            return term;
        }

        if(term == NLK_ETRUNC) {    /* skip if word size > max_word_size */
            word_idx--;
        }
    }

    return EOF;
}
