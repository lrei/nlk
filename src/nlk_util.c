/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
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


/** @file nlk_util.c
 * Utility functions
 * @TODO fix all these functions: assume sorted, create NLK_SET
 */

#include <stdbool.h>
#include <stdlib.h>

#include "nlk_err.h"

#include "nlk_util.h"


/**
 * Count unique elements in an array
 *
 * @param array     the array
 * @param length    the length of the array
 *
 * @return the number of unique elements in the array
 */
size_t
nlk_count_unique(const unsigned int *array, const size_t length)
{
    size_t unique = 0;
    bool is_unique = false;
    unsigned int *seen = NULL;

    seen = (unsigned int *) malloc(sizeof(unsigned int) * length);
    if(seen == NULL) {
        NLK_ERROR("unable to allocate memory for count unique", NLK_ENOMEM);
        /* unreachable */
    }


    /* for each element in the array */
    for(size_t ii = 0; ii < length; ii++) {
        /* assume it's unique unless the value in found in seen */
        is_unique = true;
        for(size_t jj = 0; jj < unique; jj++) {
            if(array[ii] == seen[jj]) {
                is_unique = false; /* yes we've seen this */
                break;
            }
        }
        /* if it is unique, add it to the seen array */
        if(is_unique) {
            seen[unique] = array[ii];
            unique++;
        }
    }   /* end of array */

    free(seen);
    return unique;
}


/**
 * Is value in the array?
 *
 * @param value     the value to search for
 * @param array     the array
 * @param length    the length of the array
 *
 * @return true if value in array, false otherwise
 */
bool
nlk_in(const size_t value, const size_t *array, const size_t length)
{
    for(size_t ii = 0; ii < length; ii++) {
        if(array[ii] == value) {
            return true;
        }
    }
    return false;
}

/**
 * set difference: elements in a not in b
 *
 * @note
 * terribly inefficient and ugly implementation
 * @endnote
 */
void
nlk_set_diff(const size_t *a, const size_t len_a, const size_t *b, 
             const size_t len_b, size_t *r, size_t *len_r)
{
    size_t len = 0;
    size_t ii = 0;

    r = (size_t *) malloc(len_a * sizeof(size_t));
    for(ii = 0; ii < len_a; ii++) {
        if(!nlk_in(a[ii], b, len_b)) {
            r[len] = a[ii];
            len++;
        }
    }

    *len_r = len;
    r = (size_t *) realloc(r, len * sizeof(size_t));
}

/**
 * Generate all positive integers from in [0, n [
 *
 * @param n the upper bound of the range
 *
 * @return array with all positive integers in the [0, n [ range
 */
size_t *
nlk_range(const size_t n)
{
    size_t *range = (size_t *) malloc(n * sizeof(size_t));

    for(size_t ii = 0; ii < n; ii++) {
        range[ii] = ii;
    }

    return range;
}

/**
 * Generate all positive integers from in [0, n [ that are not in a
 */
size_t *
nlk_range_not_in(const size_t *a, const size_t len_a, const size_t n, 
                 size_t *len_r)
{
    size_t len = 0;
    size_t *r = (size_t *) malloc(n * sizeof(size_t));
    if(r == NULL) {
        NLK_ERROR_NULL("unable to allocate memory", NLK_ENOMEM);
    }

    for(size_t ii = 0; ii < n; ii++) {
        if(!nlk_in(ii, a, len_a)) {
            r[len] = ii;
            len++;
        }
    }

    realloc(r, len * sizeof(size_t));
    if(r == NULL) {
        NLK_ERROR_NULL("unable to reallocate memory", NLK_ENOMEM);
        /* unreachable */
    }
    *len_r = len;

    return r;
}
