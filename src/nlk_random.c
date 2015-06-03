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


/** @file nlk_random.c
 * Pseudo Random Number Functions
 */

#include <stdint.h>
#include <time.h>


/**
 * Avalanche function (force mix) from MurmurHash3 applied 2x
 */
uint64_t
nlk_random_fmix(uint64_t k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdUL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53UL;
    k ^= k >> 33;
    
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdUL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53UL;
    k ^= k >> 33;

    return k;
}


/**
 * xorshift64* 64 bit pseudo random number generator
 *
 * @param x the previous value of the xorshift ouput or seed
 */
static inline void
nlk_random_xs64(uint64_t *x) {
	*x ^= *x >> 12; // a
	*x ^= *x << 25; // b
	*x ^= *x >> 27; // c
	*x = *x * UINT64_C(2685821657736338717);
}

static uint64_t __s[16];
static int __p;
/**
 * xorshift1024*
 * Written in 2014 by Sebastiano Vigna (vigna@acm.org)
 *
 * @note
 * Designed for "large-scale parallel simulation", abuse in NLK is probably
 * more than this was designed to handle but "meh".
 * @endnote
 */
uint64_t 
nlk_random_xs1024() { 
	uint64_t s0 = __s[__p];
	uint64_t s1 = __s[__p = ( __p + 1 ) & 15];
	s1 ^= s1 << 31; // a
	s1 ^= s1 >> 11; // b
	s0 ^= s0 >> 30; // c
	return ( __s[__p] = s0 ^ s1 ) * 1181783497276652981LL; 
}

/**
* Ramdon number Float between [0, 1[ from nlk_random_xs1024
*/
float 
nlk_random_xs1024_float() { 
   return (nlk_random_xs1024() % UINT16_MAX) / (float) UINT16_MAX;
}

/**
 * Initializes xorshift1024* state
 *
 * @param seed  the seed
 */
void
nlk_random_init_xs1024(uint64_t seed)
{
    uint64_t init;
    init = nlk_random_fmix(seed);
    for(int ii = 0; ii < 16; ii++) {
        nlk_random_xs64(&init);
        __s[ii] = init;
    }

}

/**
 * Generates a seed based on the number of seconds since epoch and processor
 * time used by the program
 */
uint64_t
nlk_random_seed()
{
    uint64_t seed;
    time_t e = time(NULL); /* seconds since epoch (POSIX) */

    seed = 6121984 * clock() * e;

    return seed;
}
