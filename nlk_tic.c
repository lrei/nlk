/*******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2014 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ******************************************************************************/


/** @file nlk_tic.c
 * For checking progress / execution time
 */


#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>


bool __nlk_tic_ticking = false;
clock_t __nlk_tic_start;
clock_t __nlk_tic_end;
time_t __nlk_time_start;


/** @fn nlk_tic(char *msg)
 * Prints the current time, the elapsed time since the first call/last call to 
 * nlk_toc() and a message
 *
 * @param msg   a message string that will be printed, use NULL for none.
 */
void
nlk_tic(char *msg, bool newline)
{
    time_t time_now;
    double tic_diff;
    double time_diff;
    

    /* First call to tic */
    if(__nlk_tic_ticking == false) {  
        __nlk_tic_ticking = true;
        __nlk_tic_start = clock();
        time(&__nlk_time_start);
        if(msg != NULL) {
            printf("nlk tic: %s\n", msg);
        }

        return;
    }

    /* All subsequent calls */
    __nlk_tic_end = clock();
    tic_diff = (double)(__nlk_tic_end - __nlk_tic_start) / 
               (double)(CLOCKS_PER_SEC * 1000);    /* milliseconds */


    time(&time_now);
    time_diff = difftime(time_now, __nlk_time_start);

    if(msg != NULL) {
        printf("\rnlk tic %.fs (elapsed %.5f): %s", time_diff , tic_diff, msg);
        if(newline) {
            printf("\n");
        }
    }
    fflush(stdout);
}

void
nlk_toc(char *msg, bool newline) {

    nlk_tic(msg, newline);
    __nlk_tic_start = clock();
}

void nlk_tic_reset()
{
    __nlk_tic_ticking = false;
}
