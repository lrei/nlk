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
 * get_monotonic_time was taken from 
 * http://stackoverflow.com/questions/21665641/ns-precision-monotonic-clock-in-c-on-linux-and-os-x/21665642#21665642
 * By Douglas B. Staple
 */


#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif


static bool ticking = false;


/**
 * Get time
 * Use clock_gettime in linux, clock_get_time in OS X.
 */
void
nlk_get_monotonic_time(struct timespec *ts)
{
#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_MONOTONIC, ts);
#endif
}

/**
 * Determine time elapsed in seconds
 */
double 
nlk_get_elapsed_time(struct timespec *before, struct timespec *after)
{
    double deltat_s  = after->tv_sec - before->tv_sec;
    double deltat_ns = after->tv_nsec - before->tv_nsec;
    return deltat_s + deltat_ns * 1e-9;
}


/**
 * Prints msg 
 *
 * @param msg       a message string that will be printed, use NULL for none
 * @param newline   print a newline after message if true
 *
 * @return total elasped time (since last call to stop)
 */
double
nlk_tic(char *msg, bool newline)
{
    static struct timespec before;
    struct timespec after;
    double diff;
    static double total;

    /** @section First call to tic */
    if(ticking == false) {  
        ticking = true;
        nlk_get_monotonic_time(&before);
        total = 0;
        if(msg != NULL) {
            printf("nlk tic: %s\n", msg);
        }
        return 0;
    }

    /** @section All subsequent calls */
    nlk_get_monotonic_time(&after);
    diff = nlk_get_elapsed_time(&before, &after);
    total += diff;
    before = after;


    if(msg != NULL) {
        printf("\rnlk (%.2f): %s", total, msg);
        if(newline) {
            printf("\n");
        }
    }
    fflush(stdout);

    return total;
}

/**
 * Stop counting time
 */
void nlk_tic_reset()
{
    ticking = false;
}
