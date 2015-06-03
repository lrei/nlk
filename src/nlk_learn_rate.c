/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2014-2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
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


/** @file nlk_learn_rate.c
 * Learning Rate update functions.
 */


#include "nlk_learn_rate.h"


/**
 * Learn rate update function from word2vec
 *
 * @param learn_rate        current learn rate
 * @param start_learn_rate  starting learn rate
 * @param epochs            total number of epochs
 * @param word_count_actual total number of words seen so far
 * @param train_words       total number of words in train file
 *
 * @return  new learn rate
 */
nlk_real
nlk_learn_rate_w2v(nlk_real learn_rate, const nlk_real start_learn_rate,
                   const size_t epochs, const uint64_t word_count_actual, 
                   const uint64_t train_words)
{
    learn_rate = start_learn_rate * (1 - word_count_actual / 
                                     (nlk_real) (epochs * train_words + 1));

    if(learn_rate < start_learn_rate * 0.0001) {
        learn_rate = start_learn_rate * 0.0001;
    }

    return learn_rate;
}


/**
 * Interval step decrease learning rate update function
 */
nlk_real
nlk_learn_rate_interval(nlk_real learn_rate, const unsigned int step,
                        const unsigned int total_steps)
{
    return ((learn_rate - 0.0001) / (total_steps - step)) + 0.0001;
}


/**
 * Learn Rate Decay update function
 */
nlk_real
nlk_learn_rate_decay(nlk_real learn_rate, const nlk_real decay)
{
    return learn_rate - (learn_rate * learn_rate_decay);
}


/**
 * Bold learning rate update function
 */
nlk_real
nlk_learn_rate_bold(nlk_real learn_rate, nlk_real err_previous, nlk_real err)
{
    nlk_real err_diff;
    /* kind of bold learning rate update */
    err_diff = err - err_previous;
    if(err_diff > 1e-10) { /* error is increasing */
        /* decrease learning rate */
        learn_rate *= 0.5;
    } else if(err_diff < -1e-10) { /* error is decreasing */
        /* increase learning rate */
        learn_rate += learn_rate * 0.05;
    }

    return learn_rate;
}
