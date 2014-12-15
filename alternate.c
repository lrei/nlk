#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "alternate.h"



    /*********** ALTERNATE *****************
    nlk_real *syn0; nlk_real *syn1; nlk_real *neu1e; real f; real g; real temp;
    long long a, b;
    unsigned long long next_random = 1;

    a = posix_memalign((void **)&syn0, 128, 
                       (long long)vocab_size * layer_size * sizeof(real));
    if (a != 0) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }

    a = posix_memalign((void **)&syn1, 128,
                       (long long)vocab_size * layer_size * sizeof(real));

    if (a != 0) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer_size; b++)
     syn1[a * layer_size + b] = 0;


    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer_size;
    }
    neu1e = (real *)calloc(layer_size, sizeof(real));


    nlk_array_init_wity_carray(lk1->weights, syn0);
    nlk_array_init_wity_carray(lk2->weights, syn1);
    /********** /ALTERNATE *****************/


                    /*** alternate ***
                    nlk_alt_sg_start(layer_size, neu1e);
                    /** /alternate **/

/*** alternate ***
                        nlk_alt_sg_step(word->index, point, layer_size, 
                                neu1e, code, syn0, syn1, learn_rate,
                                sigmoid_table->table->data, &f, &g, &temp); 

                        bool flag = false;
                        if(!nlk_carray_compare_carray(cc_out->data, 
                                                      &syn0[word->index 
                                                            * layer_size],
                                                      layer_size, 1)) {
                            printf("\ncc_out != syn0[l]\n");
                            flag = true;
                        }
                        if(!nlk_carray_compare_carray(lk1_out->data, 
                                                      &syn0[word->index 
                                                            * layer_size],
                                                      layer_size, 1)) {
                            printf("\nlk1_out != syn0[l]\n");
                            flag = true;
                        }
                        if(!nlk_carray_compare_carray(&lk1->weights->data[
                                                        word->index 
                                                      * lk1->weights->cols], 
                                                  &syn0[word->index * layer_size],
                                                  layer_size, 1)) {
                        printf("\nsyn0[l] != lk1->weights[l]\n");
                        flag = true;       
                        }
                        if(!nlk_array_compare_carray(lk1_out,
                                                     &lk1->weights->data[
                                                        word->index 
                                                      * lk1->weights->cols], 
                                                        1)) {
                        printf("\nlk1_out != lk1->weights[l]\n");
                        flag = true;       
                        }
                        if(!nlk_carray_compare_carray(&lk2->weights->data[point
                                                        * lk2->weights->cols], 
                                                      &syn1[point * layer_size],
                                                      layer_size, 1)) {
                            printf("\nlk2[point] != syn1[point]\n");
                            printf("%f != %f\n",
                                nlk_carray_abs_sum(&lk2->weights->data[point
                                                        * lk2->weights->cols],
                                                        lk2->weights->cols),
                                nlk_carray_abs_sum(&syn1[point
                                                        * layer_size],
                                                        layer_size));

                            flag = true;
                        }
                        if(fabs(temp - lk2_out) > 0.1) {
                            printf("\ntemp = %f, lk2_out = %f\n", temp, lk2_out);
                            flag = true;
                        }

                        if(fabs(f - out) > 0.1) {
                            printf("\nf = %f, out = %f\n", f, out);
                            printf("see: temp = %f, lk2_out = %f\n", temp, lk2_out);
                            flag = true;
                        }
                        if(fabs(g - grad_out) > 0.001) {
                            printf("g = %f, grad_out = %f\n", g, grad_out);
                            flag = true;
                        }
                        if(!nlk_array_compare_carray(grad_acc, neu1e, 0.1)) {
                            printf("neu1e != grad_out\n");
                            flag = true;
                        }
                        if(flag) {
                            printf("\n");
                            exit(1);
                        }
                        /** /alternate **/

                    /*** alternate ***
                    nlk_al_sg_end(word->index, layer_size, syn0, neu1e);
                    if(!nlk_array_compare_carray(grad_acc, neu1e, 1)) {
                            printf("grad_acc != neu1e\n");
                            exit(1);
                    }
                    if(!nlk_carray_compare_carray(&lk1->weights->data[word->index 
                                                      * lk1->weights->cols], 
                                                  &syn0[word->index * layer_size],
                                                  layer_size, 1)) {
                        printf("syn0 != lk1->weights\n");
                        exit(1);        
                    }
                    /** /alternate **/


void 
nlk_alt_sg_init(real *syn0, real *syn1, real *neu1e,  
                      long long vocab_size, long long layer1_size) {

}

void nlk_alt_sg_start(size_t layer1_size, real *neu1e)
{
    size_t c;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
}


void nlk_alt_sg_step(size_t input_word_idx, size_t point, 
                     size_t layer1_size, real *neu1e,
                     char code, real *syn0, 
                     real *syn1, real alpha, real *expTable,
                     real *f, real *g, real *temp) {
    size_t c;
    size_t d;
    size_t l1 = input_word_idx * layer1_size;
    size_t l2 = point * layer1_size;
    *f = 0;
    *g = 0;
    for (c = 0; c < layer1_size; c++) *f += syn0[c + l1] * syn1[c + l2];
    *temp = *f;
    // Propagate hidden -> output
    *f = 1.0 /(1.0 + exp(-*f));
    // 'g' is the gradient multiplied by the learning rate
    *g = (1 - code - *f) * alpha;
    // Propagate errors output -> hidden
    for (c = 0; c < layer1_size; c++) neu1e[c] += *g * syn1[c + l2];
    // Learn weights hidden -> output
    for (c = 0; c < layer1_size; c++) syn1[c + l2] += *g * syn0[c + l1];
}


void nlk_al_sg_end(size_t input_word_idx, size_t layer1_size, real *syn0, 
                   real *neu1e)
{
    size_t c;
    size_t l1 = input_word_idx * layer1_size;
    // Learn weights input -> hidden
    for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
}

