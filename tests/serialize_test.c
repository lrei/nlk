#include <stdio.h>
#include "minunit.h"
#include "../src/nlk_array.h"
 
int tests_run = 0;
int tests_passed = 0;


/**
 * Test writing an array to a text file and loading from it
 */
static char *
test_array_text()
{
    FILE *fp;
    size_t rows = 20;
    size_t cols = 31;
    /* create, init */
    NLK_ARRAY *origin = nlk_array_create(rows, cols);
    nlk_array_init_uniform(origin, 1, 2);

    /* save */
    fp = fopen("tmp/array.txt", "wb");
    if(fp == NULL) {
        mu_assert("unable to open file for writting: array.txt", 0);
    }

    nlk_array_save_text(origin, fp);
    fclose(fp);

    /* load */
    fp = fopen("tmp/array.txt", "rb");
    if(fp == NULL) {
        mu_assert("unable to open file for reading: array.txt ", 0);
    }

    NLK_ARRAY *loaded = nlk_array_load_text(fp);
    fclose(fp);

    /* compare headers */
    mu_assert("Array-Text Serialization: arrays rows do not match", 
              loaded->rows == origin->rows);
    mu_assert("Array-Text Serialization: arrays columns do not match", 
              loaded->cols == origin->cols);

    /* compare data */
    for(size_t ri = 0; ri < rows; ri++) {
        for(size_t ci = 0; ci < cols; ci++) {
            nlk_real ld = loaded->data[ri * cols + ci];
            nlk_real od = origin->data[ri * cols + ci];
            if(ld != od) {
                printf("original = %f\n", od);
                printf("loaded = %f\n", ld);
            }
            mu_assert("Array-Text Serialization: error in data",  ld == od);
        }
    }
    return 0;
}

/**
 * Test loading an array created with a different tool (text format)
 */
static char *
test_array_load_text() 
{
    /* create, init */
    size_t rows = 3;
    size_t cols = 3;
    nlk_real carr[9] = {0.503196, -0.482911, -0.538873,
                        -1.135939, 0.318962, -0.591220,
                        0.551616, 0.001735, -0.919580};

    NLK_ARRAY *origin = nlk_array_create(rows, cols);
    nlk_array_init_wity_carray(origin, carr);

    /* load */
    FILE *fp = fopen("data/small.vector.txt", "rb");
    if(fp == NULL) {
        mu_assert("unable to open file for reading: small.vector.txt", 0);
    }
    NLK_ARRAY *loaded = nlk_array_load_text(fp);
    fclose(fp);

    /* compare headers */
    mu_assert("Array-Load-Text: arrays rows do not match", 
              loaded->rows == origin->rows);
    mu_assert("Array-Load-Text: arrays columns do not match", 
              loaded->cols == origin->cols);

    /* compare data */
    for(size_t ri = 0; ri < rows; ri++) {
        for(size_t ci = 0; ci < cols; ci++) {
            nlk_real ld = loaded->data[ri * cols + ci];
            nlk_real od = origin->data[ri * cols + ci];
            if(ld != od) {
                printf("original = %f\n", od);
                printf("loaded = %f\n", ld);
            }
            mu_assert("Array-Load-Text: error in data",  ld == od);
        }
    }

    return 0;
}

/**
 * Function that runs all tests
 */
static char *
all_tests() {
    mu_run_test(test_array_text);
    mu_run_test(test_array_load_text);
    return 0;
}
 
int main(int argc, char **argv) {
    printf("---------------------------------------------------------\n");
    printf("Serialization Tests\n");
    printf("---------------------------------------------------------\n");
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests Passed: %d/%d\n", tests_run, tests_passed);
 
    return result != 0;
}
