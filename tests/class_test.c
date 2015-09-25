#include <stdio.h>
#include "minunit.h"
#include "../src/nlk_dataset.h"
 
int tests_run = 0;
int tests_passed = 0;


/**
 * Test reading classes
 */
static char *
test_read_classes()
{
    const size_t N_ENTRIES = 7;
    const unsigned int N_CLASSES = 3;
    const unsigned int t[7] = {1, 1, 1, 0, 2, 2, 2};
    struct nlk_dataset_t *dset;

    printf("Testing dataset read\n");

    /* read file */
    dset = nlk_dataset_load_path("data/classes_small.txt");
    mu_assert("read classes: error reading file", dset != NULL);

    /* summary */
    mu_assert("read classes: wrong number of lines read", 
              dset->size == N_ENTRIES);
    mu_assert("read classes: bad number of classes", 
              dset->n_classes = N_CLASSES);


    /* values */
    for(unsigned int ii = 0; ii < N_ENTRIES; ii++) {
        unsigned int index = dset->ids[ii];
        if(dset->classes[ii] != t[index]) {
            printf("dset[%u] = %u should test[%u] = %u\n", 
                    ii, dset->classes[ii], index, t[index]);

        }
        mu_assert("read classes: invalid class value", 
                  dset->classes[ii] == t[index]);
    }


    nlk_dataset_free(dset);
    return 0;
}

/**
 * Test shuffle
 */
static char *
test_shuffle()
{
    const unsigned int t[7] = {1, 1, 1, 0, 2, 2, 2};
    struct nlk_dataset_t *dset;

    printf("Testing dataset shuffle\n");
    dset = nlk_dataset_load_path("data/classes_small.txt");

    /* before */
    const unsigned int n_classes = dset->n_classes;
    const unsigned int size = dset->size;

    for(size_t ii = 0; ii < 10; ii++) {
        nlk_dataset_shuffle(dset);
    }
    /* @TODO test at least one value changed places during iters */

    /* summary */
    mu_assert("shuffle: bad number of classes", dset->n_classes == n_classes);
    mu_assert("shuffle: bad size", dset->size == size);

    /* values */
    for(unsigned int ii = 0; ii < size; ii++) {
        unsigned int index = dset->ids[ii];
        if(dset->classes[ii] != t[index]) {
            printf("dset[%u] = %u should be test[%u] = %u\n", 
                    ii, dset->classes[ii], index, t[index]);

        }
        mu_assert("shuffle clases: invalid class value", 
                  dset->classes[ii] == t[index]);
    }


    return 0;
}


/**
 * Test count conll words
 */
static char *
test_conll_count()
{
    size_t n_sentences = 0;
    printf("Testing CONLL counts\n");


    unsigned int *counts;
    counts = nlk_supervised_corpus_count_conll("data/conll.mini.txt", 
                                               &n_sentences);
    if(n_sentences != 12) {
        printf("%zu\n", n_sentences);

    }
    mu_assert("bad number of sentences", n_sentences == 12);

    if(counts[0] != 27) {
        printf("counts[0] = %u\n", counts[0]);
    }
    mu_assert("bad number of words for sentence 0", counts[0] == 27);

   if(counts[1] != 12) {
        printf("counts[1] = %u\n", counts[0]);
    }
    mu_assert("bad number of words for sentence 1", counts[1] == 12);


    mu_assert("bad number of words for sentence 2", counts[2] == 5);
    mu_assert("bad number of words for sentence 3", counts[3] == 9);
    mu_assert("bad number of words for sentence 11", counts[11] == 12);


    free(counts);
    return 0;
}


/**
 * Test conll
 */
static char *
test_conll_load()
{
    int ret = 0;
    struct nlk_supervised_corpus_t *corpus = NULL;
    struct nlk_vocab_t *label;

    printf("Testing CONLL loading\n");

    corpus = nlk_supervised_corpus_load_conll("data/conll.mini.txt", NULL);
    mu_assert("corpus not loaded", corpus != NULL);
    mu_assert("bad number of sentences", corpus->n_sentences == 12);

    /* words */
    ret = strcmp(corpus->words[0][0], "@paulwalk");
    if(ret != 0) {
        printf("corpus word = %s\n", corpus->words[0][0]);
        nlk_supervised_corpus_print(corpus);
    }
    mu_assert("bad word s=0 (1st), w=0 (1st)", ret == 0);

    ret = strcmp(corpus->words[0][1], "It");
    mu_assert("bad word s=0, w=1", ret == 0);


    ret = strcmp(corpus->words[0][3], "the");
    mu_assert("bad word s=0, w=4", ret == 0);

    ret = strcmp(corpus->words[1][0], "From");
    mu_assert("bad word s=1, w=0", ret == 0);

    ret = strcmp(corpus->words[2][4], "#photography");
    mu_assert("bad word s=2, w=4 (last)", ret == 0);

    ret = strcmp(corpus->words[11][11], "grub");
    mu_assert("bad word s=11 (last), w=11 (last)", ret == 0);

    /* label names */
    label = nlk_vocab_at_index(&corpus->label_map, corpus->classes[0][0]);
    ret = strcmp(label->word, "O");
    mu_assert("bad label name s=0 (1st), w=0 (1st)", ret == 0);

    label = nlk_vocab_at_index(&corpus->label_map, corpus->classes[11][11]);
    ret = strcmp(label->word, "O");
    mu_assert("bad label name s=11 (last), w=11 (last)", ret == 0);

    label = nlk_vocab_at_index(&corpus->label_map, corpus->classes[0][16]);
    ret = strcmp(label->word, "I-facility");
    mu_assert("bad label name s=11 (last), w=11 (last)", ret == 0);

    /* class id consistency */
    ret = corpus->classes[0][0] == corpus->classes[0][26];
    mu_assert("bad class s=0 w=1 == s=0 w=26", ret == 1);

    ret = corpus->classes[0][15] == corpus->classes[0][16];
    mu_assert("bad class s=0 w=15 == s=0 w=16", ret == 1);

    ret = corpus->classes[0][15] == corpus->classes[11][5];
    mu_assert("bad class s=0 w=15 == s=11 w=5", ret == 1);

    ret = corpus->classes[0][0] != corpus->classes[0][15];
    mu_assert("bad class s=0 w=0 != s=0 w=15", ret == 1);

    ret = corpus->classes[5][0] != corpus->classes[0][15];
    mu_assert("bad class s=5 w=0 != s=0 w=15", ret == 1);


    return 0;
}

/**
* Function that runs all tests
*/
static char *
all_tests() {
    mu_run_test(test_conll_count);
    mu_run_test(test_conll_load);
    mu_run_test(test_read_classes);
    mu_run_test(test_shuffle);

    return 0;
}

int 
main() {
   printf("\n-------------------------------------------------------\n");
   printf("Class Tests\n");
   printf("---------------------------------------------------------\n");
   char *result = all_tests();
   if (result != 0) {
       printf("FAIL: %s\n", result);
   }
   else {
       printf("ALL TESTS PASSED\n");
   }
   printf("Tests Passed: %d/%d\n", tests_passed, tests_run);

   return result != 0;
}
