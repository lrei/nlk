#include <stdio.h>
#include "minunit.h"
#include "../src/nlk_class.h"
 
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
        mu_assert("read claases: invalid class value", 
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
* Function that runs all tests
*/
static char *
all_tests() {
   mu_run_test(test_read_classes);
   mu_run_test(test_shuffle);
   return 0;
}

int 
main() {
   printf("---------------------------------------------------------\n");
   printf("Class Tests\n");
   printf("---------------------------------------------------------\n");
   char *result = all_tests();
   if (result != 0) {
       printf("%s\n", result);
   }
   else {
       printf("ALL TESTS PASSED\n");
   }
   printf("Tests Passed: %d/%d\n", tests_passed, tests_run);

   return result != 0;
}
