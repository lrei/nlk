#include <stdio.h>
#include "minunit.h"
#include "../src/nlk_vocabulary.h"
#include "../src/nlk_text.h"
 
int tests_run = 0;
int tests_passed = 0;


/**
 * Test file split
 *
static char *
test_text_file_split()
{
    return 0;
}
*/

/**
 * Test creating a vocabulary from a dataset and loading from disk
 * "Large" version
 */
static char *
test_vocab_create_large()
{
    struct nlk_vocab_t *created;
    struct nlk_vocab_t *loaded;
    
    /* create */
    created = nlk_vocab_create("../data/imdb.txt", false, 100, false);
    nlk_vocab_sort(&created);
    nlk_vocab_reduce(&created, 10);

    /* load, ensure sorted */
    loaded = nlk_vocab_load("data/imdb.vocab.txt", 100);
    nlk_vocab_sort(&loaded);

    /* save created since it might be necessary to debug */
    nlk_vocab_save("tmp/imdb.vocab.created.txt", &created);

    /* match stats */
    size_t csize = nlk_vocab_size(&created);
    size_t lsize = nlk_vocab_size(&loaded);
    if(csize != lsize) {
        printf("csize = %zu, lsize = %zu\n", csize, lsize);
    }
    mu_assert("Large Vocabulary: Created != Loaded Size", csize == lsize); 

    uint64_t ctotal = nlk_vocab_total(&created);
    uint64_t ltotal = nlk_vocab_total(&created);
    if(ctotal != ltotal) {
        printf("ctotal = %zu, ltotal = %zu\n", ctotal, ltotal);
    }
    mu_assert("Large Vocabulary: Created != Loaded Total", ctotal == ltotal); 

    /* iterate through and match counts */
    struct nlk_vocab_t *cv;
    struct nlk_vocab_t *lv;
    for(size_t index = 0; index < csize; index++) {
        cv = nlk_vocab_at_index(&created, index);
        lv = nlk_vocab_at_index(&loaded, index);

        if(cv->count != lv->count) {
            printf("Words at %zu have different counts: %zu != %zu\n", 
                   index, cv->count, lv->count);
        }
        mu_assert("Large Vocabulary: different word counts", 
                  cv->count == lv->count);
        
    }

    return 0;
}


/**
 * Function that runs all tests
 */
static char *
all_tests() {
    mu_run_test(test_vocab_create_large);
    return 0;
}
 
int main(int argc, char **argv) {
    printf("---------------------------------------------------------\n");
    printf("Vocabulary Tests\n");
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
