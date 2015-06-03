#include <stdio.h>
#include <inttypes.h>
#include "minunit.h"
#include "../src/nlk_vocabulary.h"
#include "../src/nlk_text.h"
 
int tests_run = 0;
int tests_passed = 0;


int
compare_vocab(struct nlk_vocab_t *created, struct nlk_vocab_t *loaded)
{

    /* match stats */
    size_t csize = nlk_vocab_size(&created);
    size_t lsize = nlk_vocab_size(&loaded);
    if(csize != lsize) {
        printf("csize = %zu, lsize = %zu\n", csize, lsize);
        return 1;
    }

    uint64_t ctotal = nlk_vocab_total(&created);
    uint64_t ltotal = nlk_vocab_total(&created);
    if(ctotal != ltotal) {
        printf("ctotal = %"SCNu64", ltotal = %"SCNu64"\n", ctotal, ltotal);
        return 2;
    }

    /* iterate through */
    struct nlk_vocab_t *cv;
    struct nlk_vocab_t *lv;
    for(size_t index = 0; index < csize; index++) {
        cv = nlk_vocab_at_index(&created, index);
        lv = nlk_vocab_at_index(&loaded, index);

        /* @TODO match strings */


        /* match counts */
        if(cv->count != lv->count) {
            printf("Words at %zu have different counts: "
                    "%"SCNu64" != %"SCNu64"\n", 
                   index, cv->count, lv->count);
            return 3;
        }

        /* @TODO match codes/points if exist */
    }

    return 0;
}

/**
 * Test creating a vocabulary from a dataset and loading from disk
 * "Large" version
 */
static char *
test_vocab_create_large()
{
    struct nlk_vocab_t *created;
    struct nlk_vocab_t *imported;
    struct nlk_vocab_t *loaded;
    int ret = 0;
    
    /* create */
    created = nlk_vocab_create("../data/imdb-id.txt", 10, false);

    /* load, ensure sorted */
    imported = nlk_vocab_import("data/imdb.vocab.txt", 100);
    nlk_vocab_sort(&imported);

    /* save created since it might be necessary to debug */
    nlk_vocab_export("tmp/imdb.vocab.created.txt", &created);

    /* Compare */
    ret = compare_vocab(created, imported);
    mu_assert("Large Vocabulary: Created != Imported Size", ret != 1);
    mu_assert("Large Vocabulary: Created != Imported Total", ret != 2); 
    mu_assert("Large Vocabulary: Created != Imported word count", ret != 3);

    nlk_vocab_free(&imported);

    /* save binary */
    FILE *vf;
    vf = fopen("tmp/imdb.vocab.created.full.txt", "wb");
    mu_assert("unable to open file for saving", vf != NULL);
    nlk_vocab_save(&created, 100, 50000, vf);
    fclose(vf);

    /* load binary */
    size_t max_word_size = 0;
    size_t max_line_size = 0;
    vf = fopen("tmp/imdb.vocab.created.full.txt", "rb");
    mu_assert("unable to open file", vf != NULL);
    loaded = nlk_vocab_load(vf, &max_word_size, &max_line_size);
    fclose(vf);
    mu_assert("bad word size", max_word_size == 100);
    mu_assert("bad line size", max_line_size == 50000);
    
    /* compare */
    ret = compare_vocab(created, loaded);
    mu_assert("Large Vocabulary: Created != Loaded Size", ret != 1);
    mu_assert("Large Vocabulary: Created != Loaded Total", ret != 2); 
    mu_assert("Large Vocabulary: Created != Loaded word count", ret != 3);


    /* free and return */
    nlk_vocab_free(&loaded);
    nlk_vocab_free(&created);
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
 
int 
main() {
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
    printf("Tests Passed: %d/%d\n", tests_passed, tests_run);
 
    return result != 0;
}
