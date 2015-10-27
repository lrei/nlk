#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <locale.h>
#include <utf8proc.h>
#include "minunit.h"
#include "../src/nlk_string.h"


#define SIZE 16

int tests_run = 0;
int tests_passed = 0;


void
print_array(const char *s)
{
    utf8proc_ssize_t bytes = 0;
    size_t len = 0;
    utf8proc_int32_t ch;

    while(s[len] != '\0') {
        bytes = utf8proc_iterate((utf8proc_uint8_t *) &s[len], -1, &ch);
        if(bytes < 0) {
            printf("BOOM\n");
            return;
        }
        len += bytes;
        printf("%"PRIx32" ", ch);
    }
    printf("\n");
}

/**
 * Test case folding
 */
static char *
test_case_folding()
{
    char str[SIZE + 1];
    char lower0[SIZE*4 + 1];
    char lower1[SIZE*4 + 1];
    char lower2[SIZE*4 + 1];

    char upper0[SIZE*4 + 1];
    ssize_t bytes = 0;
    ssize_t len = 0;
    int ret = 0;

    setlocale (LC_ALL, "");


    /* test normal cases */
    for(utf8proc_uint32_t c = 1; c <= 0x110000; ++c) {
        if(!utf8proc_codepoint_valid(c)) {
            continue;
        }

        /* fill a string with this character */
        len = 0;
        while(len + 4 <= SIZE) {
            bytes = utf8proc_encode_char(c, (utf8proc_uint8_t *) &str[len]);
            mu_assert("writting failed", bytes > 0);
            len += bytes;
        }

        /* null terminate it */
        str[len] = '\0';

        /* case fold it 1: 
         */

        /* down: */
        bytes = nlk_string_lower(str, SIZE * 4, lower0);
        if(bytes < 0) {
            printf("%s (%zd)\n", str, len);
            lower0[SIZE*4] = '\0';
            printf("%s\n", lower0);
        }
        mu_assert("bad conversion to lower case", bytes > 0);

        /* up */
        bytes = nlk_string_upper(lower0, SIZE * 4, upper0);
        if(bytes < 0) {
            printf("%s (%zd)\n", str, len);
            upper0[SIZE*4] = '\0';
            printf("%s\n", upper0);
        }
        mu_assert("bad conversion to upper case", bytes > 0);

        /*down */
        bytes = nlk_string_lower(upper0, SIZE * 4, lower1);
        if(bytes < 0) {
            printf("%s (%zd)\n", str, len);
            upper0[SIZE*4] = '\0';
            printf("%s\n", upper0);
        }
        mu_assert("bad conversion to upper case", bytes > 0);


        /* case fold it 2: 
         */
        bytes = nlk_string_lower(str, SIZE * 4, lower0);
        if(bytes < 0) {
            printf("%s (%zd)\n", str, len);
            lower0[SIZE*4] = '\0';
            printf("%s\n", lower0);
        }
        mu_assert("bad conversion to lower case", bytes > 0);

        /* up */
        bytes = nlk_string_upper(lower0, SIZE * 4, upper0);
        if(bytes < 0) {
            printf("%s (%zd)\n", str, len);
            upper0[SIZE*4] = '\0';
            printf("%s\n", upper0);
        }
        mu_assert("bad conversion to upper case", bytes > 0);

        /*down */
        bytes = nlk_string_lower(upper0, SIZE * 4, lower2);
        if(bytes < 0) {
            printf("%s (%zd)\n", str, len);
            upper0[SIZE*4] = '\0';
            printf("%s\n", upper0);
        }
        mu_assert("bad conversion to upper case", bytes > 0);

        /* now compare both */
        ret = strcmp(lower2, lower1);
        if(ret != 0) {
            printf("%s(%zu)\n%s(%zu)\n", 
                    lower0, strlen(lower0), lower1, strlen(lower1));
            print_array(lower1);
            print_array(lower2);
        }
        mu_assert("case folding failure (1)", ret == 0);





        /* test error cases */
        bytes = nlk_string_lower(str, 2, lower0);
        mu_assert("missing error", bytes < 0);
        
    }

    return 0;
}


/**
 * Function that runs all tests
 */
static char *
all_tests() {
    mu_run_test(test_case_folding);
    return 0;
}
 
int 
main() {
    printf("\n-------------------------------------------------------\n");
    printf("String Tests\n");
    printf("---------------------------------------------------------\n");

    char *result = all_tests();
    if(result != 0) {
        printf("FAIL: %s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests Passed: %d/%d\n", tests_passed, tests_run);
 
    return result != 0;
}
