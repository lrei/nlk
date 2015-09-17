#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include "minunit.h"
#include "../src/nlk_text.h"
 
int tests_run = 0;
int tests_passed = 0;


/**
 * Test reading lines
 * @note: in this test line numbers refered to are 1-indexed instead of the
 * traditional 0-index
 */
static char *
test_read_lines()
{
    size_t par_id = 0;
    int ret = 0;
    
    char **text_line = nlk_text_line_create();
    
    int fd = nlk_open("data/micro.data.2.txt");
    char buffer[NLK_BUFFER_SIZE];

    /* 1st line (1-index not 0-index) */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("1-terminator", ret == '\n');
    mu_assert("1-paragrah_id", par_id == 100);
    mu_assert("1-first word", strcmp("this", text_line[0]) == 0);
    mu_assert("1-last word", strcmp(".", text_line[10]) == 0);
    mu_assert("1-text_lines is null termed", text_line[11][0] == '\0');

    /* 2nd line */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("2-terminator", ret == '\n');
    mu_assert("2-paragrah_id", par_id == 222);
    mu_assert("2-first word", strcmp("this", text_line[0]) == 0);
    mu_assert("2-last word", strcmp(".", text_line[5]) == 0);
    mu_assert("2-text_lines is null termed", text_line[6][0] == '\0');

    /** 3rd line */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("3-terminator", ret == '\n');
    mu_assert("3-paragrah_id", par_id == 33);
    mu_assert("3-first word", strcmp("the", text_line[0]) == 0);
    mu_assert("3-last word", strcmp("...", text_line[14]) == 0);
    mu_assert("3-text_lines is null termed", text_line[15][0] == '\0');

    /** 4th line */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("4-terminator", ret == '\n');
    mu_assert("4-paragrah_id", par_id == 42);
    mu_assert("4-first word", strcmp("(", text_line[0]) == 0);
    mu_assert("4-last word", strcmp("url", text_line[8]) == 0);
    mu_assert("4-text_lines is null termed", text_line[9][0] == '\0');

     /** 5th line: skip */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("5-terminator", ret == '\n');
    mu_assert("5-paragrah_id", par_id == 5);
    mu_assert("5-first word", strcmp("in", text_line[0]) == 0);
    mu_assert("5-last word", strcmp("nothing", text_line[5]) == 0);
    mu_assert("5-text_lines is null termed", text_line[6][0] == '\0');

    /** 6th line: single newline */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("6-terminator", ret == '\n');
    mu_assert("6-paragrah_id", par_id == (size_t)-1);
    mu_assert("6-text_lines is null termed", text_line[0][0] == '\0');

    /** 7th line: empty */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("7-terminator", ret == EOF);
    mu_assert("7-paragrah_id", par_id == (size_t)-1);
    mu_assert("7-text_lines is null termed", text_line[0][0] == '\0');

    
    return 0;
}

/**
 * Test count and goto lines
 * @note: this test requires that the read_line test succeeds
 */
static char *
test_goto_lines()
{
    size_t par_id = 0;
    int ret = 0;
    size_t n_lines;
    size_t line_start = 0;
    size_t line_end = 0;
    
    char **text_line = nlk_text_line_create();
    
    int fd = nlk_open("data/micro.data.2.txt");
    char buffer[NLK_BUFFER_SIZE];

    /* Count Lines */
    n_lines = nlk_text_count_lines("data/micro.data.2.txt");
    /*printf("n_lines = %zu", n_lines);*/
    mu_assert("count lines", n_lines == 6);

    /* Goto Line (requires that previous test succeeds) */
    nlk_text_goto_line(fd, 3);
    /** 4th line in 1-index */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("4-terminator", ret == '\n');
    mu_assert("4-paragrah_id", par_id == 42);
    mu_assert("4-first word", strcmp("(", text_line[0]) == 0);
    mu_assert("4-last word", strcmp("url", text_line[8]) == 0);
    mu_assert("4-text_lines is null termed", text_line[9][0] == '\0');

    nlk_text_goto_line(fd, 1);

    /* 2nd line (1-index) */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("2-terminator", ret == '\n');
    mu_assert("2-paragrah_id", par_id == 222);
    mu_assert("2-first word", strcmp("this", text_line[0]) == 0);
    mu_assert("2-last word", strcmp(".", text_line[5]) == 0);
    mu_assert("2-text_lines is null termed", text_line[6][0] == '\0');

    nlk_text_goto_line(fd, 6);
    /** 7th line: empty */
    ret = nlk_read_line(fd, text_line, &par_id, buffer);
    mu_assert("7-terminator", ret == EOF);
    mu_assert("7-paragrah_id", par_id == (size_t)-1);
    mu_assert("7-text_lines is null termed", text_line[0][0] == '\0');


    /** check split into 3 parts 
     * 0 -> [0, 1]; 1 -> [2, 3]; 2 -> [4, 5]
     */
    /* thread 0 */
    line_start = nlk_text_get_split_start_line(n_lines, 3, 0);
    line_end = nlk_text_get_split_end_line(n_lines, 3, 0);
    mu_assert("thread 0 start", line_start == 0);
    mu_assert("thread 0 end", line_end == 1);

    /* thread 1 */
    line_start = nlk_text_get_split_start_line(n_lines, 3, 1);
    line_end = nlk_text_get_split_end_line(n_lines, 3, 1);
    mu_assert("thread 1 start", line_start == 2);
    mu_assert("thread 1 end", line_end == 3);

    /* thread 2 */
    line_start = nlk_text_get_split_start_line(n_lines, 3, 2);
    line_end = nlk_text_get_split_end_line(n_lines, 3, 2);
    mu_assert("thread 2 start", line_start == 4);
    mu_assert("thread 2 end", line_end == 5);

    return 0;
}


/**
 * Test count empty lines
 */
static char *
test_count_empty_lines()
{
    size_t empty_lines = nlk_text_count_empty_lines("data/conll.mini.txt");
    mu_assert("count empty lines", empty_lines == 12);

    return 0;
}


/**
 * Function that runs all tests
 */
static char *
all_tests() {
    mu_run_test(test_read_lines);
    mu_run_test(test_goto_lines);
    mu_run_test(test_count_empty_lines);
    return 0;
}
 
int 
main() {
    printf("---------------------------------------------------------\n");
    printf("Reading Tests\n");
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
