
#include <stdbool.h>
#include <omp.h>

#include "nlk_err.h"
#include "nlk_math.h"
#include "nlk_random.h"
#include "nlk_tic.h"


#include "nlk.h"



int __nlk_num_threads = 0;  /**< global number of threads */


/**
 * Returns the file format type given a name or the default given NULL
 *
 * @param   format_name the format name (string) or NULL
 *
 * @return the format type
 */
NLK_FILE_FORMAT
nlk_format(const char *format_name)
{
    NLK_FILE_FORMAT format;

    /* default */
    if(format_name == NULL) {
        return NLK_FILE_BIN;
    }

    /* name */
    if(strcasecmp(format_name, "w2vtxt") == 0) { 
        format = NLK_FILE_W2V_TXT;
    } else if(strcasecmp(format_name, "w2vbin") == 0) {
        format = NLK_FILE_W2V_BIN;
    } else if(strcasecmp(format_name, "nlk") == 0) {
        format = NLK_FILE_BIN;
    } else if(strcasecmp(format_name, "nlktxt") == 0) {
        format = NLK_FILE_TXT;
    } else {
        NLK_ERROR_ABORT("Invalid format type.", NLK_EINVAL);
    }

    return format;
}


void
nlk_init()
{
    nlk_random_init_xs1024(nlk_random_seed());
    nlk_table_sigmoid_create();
    nlk_tic_reset();
    nlk_tic(NULL, false);
    nlk_set_num_threads(0);
}


int
nlk_set_num_threads(int num_threads)
{
    if(num_threads <= 0) {
        num_threads = omp_get_num_procs();
    }

    __nlk_num_threads = num_threads;

    omp_set_num_threads(num_threads);
    return num_threads;
}

int
nlk_get_num_threads() {
    if(__nlk_num_threads <= 0) {
        __nlk_num_threads = omp_get_num_procs();
    }
    return __nlk_num_threads;
}
