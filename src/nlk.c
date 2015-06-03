
#include <stdbool.h>

#include "nlk_err.h"
#include "nlk_math.h"
#include "nlk_random.h"


#include "nlk.h"


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
}
