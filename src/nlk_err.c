/* 
 * ## nlk_err.c ##
 *
 * Error handling - based on the GSL error handling.
 *
 * Copyright (c) 2014 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


#include <stdio.h>
#include <stdlib.h>
#include "nlk_err.h"


/* the user defined error handler (via *set_error_handler*) */
nlk_error_handler_t *__nlk_error_handler = NULL;


/* 
 * nlk_display - displays error messages
 *
 * Params:
 *  label - the "type" of error message, e.g. "ERROR"
 *  file - the source code file name where the error occurrent
 *  line - the line in the source code file where the error occurred
 *  reason - the human-readable error string that will be displayed
 *
 * Returns:
 *  no return value (void)
 *
 */
void
nlk_display(const char *label, const char *file, int line,  const char *reason)
{
    fprintf(stderr, "nlk: %s:%d: %s: %s\n", file, line, label, reason);
}

/* 
 * nlk_error - The actual error handling function.
 *  Calls user-defined error function if set otherwise calls *nlk_display*
 *
 * Params: meant to be compatible with nlk_error_handler_t
 *  reason - the human-readable error string that will be displayed
 *  file - the source code file name where the error occurrent
 *  line - the line in the source code file where the error occurred
 *  nlk_errno - the error reason as specified in the enum from nlk_err.h
 *
 * Returns:
 *  If a user defined error handling function is set, no return value (void)
 *  Otherwise, aborts execution by calling abort() after displaying message
 *
 */
void
nlk_error(const char* reason, const char *file, int line, int nlk_errno)
{
    /* if a user-defined nlk_error_handler is defined, call it and return */
    if(__nlk_error_handler) {
        (*__nlk_error_handler)(reason, file, line, nlk_errno);
        return ;
    }
  
    /* otherwise, call nlk_display and abort execution */
    nlk_display("ERROR", file, line, reason);
    abort();
}

/* 
 * nlk_set_error_handler - Set a user defined error handler (or the default)
 *  The error_handler is responsible for calling *abort* if that is intended
 *
 * Params:
 *  new_handler - the new error handling function, if NULL, uses default
 *
 * Returns:
 *  The previous error handling function, possibly NULL if none was defined
 *
 */
nlk_error_handler_t *
nlk_set_error_handler(nlk_error_handler_t *new_handler)
{
    nlk_error_handler_t * previous_handler = __nlk_error_handler;
    __nlk_error_handler = new_handler;
    return previous_handler;
}

/* 
 * __nlk_nop_handler - do-Nothing error handler
 *  This function is not meant to be called directly by the user but to be 
 *  called by by *nlk_set_error_handler_off()* 
 *  As the name (no operation) implies, it does, literally, nothing.
 *
 * Params: meant to be compatible with nlk_error_handler_t
 *  reason - the human-readable error string that will be displayed
 *  file - the source code file name where the error occurrent
 *  line - the line in the source code file where the error occurred
 *  nlk_errno - the error reason as specified in the enum from nlk_err.h
 
 * Returns:
 *  no return value (void)
 *
 */
static void
__nlk_nop_handler(const char *reason, const char *file, int line, int nlk_errno)
{
    /* prevent unused warnings */
    (void) reason;
    (void) file;
    (void) line;
    (void) nlk_errno;
    return ;
}

/* 
 * nlk_set_error_handler_off - disable error handling
 *  This is different from *nlk_set_error_handler(NULL)*
 *
 * Returns:
 *  The previous error handling function, possibly NULL if none was defined
 *
 */
nlk_error_handler_t *
nlk_set_error_handler_off(void)
{
    return nlk_set_error_handler(__nlk_nop_handler);
}
