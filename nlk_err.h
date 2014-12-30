/* 
 * ## nlk_err.h ##
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


#ifndef __NLK_ERR_H__
#define __NLK_ERR_H__


#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif
__BEGIN_DECLS

/*
 * ### Error Types ###
 * These are compatible with the GSL errors. Since a smaller
 * number of errors is checked for, the numbers jump a bit.
 */
enum { 
    NLK_SUCCESS  = 0, 
    NLK_FAILURE  = -1,
    NLK_ETRUNC   = -10, /* truncated */  
    NLK_EDOM     = 1,   /* input domain error, e.g sqrt(-1), including OOB */
    NLK_ERANGE   = 2,   /* output range error, e.g. exp(1e100) */
    NLK_EINVAL   = 4,   /* invalid argument supplied by user */
    NLK_ENOMEM   = 8,   /* alloc failed */
    NLK_EBADLEN  = 19   /* matrix, vector lengths are not conformant */
};

/* All NLK error handlers have the type nlk_error_handler_t */
typedef void nlk_error_handler_t(const char *reason, const char *file,
				                 int line, int nlk_errno);

/*
 * ### Error Message & Handler Functions ###
 */

/* The function that displays (print) error messages */
void nlk_display(const char *label, const char *file, int line, 
                 const char *reason);

/* The actual error handling function, can call user-defined func if set */
void nlk_error(const char * reason, const char *file, int line, int nlk_errno);

/* Set a new error handler */
nlk_error_handler_t *nlk_set_error_handler(nlk_error_handler_t *new_handler);

/* Do-Nothing (no op) error handler: used by *nlk_set_error_handler_off()* */
static void __nlk_nop_handler(const char *reason, const char *file, int line, 
                              int nlk_errno);

/* Disable error handling */
nlk_error_handler_t *nlk_set_error_handler_off(void);


/*
 * ### Error Handling Macros ###
 */

/* NLK_ERROR: call the error handler and return the error code (nlk_errno) */
#define NLK_ERROR(reason, nlk_errno) \
    do { \
        nlk_error(reason, __FILE__, __LINE__, nlk_errno) ; \
        return nlk_errno ; \
    } while (0)

/* NLK_ERROR_NULL: call the error handler and return NULL */
#define NLK_ERROR_NULL(reason, nlk_errno) \
    do { \
        nlk_error(reason, __FILE__, __LINE__, nlk_errno) ; \
        return NULL ; \
    } while (0)

/* NLK_ERROR_VOID: call the error handler no return (for void funtions)  */
#define NLK_ERROR_VOID(reason, nlk_errno) \
    do { \
        nlk_error(reason, __FILE__, __LINE__, nlk_errno) ; \
        return ; \
    } while (0)

/* NLK_ERROR_VOID: call the error handler no return (for void funtions)  */
#define NLK_ERROR_ABORT(reason, nlk_errno) \
    nlk_error(reason, __FILE__, __LINE__, nlk_errno) ; \
    abort();

/*
 * OMP DEFS
 */
#ifdef DEBUG
    #define omp_get_thread_num() 0
    #define omp_get_num_threads() 1
    #define omp_get_num_procs() 1
#endif

__END_DECLS
#endif /* __NLK_ERR_H__ */
