/******************************************************************************
 * NLK - Neural Language Kit
 *
 * Copyright (c) 2015 Luis Rei <me@luisrei.com> http://luisrei.com @lmrei
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to 
 * deal in the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/


/** @file nlk_class.h
 * Supervised Classification helper definitions
 */


#ifndef __NLK_CLASS_H__
#define __NLK_CLASS_H__


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


/** @struct nlk_class_set_t
 * A supervised classification dataset mapping an id to a class
 */
struct nlk_dataset_t {
    size_t           size;          /**< the number of examples */
    unsigned int     n_classes;     /**< the number of distinct classes */
    size_t          *ids;           /**< the ids array */
    unsigned int    *classes;       /**< the classes array */
};
typedef struct nlk_dataset_t NLK_DATASET;


/* create/copy */
struct nlk_dataset_t    *nlk_dataset_create(size_t);
struct nlk_dataset_t    *nlk_dataset_create_copy(const struct nlk_dataset_t *);

/* splits */
void nlk_dataset_split_r(const struct nlk_dataset_t *, const float,
                         struct nlk_dataset_t **, struct nlk_dataset_t **);
void nlk_dataset_split(const struct nlk_dataset_t *, const float,
                       struct nlk_dataset_t **, struct nlk_dataset_t **);
void nlk_dataset_swap(struct nlk_dataset_t *, struct nlk_dataset_t *);


/* load */
void                     nlk_dataset_free(struct nlk_dataset_t *);
struct nlk_dataset_t    *nlk_dataset_load(FILE *);
struct nlk_dataset_t    *nlk_dataset_load_path(const char *);

/* misc */
void                     nlk_dataset_shuffle(struct nlk_dataset_t *);
double                   nlk_class_score_accuracy(const unsigned int *, 
                                                  const unsigned int *, 
                                                  const size_t);


__END_DECLS
#endif /* __NLK_CLASS_H__ */
