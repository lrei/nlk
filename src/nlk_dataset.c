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


/** @file nlk_dataset.c
 * Dataset functions
 */

#include <stdlib.h>
#include <inttypes.h>

#include "nlk_err.h"
#include "nlk_text.h"
#include "nlk_util.h"
#include "nlk_random.h"

#include "nlk_dataset.h"


/**
 * Creates a dataset structure that holds a supervised dataset in the form of
 * an id -> class map
 *
 * @param size  the size (number of examples) of the dataset
 *
 * @return the dataset structure
 */
struct nlk_dataset_t *
nlk_dataset_create(size_t size)
{
    /* create structure */
    struct nlk_dataset_t *dset = NULL;
    dset = (struct nlk_dataset_t *) malloc(sizeof(struct nlk_dataset_t));
    nlk_assert_silent(dset != NULL);
    
    /* set size */
    dset->size = size;

    /* allocate memory for components */
    dset->ids = (size_t *) malloc(sizeof(size_t) * size);
    nlk_assert_silent(dset->ids != NULL);
    dset->classes = (unsigned int *) malloc(sizeof(size_t) * size);
    nlk_assert_silent(dset->classes != NULL);

    return dset;

error:
    NLK_ERROR_NULL("unable to allocate memory for dataset", NLK_ENOMEM);
    /* unreachable */
}

/**
 * Create a copy
 */
struct nlk_dataset_t *
nlk_dataset_create_copy(const struct nlk_dataset_t *dset)
{
    struct nlk_dataset_t *cp = nlk_dataset_create(dset->size);

    memcpy(cp->ids, dset->ids, sizeof(size_t) * dset->size);
    memcpy(cp->classes, dset->classes, sizeof(unsigned int) * dset->size);
    cp->n_classes = dset->n_classes;

    return cp;
}

/**
 * Creates a random split
 */
void
nlk_dataset_split_r(const struct nlk_dataset_t *dset, const float percentage, 
                    struct nlk_dataset_t **s1, struct nlk_dataset_t **s2)
{
    struct nlk_dataset_t *shuffled;
    size_t s1_size;
    size_t s2_size;
    
    /* determine size */
    if(percentage >= 1) {
        /* @TODO handle stupidity */
    }

    s1_size = dset->size * percentage;
    s2_size = dset->size - s1_size;

    /* create shuffled copy */
    shuffled = nlk_dataset_create_copy(dset);
    nlk_dataset_shuffle(shuffled);

    /* copy s1 */
    shuffled->size = s1_size;
    *s1 = nlk_dataset_create_copy(shuffled);
    shuffled->size = dset->size;

    /* copy s2 */
    shuffled->ids = shuffled->ids + s1_size;
    shuffled->classes = shuffled->classes + s1_size;
    shuffled->size = s2_size;
    *s2 = nlk_dataset_create_copy(shuffled);

    /* free */
    shuffled->ids = shuffled->ids - s1_size;
    shuffled->classes = shuffled->classes - s1_size;
    nlk_dataset_free(shuffled);

}

/**
 * Creates a split
 */
void
nlk_dataset_split(const struct nlk_dataset_t *dset, const float percentage, 
                  struct nlk_dataset_t **s1, struct nlk_dataset_t **s2)
{
    struct nlk_dataset_t *cp;
    size_t s1_size;
    size_t s2_size;
    
    /* determine size */
    if(percentage >= 1) {
        /* @TODO handle stupidity */
    }

    s1_size = dset->size * percentage;
    s2_size = dset->size - s1_size;

    /* create shuffled copy */
    cp = nlk_dataset_create_copy(dset);

    /* copy s1 */
    cp->size = s1_size;
    *s1 = nlk_dataset_create_copy(dset);
    cp->size = dset->size;

    /* copy s2 */
    cp->ids = cp->ids + s1_size;
    cp->classes = cp->classes + s1_size;
    cp->size = s2_size;
    *s2 = nlk_dataset_create_copy(cp);

    /* free */
    cp->ids = cp->ids - s1_size;
    cp->classes = cp->classes - s1_size;
    nlk_dataset_free(cp);
}

void
nlk_dataset_swap(struct nlk_dataset_t *s1, struct nlk_dataset_t *s2)
{
    struct nlk_dataset_t tmp;

    /* backup s1 */
    tmp.size = s1->size;
    tmp.n_classes = s1->n_classes;
    tmp.ids = s1->ids;
    tmp.classes = s1->classes;

    /* replace s1 */
    s1->size = s2->size;
    s1->n_classes = s2->n_classes;
    s1->ids = s2->ids;
    s1->classes = s2->classes;

    /* replace s2 */
    s2->size = tmp.size;
    s2->n_classes = tmp.n_classes;
    s2->ids = tmp.ids;
    s2->classes = tmp.classes;
}


/**
 * Free a dataset structure
 *
 * @param the dataset structure
 */
void
nlk_dataset_free(struct nlk_dataset_t *dset)
{
    if(dset != NULL) {
        if(dset->ids != NULL) {
            free(dset->ids);
            dset->ids = NULL;
        }
        if(dset->classes != NULL) {
            free(dset->classes);
            dset->classes = NULL;
        }
        free(dset);
        dset = NULL;
    }
}


/**
 * Reads a class map file: a file that maps an id (e.g. line number) to a 
 * class (number). The file should be line delimited:
 * id class \\n -> explanation, do not include header in the file
 * ...
 * 33 5
 * 66 2
 * ...
 *
 * Ids should start at 0 and end at number of lines - 1.
 * Order can be arbitrary.
 *
 * @note
 * User is responsible for calling free() on the resulting arrays
 * @endnote
 *
 * @param file      file pointer to the class file
 *
 * @return the dataset 
 *
 */
struct nlk_dataset_t *
nlk_dataset_load(FILE *file, const size_t n)
{
    struct nlk_dataset_t *dset;
    int ret = 0;            /* the return from fscanf */
    size_t counter = 0;     /* line counter */
    size_t index = 0;       /* the id read in the current line */
    int class = 0;          /* the class read in the current line */


    /* get the number of lines = number of values = class array size */
    if(n == 0) {
        NLK_ERROR_NULL("file is empty", NLK_EINVAL);
    }

    /* create dataset */
    dset = nlk_dataset_create(n);
    if(dset == NULL) {
        return NULL;
    }

 
    /* read file cycle */
    while(counter < n) {
        ret = fscanf(file, "%zu ", &index);
        if(ret <= 0) {
            NLK_ERROR_NULL("invalid file", NLK_EINVAL);
            /* unreachable */
        }
        dset->ids[counter] = index;
        ret = fscanf(file, "%d", &class);
        if(ret <= 0) {
            NLK_ERROR_NULL("invalid file", NLK_EINVAL);
            /* unreachable */
        }
        dset->classes[counter] = class;

        counter += 1;

        /* read newline */
        if(counter < n) { 
            ret = fscanf(file, "\n");
        }
    }

    /* determine the number of classes */
    dset->n_classes = nlk_count_unique(dset->classes, dset->size);

    return dset;
}


struct nlk_dataset_t *
nlk_dataset_load_path(const char *file_path)
{
    struct nlk_dataset_t *dset;
    

    const size_t n_lines = nlk_text_count_lines(file_path);

    /* open file */
    errno = 0;
    FILE *tfp = fopen(file_path, "r");
    if(tfp == NULL) {
        NLK_ERROR_ABORT(strerror(errno), errno);
        /* unreachable */
    }
    /* load */
    dset = nlk_dataset_load(tfp, n_lines);

    if(dset == NULL) {
        NLK_ERROR_ABORT("unable to read class file", NLK_EINVAL);
        /* unreachable */
    }
    fclose(tfp);
    tfp = NULL;

    return dset;
}


/**
 * Save a dataset to a file by passing just the ids & classes
 */
void
nlk_dataset_save_map(FILE *fp, const size_t *ids, 
                     const unsigned int *classes,
                     const size_t size)
{
    /* write to file cycle */
    for(size_t ii = 0; ii < size; ii++) {
        fprintf(fp, "%zu %u\n", ids[ii], classes[ii]);
    }
}


void
nlk_dataset_save_map_path(const char *filepath, const size_t *ids, 
                          const unsigned int *classes, const size_t size)
{
    /* open file */
    FILE *fp = fopen(filepath, "w");
    if(fp == NULL) {
        NLK_ERROR_VOID(strerror(errno), errno);
        /* unreachable */
    }

    /* write */
    nlk_dataset_save_map(fp, ids, classes, size);

    /* close */
    fclose(fp);
}


/**
 * Dataset Random Shuffle
 * 
 * @param dset  the dataset to random shuffle (modified)
 */
void
nlk_dataset_shuffle(struct nlk_dataset_t *dset)
{
    size_t idx = 0;
    size_t tmp_id = 0;
    unsigned int tmp_class = 0;

    for(size_t ii = 0; ii < dset->size; ii++) {
        /* store element at ii */
        tmp_id = dset->ids[ii];
        tmp_class = dset->classes[ii];

        /* get a random index */
        idx = nlk_random_xs1024() % dset->size;

        /*  replace element at ii with element at idx */
        dset->ids[ii] = dset->ids[idx];
        dset->classes[ii] = dset->classes[idx];

        /* replace element at idx with stored element */
        dset->ids[idx] = tmp_id;
        dset->classes[idx] = tmp_class;
    }
}

/**
 * Print dataset stats (examples per class
 */
int
nlk_dataset_print_class_dist(struct nlk_dataset_t *dset)
{
    size_t ii;
    size_t *examples_per_class;
    examples_per_class = (size_t *) calloc(dset->n_classes, sizeof(size_t));
    nlk_assert_silent(examples_per_class != NULL);

    /* count examples per class */
    for(ii = 0; ii < dset->size; ii++) {
        examples_per_class[dset->classes[ii]] += 1;
    }


    printf("class\texamples\n");
    for(ii = 0; ii < dset->n_classes; ii++) {
        printf("%zu\t%zu\n", ii, examples_per_class[ii]);
    }

    return 0;
error:
    NLK_ERROR("unable to allocate memory", NLK_ENOMEM);
    /* unreachable */
}

/**
 * Undersample a Dataset in order to balance it
 * @param dset the dataset to balance
 *
 * @return a new dataset that is balanced (must be freed)
 */
struct nlk_dataset_t *
nlk_dataset_undersample(struct nlk_dataset_t *dset, bool verbose)
{
    struct nlk_dataset_t *balanced;
    size_t *examples_per_class;
    size_t ii;
    size_t min_examples = 0;
    unsigned int min_class = 0;
    size_t new_size = 0;

    /**
     * @section Determine the smallest class 
     */
    examples_per_class = (size_t *) calloc(dset->n_classes, sizeof(size_t));
    nlk_assert_silent(examples_per_class != NULL);

    /* count examples per class */
    for(ii = 0; ii < dset->size; ii++) {
        examples_per_class[dset->classes[ii]] += 1;
    }

    /* determine the minimum count */
    min_examples = dset->size;
    for(ii = 0; ii < dset->n_classes; ii++) {
        if(examples_per_class[ii] < min_examples) {
            min_examples = examples_per_class[ii];
            min_class = ii;
        }
    }

    /**
     * @section Create balanced (undersampled) dataset
     */
    /* allocate memory for new dataset struct */
    new_size = min_examples * dset->n_classes;
    balanced = nlk_dataset_create(new_size);
    nlk_assert_silent(balanced != NULL);

    balanced->n_classes = dset->n_classes;
    balanced->size = 0; /* we'll use this to break the cycle */

    /* re-use examples_per_class to keep track of count of examples added */
    memset(examples_per_class, 0, dset->n_classes * sizeof(size_t));

    /* copy data to new balanced dataset */
    for(ii = 0; ii < dset->size; ii++) {
        if(examples_per_class[dset->classes[ii]] < min_examples) {
            examples_per_class[dset->classes[ii]] += 1;
            balanced->classes[balanced->size] = dset->classes[ii];
            balanced->ids[balanced->size] = dset->ids[ii];
            balanced->size = balanced->size + 1;
            if(balanced->size == new_size) {
                break;
            }
        }
    }

    if(verbose) {
        printf("Undersample: min class is %u with %zu examples (ntotal=%zu)\n",
                min_class, min_examples, balanced->size);
    }

    free(examples_per_class);

    return balanced;

error:
    NLK_ERROR_NULL("unable to allocate memory for undersampling", NLK_ENOMEM);
    /* unreachable */
}

/**
 * Calculate supervised classification accuracy
 * @param pred  array of predictions
 * @param truth array containing the ground truth
 * @param n     the number of test cases (size of pred/truth arrays)
 *
 * @return the accuracy
 */
float
nlk_class_score_accuracy(const unsigned int *pred, const unsigned int *truth, 
                         const size_t n)
{
    size_t correct = 0;
    float accuracy = 0;

    for(size_t ii = 0; ii < n; ii++) {
        if(pred[ii] == truth[ii]) {
            correct++;
        }
    }

    accuracy = correct;
    accuracy /= (double) n;

    return (float) accuracy;
}

/**
 * Calculates f1, precision, recall for a given class
 * where f1 = 2 * (precion + recall) / (precision * recall)
 *
 * @param pred        array of predictions
 * @param truth       array containing the ground truth
 * @param n           the number of test cases (size of pred/truth arrays)
 * @param class_val   the value of the class to calculate for
 * @param precion     the calculated precision (result)
 * @param recall      the calculated recall (result)
 *
 * @return the f1 score
 */
float
nlk_class_score_f1pr_class(const unsigned int *pred, 
                           const unsigned int *truth, 
                           const size_t n, const unsigned int class_val,
                           float *precision, float *recall)
{
    size_t truth_class = 0; /* # of class in truth array */
    size_t pred_class = 0;  /* # of class in pred array */
    size_t tp = 0;          /* # of true class in pred (true positives) */
    size_t ii;

    /** @section Calculate Precision, Recall, F1
     */
    for(ii = 0; ii < n; ii++) {
        if(truth[ii] == class_val) {
            truth_class++;
            if(truth[ii] == pred[ii]) {
                tp++;
                pred_class++;
                continue;
            }
        }
        if(pred[ii] == class_val) {
            pred_class++;
        }
    }

    *precision = tp / (double) pred_class;
    *recall = tp / (double) truth_class;

    return (2.0 * *precision * *recall) / (*precision + *recall);
}

/**
 * Calculate semeval sentiment f1 score
 * This is the SEMEVAL TASK 9 scoreing function. Described in 
 *
 *      SemEval-2014 Task 9: Sentiment Analysis in Twitter
 *      Rosenhal et al
 *      http://www.aclweb.org/anthology/S/S14/S14-2009.pdf
 *
 * @param pred  array of prediction
 * @param truth array containing the ground truth
 * @param n     the number of test cases (size of pred/truth arrays)
 * @param pos   the positive class
 * @param neg   the negative class
 *
 * @return the accuracy
 */
float
nlk_class_score_semeval_senti_f1(const unsigned int *pred, 
                                 const unsigned int *truth, 
                                 const size_t n, const unsigned int pos,
                                 const unsigned int neg)
{
    float precision_pos = 0;    /* precision of positive class */
    float recall_pos = 0;       /* recall of positive class */
    float f1_pos = 0;           /* f1 of the positive class */
    float precision_neg = 0;    /* precision of negative class */
    float recall_neg = 0;       /* recall of negative class */
    float f1_neg = 0;           /* f1 of the negative class */

    f1_pos = nlk_class_score_f1pr_class(pred, truth, n, pos, 
                                        &precision_pos, &recall_pos);
    f1_neg = nlk_class_score_f1pr_class(pred, truth, n, neg, 
                                        &precision_neg, &recall_neg);

    return (f1_pos + f1_neg) / 2.0;
}

/**
 * Create a Confusion Matrix
 */
static uint64_t **
nlk_cm_create(const size_t n_classes)
{
    uint64_t **cm;
    cm = (uint64_t **) calloc(n_classes, sizeof(uint64_t *));
    for(unsigned int ii = 0; ii < n_classes; ii++) {
        cm[ii] = (uint64_t *) calloc(n_classes, sizeof(uint64_t));
    }

    return cm;
}

/**
 * Free a Confusion Matrix
 */
static void
nlk_cm_free(uint64_t **cm, const unsigned int n_classes)
{
    for(unsigned int ii = 0; ii < n_classes; ii++) {
        free(cm[ii]);
        cm[ii] = NULL;
    }

    free(cm);
    cm = NULL;
}

/**
 * Create and Print a Confusion Matrix
 * Row = Truth
 * Col = Predicted
 */
void
nlk_class_score_cm_print(const unsigned int *pred, const unsigned int *truth, 
                         const size_t n)
{
    unsigned int n_classes = nlk_count_unique(truth, n);
    uint64_t **cm = nlk_cm_create(n_classes);
    size_t total = 0;
    size_t errors = 0;
    size_t total_fp = 0;
    size_t total_fn = 0;

    /* set values */
    for(unsigned int ii = 0; ii < n; ii++) {
        cm[truth[ii]][pred[ii]]++;
    }

    /* header */
    printf("\nT\\P:\t");
    for(unsigned int ii = 0; ii < n_classes; ii++) {
        printf("%u:\t", ii); /* col index */
    }
    printf("\t|Total:\tE(FN):"); /* col index */
    printf("\n");

    /* rows/cells */
    for(unsigned int ii = 0; ii < n_classes; ii++) {
        /* row ii */
        total = 0;
        errors = 0;
        printf("%u:\t", ii); /* row index */
        for(unsigned int jj = 0; jj < n_classes; jj++) {
            printf("%"PRIu64"\t", cm[ii][jj]); /* cell value */
            total += cm[ii][jj];
            if(ii != jj) {
                errors += cm[ii][jj];
            }
        }
        total_fn += errors;
        printf("\t|%"PRIu64"\t%"PRIu64"", total, errors); /* cell value */
        printf("\n");
    }

    /* Print error row */
    printf("-\n");
    printf("E(FP):\t");
    for(unsigned int jj = 0; jj < n_classes; jj++) {
        errors = 0;
        for(unsigned int ii = 0; ii < n_classes; ii++) {
            if(ii != jj) {
                errors += cm[ii][jj];
            }
        }
        total_fp += errors;
        printf("%"PRIu64"\t", errors); /* cell value */
    }
    printf("\t|%"PRIu64"\t\\%"PRIu64"", total_fp, total_fn);
    printf("\n");


    nlk_cm_free(cm, n_classes);
}


/**
 * Read a Supervised Corpus in CONLL format
 * The format consists space delimited columns and sentences separated by
 * empty lines:
 * s1_token_1 POS NER
 * s1_token_2 POS NER
 * s1_token_n POS NER
 *
 * s2_token_1 POS NER
 *
 * However either POS or NER can be absent
 *
 * @param corpus_path  the path file in conll format
 *
 * @return nlk_corpus_t
 */
void
nlk_dataset_load_conll()
{

}
