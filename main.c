#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_text.h"
#include "nlk_window.h"
#include "nlk_layer_linear.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_tic.h"
#include "nlk_eval.h"

#include "nlk_w2v.h"

int main(int argc, char **argv)
{
    nlk_Vocab *vocab;
    size_t vocab_size = 0;
    size_t vocab_total = 0;
    size_t ii = 0;
    size_t jj = 0;
    size_t kk = 0;
    nlk_Vocab *vi = vocab;

    /*
     * Parse command line options
     */
    int c;
    static const char *opt_string = "f:v";
    struct args_t {
        char        *filename;      /* -f option */
        int          verbosity;     /* -v option */
    } args;

    /* set the defaults */
    args.verbosity = 1;
    args.filename = NULL;

    while((c = getopt (argc, argv, opt_string)) != -1) {
        switch (c) {
            case 'f':
                args.filename = optarg;
                break;
            case '?':
                /* display usage */
                if (optopt == 'f') {
                  fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                } else {
                  fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                }
                break;
              default:
                /* unreachable */
                abort();
        }
    }
    if(args.filename == NULL) {
        fprintf(stderr, "Missing required input file.\n");
        return 0;
    }

    /*
     * Main
     */
    nlk_tic(NULL, false);
    vocab = nlk_vocab_create(args.filename, NLK_LM_MAX_WORD_SIZE );
    nlk_tic("create", true);

    vocab_size = nlk_vocab_size(&vocab);

    nlk_tic("count size", true);
    printf("\n%zu\n", vocab_size);

    
    nlk_vocab_reduce(&vocab, 5);
    vocab_size = nlk_vocab_size(&vocab);
    
    nlk_tic("reduce-count", true);
    printf("\n%zu\n", vocab_size);
    
    

    vocab_total = nlk_vocab_total(&vocab);
    printf("\n%zu\n", vocab_total);
 
    nlk_vocab_sort(&vocab);
    nlk_tic("sort", true);
    ii = 0;

    nlk_tic(NULL, false);
    nlk_vocab_encode_huffman(&vocab);
    nlk_tic("huffman", true);

    /*
    size_t maxpoint = 0;
    for(vi = vocab; vi != NULL; vi=vi->hh.next) {
        for(jj = 0; jj < vi->code_length; jj++) {
            if(vi->point[jj] > maxpoint) maxpoint = vi->point[jj];
        }
        if(ii < vocab_size - 20) { 
            ii++; 
            continue; 
        }
        printf("id=%zu s='%s': ", vi->id, vi->word);
        for(jj = 0; jj < vi->code_length; jj++)
            printf("%zu ", vi->point[jj]);
        printf("\n");
        ii++;
    }
    printf("maxpoint = %zu\n", maxpoint);
    exit(0);
    */
    

    printf("\n");

    //nlk_vocab_save_full("vocab.txt", &vocab);
    size_t before = 8;
    size_t after = 8;
    float sample_rate = 10e-3;
    size_t layer_size = 200;
    nlk_real learn_rate = 0.025; learn_rate = 0.05;
    int epochs = 1;
    int num_threads = 0;
    int verbose = 1;
    
    /* *
    nlk_word2vec(NLK_SKIPGRAM, args.filename, &vocab, before, after, sample_rate, layer_size, 
                 learn_rate, epochs, num_threads, verbose, "vectors.nlk.bin", NLK_FILE_BIN);
    /**/

    nlk_tic_reset();
    nlk_Layer_Lookup *lk1 = nlk_layer_lookup_load("vectors.nlk.bin");
    float accuracy;
    nlk_tic("evaluating", true);
    nlk_eval_on_questions("questions.txt", &vocab, lk1->weights, 30000, 
                          true, &accuracy);
    printf("accuracy = %f%%\n", accuracy * 100);
    nlk_tic("tested", true);


    /*printf("\nii=%zu\n", ii);*/
    
    return 0;
}
