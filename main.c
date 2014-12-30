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
    size_t vocab_size = 0;
    size_t vocab_total = 0;
    size_t ii = 0;
    size_t jj = 0;
    size_t kk = 0;

    /*
     * Parse command line options
     */
    int c;
    static const char *opt_string = "f:vp:";
    struct args_t {
        char        *filename;      /* -f option */
        char        *parafile;      /* -p option */
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
            case 'p':
                args.parafile = optarg;
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

    /**
    nlk_Vocab *vocab = nlk_vocab_create(args.filename, NLK_LM_MAX_WORD_SIZE, 
                             0, true);
    nlk_tic("created", true);
    /**/
   /**/
    nlk_Vocab *vocab = vocab = nlk_vocab_load("vocab.txt", NLK_LM_MAX_WORD_SIZE);
    printf("LOADED\n");
    /**/

    vocab_size = nlk_vocab_size(&vocab);
    size_t vocab_words = nlk_vocab_words_size(&vocab);
    vocab_total = nlk_vocab_total(&vocab);

    nlk_tic("count unique items", true);
    printf("items: %zu, words: %zu total count: %zu\n", vocab_size, vocab_words, vocab_total);
    
    nlk_vocab_reduce(&vocab, 40);
    vocab_size = nlk_vocab_size(&vocab);
    vocab_words = nlk_vocab_words_size(&vocab);
    vocab_total = nlk_vocab_total(&vocab);
    
    nlk_tic("reduced", true);
    printf("items: %zu, words: %zu total count: %zu\n", vocab_size, vocab_words, vocab_total);
    /**/
     
    nlk_vocab_sort(&vocab);
    nlk_tic("sorted", true);

    /**/
    nlk_vocab_save("vocab.txt", &vocab);
    nlk_tic("saved", true);
    /**/
    

    nlk_tic(NULL, false);
    nlk_vocab_encode_huffman(&vocab);
    nlk_tic("huffman", true);
    /*nlk_vocab_save_full("vocab.full.txt", &vocab);*/
    printf("\n");

    size_t before = 12;
    size_t after = 12;
    float sample_rate = 10e-5;
    size_t layer_size = 600;
    nlk_real learn_rate = 0.025; learn_rate = 0.05;
    int epochs = 1;
    int verbose = 1;
    bool learn_par = false;
    nlk_real accuracy;
    nlk_real tol = 0.01;
    nlk_real ret = 1;
    size_t iter = 0;
    bool lower_words = true;
    bool freeze = false;
    
    /* first pass */
    nlk_word2vec(NLK_SKIPGRAM, NULL, NULL, learn_par, freeze, args.filename, &vocab, 
                 before, after, sample_rate, layer_size, 
                 learn_rate, epochs, verbose, 
                 "vectors.1.nlk.bin", "vectors.2.nlk.bin", NLK_FILE_BIN);

    nlk_Layer_Lookup *lk1 = nlk_layer_lookup_load("vectors.1.nlk.bin");
    nlk_tic_reset();
    nlk_tic("evaluating", true);
    nlk_eval_on_questions("questions.txt", &vocab, lk1->weights, 30000, 
                          lower_words, &accuracy);
    printf("accuracy = %f%%\n", accuracy * 100);
    nlk_tic("tested", true);

    /* pass over test */
    freeze = true;
    learn_rate = 0.01;
    learn_par = true;
    /* extend the vocabulary with paragraphs */
    size_t max_iter = 1000;

    nlk_vocab_extend(&vocab, args.parafile, NLK_LM_MAX_WORD_SIZE,
                     NLK_LM_MAX_LINE_SIZE, lower_words);

    /* increase size of layers */
    size_t new_table_size = nlk_vocab_size(&vocab);
    nlk_layer_lookup_resize(lk1, new_table_size);

    nlk_Layer_Lookup *lk2 = nlk_layer_lookup_load("vectors.2.nlk.bin");

    nlk_tic_reset();
    nlk_tic("Learning PVs", true);

    while(ret > tol && iter < max_iter) {
        ret = nlk_word2vec(NLK_CBOW, lk1, lk2, learn_par, freeze, 
                          args.parafile, &vocab, before, after, 0, layer_size, 
                           learn_rate, 10, verbose, NULL, NULL, NLK_FILE_BIN);
        iter++;
    }
    nlk_tic("learned PVs", true);
    printf("ret = %f\n", ret);

    nlk_tic_reset();
    nlk_tic("eval PVs", true);
    nlk_eval_on_paraphrases(args.parafile, &vocab, lk1->weights,
                            lower_words, &accuracy);
    

    /*printf("\nii=%zu\n", ii);*/
    
    return 0;
}
