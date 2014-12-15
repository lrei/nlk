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


#define MAX_WORD_SIZE 128


nlk_real
nlk_w2v_learn_rate_update(nlk_real learn_rate, nlk_real start_learn_rate,
                          size_t epochs, size_t word_count_actual, 
                          size_t train_words)
{
    learn_rate = start_learn_rate * (1 - word_count_actual / 
                                     (nlk_real) (epochs * train_words + 1));

    if(learn_rate < start_learn_rate * 0.0001) {
        learn_rate = start_learn_rate * 0.0001;
    }

    return learn_rate;
}


void
skipgram(char *filepath, nlk_Vocab **vocab)
{
    size_t zz;
    size_t train_words = nlk_vocab_total(vocab); /* change this if train!=vocab) */
    size_t word_count = 0;
    size_t word_count_actual = 0;
    size_t last_word_count = 0;
    double speed;
    double progress;
    clock_t now;
    clock_t start;
    char display_str[256];
    nlk_real accuracy = 0;

    /*
     * text input related allocations/initializations
     */
    size_t examples;
    size_t line_len;
    size_t max_line_size = 1024;
    size_t max_word_size = MAX_WORD_SIZE;
    /* context window variables */
    size_t after = 8;                   /* max words after current word */
    size_t before = 8;                  /* max words before current word */
    size_t ctx_size = after + before;   /* max context size */

    /* allocate memory for reading from the input file */
    char **text_line = (char **) calloc(max_line_size, sizeof(char *));
    for(zz = 0; zz < max_line_size; zz++) {
        text_line[zz] = calloc(max_word_size, sizeof(char));
    }
    /* for converting to a vectorized representation of text */
    nlk_Vocab *vectorized[max_line_size];

    /* for converting a sentence to a series of training contexts */
    nlk_Context **contexts = (nlk_Context **) calloc(max_line_size,
                                                     sizeof(nlk_Context *));
    for(zz = 0; zz < max_line_size; zz++) {
        contexts[zz] = nlk_context_create(ctx_size);
    }
    /* open the input file */
    FILE *text_file = fopen(filepath, "rb");
    
    /* 
     * create / initialize neural net and associated variables 
     */
    size_t vocab_size = nlk_vocab_size(vocab);
    size_t max_code_length = vocab_max_code_length(vocab);

    size_t layer_size         = 200;        /* size of the hidden layer */
    nlk_real learn_rate_start = 0.025;      /* initial learning rate */
    nlk_real learn_rate       = 0.025;      /* learning rate */
    int epochs = 1;                         /* total epochs */
    int epoch_current = 0;                  /* current epoch */

    nlk_set_seed(6121984);
    /* the first layer: vanilla loopup table - word vectors*/
    nlk_Layer_Lookup *lk1 = nlk_layer_lookup_create(vocab_size, layer_size, 1);
    nlk_layer_lookup_init(lk1);
    nlk_Array *lk1_out = nlk_array_create(1, layer_size);
    /* concat "layer" */
    nlk_Array *cc_out = nlk_array_create(layer_size, 1);
    /* the second layer, hierarchical softmax - hs point vectors */
    nlk_Layer_Lookup *lk2 = nlk_layer_lookup_create(vocab_size, layer_size, 1);

    nlk_Array *lk2_grad = nlk_array_create(1, layer_size);
    nlk_Array *grad_acc = nlk_array_create(1, layer_size);
    nlk_Array *lk2_temp = nlk_array_create(layer_size, 1);
    /* sigmoid table */
    nlk_Table *sigmoid_table = nlk_table_sigmoid_create();



    /*
     * thread specific variables
     */
    /*int num_cpus = omp_get_num_procs();   */

    /* 
     * train
     */
    nlk_tic_reset();
    nlk_tic("starting", true);

    while(epoch_current < epochs) {
        epoch_current += 1;
        word_count_actual += word_count - last_word_count;
        word_count = 0;
        last_word_count = 0;
        rewind(text_file);

        /* one epoch */
        while(!feof(text_file)) {

            /* update learning rate */
            if (word_count - last_word_count > 10000) {
                word_count_actual += word_count - last_word_count;
                last_word_count = word_count;

                now = clock();

                progress =  word_count_actual / 
                            (double)(epochs * train_words + 1) * 100;
                speed = word_count_actual / ((double)(now - start + 1) / 
                        (double)CLOCKS_PER_SEC * 1000),
                snprintf(display_str, 256,
                        "Alpha: %f  Progress: %.2f%%  Words/sec: %.2fk "
                        "Total Words: %zu Epoch = %d accuracy = %0.2f ", 
                        learn_rate, progress, speed, word_count_actual, 
                        epoch_current, accuracy);
                nlk_tic(display_str, false);

                learn_rate = nlk_w2v_learn_rate_update(learn_rate,
                                                       learn_rate_start,
                                                       epochs,
                                                       word_count_actual, 
                                                       train_words);
            }

            /* read line */
            nlk_read_line(text_file, text_line, max_word_size, max_line_size);
            
            /* vocabularize */
            line_len = nlk_vocab_vocabularize(vocab, text_line, NULL, false, 
                                              vectorized); 

            /* window */
            examples = nlk_context_window(vectorized, line_len, false, before, 
                                          after, true, contexts);
            word_count += examples;

            /* loop through each context word 
             * each example is one word + the context surrounding it
             */
/*#pragma omp parallel for*/
            for(zz = 0; zz < examples; zz++) {
                /* loop variables */
                nlk_real lk2_out;
                nlk_real out;
                nlk_real grad_out;
                nlk_Context *context = contexts[zz];
                nlk_Vocab *center_word = context->word;
                nlk_Vocab *word;
                size_t point;
                uint8_t code;
                size_t ii;
                size_t jj;
                size_t pp;


                /* for each context word jj */
                for(jj = 0; jj < context->size; jj++) {
                    word = context->context_words[jj];
                    nlk_array_zero(grad_acc);

                    /* 
                     * forward 
                     */
                    nlk_layer_lookup_forward_lookup(lk1, &word->index, 1,
                                                    lk1_out);

                    /*  [1, layer_size] -> [layer_size, 1] */
                    nlk_transfer_concat_forward(lk1_out, cc_out);

                    
                    /* hierarchical softmax: for each point of center word */
                    for(pp = 0; pp < center_word->code_length; pp++) {
                        point = center_word->point[pp];
                        code = center_word->code[pp];

                        /* forward with lookup for point pp */
                        nlk_layer_lookup_forward(lk2, cc_out, point, &lk2_out);
                        
                        /* ignore points with outputs outside of sigm bounds */
                        if(lk2_out >= sigmoid_table->max) {
                            continue;
                        } else if(lk2_out <= sigmoid_table->min) {
                            continue;
                        }
                        /**/
                        out =  nlk_sigmoid_table(sigmoid_table, lk2_out);
                        //out =  nlk_sigmoid_table(NULL, lk2_out);

                        /*
                         * backprop
                         * Conveniently, using the negative log likelihood,
                         * the gradient simplifies to the same formulation/code
                         * as the binary neg likelihood:
                         *
                         * log(sigma(z=v'n(w,j))'vwi) = 
                         * = (1 - code) * z - log(1 + e^z)
                         * d/dx = 1 - code  - sigmoid(z)
                         */

                        nlk_bin_nl_sgradient(out, code, &grad_out);

                        /* multiply by learning rate */
                        grad_out *= learn_rate;
                        
                        /* layer2 backprop */
                        nlk_layer_lookup_backprop_acc(lk2, cc_out, point,
                                                      grad_out, lk2_grad, 
                                                      grad_acc, lk2_temp);

                    } /* end of points/codes */
                    /* learn layer1 weights */
                    nlk_layer_lookup_backprop_lookup(lk1, word->index,
                                                     grad_acc);

               } /* end of context words for center_word */
            } /* contexts (contexts for line) */
        } /* end of file (lines) */

        nlk_tic_reset();
        /* save vectors */
        nlk_layer_lookup_save("vectors.bin", true, vocab, lk1);

        nlk_tic("evaluating", true);
        /*
        nlk_eval_on_questions("questions.txt", vocab, lk1->weights, 1000, 
                              false, true, &accuracy);
        */
        nlk_tic("done", true);
        printf("accuracy = %f\n", accuracy);
        nlk_tic_reset();
    } /* end of all epochs */
}



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
    vocab = nlk_vocab_create(args.filename, MAX_WORD_SIZE);
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
    skipgram(args.filename, &vocab);

    /*printf("\nii=%zu\n", ii);*/
    
    return 0;
}
