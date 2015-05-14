#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <locale.h>
#include <getopt.h>

#include "nlk_err.h"
#include "nlk_array.h"
#include "nlk_vocabulary.h"
#include "nlk_text.h"
#include "nlk_window.h"
#include "nlk_neuralnet.h"
#include "nlk_layer_linear.h"
#include "nlk_transfer.h"
#include "nlk_criterion.h"
#include "nlk_tic.h"
#include "nlk_eval.h"
#include "nlk_w2v.h"
#include "nlk_pv.h"

#include "nlk_random.h"


/* program info */
#define PROGRAM_NAME "nlktool"
#define PROGRAM_FULLNAME "(N)eural (L)anguage (K)it Tool"
#define PROGRAM_VERSION "0.0.1"
#define AUTHOR "Luis Rei\n<me@luisrei.com>\nhttp://luisrei.com"
#define URL "http://github.com/lrei/nlk"
#define LICENSE "MIT"


/* command line options */
enum cmd_opts_t { 
    /* train/nn options */
    CMD_OPTS_MODEL = 1,     /**< specify language model */
    CMD_OPTS_TRAIN,         /**< train from file */
    CMD_OPTS_EPOCHS,        /**< number of epochs */
    CMD_OPTS_LEARN_RATE,    /**< starting learning rate */
    CMD_OPTS_NEGATIVE,      /**< number of negative examples to use */
    CMD_OPTS_VECTOR_SIZE,   /**< word/pv size */
    CMD_OPTS_WINDOW,        /**< context window  size = [window, window]*/
    CMD_OPTS_SAMPLE,        /**< undersample rate for words */
    /* vocabulary */
    CMD_OPTS_SAVE_VOCAB,    /**< save vocab to file */
    CMD_OPTS_LOAD_VOCAB,    /**< load vocab from file */
    CMD_OPTS_MIN_COUNT,     /**< minimum word frequency */
    /* serialization/output */
    CMD_OPTS_SAVE_NET,      /**< save neural net to file */
    CMD_OPTS_LOAD_NET,      /**< load neural net from file */
    CMD_OPTS_OUT_WORDS,     /**< save/export word vectors */
    CMD_OPTS_OUT_PVS,       /**< save/export PVs */
    CMD_OPTS_OUT_LIMIT,     /**< limit output */
    CMD_OPTS_OUT_FORMAT,    /**< output format */
    CMD_OPTS_PREFIX_PVS,    /**< prefix paragraph ids with string in export */
    /* PV generation (inference/test) */
    CMD_OPTS_GEN_PVS,       /**< generate paragraph vectors for file */
    CMD_OPTS_GEN_ITER,      /**< number of iterations for PV generation */
    CMD_OPTS_GEN_SAVE,      /**< save generated PVs according to FORMAT */
    /* evaluation */
    CMD_OPTS_EVAL_QUESTIONS,    /**< eval model on question answering */
    CMD_OPTS_EVAL_PARAPHRASES,  /**< eval model on paraphrase corpus */
    CMD_OPTS_EVAL_PVS,          /**< eval pre-generated PVs as paraphrases */
    CMD_OPTS_EVAL_LIMIT,        /**< limit evaluation to first n elements */
};


static void print_version()
{
    printf("%s - v %s\n"
           "%s\n\n"
           "Copyright (C) 2014-2015\n%s\n"
           "License: %s\n\n",
           PROGRAM_FULLNAME, PROGRAM_VERSION, URL, AUTHOR, LICENSE);
}

static void 
print_usage()
{
    printf ("Usage:\n%s --help\n\n", PROGRAM_NAME);
}
static void print_help()
{
    print_version();
    fputs("\
Neural Network Language Model Tool.\n\
\n\
Model/Training:\n\
  --model [STRING]      language model: CBOW, SG, PVDM, PVDBOW\n\
  --concat              use concatenate variant (valid if --model PVDM)\n\
  --train [FILE]        train model with this (text) file\n\
  --numbered            the training file consists of line-delimed id text\n\
  --iter [INT]          number of train epochs\n\
  --alpha [FLOAT]       the initial learning rate\n\
  --hs                  use hierarchical softmax\n\
  --negative [INT]      the number of negative sampling examples\n\
  --size [INT]          the size of word/paragraph vectors\n\
  --window [INT]        the size of the context window\n\
  --sample [FLOAT]      the word undersampling rate\n\
\n\
Vocabulary:\n\
  --min-count  [INT]    minimum word count\n\
  --save-vocab [FILE]   save the vocabulary to file\n\
  --load-vocab [FILE]   load the vocabulary from file\n\
\n\
Serialization/Export:\n\
  --save-net [FILE]     save the neural network\n\
  --load-net [FILE]     load an existing neural network\n\
  --keep-pvs            keep paragraph vectors in saved network\n\
  --output [FILE]       output word+paragraph vectors\n\
  --output-words [FILE] output word vectors\n\
  --output-pvs [FILE]   output paragraph vectors\n\
  --output-limit [INT]  limit output of vectors to the first [INT]\n\
  --par-prefix [STR]    prefix paragraph ids when exporting\n\
\n\
Paragraph Vector Inference:\n\
  --gen-pvs [FILE]      generate paragraph vectors for this file\n\
  --gen-iter [FILE]     number of iterations to generate paragraph vectors\n\
  --gen-output [FILE]   output generated paragraph vectors\n\
\n\
Evaluation:\n\
  --questions [FILE]    evaluate on question corpus\n\
  --paraphrases [FILE]  evaluate on question corpus\n\
  --eval-pvs [FILE]     evaluate similarity of pre-generated PVs\n\
  --eval-limit [INT]    limit evaluation to the first vectors\n\
\n\
Miscellaneous:\n\
  --format [STR]        format for the output options: w2vtxt, w2vbin, nlk,\n\
                        nlktext (for PVs only)\n\
  --verbose             print status related messages during execution\n\
  --help                print this message and quit\n\
  --version             print program version information and quit\n\
\n\n\
", stdout);

    printf("Example:\n");
}


int main(int argc, char **argv)
{
    /** @section Variable Declaration and Initialization
     */

    /** @subsection Vocabulary Options
     */
    struct nlk_vocab_t *vocab;          /* the vocabulary structure */
    char *vocab_load_file       = NULL; /* load vocab from this file */
    char *vocab_save_file       = NULL; /* save vocab to this file */
    size_t vocab_size           = 0;    /* size of the vocabulary */
    size_t vocab_total          = 0;    /* sum of all word counts in vocab */
    size_t total_lines          = 0;    /* total lines in train file */

    /** @subsection Model Options
     */
    NLK_LM lm_type              = NLK_MODEL_NULL; /* the model type */
    char *model_name            = NULL; /* model type (string) - name (opt) */
    static int concat           = 0;    /* concatenate (e.g. PVDM_CONCAT) */

    /** @subsection Training Options
     */
    char *train_file            = NULL; /* train model on this file */
    static int numbered         = 0;    /* train lines are numbered (id) */
    static int learn_par        = 0;    /* learn paragraph vectors */
    static int hs               = 0;    /* use hierarchical softmax */
    size_t vector_size          = 100;  /* word vector size */    
    int window                  = 8;    /* window, words before and after */    
    int negative                = 0;    /* number of negative examples */
    int epochs                  = 5;    /* number of iterations/epochs */
    int min_count               = 0;    /* minimum word frequency (count) */
    nlk_real learn_rate         = 0;    /* learning rate (start) */
    float sample_rate           = 1e-3; /* random undersample of freq words */

    /** @subsection Serialization
     */
    static int keep_pvs        = 0;    /* save NN with the sentence vectors */
    char *nn_load_file          = NULL; /* load neuralnet from this file */
    char *nn_save_file          = NULL; /* save neuralnet to this file */
    char *output_words_file     = NULL; /* export word vectors to this file */
    char *output_pvs_file       = NULL; /* save PVs to this file */
    size_t output_limit         = 0;    /* max vectors to output */
    char *par_prefix            = "";   /* prefix for paragraph ids */
    char *format_name           = NULL; /* format option as a string */
    NLK_FILE_FORMAT format      = NLK_FILE_BIN;
    
    /** @subsection Paragraph Vector Generation (Inferance/Test) Variables
     */
    char *gen_paragraphs_file   = NULL; /* Line delimited paragraphs file */
    char *pvs_save_file         = NULL; /* save generated PVs to this file */
    unsigned int gen_iter       = 100;  /* PV inference iterations (steps) */

    /** @subsection Evaluation
     */
    char *questions_file        = NULL; /* word2vec word-analogy tests */
    char *paraphrases_file      = NULL; /* MSR paraphrases style file */
    char *pvs_input_file        = NULL; /* read PVs from this file (eval) */
    size_t eval_limit           = 0;    /* limit eval to first n elements */

    /** @subsection Miscellaneous
     */
    NLK_LAYER_LOOKUP *word_table = NULL;
    NLK_LAYER_LOOKUP *paragraph_table = NULL;
    static int show_help        = 0;    /* show help */
    static int show_version     = 0;    /* show version information */
    static int verbose          = 0;    /* print status during execution */
    int c                       = 0;    /* used by getop */
    int option_index            = 0;    /* getopt option index */



    /** @section Define and Parse Command Line Options Defintions
     */
    while(1) {
        static struct option long_options[] = {
            /* These options set a flag. */
            {"numbered",        no_argument,       &numbered,       1  },
            {"concat",          no_argument,       &concat,         1  },
            {"hs",              no_argument,       &hs,             1  },
            {"keep-pvs",        no_argument,       &keep_pvs,       1  },
            {"help",            no_argument,       &show_help,      1  },
            {"version",         no_argument,       &show_version,   1  },
            {"verbose",         no_argument,       &verbose,        1  },
            /* 
             * These options don’t set a flag 
             */
            /* train/nn/context options */
            {"model",           required_argument, 0, CMD_OPTS_MODEL         },
            {"train",           required_argument, 0, CMD_OPTS_TRAIN         },
            {"iter",            required_argument, 0, CMD_OPTS_EPOCHS        },
            {"alpha",           required_argument, 0, CMD_OPTS_LEARN_RATE    },
            {"negative",        required_argument, 0, CMD_OPTS_NEGATIVE      },
            {"size",            required_argument, 0, CMD_OPTS_VECTOR_SIZE   },
            {"window",          required_argument, 0, CMD_OPTS_WINDOW        },
            {"sample",          required_argument, 0, CMD_OPTS_SAMPLE        },
            /* vocabulary */
            {"save-vocab",      required_argument, 0, CMD_OPTS_SAVE_VOCAB    },
            {"load-vocab",      required_argument, 0, CMD_OPTS_LOAD_VOCAB    },
            {"min-count",       required_argument, 0, CMD_OPTS_MIN_COUNT     },
            /* serialization */
            {"save-net",        required_argument, 0, CMD_OPTS_SAVE_NET      },
            {"load-net",        required_argument, 0, CMD_OPTS_LOAD_NET      },
            {"output-words",    required_argument, 0, CMD_OPTS_OUT_WORDS     },
            {"output-pvs",      required_argument, 0, CMD_OPTS_OUT_PVS       },
            {"output-limit",    required_argument, 0, CMD_OPTS_OUT_LIMIT     },
            {"format",          required_argument, 0, CMD_OPTS_OUT_FORMAT    },
            {"par-prefix",      required_argument, 0, CMD_OPTS_PREFIX_PVS    },
            /* PV generation (inferance/test) */
            {"gen-pvs",         required_argument, 0, CMD_OPTS_GEN_PVS       },
            {"gen-iter",        required_argument, 0, CMD_OPTS_GEN_ITER      },
            {"gen-output",      required_argument, 0, CMD_OPTS_GEN_SAVE      },
            /* evaluation */
            {"questions",  required_argument, 0, CMD_OPTS_EVAL_QUESTIONS     },
            {"paraphrases",required_argument, 0, CMD_OPTS_EVAL_PARAPHRASES   },
            {"eval-pvs",   required_argument, 0, CMD_OPTS_EVAL_PVS           },
            {"eval-limit", required_argument, 0, CMD_OPTS_EVAL_LIMIT         },
            /* global */
            {0,                     0,                 0,                   0}
        };
        
        c = getopt_long_only(argc, argv, "", long_options, &option_index);

        /* Detect the end of the options. */
        if(c == -1) {
            break;
        }

        switch(c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if(long_options[option_index].flag != 0) {
                    break;
                }
                break;
            /* train/nn options */
            case CMD_OPTS_MODEL:
                model_name = optarg;
                break;
            case CMD_OPTS_TRAIN:
                train_file = optarg;
                break;
            case CMD_OPTS_EPOCHS:
                epochs = atoi(optarg);
                break;
            case CMD_OPTS_LEARN_RATE:
                learn_rate = atof(optarg);
                break;
            case CMD_OPTS_NEGATIVE:
                negative = atoi(optarg);
                break;
            case CMD_OPTS_VECTOR_SIZE:
                vector_size = atoi(optarg);
                break;
            case CMD_OPTS_WINDOW:
                window = atoi(optarg);
                break;
            case CMD_OPTS_SAMPLE:
                sample_rate = atof(optarg);
                break;
            /* vocabulary */
            case CMD_OPTS_SAVE_VOCAB:
                vocab_save_file = optarg;
                break;
            case CMD_OPTS_LOAD_VOCAB:
                vocab_load_file = optarg;
                break;
            case CMD_OPTS_MIN_COUNT:
                min_count = atoi(optarg);
                break;
            /* serialization/output */
            case CMD_OPTS_SAVE_NET:
                nn_save_file = optarg;
                break;
            case CMD_OPTS_LOAD_NET:
                nn_load_file = optarg;
                break;
            case CMD_OPTS_OUT_WORDS:
                output_words_file = optarg;
                break;
            case CMD_OPTS_OUT_PVS:
                output_pvs_file = optarg; /* save PVs to this file */
                break;
            case CMD_OPTS_OUT_LIMIT:
                output_limit = atoll(optarg);
                break;
            case CMD_OPTS_OUT_FORMAT:
                format_name = optarg;
                break;
            case CMD_OPTS_PREFIX_PVS:
                par_prefix = optarg;
                break;
            /* paragraph vector inference (generate/test) */
            case CMD_OPTS_GEN_PVS:
                gen_paragraphs_file = optarg;
                break;
            case CMD_OPTS_GEN_ITER:
                gen_iter = (unsigned int) atoi(optarg);
                break;
            case CMD_OPTS_GEN_SAVE:
                pvs_save_file = optarg;
                break;
            /* evaluation */
            case CMD_OPTS_EVAL_QUESTIONS:
                questions_file = optarg;
                break;
            case CMD_OPTS_EVAL_PARAPHRASES:
                paraphrases_file = optarg;
                break;
            case CMD_OPTS_EVAL_PVS:
                pvs_input_file  = optarg;
                break;
            case CMD_OPTS_EVAL_LIMIT:
                eval_limit  = atoi(optarg);
                break;
            /* miscellaneous */
            default:
                print_usage();
                exit(1);
        }
    }

    /* check for no options */
    if(optind == 0) {
        print_usage();
        exit(1);
    }

    /* show help and quit */
    if(show_help) {
        print_help();
        exit(0);
    }

    /* show version and quit */
    if(show_help) {
        print_version();
        exit(0);
    }


    /** @subsection Process some options */
#ifndef NCHECKS
    if(verbose) {
        printf("CHECKS enabled!\n");
    }
#endif
#ifdef CHECK_NANS
    if(verbose) {
        printf("CHECK_NANS enabled\n");
    }
#endif
#ifdef DEBUG
    if(verbose) {
        printf("Running in DEBUG mode!\n");
        printf("MAX WORD SIZE = %d chars\n", NLK_LM_MAX_WORD_SIZE);
        printf("MAX LINE SIZE = %d words\n", NLK_LM_MAX_LINE_SIZE);
    }
#endif


    /** @section Global Init
     * @TODO move this to a init() function
     * @TODO make neg_table and exp_table globals
     */
    nlk_random_init_xs1024(nlk_random_seed());


    /** @section Model Type 
     * Model Type and associated options (e.g. default learn rate)
     */
    char *model_type = "NULL";
    if(model_name == NULL) {
        /* do nothing */
    }
    else if(strcasecmp(model_name, "cbow") == 0) { 
        lm_type = NLK_CBOW;
        model_type = "CBOW";
        learn_par = false;
        if(learn_rate == 0) {
            learn_rate = 0.025; 
        }
 
    } else if(strcasecmp(model_name, "sg") == 0) {
        lm_type = NLK_SKIPGRAM;
        model_type = "SKIPGRAM";
        learn_par = false;
        if(learn_rate == 0) {
            learn_rate = 0.05; 
        }
    } else if(strcasecmp(model_name, "pvdm") == 0) {
        if(concat) {
            model_type = "PVDM_CONCAT";
            lm_type = NLK_PVDM_CONCAT;
        } else {
            lm_type = NLK_PVDM;
            model_type = "PVDM";
        }
        learn_par = true;
        if(learn_rate == 0) {
            learn_rate = 0.05; 
        }
    } else if(strcasecmp(model_name, "pvdbow") == 0) {
        lm_type = NLK_PVDBOW;
        model_type = "PVDBOW";
        learn_par = true;
        if(learn_rate == 0) {
            learn_rate = 0.025; 
        }
    } else {
        print_usage();
        NLK_ERROR_ABORT("Invalid model type.", NLK_EINVAL);
    }

    /** @section Output File Format
     */
   if(format_name == NULL) {
        /* do nothing defaults to NLK */
    }
    else if(strcasecmp(format_name, "w2vtxt") == 0) { 
        format = NLK_FILE_W2V_TXT;
    } else if(strcasecmp(format_name, "w2vbin") == 0) {
        format = NLK_FILE_W2V_BIN;
    } else if(strcasecmp(format_name, "nlk") == 0) {
        format = NLK_FILE_BIN;
    } else if(strcasecmp(format_name, "nlktxt") == 0) {
        format = NLK_FILE_TXT;
    } else {
        print_usage();
        NLK_ERROR_ABORT("Invalid format type.", NLK_EINVAL);
    }

    /** @section Vocabulary
     */
    bool vocab_changes = false;

    if(vocab_load_file == NULL && train_file != NULL) {
        if(verbose) {
            nlk_tic(NULL, false);
        }
        vocab = nlk_vocab_create(train_file, numbered, NLK_LM_MAX_WORD_SIZE, 
                                verbose);
        nlk_vocab_sort(&vocab);
        nlk_vocab_encode_huffman(&vocab);

        vocab_changes = true;
        nlk_tic_reset();

        if(verbose) { /* since tic did not print newlines */
            printf("\n");
        }
    } else if(vocab_load_file != NULL) {
        vocab = nlk_vocab_load(vocab_load_file, NLK_LM_MAX_WORD_SIZE);
        nlk_vocab_sort(&vocab);
        nlk_vocab_encode_huffman(&vocab);
    } else {
        vocab = NULL;
        vocab_changes = false;
    }

    if(vocab != NULL) {
        vocab_size = nlk_vocab_size(&vocab);
        vocab_total = nlk_vocab_total(&vocab);

        if(verbose) {
            printf("Vocabulary words: %zu (total count: %zu)\n", 
                    vocab_size, vocab_total);
        }
    
        if(min_count > 0) {
            vocab_changes = true;
            nlk_vocab_reduce(&vocab, min_count);
            vocab_size = nlk_vocab_size(&vocab);
            vocab_total = nlk_vocab_total(&vocab);
        
            if(verbose) {
                printf("Vocabulary reduced to words: %zu (total count: %zu)\n", 
                        vocab_size, vocab_total);
            }
        }
#ifndef NCHECKS
        size_t last_index = nlk_vocab_last_index(&vocab);
        if(last_index + 1 != vocab_size) {
            printf("last index = %zu\n", last_index);
            NLK_ERROR_ABORT("Invalid vocabulary.", NLK_FAILURE);
        }
#endif
    }

    /* sort and huffman encode if newly created or reduced */
    if(vocab_changes) {
        nlk_vocab_encode_huffman(&vocab);
        if(verbose) {
            printf("Vocabulary sorted & huffman encoded\n");
        }

        if(nn_load_file != NULL) {
            printf("WARNING: Vocabulary might have changed for previously " 
                   "trained network\n");
        }
    }

    if(vocab_save_file != NULL && vocab != NULL) {
        nlk_vocab_save(vocab_save_file, &vocab);
        if(verbose) {
            printf("Vocabulary saved to: %s\n", vocab_save_file);
        }
    }

    /* paragraphs */
    if(train_file != NULL) {
        errno = 0;
        FILE *lc = fopen(train_file, "rb");
        if(lc == NULL) {
            NLK_ERROR_ABORT(strerror(errno), errno);
            /* unreachable */
        }
        total_lines = nlk_text_count_lines(lc);
        fclose(lc);
        if(verbose) {
            printf("Lines: %zu\n", total_lines);
        }
    }

    /** @ section Neural Network
     */
    struct nlk_neuralnet_t *nn;

    if(nn_load_file != NULL) {
        printf("Loading Neural Network from %s\n", nn_load_file);
        /* load */
        nn = nlk_neuralnet_load_path(nn_load_file, verbose);
        if(nn == NULL) {
            printf("Unable to load neural network from %s\n", nn_load_file);
            exit(1);
        }
        if(verbose) {
            printf("Neural Network loaded from %s\n", nn_load_file);
        }
    /* create */
    } else if(vocab != NULL && train_file != NULL && lm_type != NLK_MODEL_NULL) {
        nn = nlk_w2v_create(lm_type, concat, window, sample_rate, learn_rate, 
                            hs, negative, vector_size, vocab_size, total_lines, 
                            verbose);
    } else {
        nn = NULL;
    }

    /* train */
    if(train_file != NULL && nn != NULL && lm_type != NLK_MODEL_NULL) {
        if(verbose) {
            printf("training %s with\nlearning rate = %f\nsample_rate=%f\n"
                   "window=%d\n", model_type, learn_rate, sample_rate, window);
        }

        nlk_w2v_train(nn, train_file, numbered, &vocab, total_lines, epochs, 
                      verbose);
    }
    if(verbose) { 
        nlk_tic("\nTraining finished", true);
    }

     /** @section Save & Export Vectors
     */
    size_t _output_limit = 0;
    if(nn != NULL) {
        word_table = nn->words;
        paragraph_table = nn->paragraphs;

        /* save paragraph vectors */
        if(output_pvs_file != NULL) {
            if(verbose) {
                nlk_tic("Saving paragraph vectors", true);
                printf("%s\n", output_pvs_file);
            }

            if(output_limit == 0) {
                _output_limit = word_table->weights->rows;
            } else {
                _output_limit = vocab_size + output_limit;

            }
            if(format == NLK_FILE_W2V_BIN || format == NLK_FILE_W2V_TXT) {
                nlk_w2v_export_paragraph_vectors(paragraph_table->weights, 
                                                 format, par_prefix, 
                                                 output_pvs_file);
            } else {
                nlk_layer_lookup_save_rows_path(paragraph_table, 
                                                output_pvs_file, 0, 
                                                _output_limit);
            }
        }

        /* save word vectors */
        if(output_words_file != NULL) {
            if(verbose) {
                nlk_tic("Saving word vectors", true);
                printf("%s\n", output_words_file);
            }

            if(output_limit == 0 || output_limit > vocab_size) {
                _output_limit = vocab_size;
            } else {
                _output_limit = output_limit;
            }

            if(format == NLK_FILE_W2V_BIN || format == NLK_FILE_W2V_TXT) {
                nlk_w2v_export_word_vectors(word_table->weights, format, 
                                            &vocab, output_words_file);
            } else {
                nlk_layer_lookup_save_rows_path(word_table, output_words_file, 
                                                0, _output_limit);
            }
        }
        
        /* remove or keep paragraph vectors in NN */
        if(learn_par && keep_pvs) {
            /* keep_pvs is required to allow training to continue */
            /* do nothing */
            if(verbose) {
                nlk_tic("Keeping sentence vectors in NN", true);
            }
        } else if(learn_par && !keep_pvs) {
            if(verbose) {
                nlk_tic("Removing sentence vectors from NN", true);
            }
            /* remove PVs from lookup */
            nlk_layer_lookup_free(nn->paragraphs);
            nn->paragraphs = NULL;
        }

        /* save neural net */
        if(nn_save_file != NULL) {
            if(verbose) {
                nlk_tic("Saving Neural Network", true);
                printf("%s\n", nn_save_file);
            }
            if(nlk_neuralnet_save_path(nn, nn_save_file) != 0) {
                printf("Unable to save neural network to %s\n", nn_save_file);
            }
        }
    } /* end of save/export if(nn != null) */


    /** @section Paragraph Vector Inference
     */
    if(gen_paragraphs_file != NULL) {
        if(nn == NULL) {
            NLK_ERROR_ABORT("no neural network loaded.", NLK_EINVAL);
        }
        if(verbose) {
            printf("Generating paragraph vectors\n");
        }

        /* generate (infer) paragraph vectors */
        NLK_ARRAY *par_vectors = nlk_pv(nn, gen_paragraphs_file, numbered,
                                        &vocab, gen_iter, verbose);
        if(verbose) {
            printf("\n");
        }

        if(pvs_save_file != NULL) {
            if(verbose) {
                printf("Saving paragraph vectors to %s\n", pvs_save_file);
            }
            if(format == NLK_FILE_W2V_BIN || format == NLK_FILE_W2V_TXT) {
                nlk_w2v_export_paragraph_vectors(par_vectors, format,
                                                 par_prefix, pvs_save_file);
            } else {
                FILE *fp_pv = fopen(pvs_save_file, "wb");
                if(fp_pv == NULL) {
                    NLK_ERROR_ABORT("unable to open file.", NLK_FAILURE);
                    /* unreachable */
                }
                nlk_array_save(par_vectors, fp_pv);
                fclose(fp_pv);
            }
        }
        nlk_array_free(par_vectors);
    }

    /** @section Evaluation  (Intrinsic)
     */

    nlk_real accuracy = 0;

    if(questions_file != NULL && nn != NULL) {
        nlk_tic_reset();
        nlk_tic(NULL, false);
        nlk_tic("evaluating word-analogy", true);
        nlk_eval_on_questions(questions_file, &vocab, word_table->weights, 
                              eval_limit, true, &accuracy);
        printf("accuracy = %f%%\n", accuracy * 100);
    }
    
    accuracy = 0;
    if(pvs_input_file != NULL) {
        NLK_ARRAY *pvs = NULL;
        FILE *fin_pv = NULL;

        errno = 0;
        fin_pv = fopen(pvs_input_file, "rb");
        if(fin_pv == NULL) {
            NLK_ERROR_ABORT(strerror(errno), errno);
        }

        if(format == NLK_FILE_TXT) {
            pvs = nlk_array_load_text(fin_pv);
        } else if(format == NLK_FILE_BIN) {
            pvs = nlk_array_load(fin_pv);
        } else {
            NLK_ERROR_ABORT("invalid format for loading paragraph vectors",
                            NLK_FAILURE);
        }
        fclose(fin_pv);

        nlk_eval_on_paraphrases_pre_gen(pvs, eval_limit, verbose, &accuracy);
        nlk_array_free(pvs);
        printf("accuracy = %f%%\n", accuracy * 100);
    }

    accuracy = 0;
    if(paraphrases_file != NULL && nn != NULL) {
        nlk_tic_reset();
        nlk_tic(NULL, false);
        nlk_tic("evaluating paraphrases", true);
        nlk_eval_on_paraphrases(nn, gen_iter, paraphrases_file, numbered, 
                                &vocab, verbose, &accuracy);
        printf("accuracy = %f%%\n", accuracy * 100);
        
    }

       
    return 0;
}
