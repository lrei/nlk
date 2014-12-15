
typedef float real;

#define MAX_SENTENCE_LENGTH 1024
#define MAX_EXP 6
#define EXP_TABLE_SIZE 10000



void nlk_alt_sg_start(size_t layer1_size, real *neu1e);


void nlk_alt_sg_step(size_t input_word_idx, size_t point, 
                     size_t layer1_size, real *neu1e,
                     char code, real *syn0, 
                     real *syn1, real alpha, real *expTable, real *f, real *g,
                     real *temp);


void nlk_al_sg_end(size_t input_word_idx, size_t layer1_size, real *syn0, 
                   real *neu1e);

