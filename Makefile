BLAS_INCLUDE = /opt/OpenBLAS/include/
BLAS_LIB = /opt/OpenBLAS/lib

INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
COMMON_FLAGS += -pthread -march=native -std=gnu11 

LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
$(foreach library,$(LIBRARIES),-l$(library))
LDFLAGS += -lm -lopenblas -fopenmp

CC = gcc
UTHASHOPTS = -DHASH_BLOOM=28
OBJS = main.o nlk_err.o nlk_array.o tinymt32.o nlk_layer_linear.o nlk_text.o \
	   nlk_vocabulary.o nlk_tic.o nlk_window.o nlk_transfer.o nlk_criterion.o \
	   nlk_eval.o nlk_w2v.o MurmurHash3.o


all: release

release: CFLAGS = -O3 -fno-strict-aliasing -ffast-math -funroll-loops
release: nlk

debug: CFLAGS = -g -pg -static-libgcc
debug: nlk

nlk: $(OBJS)
	$(CC) -o main $(OBJS) $(CFLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(UTHASHOPTS)

%.o: %.c
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(UTHASHOPTS) $<

clean:
	rm -f *.o
clear: clean
	rm main
