# OS
OSNAME := $(shell uname -s)

# Compiler, default GCC, clang for OSX
CC = gcc
ifeq ($(OSNAME),Darwin)
CC = clang
endif

#
# Miscellaneous OS dependent options
#
# Linux
ifeq ($(OSNAME), Linux)
EXTRALIB += -lm 

EXTRALIB += -lopenblas
BLAS_INCLUDE = /opt/OpenBLAS/include/
BLAS_LIB = /opt/OpenBLAS/lib
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)


#unfinished
#EXTRALIB += -latlas
#ATLAS_INCLUDE = /usr/lib
#ATLAS_LIB = /usr/lib/atlas-base/atlas/
#INCLUDE_DIRS += $(BLAS_INCLUDE)
#LIBRARY_DIRS += $(BLAS_LIB)



endif

# OSX
ifeq ($(OSNAME),Darwin)
EXTRALIB += -framework Accelerate
CFLAGS += -DACCELERATE=1
endif


#
# Compiler dependent options
#
ifeq ($(CC), gcc)
endif


ifeq ($(CC), clang)
endif


COMMON_FLAGS += -Wall
BUILD = build


COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
COMMON_FLAGS += -pthread -march=native -std=gnu11 -mtune=native

OPT_FLAGS = -O3 -fno-strict-aliasing -ffast-math -funroll-loops -DNCHECKS


LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
$(foreach library,$(LIBRARIES),-l$(library))
LDFLAGS += $(EXTRALIB)

#
# UTHASH
#
UTHASHOPTS = -DHASH_BLOOM=32


OBJS = main.o nlk_err.o nlk_array.o tinymt32.o nlk_layer_linear.o nlk_text.o \
	   nlk_vocabulary.o nlk_tic.o nlk_window.o nlk_transfer.o nlk_criterion.o \
	   nlk_eval.o nlk_w2v.o MurmurHash3.o nlk_neuralnet.o nlk_random.o


all: release

release: CFLAGS += $(OPT_FLAGS)
release: LDFLAGS += -fopenmp
release: nlk

debug: CFLAGS = -g -static-libgcc -DDEBUG
debug: nlk

debug-multi: CFLAGS = -g -static-libgcc
debug-multi: LDFLAGS += -fopenmp
debug-multi: nlk

nlk: $(OBJS)
	$(CC) -o main $(OBJS) $(CFLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(UTHASHOPTS)
	mv main *.o $(BUILD)

%.o: %.c
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) $(LDFLAGS) $(UTHASHOPTS) $<

clean:
	rm -f *.o
clear: clean
	rm main
