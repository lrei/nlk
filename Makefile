CFLAGS = $(OPTFLAGS)

# OS
OSNAME := $(shell uname -s)

# Compiler, default GCC
CC = gcc

# clang for OSX
#ifeq ($(OSNAME),Darwin)
#CC = clang
#endif

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
CFLAGS += -march=native
endif

# OSX
ifeq ($(OSNAME),Darwin)
# Currently not using Accelerate
#EXTRALIB += -framework Accelerate
#CFLAGS += -DACCELERATE=1

# Using homebrew
# brew install gcc --without-multilib
# brew install homebrew/science/openblas
# brew reinstall openblas --build
CC = /usr/local/bin/gcc-5
EXTRALIB += -lm 
EXTRALIB += -lopenblas
BLAS_INCLUDE = /usr/local/opt/openblas/include/
BLAS_LIB = /usr/local/opt/openblas/lib/
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)
CFLAGS += -msse4.2
endif


#
# Compiler dependent options
#
ifeq ($(CC), gcc)
endif


ifeq ($(CC), clang)
endif

#
# General Options
#

# Common Flags
CFLAGS += -Wall -Wextra -Wno-unused-result
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CFLAGS += -std=gnu11 -mtune=native

# Flags for relase w/o omp
REL_FLAGS = -O3 -fno-strict-aliasing -ffast-math -pthread -funroll-loops \
			-DNCHECKS -Werror

# Debug Flags
# add -DCHECK_NANS=1,2 to check for NaNs
DEB_FLAGS = -g -static-libgcc -DDEBUG 

# LDFlags
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
$(foreach library,$(LIBRARIES),-l$(library))
LDFLAGS += $(EXTRALIB)
CFLAGS += $(LDFLAGS)

# UTHASH
UTHASH_OPTS = -DHASH_BLOOM=32
CLAGS += $(UTHASH_OPTS)


#
# Files
#
PRG_NAME = nlktool
BIN_DIR = bin
BUILD_DIR = build
SOURCE_DIR = src
TEST_DIR = tests

SOURCES = $(wildcard $(SOURCE_DIR)/**/*.c $(SOURCE_DIR)/*.c)
OBJECTS = $(patsubst $(SOURCE_DIR)%.c,$(BUILD_DIR)%.o,$(SOURCES))

TEST_SRC=$(wildcard $(TEST_DIR)/*.c)
TESTS=$(patsubst %.c,%,$(TEST_SRC))

TARGET=$(BIN_DIR)/$(PRG_NAME)

#
# Targets
#
all: release

$(TARGET): build $(OBJECTS)
		   $(CC) -o $@ $(OBJECTS) $(CFLAGS)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.c
	$(CC) -c -o $@ $< $(CFLAGS)

build:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

release: CFLAGS += $(REL_FLAGS)
release: LDFLAGS += -fopenmp
release: $(TARGET)
release: build-tests

# For Monitoring NN Training
monitor: CFLAGS += $(REL_FLAGS) -DNLK_MONITOR=1
monitor: LDFLAGS += -fopenmp
monitor: $(TARGET)
monitor: build-tests

# Debug Builds
debug: CFLAGS += $(DEB_FLAGS) -DNOMP
debug: $(TARGET)
debug: build-tests

debug-nan: CFLAGS += $(DEB_FLAGS) -DNOMP -DCHECK_NANS
debug-nan: $(TARGET)
debug-nan: build-tests

debug-multi: CFLAGS += $(DEB_FLAGS)
debug-multi: LDFLAGS += -fopenmp
debug-multi: $(TARGET)
debug-multi: build-tests

$(TEST_DIR)/%: $(TEST_DIR)/%.c
	$(CC) -o $@ $< $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS)) $(CFLAGS)

build-tests: $(TESTS)

.PHONY: tests
tests: 
	cd $(TEST_DIR); \
	sh runtests.sh

clean: 
	rm -rf build $(TESTS)

# allow typing make print-var
print-%: ; @echo $*=$($*)
