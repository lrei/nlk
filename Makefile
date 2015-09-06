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
CFLAGS += -D__MACH__
CC = /usr/local/Cellar/gcc/5.2.0/bin/gcc-5
EXTRALIB += -lm 
CFLAGS += -msse4.2 -flax-vector-conversions
#EXTRALIB += -lopenblas
#BLAS_LIB  = /usr/local/opt/openblas/lib
#BLAS_INCLUDE = /usr/local/opt/openblas/include
#INCLUDE_DIRS += $(BLAS_INCLUDE)
#LIBRARY_DIRS += $(BLAS_LIB)
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

#
# General Options
#

# Common Flags
CFLAGS += -Wall -Wextra -Wno-unused-result
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CFLAGS += -std=gnu11 -mtune=native

# 
# Flags for relase
#
REL_FLAGS = -O3 -fno-strict-aliasing  -pthread -funroll-loops \
			-funsafe-loop-optimizations \
# REL MATH OPTION
REL_FLAGS += -fno-math-errno -ffast-math -funsafe-math-optimizations \
			 -ffinite-math-only -fno-signed-zeros
# ERRORS
REL_FLAGS += -DNCHECKS -Werror


#
# Debug Flags
#
# add -DCHECK_NANS=1,2 to check for NaNs
DEB_FLAGS = -O0 -ggdb3 -static-libgcc -DDEBUG

#
# LDFlags
#
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
