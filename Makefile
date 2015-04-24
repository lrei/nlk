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

#
# General Options
#

# Common Flags
CFLAGS += -Wall 
#CFLAGS += -Wextra
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CFLAGS += -pthread -march=native -std=gnu11 -mtune=native

# Flags for relase w/o omp
REL_FLAGS = -O3 -fno-strict-aliasing -ffast-math -funroll-loops \
			-Wno-unused-result -DNCHECKS

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

SOURCES = $(wildcard $(SOURCE_DIR)/**/*.c $(SOURCE_DIR)/*.c)
OBJECTS = $(patsubst $(SOURCE_DIR)%.c,$(BUILD_DIR)%.o,$(SOURCES))

TEST_SRC=$(wildcard tests/*.c)
TESTS=$(patsubst %.c,%,$(TEST_SRC))

TARGET=$(BIN_DIR)/$(PRG_NAME)

#
# Targets
#
all: release $(TARGET)

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

release-single: CFLAGS += $(REL_FLAGS) -DNOMP
release-single: $(TARGET)

debug: CFLAGS += $(DEB_FLAGS) -DNOMP
debug: $(TARGET)

debug-multi: CFLAGS += -g -static-libgcc -DDEBUG
debug-multi: LDFLAGS += -fopenmp
debug-multi: $(TARGET)

.PHONY: tests
tests: $(TARGET)
tests: $(TESTS)
	   sh ./tests/runtests.sh

clean: 
	rm -rf build $(TESTS)

# allow typing make print-var
print-%: ; @echo $*=$($*)
