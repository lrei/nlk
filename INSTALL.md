# NLK Install

## Pre-Requisites

### Supported Platforms

    * Linux AMD64 (tested with Ubuntu 14.04)

### Compiler

    * gcc with OpenMP support

### External Library Dependencies

    * OpenBLAS
    * UTF8Proc

## Instructions

Install gcc & make normally via the package manager: on Ubuntu:

'''
sudo apt-get install build-essential
'''

Install OpenBLAS to the default directory: /opt/OpenBLAS.

Install UTF8Proc to the default directory: /usr/local/


make


## Testing

make tests
