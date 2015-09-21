#!/bin/bash

echo "Running unit tests:"
# Clean log
if test -f tests.log
then
    rm tests.log
fi
if test -d ./tmp
then
    rm ./tmp/*
fi

# run one by one
#TESTS_ENABLED=( 'class_test' )
for i in $(find ./ -type f -not -name \*\.\*)
#for i in ${TESTS_ENABLED[@]}
do
    if test -f $i
    then
        echo $i
        ./$i >> tests.log
    fi
done

cat tests.log

echo ""
