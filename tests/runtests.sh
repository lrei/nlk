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
for i in $(find ./ -type f -not -name \*\.\*)
do
    if test -f $i
    then
        ./$i >> tests.log
    fi
done

cat tests.log

echo ""
