rm -rf *.csv

for j in {0..4}
do
    for i in {1..20}
    do
        ./mmOPT --sizemult=$i $j
    done
done

for i in {1..20}
do
    ./mmCUBLAS --sizemult$i
done

echo "Completed"