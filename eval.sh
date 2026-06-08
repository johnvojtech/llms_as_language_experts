#!/bin/bash
ls $1 > a.txt
python3 a.py a.txt $1 $2 > a.sh
bash a.sh > results.tsv
cat a.sh|sed -e "s/evaluation/evaluation-multi/g" > b.sh
bash b.sh > results_multi.tsv
python3 tsv_to_csv.py results.tsv > results.csv
python3 tsv_to_csv.py results_multi.tsv > results_multi.csv
rm a.txt a.sh b.sh
