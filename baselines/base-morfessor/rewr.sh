#!/bin/bash
python3 rewr.py $1/$1-words.txt > training_data/$2.tsv
rm -r $1
rm $1.tar.gz
