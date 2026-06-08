#!/bin/bash
morfessor -l models/$1.pickle -T test_src/$1.tsv -o test_predicted/$1.txt
