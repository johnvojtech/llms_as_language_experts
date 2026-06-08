#!python3
import sys
data = {}
with open(sys.argv[1], "r") as r:
    for line in r:
        ln = line.strip().split()
        data[ln[0].replace(".tsv", "").split("/")[-1]] = float(ln[-1])
print(data)
