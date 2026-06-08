#!python3
import sys
with open(sys.argv[1], "r") as r:
    for line in r:
        ln = line.strip().split("\t")
        if ln[0] != "#ID":
            ln[0] = ln[0].split("-")[3].split(":")[1]
        print("\t".join(ln))
