#!python3
import sys

c, t = 0, 0
tp, fp, tn, fn = 
with open(sys.argv[1], "r") as r:
    with open(sys.argv[2], "r") as r2:
        for a, b in zip(r, r2):
            a = a.split("\t")[-1]
            b = b.split("\t")[-1]
            if a.strip() == b.strip():
                c += 1
            t += 1
print("Word accuracy", c/t)
