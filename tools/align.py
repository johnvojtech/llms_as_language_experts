#!python3
import sys
from collections import defaultdict
annotations = defaultdict(list)

with open(sys.argv[1], "r") as r:
    for line in r:
        ln = line.strip().split("\t")
        form = ln[0]
        segm = ln[2]
        annotations[form] = segm.replace(" ", "")

with open(sys.argv[2], "r") as r:
    with open(sys.argv[3], "w") as w:
        print(r.readline().strip("\n"), file=w)
        for line in r:
            ln = line.strip("\n").split("\t")
            form = ln[1]
            ln[8] = annotations[form]
            while len(ln[:10]) < 10:
                ln.append("")
            print("\t".join(ln), file=w)

