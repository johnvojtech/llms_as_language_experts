#!python3
import sys
from collections import defaultdict

data_sorted = defaultdict(list)
with open(sys.argv[1], "r") as r:
    for line in r:
        ln = line.strip().split("\t")
        data = [float(x) for x in ln[1:]]
        settings = ln[0].split(".")
        lang, model, guidelines, examples = settings[0], settings[1], settings[2], settings[3]
        data_sorted[(model, guidelines, examples)].append(data)

for key in data_sorted.keys():
    print("|".join(key) + "\t" +str(len(data_sorted[key])) + "\t" + "\t".join([str(sum([x[i] for x in data_sorted[key]])/len(data_sorted[key])) for i in range(4)]))

