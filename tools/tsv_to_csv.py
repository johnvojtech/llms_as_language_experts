#!python3
import sys
header = ["LANGUAGE", "LLM", "GUIDELINES", "EXAMPLES", "UD", "WORD_ACCURACY", "BOUNDARY_PRECISION", "BOUNDARY_RECALL", "BOUNDARY_F1", "PRESENT"]
print("\t".join(header))
with open(sys.argv[1], "r") as r:
    for line in r:
        ln = line.strip().split("\t")
        effects = ln[0].replace("results/","").replace(".tsv", "").split(".")
        if "_no_ud" in effects[2]:
            effects[2] = effects[2].replace("_no_ud", "")
            effects.append("0")
        else:
            effects.append("1")
        data = ln[1:]
        print("\t".join(effects + data))

