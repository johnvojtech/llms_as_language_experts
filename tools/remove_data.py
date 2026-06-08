#!python3
import sys
import pandas as pd

selected_words = []
with open(sys.argv[1], "r") as r:
    for line in r:
        selected_words.append(int(line.strip()))

with open(sys.argv[2], "r") as r:
    for line in r:
        filename = line.strip()
        new_filename = filename.replace("results_2", "results_3")
        df = pd.read_csv(filename, sep="\t")
        #print(df)
        #print(df["#ID"])
        df = df[df["#ID"].isin(selected_words)].copy()
        #print(df)
        #print(df, selected_words)
        df.to_csv(new_filename, sep="\t", index=False)
    #df = df[df["LLM"] == sys.argv[2]]

 
