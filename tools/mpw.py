#!python3
import sys

words = 0
morphs = 0
morphlens = 0
morphset = set()
with open(sys.argv[1], "r") as r:
   r.readline()
   for line in r:
       ln = line.strip().split("\t")
       m = [x.strip() for x in ln[8].split("+")]
       for item in m:
           morphset.add(item)
       morphs += len(m)
       words += 1
       morphlens += len(ln[1].strip())

# morphs per word, normalized by word length?
# morphs per word
print(sys.argv[1], words, morphs, morphlens, morphs/words, morphlens/morphs, len(morphset)/morphs)


