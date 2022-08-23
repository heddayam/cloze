from nltk.corpus import wordnet as wn
import pdb

res = wn.synsets('dog', pos=wn.VERB)
res = [r.name().split('.')[0] for r in res]
pdb.set_trace()