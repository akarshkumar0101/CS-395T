from matplotlib.pyplot import figure, cm
import numpy as np
import logging
from SemanticModel import SemanticModel
from stimulus_utils import load_grids_for_stories
from stimulus_utils import load_generic_trfiles
from dsutils import make_word_ds, make_phoneme_ds
from dsutils import make_semantic_model
from npp import zscore
from util import make_delayed
from helper import getTimestampDict, listToString, numUniqueWords
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import tables
from bert_serving.client import BertClient
logging.basicConfig(level=logging.DEBUG)

bc = BertClient()

print("Pre-loading model")

eng1000 = SemanticModel.load("data/english1000sm.hf5")

print("Post-loading model")
# These are lists of the stories
# Rstories are the names of the training (or Regression) stories, which we will use to fit our models
Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy', 
            'life', 'myfirstdaywiththeyankees', 'naked', 
            'odetostepfather', 'souls', 'undertheinfluence']

# Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
Pstories = ['wheretheressmoke']

allstories = Rstories + Pstories

# Load TextGrids
grids = load_grids_for_stories(allstories)
# Load TRfiles
trfiles = load_generic_trfiles(allstories)

# Make word and phoneme datasequences
wordseqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
phonseqs = make_phoneme_ds(grids, trfiles) # dictionary of {storyname : phoneme DataSequence}


#Preprocess Stimuli
text = []
paragraph = wordseqs["naked"]
input_list = list(paragraph.data)
print("how many words?")
print(len(input_list))

# for i in range(len(list(paragraph.data))):
# 	if list(paragraph.data)[i] == " ":
# 		print(i)

fold = len(input_list) // 400

for i in range(fold):
	text.append(" ".join(list(paragraph.data[400*i:400*(i+1)])).replace("'","").replace("-","").replace(".",""))
text.append(" ".join(list(paragraph.data[400*fold:])).replace("'","").replace("-","").replace(".",""))

print("length of text: ")
print(len(text))

print("Pre-encoding")

word_embeddings = bc.encode(text, show_tokens=True)

word_embedding = word_embeddings[0]
tokens = word_embeddings[1]

print("TOKENS[0] SHAPE:: ")
print(len(tokens[0]))


print("Shape of word embeddings for story")
shape = word_embedding.shape
print(shape)

word_embedding = word_embedding.reshape(shape[0]*shape[1], shape[2])
print("Shape of word embeddings after reshape")
print(word_embedding.shape)

index_to_be_removed = []

base = 0
for k in range(len(tokens)):
	for i in range(len(tokens[k])):
		if not tokens[k][i].isalpha():
			index_to_be_removed.append(base + i)
		else:
			if not tokens[k][i] in input_list:
				print("not in orig list ::")
				print(tokens[k][i])
	base += 512

for index in range(word_embedding.shape[0]):
	if np.sum(word_embedding[index]) == 0 and index not in index_to_be_removed:
		index_to_be_removed.append(index)

print("number of rows to be removed: ")
print(len(index_to_be_removed))
print(index_to_be_removed)
word_embedding = np.delete(word_embedding, index_to_be_removed, 0)

print("Shape of word embeddings after remove")
print(word_embedding.shape)

np.save("naked", word_embedding)