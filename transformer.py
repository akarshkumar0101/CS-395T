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
from helper import getTimestampDict, listToString
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import tables
from bert_serving.client import BertClient
logging.basicConfig(level=logging.DEBUG)

bc = BertClient()


eng1000 = SemanticModel.load("data/english1000sm.hf5")

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

naked = wordseqs["naked"]

print("data time length: ")
print(len(naked.data_times))
print("tr time length: ")
print(len(naked.tr_times))

timestampDict = getTimestampDict(naked)
for t in naked.tr_times[:10]:
	print(timestampDict[t])

#Preprocess Stimuli

Rstim = []
Pstim = []
trim = 5

for story in Rstories:
	timestampDict = getTimestampDict(wordseqs[story])
	temp_list = []
	for key in timestampDict:
		temp_list.append(timestampDict[key])
	for item in temp_list[5+trim:-trim]:
		Rstim.append(item)

for story in Pstories:
	timestampDict = getTimestampDict(wordseqs[story])
	temp_list = []
	for key in timestampDict:
		temp_list.append(timestampDict[key])
	for item in temp_list[5+trim:-trim]:
		Pstim.append(item)

print("Rstim length: ")
print(len(Rstim))
print("Pstim length: ")
print(len(Pstim))

#load responses
resptf = tables.open_file("data/fmri-responses.hf5")
zRresp = resptf.root.zRresp.read()
zPresp = resptf.root.zPresp.read()
mask = resptf.root.mask.read()

# Print matrix shapes
print ("zRresp shape (num time points, num voxels): ", zRresp.shape)
print ("zPresp shape (num time points, num voxels): ", zPresp.shape)
print ("mask shape (Z, Y, X): ", mask.shape)


R_texts = []
P_texts = []
R_voxels_to_be_removed = []
P_voxels_to_be_removed = []
for index in range(len(Rstim)):
	sList = Rstim[index]
	if len(sList) > 0:
		R_texts.append(listToString(sList))
	else:
		R_voxels_to_be_removed.append(index)
#Not sure if we need to add held-out set stimuli
for index in range(len(Pstim)):
	sList = Pstim[index]
	if len(sList) > 0:
		P_texts.append(listToString(sList))
	else:
		P_voxels_to_be_removed.append(index)

print("Number of R sentences: ")
print(len(R_texts))
print("Number of P sentences: ")
print(len(P_texts))

Rresp = np.delete(zRresp, np.array(R_voxels_to_be_removed), 0)
Presp = np.delete(zPresp, np.array(P_voxels_to_be_removed), 0)

print("Shape of R responses: ")
print(Rresp.shape)
print("Shape of P responses: ")
print(Presp.shape)

R_word_embeddings = bc.encode(R_texts)
P_word_embeddings = bc.encode(P_texts)

print("Shape of word embeddings for Rstories")
print(R_word_embeddings.shape)
print("Shape of word embeddings for Pstories")
print(P_word_embeddings.shape)

np.save("R_word_embedding_vectors", R_word_embeddings)
np.save("P_word_embedding_vectors", P_word_embeddings)