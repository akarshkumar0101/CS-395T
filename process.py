import os
import tables
import numpy as np
import torch
import matplotlib.pyplot as plt

from stimulus_utils import load_grids_for_stories
from stimulus_utils import load_generic_trfiles
from dsutils import make_word_ds, make_phoneme_ds
from dsutils import make_semantic_model
from npp import zscore
from util import make_delayed

from SemanticModel import SemanticModel

def remove_blank_strs(wordseqs):
    wordseqs_fixed = {}
    for story in wordseqs.keys():
#         print(story)
        wordseq = wordseqs[story].copy()
        wordseq.split_inds = np.array(wordseq.split_inds)
        wordseqs_fixed[story] = wordseq

        init_total_words = len(wordseq.data)
        total_words = len(wordseq.data)
        i=0
        while i!=total_words:
            w = wordseq.data[i]
            if w=='' or w==' ':
#                 print(' ---- found', i, wordseq.data_to_chunk_ind(i))
                wordseq.data = np.delete(wordseq.data, i)
                wordseq.data_times = np.delete(wordseq.data_times, i)
                wordseq.split_inds[wordseq.split_inds>i] -= 1
                i-=1
                total_words-=1
            i+=1
#         print(f'Removed {init_total_words-total_words} words from {story}')
    return wordseqs_fixed

"""
Load the stimuli and fMRI response into torch tensors.
`use_bert_embeddings` determines to use English1000 word embeddings or BERT embeddings.
It loads BERT embeddings from the directory called 'bert_word_embeddings'.

`trim` determines how much to trim the data.
`ndelays` determines how many previous timepoints to include in the fMRI response.
`make_validation_set` will return X_val, Y_val also to use as a validation set
if True, method will return X_train, X_val, X_test, Y_train, Y_val, Y_test, mask
else, method will return X_train, X_test, Y_train, Y_test, mask
"""
def load_data(use_bert_embeddings=True, trim=5, ndelays=4, make_validation_set=True):
    interptype = "lanczos" # filter type
    window = 3 # number of lobes in Lanczos filter

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

    wordseqs = remove_blank_strs(wordseqs)

    # Project stimuli
    semanticseqs = dict() # dictionary to hold projected stimuli {story name : projected DataSequence}
    for story in allstories:
        semanticseqs[story] = make_semantic_model(wordseqs[story], eng1000)

    if use_bert_embeddings:
        for story in allstories:
            semanticseqs[story].data = np.load(os.path.join('bert_word_embeddings/', story+'.npy'))


    # Downsample stimuli
    downsampled_semanticseqs = dict() # dictionary to hold downsampled stimuli
    for story in allstories:
        downsampled_semanticseqs[story] = semanticseqs[story].chunksums(interptype, window=window)

    # Combine stimuli
    Rstim = np.vstack([zscore(downsampled_semanticseqs[story][5+trim:-trim]) for story in Rstories])
    Pstim = np.vstack([zscore(downsampled_semanticseqs[story][5+trim:-trim]) for story in Pstories])

    storylens = [len(downsampled_semanticseqs[story][5+trim:-trim]) for story in Rstories]

    # Delay stimuli
    delays = range(1, ndelays+1)

    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)

    # Load responses
    resptf = tables.open_file("data/fmri-responses.hf5")
    zRresp = resptf.root.zRresp.read()
    zPresp = resptf.root.zPresp.read()
    mask = resptf.root.mask.read()
    
    # rename these
    X_test = torch.from_numpy(delPstim).float()
    Y_test = torch.from_numpy(zPresp).float()
    mask = torch.from_numpy(mask)
    
    if make_validation_set:
        val_size = len(delPstim)
        X_train = torch.from_numpy(delRstim[:-val_size]).float()
        X_val = torch.from_numpy(delRstim[-val_size:]).float()
        Y_train = torch.from_numpy(zRresp[:-val_size]).float()
        Y_val = torch.from_numpy(zRresp[-val_size:]).float()
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, mask
    else:
        X_train = torch.from_numpy(delRstim).float()
        Y_train = torch.from_numpy(zRresp).float()
        return X_train, X_test, Y_train, Y_test, mask
        

"""
Calculate the stats of the predicted voxel data.
Returns the MSE and the correlation of the predicted responses versus the ground truth.

`Y` should be the ground truth data of shape (T, M)
`Y_pred` should be the predicted data of shape (T, M)


T is the number of fMRI responses.
M is the number of voxels.
"""
def calc_stats(Y, Y_pred, print_stats=True, show_vox_corr_hist=False):
    squared_error = (Y-Y_pred)**2
    
    top = ((Y-Y.mean(dim=0))*(Y_pred-Y_pred.mean(dim=0))).sum(dim=0)
    bot = torch.sqrt((Y-Y.mean(dim=0)).pow(2).sum(dim=0)*(Y_pred-Y_pred.mean(dim=0)).pow(2).sum(dim=0))
    voxcorrs = top/bot
    return squared_error, voxcorrs

"""
Shows the stats for the current predictions
`Y` should be the ground truth data of shape (T, M)
`Y_pred` should be the predicted data of shape (T, M)
"""
def show_stats(Y, Y_pred, show_vox_corr_hist=False):
    squared_error, voxcorrs = calc_stats(Y, Y_pred)
    mse = squared_error.mean()
    r = voxcorrs.mean()
    print('MSE: ', mse.item())
    print('Mean Correlation: ', r.item())
    if show_vox_corr_hist:
        plt.title('Correlation over voxels')
        plt.hist(voxcorrs.numpy(), bins=100)
        plt.show()
    

