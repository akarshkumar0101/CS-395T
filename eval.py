

"""
Calculate the stats of the predicted voxel data.
Returns the MSE and the correlation of the predicted responses versus the ground truth.

`gt` should be the ground truth data of shape (T, M)
`pred` should be the predicted data of shape (T, M)


T is the number of fMRI responses.
M is the number of voxels.
"""
def calc_stats(gt, pred, print_stats=True, show_vox_corr_hist=False):
    mse = ((gt-pred)**2).mean()
        
    top = ((gt-gt.mean(dim=0))*(pred-pred.mean(dim=0))).sum(dim=0)
    bot = torch.sqrt((gt-gt.mean(dim=0)).pow(2).sum(dim=0)*(pred-pred.mean(dim=0)).pow(2).sum(dim=0))
    voxcorrs = top/bot
    r = voxcorrs.mean()
    
    if print_stats:
        print('MSE: ', mse.item())
        print('Mean Correlation: ', r.item())
    if show_vox_corr_hist:
        plt.title('Correlation over voxels')
        plt.hist(voxcorrs, bins=100)
        plt.show()
    return mse.item(), r.item()
    


