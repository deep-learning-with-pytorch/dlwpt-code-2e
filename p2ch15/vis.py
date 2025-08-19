# import matplotlib
# matplotlib.use('nbagg')

import numpy as np
import matplotlib.pyplot as plt

from .dsets import Ct, LunaDataset

clim=(-1000.0, 300)

def findPositiveSamples(start_ndx=0, limit=100):
    ds = LunaDataset(sortby_str='label_and_size')

    positiveSample_list = []
    for sample_tup in ds.candidateInfo_list:
        if sample_tup.isNodule_bool:
            print(len(positiveSample_list), sample_tup)
            positiveSample_list.append(sample_tup)

        if len(positiveSample_list) >= limit:
            break

    return positiveSample_list

def showCandidate(series_uid, batch_ndx=None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]

    def determine_batch_index(batch_ndx, pos_list):
        if batch_ndx is None:
            if pos_list:
                return pos_list[0]
            else:
                print("Warning: no positive samples found; using first negative sample.")
                return 0
        return batch_ndx

    def plot_ct_slices(fig, ct, i, r, c):
        plot_slice(fig, 1, 'ct index {}'.format(int(i)), ct.hu_a[int(i)])
        plot_slice(fig, 2, 'ct row {}'.format(int(r)), ct.hu_a[:, int(r)], invert_y=True)
        plot_slice(fig, 3, 'ct col {}'.format(int(c)), ct.hu_a[:, :, int(c)], invert_y=True)

    def plot_smaller_scale_slices(fig, ct_a, i, r, c):
        plot_slice(fig, 4, 'ct index {} (smaller scale)'.format(int(i)), ct_a[ct_a.shape[0] // 2], smaller_scale=True)
        plot_slice(fig, 5, 'ct row {} (smaller scale)'.format(int(r)), ct_a[:, ct_a.shape[1] // 2], invert_y=True)
        plot_slice(fig, 6, 'ct col {} (smaller scale)'.format(int(c)), ct_a[:, :, ct_a.shape[2] // 2], invert_y=True)
    
    def plot_group_slices(fig, ct_a, group_list):
        for row, index_list in enumerate(group_list):
            for col, index in enumerate(index_list):
                plot_slice(fig, row * 3 + col + 7, 'slice {}'.format(index), ct_a[index])
                
    def plot_slice(fig, position, title, data, invert_y=False, smaller_scale=False):
        subplot = fig.add_subplot(len(group_list) + 2, 3, position)
        subplot.set_title(title, fontsize=30)
        for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
            label.set_fontsize(20)
        clim=(-1000.0, 300)
        plt.imshow(data, clim=clim, cmap='gray')
        if invert_y:
            plt.gca().invert_yaxis()
        if smaller_scale:
            subplot.set_xticks(np.arange(0, data.shape[1], step=10))
            subplot.set_yticks(np.arange(0, data.shape[0], step=10))

    batch_ndx = determine_batch_index(batch_ndx, pos_list)
    ct = Ct(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()
    fig = plt.figure(figsize=(30, 50))
    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]
    i, r, c = center_irc
    plot_ct_slices(fig, ct, i, r, c)
    plot_smaller_scale_slices(fig, ct_a, i, r, c)
    plot_group_slices(fig, ct_a, group_list)
    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)


