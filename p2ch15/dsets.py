import copy
import csv
import functools
import glob
import os
import random
from pathlib import Path

from collections import namedtuple

import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch15_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

mhd_data_folder = 'data-unversioned/part2/luna/'

@functools.lru_cache(1)
def getCandidateInfoList(require_on_disk=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.

    data_directory = Path(f"{mhd_data_folder}")
    mhd_files = list(data_directory.rglob("subset*/*.mhd"))
    if not mhd_files:
        print(f"Warning: No .mhd files found under {data_directory}")
    present_on_disk_set = {path.stem for path in mhd_files}
    
    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm),
            )

    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in present_on_disk_set and require_on_disk:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Ct:
    def __init__(self, series_uid):
        import SimpleITK as sitk
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getSingleSlice(self, center_xyz, axis=0):
        """
        Extracts a single 2D slice from a 3D array based on the given center coordinates.

        Parameters:
        - center_xyz: The center coordinates in the 3D space.
        - axis: The axis along which to extract the slice (default is 0).

        Returns:
        - A 2D slice from the 3D array.
        """
        # Convert center coordinates from xyz to irc
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        # Ensure the center index is within bounds
        center_val = int(round(center_irc[axis]))

        # Extract the 2D slice along the specified axis
        if axis == 0:
            ct_slice = self.hu_a[center_val, :, :]
        elif axis == 1:
            ct_slice = self.hu_a[:, center_val, :]
        elif axis == 2:
            ct_slice = self.hu_a[:, :, center_val]
        else:
            raise ValueError("Invalid axis value. Must be 0, 1, or 2.")

        return ct_slice, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtSlice(series_uid, center_xyz):
    ct = getCt(series_uid)
    ct_slice, center_irc = ct.getSingleSlice(center_xyz)
    return ct_slice, center_irc

class LunaDataset(Dataset):
    def __init__(self,
                 candidate_info_list=None,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
            ):
        if candidate_info_list is not None:
            self.candidateInfo_list = candidate_info_list
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        if isinstance(ndx, slice):
            # Handle slicing
            return [self._get_single_item(i) for i in range(*ndx.indices(len(self)))]
        elif isinstance(ndx, int):
            # Handle single index
            return self._get_single_item(ndx)
        else:
            raise TypeError("Invalid argument type.")

    def _get_single_item(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        ct_slice, center_irc = getCtSlice(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
        )
        ct_slice_tensor = torch.from_numpy(ct_slice).to(torch.float32)
        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )
        return ct_slice_tensor, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)