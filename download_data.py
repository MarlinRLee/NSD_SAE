# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import trange


# Helper to load 'nsd_expdesign.mat' file Copy-pasted,
# from https://github.com/ozcelikfu/brain-diffuser/blob/main/data/prepare_nsddata.py
# Commit 1c07200
def _loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    import scipy.io as spio

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def download(nsd_bucket: str, path: str, aws_args: str, subjects = [1]) -> None:
    path = Path(path)
    download_nsd_timeseries_dataset(nsd_bucket, path, aws_args, subjects = subjects)
    prepare_dataset(path)


def download_nsd_timeseries_dataset(nsd_bucket: str, path: Path, aws_args: str, subjects = [1]) -> None:
    path.mkdir(exist_ok=True, parents=True)

    aws_cmds = []
    
    nsd_bucket = nsd_bucket.rstrip('/')
    for subject in subjects:
        # timeseries data
        for data_type in ["timeseries", "design"]:
            aws_cmd = (
                f"aws s3 {aws_args} sync"
                f" {nsd_bucket}/nsddata_timeseries/ppdata"
                f"/subj{subject:02}"
                f"/func1pt8mm/{data_type}/"
                f" {path}/nsddata_timeseries/ppdata/subj{subject:02}"
                f"/func1pt8mm/{data_type}/"
                " --exclude '*'"
                f" --include '{data_type}_session*'"
            )
            aws_cmds.append(aws_cmd)
        # rois
        roi_aws_cmd = (
            f"aws s3 {aws_args} sync"
            f" {nsd_bucket}/nsddata/ppdata/subj{subject:02}"
            "/func1pt8mm/roi/"
            f" {path}/nsddata/ppdata/subj{subject:02}/func1pt8mm/roi/"
        )
        aws_cmds.append(roi_aws_cmd)

    # Experimental design matrix
    expdesign_mat_aws_cmd = (
        f"aws s3 {aws_args} cp"
        f" {nsd_bucket}/nsddata/experiments/nsd/nsd_expdesign.mat"
        f" {path}"
    )
    aws_cmds.append(expdesign_mat_aws_cmd)

    # Stimulus matrix
    stimuli_mat_aws_cmd = (
        f"aws s3 {aws_args} cp"
        f" {nsd_bucket}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
        f" {path}"
    )
    aws_cmds.append(stimuli_mat_aws_cmd)

    for aws_cmd in aws_cmds:
        subprocess.run(aws_cmd, shell=True)


def prepare_dataset(path: Path) -> None:
    extract_test_images_ids(path)


def extract_test_images_ids(path: Path) -> None:
    path_to_expdesign_mat = path / "nsd_expdesign.mat"
    expdesign_mat = _loadmat(path_to_expdesign_mat)
    np.save(path / "test_images_ids.npy", expdesign_mat["sharedix"])


if __name__ == "__main__":

    nsd_bucket = ""#required to replace with real nsd bucket
    path = "nsddata"#required to replace with saved location. default is nsddata
    aws_args = ""
    subjects = [1]#,2,5,7
    download(nsd_bucket, path, aws_args, subjects)
    print("Done downloading and preparing NSD data.")