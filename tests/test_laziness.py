# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import shutil
from typing import Tuple

import pytest

from streaming import MDSWriter, StreamingDataset


@pytest.mark.usefixtures('remote_local')
def test_laziness(remote_local: Tuple[str, str]):
    num_samples = 100_000
    remote, local = remote_local
    columns = {'value': 'int'}
    compression = None
    hashes = None
    size_limit = 10_000

    with MDSWriter(remote, columns, compression, hashes, size_limit) as out:
        for i in range(num_samples):
            sample = {'value': i}
            out.write(sample)

    # Verify __getitem__ accesses.
    dataset = StreamingDataset(local=remote)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i

    # Verify __iter__ -> __getitem__ accesses.
    dataset = StreamingDataset(local=remote, shared_dir_seed=1234)
    for i, sample in zip(range(num_samples), dataset):
        assert sample['value'] == i

    # Verify __getitem__ downloads/accesses.
    dataset = StreamingDataset(local=local, remote=remote, shared_dir_seed=5678)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i

    shutil.rmtree(local)

    # Verify __iter__ -> __getitem__ downloads/accesses.
    dataset = StreamingDataset(local=local, remote=remote, shared_dir_seed=910)
    for i, sample in zip(range(num_samples), dataset):
        assert sample['value'] == i

    # Re-verify __getitem__ downloads/accesses.
    dataset = StreamingDataset(local=local, remote=remote, shared_dir_seed=2468)
    for i in range(num_samples):
        sample = dataset[i]
        assert sample['value'] == i
