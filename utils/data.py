import numpy as np
import h5py


__all__ = ["get_dataset", "random_split_dataset", "short_edge_section_split_dataset", "long_edge_section_split_dataset", "within_area_split_dataset"]


def get_dataset(dataset_dir):
    h_filename = 'h_Estimated_CTW_Train.h5'
    h_path = dataset_dir.joinpath(h_filename)
    assert h_path.exists(), f'{h_filename} is missing!'
    pos_filename = 'r_Position_CTW_Train.h5'
    pos_path = dataset_dir.joinpath(pos_filename)
    assert pos_path.exists(), f'{pos_filename} is missing!'
    #snr_filename = 'SNR_Est_test.pickel'  #snr_filename = 'SNR_CTW_test.h5'
    #snr_path = dataset_dir.joinpath(snr_filename)
    #assert snr_path.exists(), f'{snr_filename} is missing!'
    kwargs = dict(mode='r', swmr=True)

    with h5py.File(h_path, **kwargs) as hf:
        h = hf['h_Estimated'][:].T
        h = h.astype(np.float32)

        # Fix #1: Correct the order of FFT components. In Data: (1 to 511, -512 to 0)
        h = np.fft.fftshift(h, axes=2)

        assert h.shape[1:] == (16, 924, 2)

    #with h5py.File(snr_path, **kwargs) as hf:
    #    snr = hf['SNR_Est'][:].T
    #    snr = snr.astype(np.float32)
    #    assert snr.shape[1:] == (16, )

    with h5py.File(pos_path, **kwargs) as hf:
        pos = hf['r_Position'][:].T
        pos = pos.astype(np.float32)

        # Fix #2: Correction of position data. Antenna will now be in the center.
        offset = (3.5, -3.15, 1.8)
        pos[:,0] -= offset[0]
        pos[:,1] -= offset[1]
        pos[:,2] -= offset[2]
        assert pos.shape[1:] == (3, )
    return h, pos


def random_split_dataset(dataset_dir, split=0.9, shuffle=True, memory=None, verbose=True, random_state=None):
    _get_dataset = memory.cache(get_dataset) if memory else get_dataset

    h, pos = _get_dataset(dataset_dir=dataset_dir)

    # Fix #1: remove the Z lower than 2.31(The height of the table)
    idx = np.argwhere(abs(pos[:,2]) > 2.31).flatten()
    pos = pos[idx]
    h = h[idx]

    # Random generator
    rng = np.random.default_rng(seed=random_state)

    # How many samples are we dealing with
    n_samples = len(pos)

    # We will operate over indexes instead with dataset(s) directly
    idx = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

    # if there is only 1 split make it as array for consistency
    if isinstance(split, (float, int)):
        split = (split, )

    # define split indexes
    split = [int(s * n_samples) for s in split]

    # add zero and n_samples for consistency
    split = (0, *split, n_samples)

    # Finally construct batches
    output = []
    for i in range(1, len(split)):
        start_idx = split[i-1]
        end_idx = split[i]
        batch = idx[start_idx : end_idx]
        output.append((h[batch], pos[batch]))
    output = list(output)

    # Sanity check before exits
    assert len(output) == (len(split) - 1), f'{len(output)} != {len(split) - 1}'
    assert sum(map(lambda item: len(item[-1]), output)) == n_samples
    #if verbose: print(f'Total: {n_samples}; Train: {split/n_samples*100:.2f}%; Test: {(n_samples - split)/n_samples*100:.2f}%')
    return output


def short_edge_section_split_dataset(dataset_dir, shuffle=True, memory=None, verbose=True, random_state=None):
    r""" narrow split
    """
    _get_dataset = memory.cache(get_dataset) if memory else get_dataset

    h, pos = _get_dataset(dataset_dir=dataset_dir)

    idx = np.argwhere(abs(pos[:,2]) > 2.31).flatten()
    pos = pos[idx]
    h = h[idx]
    # print("pos num = ", len(pos))
    # print("h num = ", len(h))
    # Random generator
    rng = np.random.default_rng(seed=random_state)

    # How many samples are we dealing with
    n_samples = len(pos)

    # We will operate over indexes instead with dataset(s) directly
    idx = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

    # Reshuffle
    h, pos = h[idx], pos[idx]

    
    x, y = pos[...,0], pos[...,1]
    threshold = -1.0 * x + 1.9583

    train_idx = np.argwhere(y > threshold).flatten()

    # train mask
    mask = np.zeros(n_samples, np.bool_)
    mask[train_idx] = 1

    if verbose == True:
        print(f'Total: {n_samples}; Train: {sum(mask)/n_samples*100:.2f}%; Test: {sum(~mask)/n_samples*100:.2f}%')

    assert len(pos[mask]) > len(pos[~mask])

    return (
        (h[mask], pos[mask]),
        (h[~mask], pos[~mask]),
    )


def long_edge_section_split_dataset(dataset_dir, shuffle=True, memory=None, verbose=True, random_state=None):
    r""" wide split
    """
    _get_dataset = memory.cache(get_dataset) if memory else get_dataset

    h, pos = _get_dataset(dataset_dir=dataset_dir)

    # Random generator
    rng = np.random.default_rng(seed=random_state)

    # How many samples are we dealing with
    n_samples = len(pos)

    # We will operate over indexes instead with dataset(s) directly
    idx = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

    # Reshuffle
    h, pos = h[idx], pos[idx]


    x, y = pos[...,0], pos[...,1]
    threshold = 0.95 * x + 4.1445

    train_idx = np.argwhere(y < threshold).flatten()

    # train mask
    mask = np.zeros(n_samples, np.bool_)
    mask[train_idx] = 1

    if verbose: print(f'Total: {n_samples}; Train: {sum(mask)/n_samples*100:.2f}%; Test: {sum(~mask)/n_samples*100:.2f}%')

    assert len(pos[mask]) > len(pos[~mask])

    return (
        (h[mask], pos[mask]),
        (h[~mask], pos[~mask]),
    )


def within_area_split_dataset(dataset_dir, shuffle=True, memory=None, verbose=True, random_state=None):
    _get_dataset = memory.cache(get_dataset) if memory else get_dataset

    h, pos = _get_dataset(dataset_dir=dataset_dir)

    # Random generator
    rng = np.random.default_rng(seed=random_state)

    # How many samples are we dealing with
    n_samples = len(pos)

    # We will operate over indexes instead with dataset(s) directly
    idx = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

    # Reshuffle
    h, pos = h[idx], pos[idx]


    # Follow equation:
    # r = 0.4
    # t = np.linspace(0, 1, 360)
    # x = r * np.cos(2*np.pi*t) + 0.8
    # y = r * np.sin(2*np.pi*t) + 4
    x, y = pos[...,0], pos[...,1]

    radius = 0.48
    conditions = np.sqrt((y - 4)**2 + (x - 0.8)**2) > radius

    train_idx = np.argwhere(conditions).flatten()

    # train mask
    mask = np.zeros(n_samples, np.bool_)
    mask[train_idx] = 1

    if verbose: print(f'Total: {n_samples}; Train: {sum(mask)/n_samples*100:.2f}%; Test: {sum(~mask)/n_samples*100:.2f}%')

    assert len(pos[mask]) > len(pos[~mask])

    return (
        (h[mask], pos[mask]),
        (h[~mask], pos[~mask]),
    )