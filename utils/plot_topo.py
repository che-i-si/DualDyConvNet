import scipy.signal as sig
import numpy as np
from scipy.interpolate import griddata
import mne
import matplotlib.pyplot as plt

DEFAULT_CHNAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
                   'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

def findstr(s, L):
    """Find string in list of strings, returns indices.

    Args:
        s: query string
        L: list of strings to search
    Returns:
        x: list of indices where s is found in L
    """

    x = [i for i, l in enumerate(L) if (l==s)]
    return x

def read_xyz(filename):
    """Read EEG electrode locations in xyz format

    Args:
        filename: full path to the '.xyz' file
    Returns:
        locs: n_channels x 3 (numpy.array)
    """
    ch_names = []
    locs = []
    with open(filename, 'r') as f:
        l = f.readline()  # header line
        while l:
            l = f.readline().strip().split("\t")
            if (l != ['']):
                ch_names.append(l[4].replace(' ', ''))
                locs.append([float(l[1]), float(l[2]), float(l[3])])
            else:
                l = None
    return ch_names, np.array(locs)

def read_ced(filename):
    ch_names = []
    locs = []
    with open(r"D:\data\0_Data\0_Data_SMC\Description\Standard-10-10-Cap27.ced", "r") as f:
        l = f.readline()  # header line
        while l:
            l = f.readline().strip().split("\t")
            if (l != ['']):
                ch_names.append(l[1].replace(' ', ''))
                locs.append([float(l[4]), float(l[5]), float(l[6])])
            else:
                l = None

    return ch_names, np.array(locs)

def plot_topo(data, channels, locs, n_grid=64, nan=None):
    """Interpolate EEG topography onto a regularly spaced grid

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: integer, interpolate to n_grid x n_grid array, default=64
    Returns:
        data_interpol: cubic interpolation of EEG topography, n_grid x n_grid
                       contains nan values
    """
    locs /= np.linalg.norm(locs, 2, axis=1, keepdims=True)
    c = findstr('Cz', channels)[0]
    w = np.linalg.norm(locs-locs[c], 2, axis=1)
    arclen = np.arcsin(w/2.*np.sqrt(4.-w*w))
    phi_re = locs[:,0]-locs[c][0]
    phi_im = locs[:,1]-locs[c][1]
    tmp = phi_re + 1j*phi_im
    phi = np.angle(tmp)
    X = arclen*np.real(np.exp(1j*phi))
    Y = arclen*np.imag(np.exp(1j*phi))
    r = max([max(X),max(Y)])
    Xi = np.linspace(-r,r,n_grid)
    Yi = np.linspace(-r,r,n_grid)
    data_ip = griddata((X, Y), data, (Xi[None,:], Yi[:,None]), method='cubic')
    if nan is not None:
        data_ip = np.nan_to_num(data_ip, nan=nan, copy=True)
    return data_ip


def get_topo_epochs(data, chans, locs, n_grid=64, nan=None):
    topo_epochs = []
    B = data.shape[0]

    for i in range(B):
        topo_epochs.append(
            plot_topo(data[i], chans, locs, n_grid=n_grid, nan=nan)
        )
    topo_epochs = np.stack(topo_epochs, axis=0)
    return topo_epochs

# %% TRANSFORMATION
def devide_fbands(X:np.ndarray, freqs:np.ndarray, axis:int=1,
                  fbands:tuple=((1, 4), (4, 8), (8, 12), (12, 20), (20, 30), (30, 40))) -> np.ndarray:
    """
    fband 안의 값을 평균해서 반환
    ex) X (size: (B, freqs, N, T)) -> X_f (size: (B, len(fbands), N, T))
    :return:    np.ndarray
    """
    if X.shape[axis] != len(freqs):
        raise ValueError('X and freqs must have same length')
    X_c = np.swapaxes(X, axis, 1)
    # -----
    X_f = []
    for band in fbands:
        X_f.append(
            X_c[:, (freqs <= band[1])&(freqs >= band[0]), ...].mean(axis=1)
        )
    X_f = np.stack(X_f, axis=1)
    X_f = np.swapaxes(X_f, axis, 1)
    return X_f

def get_freqs(n, fmin, fmax, fs=500):
    freqs = np.fft.fftfreq(n=n, d=1/fs)
    freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]
    return freqs

# %%
def convert2topomat(channel_values: np.ndarray, xy_locs: np.ndarray, resolution=27, ):
    topo, _2 = mne.viz.plot_topomap(channel_values, cmap='RdBu_r', pos=xy_locs, res=resolution, show=False)
    array = topo.get_array().data.copy()  # (H, W)
    plt.close()

    return array


def prepare_topomat(data: np.ndarray, xy_locs: np.ndarray, resolution=27, ):
    """

    :param data: array (-1, N)
    :param xy_locs: array (N, 2)
    :param resolution: size of the grid
    :return: (-1, H, W)
    """

    topos = np.zeros((data.shape[0], resolution, resolution))
    for i in range(data.shape[0]):
        topos[i] = convert2topomat(data[i], xy_locs, resolution, )
    return topos

def get_xyz(chnames: list=None, kind: str = 'standard_1020', ):
    if chnames is None:
        chnames = DEFAULT_CHNAMES
    all_positions = mne.channels.make_standard_montage(kind).get_positions()['ch_pos']
    locs = [all_positions[ch] for ch in chnames]
    locs = np.stack(locs, axis=0)
    return locs

def plot_topomat(topomat: np.ndarray, pos: np.ndarray, show: bool = False, ax=None,
                 mask=None, sensors=True, contours=6, names: list | None = None, cmap='RdBu_r',
                 vmin=None, vmax=None, sphere=None, outlines='head', mask_params=None):
    """
    Plot topology map (circular) from topomat (2-d array)

    :param topomat: array (res, res)
    :param pos:     array (N, 2)
    :param mask:
    :param sensors:
    :param names:   Labels of EEG electrodes. If None, location is not marked.

    :return: ax:      Matplotlib Axes instance
    """
    import mne.viz.topomap as mne_topo
    from mne import defaults

    image_interp = defaults._INTERPOLATION_DEFAULT
    border = defaults._BORDER_DEFAULT
    extrapolate = defaults._EXTRAPOLATE_DEFAULT
    res = topomat.shape[0]

    # ===
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")

    # === Fig SETTINGS
    sphere = mne.viz.utils._check_sphere(sphere, pos if isinstance(pos, mne.Info) else None)
    extrapolate = mne_topo._check_extrapolate(extrapolate, 'eeg')
    cmap = mne.viz.utils._get_cmap(cmap)
    outlines = mne_topo._make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    assert isinstance(outlines, dict)

    mne_topo._prepare_topomap(pos, ax)

    # === 좌표 및 Interpolation 구성
    mask_params = defaults._handle_default("mask_params", mask_params)

    extent, Xi, Yi, interp = mne_topo._setup_interp(
        pos, res, image_interp, extrapolate, outlines, border
    )

    patch_ = mne_topo._get_patch(outlines, extrapolate, interp, ax)

    # === 등고선 그리기
    linewidth = mask_params["markeredgewidth"]
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
    #########등고선 원형으로 맞춰주기 위함
    xx, yy = np.meshgrid(np.arange(0, res, 1), np.arange(0, res, 1))
    # 원형 마스크 생성 (중심으로부터의 거리 계산)
    radius = (res * (0.148 / 0.3))
    if res % 2 == 1:
        distance = np.sqrt((xx - (res // 2)) ** 2 + (yy - (res // 2)) ** 2)
        cont_mask = (distance <= radius).astype(float)  # boolean을 0과 1로 변환
    else:
        distance = np.sqrt((xx - (res / 2 - 0.5)) ** 2 + (yy - (res / 2 - 0.5)) ** 2)
        cont_mask = (distance <= radius).astype(float)  # boolean을 0과 1로 변환
    cont_mask[cont_mask == 0] = np.nan
    ax.contour(Xi, Yi, topomat * cont_mask, contours, colors='k', linewidths=linewidth / 2.0)  # 등고선 그림

    # === 값 채우기
    im = ax.imshow(topomat, cmap=cmap, origin='lower', interpolation='bilinear',
                   extent=extent, aspect='equal', vmin=vmin, vmax=vmax,
                   )  # voxel 채움

    # ===
    cont = None
    if cont:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            cont = ax.contour(
                Xi, Yi, topomat, contours, colors="k", linewidths=linewidth / 2.0
            )

    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in mne_topo._cont_collections(cont):
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T
    # === 선 그리기?
    mask = mask.astype(bool, copy=False) if mask is not None else None  # None
    if sensors is not False and mask is None:
        mne_topo._topomap_plot_sensors(pos_x, pos_y, sensors=sensors, ax=ax)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        mne_topo._topomap_plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=ax)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)

    # === 머리 모양 그리기
    if isinstance(outlines, dict):
        mne_topo._draw_outlines(ax, outlines)

    # === 전극 이름 표시
    if names is not None:
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (_pos, _name) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            # axes.scatter(_pos[0], _pos[1], c='black', s=2, marker='o', label=_name)
            ax.text(
                _pos[0],
                _pos[1],
                _name,
                horizontalalignment="center",
                verticalalignment="center",
                size="small",
            )

    # ========= 원 외부 지우기
    if show:
        plt.show()
    return ax