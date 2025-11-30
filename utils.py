# utils.py
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def load_ply_as_array(filename):
    ply = PlyData.read(filename)
    arr = np.stack([
        ply['vertex'].data['x'],
        ply['vertex'].data['y'],
        ply['vertex'].data['z'],
        ply['vertex'].data['intensity']
    ], axis=-1)
    return arr

def write_ply_xyz_intensity(filename, coords, intensity, as_float=True):
    # coords: (N,3), intensity:(N,) or (N,1)
    intensity = intensity.reshape(-1)
    N = coords.shape[0]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
    data = np.empty(N, dtype=dtype)
    data['x'] = coords[:,0]
    data['y'] = coords[:,1]
    data['z'] = coords[:,2]
    data['intensity'] = intensity.astype(np.float32)
    el = PlyElement.describe(data, 'vertex')
    PlyData([el]).write(filename)

def plot_and_save_error_scatter(coords, values, outpng, title="error", elev=30, azim=45, s=1.5):
    # coords: (N,3), values: scalar per point
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=values, marker='o', s=s)
    ax.view_init(elev=elev, azim=azim)
    plt.title(title)
    plt.colorbar(sc, shrink=0.6)
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close(fig)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_loss_curve(history, outpath, title='Loss'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    if isinstance(history, dict):
        keys = list(history.keys())
        for k in keys:
            plt.plot(history[k], label=k)
    else:
        # Keras History object
        h = history.history
        for k,v in h.items():
            plt.plot(v, label=k)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
