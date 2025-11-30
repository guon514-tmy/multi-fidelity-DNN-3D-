# infer.py
import argparse
import os
import numpy as np
import joblib
from plyfile import PlyData
from utils import load_ply_as_array, write_ply_xyz_intensity, plot_and_save_error_scatter, ensure_dir
import tensorflow as tf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hf', required=True, help='HF ply file path (will predict on this coordinate set)')
    p.add_argument('--lf', required=False, default=None, help='Optional matching LF ply (if available) to compute LF_pred properly)')
    p.add_argument('--models_dir', default='models')
    p.add_argument('--outdir', default='results')
    args = p.parse_args()

    ensure_dir(args.outdir)

    # load HF point cloud
    hf = load_ply_as_array(args.hf)
    coords_hf_real = hf[:, :3]
    hf_int_real = hf[:, 3:4]

    # load models & scalers
    lf_model = tf.keras.models.load_model(os.path.join(args.models_dir, 'lf_model'))
    hf_model = tf.keras.models.load_model(os.path.join(args.models_dir, 'hf_model'))
    scalers_per_group = joblib.load(os.path.join(args.models_dir, 'scalers_per_group.pkl'))

    # If user provides a corresponding LF ply, try to find the group whose HF scaler best matches
    # Otherwise we use the first group's HF scaler as fallback (user should ensure consistent ordering).
    sc_for_use = scalers_per_group[0]
    if args.lf is not None:
        # try to match exact filenames (basic heuristic)
        # fallback: use first group
        for (sc_xyz_l, sc_i_l, sc_xyz_h, sc_i_h) in scalers_per_group:
            # We can check bounding box overlap: transform coords_hf_real into candidate hf-normalized and inverse -> compare
            sc_for_use = (sc_xyz_l, sc_i_l, sc_xyz_h, sc_i_h)
            break

    sc_xyz_l, sc_i_l, sc_xyz_h, sc_i_h = sc_for_use

    # normalize HF coords with its HF scaler (used for measuring error/visualization consistency)
    coords_hf_n = sc_xyz_h.transform(coords_hf_real)
    # get LF prediction for HF coords:
    # To feed into LF model, we must express coords in the LF scaler space (LF model was trained on coords normalized by sc_xyz_l)
    coords_hf_in_l_space = sc_xyz_l.transform(coords_hf_real)
    lf_pred_on_hf = lf_model.predict(coords_hf_in_l_space, batch_size=2048)

    # Now feed HF model. hf_model may expect [coords, lf_pred] (PointNet wrapper) or a single concatenated vector.
    try:
        hf_pred_n = hf_model.predict([coords_hf_in_l_space, lf_pred_on_hf], batch_size=2048)
    except Exception:
        # maybe hf_model expects concatenated normalized coords in HF normal space + lf_pred_on_hf
        # We'll build input as HF coords normalized with HF scaler and LF pred (both normalized or not?)
        inp = np.hstack([coords_hf_n, lf_pred_on_hf])
        hf_pred_n = hf_model.predict(inp, batch_size=2048)

    # hf_pred_n is normalized by HF intensity scaler sc_i_h (since we used that scaler in training)
    hf_pred_real = sc_i_h.inverse_transform(hf_pred_n)

    # compute errors
    mse = np.mean((hf_pred_real - hf_int_real) ** 2)
    print(f"HF prediction MSE on provided HF file: {mse:.6e}")

    # save PLYs:
    #  - predicted intensity as PLY (same coords)
    write_ply_xyz_intensity(os.path.join(args.outdir, 'hf_coords_predicted.ply'), coords_hf_real, hf_pred_real)
    #  - error map as intensity
    error_vals = np.abs((hf_pred_real - hf_int_real)).reshape(-1)
    write_ply_xyz_intensity(os.path.join(args.outdir, 'hf_coords_error.ply'), coords_hf_real, error_vals)

    # visualizations: color by error
    plot_and_save_error_scatter(coords_hf_real, error_vals, os.path.join(args.outdir, 'error_scatter.png'), title=f'Error (MSE={mse:.4e})')

    # also save a comparison png arrays (pred vs gt) using simple projection for quick check
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.hist(hf_int_real.reshape(-1), bins=80)
    plt.title('HF ground-truth intensity')
    plt.subplot(1,2,2)
    plt.hist(hf_pred_real.reshape(-1), bins=80)
    plt.title('Predicted HF intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'hist_pred_vs_gt.png'), dpi=200)
    plt.close()

    print("Outputs written to", args.outdir)

if __name__ == '__main__':
    main()
