# train.py
import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from models import build_lf_mlp, build_hf_mlp, build_hf_with_pointnet
from utils import load_ply_as_array, ensure_dir, save_loss_curve

# --------------- 配置区 ---------------
# 支持多组数据：每组是 (lf_path, hf_path)
dataset = [
    ("data/lf1.ply","data/hf1.ply"),
    #("data/lf2.ply","data/hf2.ply"),
]

train_mode = 'joint'  # 'separate' or 'joint'
use_pointnet_hf = True  # Whether to use pointnet-based HF head or simple MLP

outdir = 'models'
ensure_dir(outdir)

# hyperparams
batch_size = 4096
epochs = 800

# --------------- 读取并准备数据 ---------------
LF_X_list, LF_Y_list = [], []
HF_X_list, HF_Y_list = [], []
# For scalers, we will keep per-group scalers so we can inverse transform correctly later
scalers_per_group = []

for (lf_file, hf_file) in dataset:
    lf = load_ply_as_array(lf_file)
    hf = load_ply_as_array(hf_file)

    coords_lf, intens_lf = lf[:, :3], lf[:, 3:4]
    coords_hf, intens_hf = hf[:, :3], hf[:, 3:4]

    # per-group scalers
    sc_xyz_l = MinMaxScaler().fit(coords_lf)
    sc_i_l = MinMaxScaler().fit(intens_lf)
    sc_xyz_h = MinMaxScaler().fit(coords_hf)
    sc_i_h = MinMaxScaler().fit(intens_hf)
    scalers_per_group.append((sc_xyz_l, sc_i_l, sc_xyz_h, sc_i_h))

    coords_lf_n = sc_xyz_l.transform(coords_lf)
    intens_lf_n = sc_i_l.transform(intens_lf)

    coords_hf_n = sc_xyz_h.transform(coords_hf)
    intens_hf_n = sc_i_h.transform(intens_hf)

    # append full LF (we train LF on all its points)
    LF_X_list.append(coords_lf_n)
    LF_Y_list.append(intens_lf_n)

    # for HF, sample up to 3000 points per group to limit training cost
    idx = np.random.choice(len(coords_hf_n), min(3000, len(coords_hf_n)), replace=False)
    HF_X_list.append(coords_hf_n[idx])
    HF_Y_list.append(intens_hf_n[idx])

# stack across groups
LF_X = np.vstack(LF_X_list)
LF_Y = np.vstack(LF_Y_list)
HF_X = np.vstack(HF_X_list)
HF_Y = np.vstack(HF_Y_list)

print("LF samples:", LF_X.shape, "HF samples:", HF_X.shape)

# --------------- 模型 ---------------
lf_model = build_lf_mlp()
if use_pointnet_hf:
    hf_model = build_hf_with_pointnet()
else:
    hf_model = build_hf_mlp()

# --------------- 训练流程 ---------------
if train_mode == 'separate':
    print("Training LF model separately...")
    lf_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    history_lf = lf_model.fit(LF_X, LF_Y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    save_loss_curve(history_lf, os.path.join(outdir, 'loss_lf.png'), title='LF Loss')
    # HF training: compute LF predictions on HF coords (note: LF and HF coords are normalized differently per group)
    # We need to predict per-group: map HF coords through the group's LF scaler? Here we trained LF on per-group normalized coords combined,
    # so we assume LF model expects coords normalized the same way as LF_X (we used per-group scalers before stacking).
    # For simplicity we will predict HF LF-values by applying corresponding group's LF scaler: we'll reconstruct inputs per group below.
    # --- construct lf predictions for HF samples by group
    lf_preds_for_hf_list = []
    start = 0
    # reconstruct group sizes of HF_X_list
    for g_idx, (coords_hf_group) in enumerate(HF_X_list):
        n = coords_hf_group.shape[0]
        # we trained LF model on stacked per-group LF_X, the LF model expects coords normalized in the per-group LF scaler space.
        # But coords_hf_group was normalized with its own HF scaler. To get proper LF prediction, transform HF coords back to real coords
        # then re-normalize with the LF scaler for that group.
        sc_xyz_l, sc_i_l, sc_xyz_h, sc_i_h = scalers_per_group[g_idx]
        # inverse HF normalized coords -> real coords:
        real_coords_hf = sc_xyz_h.inverse_transform(coords_hf_group)
        coords_for_lf = sc_xyz_l.transform(real_coords_hf)
        lf_pred = lf_model.predict(coords_for_lf, batch_size=1024)
        lf_preds_for_hf_list.append(lf_pred)
        start += n
    lf_preds_for_hf = np.vstack(lf_preds_for_hf_list)
    hf_input = np.hstack([HF_X, lf_preds_for_hf])
    print("Training HF model...")
    hf_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    history_hf = hf_model.fit(hf_input, HF_Y, epochs=epochs, batch_size=1024, validation_split=0.2, verbose=1)
    save_loss_curve(history_hf, os.path.join(outdir, 'loss_hf.png'), title='HF Loss')
else:
    # joint training
    print("Joint training LF and HF models...")
    # We'll build a Keras Model that takes two inputs: LF coords (many points) and HF coords (sampled),
    # outputs LF predictions for LF coords and HF predictions for HF coords.
    inp_l = tf.keras.Input(shape=(3,), name='lf_coords')      # will feed LF_X
    inp_h = tf.keras.Input(shape=(3,), name='hf_coords')      # will feed HF_X
    lf_out = lf_model(inp_l)  # LF prediction for LF points
    # For HF head, we need lf_model applied to hf coords to form auxiliary feature:
    lf_on_h = lf_model(inp_h)  # shared weights
    if use_pointnet_hf:
        # hf_model expects [coords, lf_pred] as inputs
        # build a small wrapper that accepts inp_h and lf_on_h
        hf_out = hf_model([inp_h, lf_on_h])
    else:
        hf_input = tf.keras.layers.Concatenate()([inp_h, lf_on_h])
        hf_out = hf_model(hf_input)
    joint = tf.keras.Model(inputs=[inp_l, inp_h], outputs=[lf_out, hf_out])
    joint.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=['mse','mse'],
                  loss_weights=[1.0, 1.5])
    history = joint.fit([LF_X, HF_X], [LF_Y, HF_Y], epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    save_loss_curve(history, os.path.join(outdir, 'loss_joint.png'), title='Joint Loss')

# --------------- 保存模型与 scalers ---------------
lf_model.save(os.path.join(outdir, 'lf_model'))
hf_model.save(os.path.join(outdir, 'hf_model'))
joblib.dump(scalers_per_group, os.path.join(outdir, 'scalers_per_group.pkl'))
print("Saved models and scalers to", outdir)
