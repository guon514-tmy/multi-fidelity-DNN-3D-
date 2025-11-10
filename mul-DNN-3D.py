import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from plyfile import PlyData
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ============================================================
# 1. 读取 .ply 文件并做归一化
# ============================================================

def load_ply_as_array(filename):
    ply = PlyData.read(filename)
    data = np.stack([
        ply['vertex'].data['x'],
        ply['vertex'].data['y'],
        ply['vertex'].data['z'],
        ply['vertex'].data['intensity']
    ], axis=-1)
    return data

# ⚠️ 修改为你的文件路径
lf_file = 'low_fidelity.ply'
hf_file = 'high_fidelity.ply'

lf_data = load_ply_as_array(lf_file)
hf_data = load_ply_as_array(hf_file)

# 检查坐标是否完全一致
assert np.allclose(lf_data[:, :3], hf_data[:, :3], atol=1e-8)

coords = lf_data[:, :3]
lf_intensity = lf_data[:, 3:4]
hf_intensity = hf_data[:, 3:4]

N = coords.shape[0]
print(f"Loaded {N} points from PLY files")

# ============================================================
# 2. 数据归一化
# ============================================================

scaler_xyz = MinMaxScaler()
scaler_lf = MinMaxScaler()
scaler_hf = MinMaxScaler()

coords_norm = scaler_xyz.fit_transform(coords)
lf_norm = scaler_lf.fit_transform(lf_intensity)
hf_norm = scaler_hf.fit_transform(hf_intensity)

# ============================================================
# 3. 数据划分
# ============================================================

x_train_lf = coords_norm
y_train_lf = lf_norm

N2 = min(3000, N // 30)
idx_hf = np.random.choice(N, N2, replace=False)
x_train_hf = coords_norm[idx_hf]
y_train_hf = hf_norm[idx_hf]

print(f"Using {N2} high-fidelity samples for HF training")

# ============================================================
# 4. 模型定义函数
# ============================================================

def build_lf_net():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    return model

def build_hf_net():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    return model

# ============================================================
# 5. 训练低保真网络
# ============================================================

model_lf = build_lf_net()
model_lf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

early_stop_lf = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True
)

print("Training Low-Fidelity Network...")
history_lf = model_lf.fit(
    x_train_lf, y_train_lf,
    batch_size=1024,
    epochs=2000,
    validation_split=0.1,
    callbacks=[early_stop_lf],
    verbose=1
)

# 可视化损失
plt.figure(figsize=(6, 5))
plt.plot(history_lf.history['loss'], label='Train Loss')
plt.plot(history_lf.history['val_loss'], label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Low-Fidelity Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve_LF.png', dpi=300)
plt.show()

print(f"Best LF val_loss = {min(history_lf.history['val_loss']):.3e}")

# ============================================================
# 6. 训练高保真修正网络
# ============================================================

# 生成低保真预测作为输入
lf_on_hf = model_lf.predict(x_train_hf, batch_size=1024)
x_input_hf = np.hstack([x_train_hf, lf_on_hf])

model_hf = build_hf_net()
model_hf.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

early_stop_hf = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True
)

print("Training High-Fidelity Correction Network...")
history_hf = model_hf.fit(
    x_input_hf, y_train_hf,
    batch_size=512,
    epochs=5000,
    validation_split=0.2,
    callbacks=[early_stop_hf],
    verbose=1
)

# 可视化损失
plt.figure(figsize=(6, 5))
plt.plot(history_hf.history['loss'], label='Train Loss')
plt.plot(history_hf.history['val_loss'], label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('High-Fidelity Correction Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve_HF.png', dpi=300)
plt.show()

print(f"Best HF val_loss = {min(history_hf.history['val_loss']):.3e}")

# ============================================================
# 7. 全场预测与保存
# ============================================================

lf_pred = model_lf.predict(coords_norm, batch_size=2048)
hf_input = np.hstack([coords_norm, lf_pred])
hf_pred = model_hf.predict(hf_input, batch_size=2048)

hf_pred_real = scaler_hf.inverse_transform(hf_pred)
mse = np.mean((hf_intensity - hf_pred_real) ** 2)
print(f'Overall HF prediction MSE: {mse:.4e}')

# ============================================================
# 8. 保存模型与归一化器
# ============================================================

os.makedirs('models', exist_ok=True)
model_lf.save('models/lf_model.h5')
model_hf.save('models/hf_model.h5')
joblib.dump((scaler_xyz, scaler_lf, scaler_hf), 'models/scalers.pkl')
print("Models and scalers saved in ./models/")
