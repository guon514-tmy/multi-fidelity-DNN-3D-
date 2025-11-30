# models.py
import tensorflow as tf

def build_lf_mlp(input_dim=3, hidden=[64,64,64]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for h in hidden:
        model.add(tf.keras.layers.Dense(h, activation='tanh'))
    model.add(tf.keras.layers.Dense(1))
    return model

def build_hf_mlp(input_dim=4, hidden=[64,64,64]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for h in hidden:
        model.add(tf.keras.layers.Dense(h, activation='tanh'))
    model.add(tf.keras.layers.Dense(1))
    return model

# -------------------------
# A compact PointNet-like encoder for point coordinates only.
# Note: This is a lightweight PointNet inspired block (not full paper code),
# designed to extract permutation-invariant features from point sets.
# We use it per-point (shared MLP) + global max-pooling.
# -------------------------
def pointnet_encoder(point_dim=3, mlp_layers=[64,128,256]):
    inp = tf.keras.Input(shape=(None, point_dim))  # (B, N, 3)
    x = inp
    for ch in mlp_layers:
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(ch, activation='relu'))(x)
    # per-point features now (B,N,C). Global feature:
    global_feat = tf.keras.layers.GlobalMaxPooling1D()(x)  # (B, C)
    # replicate global back to per-point if needed; here we'll return global only
    model = tf.keras.Model(inputs=inp, outputs=global_feat, name='pointnet_encoder')
    return model

# Use PointNet features concatenated with coords and LF-pred for HF head
def build_hf_with_pointnet(point_dim=3, mlp_point=[64,128], hf_head=[128,64]):
    # per-point input (single point treated as N=1 for encoder: make shape (B,1,3))
    coords_in = tf.keras.Input(shape=(3,), name='coords_in')
    lf_pred_in = tf.keras.Input(shape=(1,), name='lf_pred_in')
    # create a 1-point sequence for encoder
    coords_seq = tf.keras.layers.Reshape((1,3))(coords_in)  # (B,1,3)
    encoder = pointnet_encoder(point_dim=3, mlp_layers=mlp_point)
    global_feat = encoder(coords_seq)  # (B, C)
    x = tf.keras.layers.Concatenate()([coords_in, lf_pred_in, global_feat])
    for h in hf_head:
        x = tf.keras.layers.Dense(h, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=[coords_in, lf_pred_in], outputs=out, name='hf_pointnet')
    return model
