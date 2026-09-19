import tensorflow as tf
import numpy as np

target_h, target_w = int(112), int(112)

def _read_image_or_pixels(p_tensor):
    p_str = p_tensor.numpy().decode("utf-8") if hasattr(p_tensor, "numpy") else str(p_tensor)
    vals = np.ones(100, dtype=np.float32)
    side = int(np.round(np.sqrt(len(vals))))
    img = vals.reshape(side, side, 1)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [target_h, target_w], method="bilinear")
    img = tf.image.grayscale_to_rgb(img)
    return img

def map_fn(p):
    return tf.py_function(func=_read_image_or_pixels, inp=[p], Tout=tf.float32)

ds = tf.data.Dataset.from_tensor_slices(tf.constant(["a", "b", "c"]))
ds = ds.map(map_fn)

for batch in ds:
    print(batch.shape)
