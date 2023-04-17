class MultiHeadAttention(tf.keras.layers.Layer):
  """
  Usage:
  temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
  y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
  out, attn = temp_mha(y, k=y, q=y, mask=None)
  out.shape, attn.shape
  """
def split_heads(self, x, batch_size):
    """
    Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    
def point_wise_feed_forward_network(d_model, dff):
   """
    The purpose of this layer is to apply non-linear functions to the input data at
	each position independently and identically.
   """