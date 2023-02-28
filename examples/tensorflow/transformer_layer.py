import argparse
import tensorflow as tf
import time
import transformer_engine.tensorflow as te

from keras import layers
from keras import Model
from typing import Optional

parser = argparse.ArgumentParser(description="Benchmark TransformerLayer.")
parser.add_argument(
    '-t', '--type', type=int, default=0,
    help="""Pick TE implementation (0:all|1:TF-fp16|2:TE-fp16|3:TE-fp8)""")
args, _ = parser.parse_known_args()
tl_type = args.type

tf.keras.mixed_precision.set_global_policy('mixed_float16')

dropout_rate = 0.0

class DotProductAttention(tf.keras.Model):
  """Attention operation in Transformer layer
  """
  def __init__(
      self,
      num_attention_heads: int,
      kv_channels: int,
      attention_dropout: float,
  ):
    super().__init__()
    self.projection_size = kv_channels * num_attention_heads
    self.hidden_size_per_attention_head = float(kv_channels)
    self.norm_factor = tf.math.sqrt(self.hidden_size_per_attention_head)
    self.dropout = layers.Dropout(attention_dropout)
    if self.dropout.dtype_policy.name == 'mixed_float16':
      self.norm_factor = tf.cast(self.norm_factor, dtype=tf.float16)

  def masked_softmax(
      self,
      inp: tf.Tensor,
      mask: Optional[tf.Tensor]
  ) -> tf.Tensor:
    if mask is not None:
      inp = tf.where(mask, -10000.0, inp)
    return tf.nn.softmax(inp, axis=-1)

  def call(
      self,
      query: tf.Tensor,
      key: tf.Tensor,
      value: tf.Tensor,
      attention_mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    b = query.shape[1]
    np = query.shape[2]
    sq = query.shape[0]
    sk = key.shape[0]
    hn = value.shape[3]

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query = tf.reshape(query, shape=(sq, b * np, hn))
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key = tf.reshape(key, shape=(sk, b * np, hn))

    bmm1 = tf.matmul(tf.transpose(query, perm=(1, 0, 2)),
                     tf.transpose(key, perm=(1, 2, 0))) / self.norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = tf.reshape(bmm1, shape=(b, np, sq, sk))

    attention_probs = self.masked_softmax(attention_scores, attention_mask)

    attention_probs = self.dropout(attention_probs)

    # change view [sk, b * np, hn]
    value = tf.reshape(value, shape=(sk, b * np, hn))

    # change view [b * np, sq, sk]
    attention_probs = tf.reshape(attention_probs, shape=(b * np, sq, sk))

    # matmul: [b * np, sq, hn]
    context = tf.matmul(attention_probs,
                        tf.transpose(value, perm=(1, 0, 2)))

    # change view [b, np, sq, hn]
    context = tf.reshape(context, shape=(b, np, sq, hn))

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context = tf.transpose(context, perm=(2, 0, 1, 3))

    # [sq, b, np, hn] --> [sq, b, hp]
    context = tf.reshape(context, shape=(sq, b, self.projection_size))

    return context

class BasicMLP(tf.keras.Model):
  """Feed-forward network in Transformer layer
  """
  def __init__(
      self,
      hidden_size: int,
      ffn_hidden_size: int,
  ):
    super().__init__()
    self.linear1 = layers.Dense(ffn_hidden_size, use_bias=True)
    self.linear2 = layers.Dense(hidden_size, use_bias=True)

  def call(
      self,
      x: tf.Tensor
  ) -> tf.Tensor:
    x = self.linear1(x)
    x = tf.nn.gelu(x, approximate=True)
    x = self.linear2(x)
    return x

class BasicTransformer(tf.keras.Model):
  def __init__(
      self,
      hidden_size: int,
      ffn_hidden_size: int,
      num_attention_heads: int,
      layernorm_eps: int = 1e-5,
      attention_dropout: float = 0.1,
      hidden_dropout: float = 0.1,
  ):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.kv_channels = hidden_size // num_attention_heads
    self.ln1 = layers.LayerNormalization(epsilon=layernorm_eps)
    self.qkv_projection = layers.Dense(3 * hidden_size, use_bias=True)
    self.attention = DotProductAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=self.kv_channels,
        attention_dropout=attention_dropout,
    )
    self.projection = layers.Dense(hidden_size, use_bias=True)
    self.dropout = layers.Dropout(hidden_dropout)
    self.ln2 = layers.LayerNormalization(epsilon=layernorm_eps)
    self.mlp = BasicMLP(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
    )

  def call(
      self,
      x: tf.Tensor,
      attention_mask: tf.Tensor,
  ) -> tf.Tensor:
    res = x
    x = self.ln1(x)

    # Fused QKV projection
    qkv = self.qkv_projection(x)
    qkv_shape = qkv.shape
    qkv = tf.reshape(qkv,
                     shape=(qkv_shape[0], qkv_shape[1],
                     self.num_attention_heads, 3 * self.kv_channels))
    q, k, v = tf.split(qkv, 3, axis=3)

    x = self.attention(q, k, v, attention_mask)
    x = self.projection(x)
    x = self.dropout(x)
    x = res + x
    res = x
    x = self.ln2(x)
    x = self.mlp(x)

    return x + res

class FusedTETransformer(tf.keras.Model):
  def __init__(
      self,
      hidden_size: int,
      ffn_hidden_size: int,
      num_attention_heads: int,
      layernorm_eps: int = 1e-5,
      attention_dropout: float = 0.1,
      hidden_dropout: float = 0.1,
  ):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.kv_channels = hidden_size // num_attention_heads
    self.ln_qkv = te.LayerNormDense(3 * hidden_size, epsilon=layernorm_eps,
                                    use_bias=True)
    self.attention = DotProductAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=self.kv_channels,
        attention_dropout=attention_dropout,
    )
    self.projection = te.Dense(hidden_size, use_bias=True)
    self.dropout = layers.Dropout(hidden_dropout)
    self.ln_mlp = te.LayerNormMLP(ffn_hidden_size, hidden_size,
                                  epsilon=layernorm_eps, use_bias=True,
                                  return_layernorm_output=False)

  def call(
      self,
      x: tf.Tensor,
      attention_mask: tf.Tensor,
  ) -> tf.Tensor:
    res = x
    qkv = self.ln_qkv(x)

    # Split qkv into query, key and value
    qkv_shape = qkv.shape
    qkv = tf.reshape(qkv,
                     shape=(qkv_shape[0], qkv_shape[1],
                     self.num_attention_heads, 3 * self.kv_channels))
    q, k, v = tf.split(qkv, 3, axis=3)

    x = self.attention(q, k, v, attention_mask)
    x = self.projection(x)
    x = self.dropout(x)
    x = res + x
    res = x
    x = self.ln_mlp(x)

    return x + res

# Layer configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 4
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = tf.float32

def speedometer(
    model: tf.keras.Model,
    input: tf.Tensor,
    forward_kwargs: dict = {},
    fp8_autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
) -> None:
  """Measure average run time for a TF model

  Performs forward and backward passes.
  """
  if fp8_autocast_kwargs is None:
      fp8_autocast_kwargs = { "enabled": False }

  p = tf.constant(0.)  # Create small tensor to force GPU resync 

  # Warmup runs
  for _ in range(warmup_iters):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(input)
      with te.fp8_autocast(**fp8_autocast_kwargs):
        output = model(input, **forward_kwargs)
      loss = tf.reduce_sum(output)
    dx, dvars = tape.gradient(loss, [input, model.variables])

  (p + 1.).numpy() # Sync the GPU

  # Timing runs
  start = time.time()
  for _ in range(timing_iters):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(input)
      with te.fp8_autocast(**fp8_autocast_kwargs):
        output = model(input, **forward_kwargs)
      loss = tf.reduce_sum(output)
    dx, dvars = tape.gradient(loss, [input, model.variables])

  (p + 1.).numpy() # Sync the GPU
  end = time.time()

  elapsed_time = (end - start) / timing_iters * 1000

  print(f"Mean time: {elapsed_time} ms")

tf.random.set_seed(12)
tf.keras.utils.set_random_seed(1)
# Synthetic data
x = tf.random.normal(shape=(sequence_length, batch_size, hidden_size),
                     dtype=dtype)

basic_transformer = BasicTransformer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    attention_dropout=dropout_rate,
    hidden_dropout=dropout_rate,
)

y = basic_transformer(x, attention_mask=None)
print(y[-1])

if tl_type in (0, 1):
  speedometer(
      basic_transformer,
      x,
      forward_kwargs = { "attention_mask": None, "training": True },
  )

te_transformer = FusedTETransformer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    attention_dropout=dropout_rate,
    hidden_dropout=dropout_rate,
)

from transformer_engine.tensorflow import Format, DelayedScaling

fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.HYBRID,
                            amax_compute_algo='max', amax_history_len=16)

# Run once to build the variables.
te_transformer(x, attention_mask=None)
# Sync the variables with the reference.
for v0, v1 in zip(basic_transformer.variables, te_transformer.variables):
  v1.assign(v0)
  tf.debugging.assert_near(v1, v0)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
  y_te = te_transformer(x, attention_mask=None)
print(y_te[-1])

if tl_type in (0, 3):
  speedometer(
      te_transformer,
      x,
      forward_kwargs = { "attention_mask": None, "training": True },
      fp8_autocast_kwargs = { "enabled": True, "fp8_recipe": fp8_recipe },
  )

y_te = te_transformer(x, attention_mask=None)
print(y_te[-1])

if tl_type in (0, 2):
  speedometer(
      te_transformer,
      x,
      forward_kwargs = { "attention_mask": None, "training": True },
      fp8_autocast_kwargs = { "enabled": False, "fp8_recipe": None },
  )

