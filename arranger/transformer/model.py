"""Transformer model."""
import numpy as np
import tensorflow as tf

# pylint:disable=arguments-differ


def get_angles(pos, i, d_model):
    """Copied from https://www.tensorflow.org/tutorials/text/transformer ."""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """Copied from https://www.tensorflow.org/tutorials/text/transformer ."""
    pos_encoding = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )

    # apply sin to even indices in the array; 2i
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

    return tf.cast(pos_encoding, dtype=tf.float32)  # pylint: disable=all


def scaled_dot_product_attention(q, k, v, mask):  # noqa
    """Calculate the attention weights.

    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights

    Copied from https://www.tensorflow.org/tutorials/text/transformer .
    """

    matmul_qk = tf.matmul(
        q, k, transpose_b=True
    )  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):  # noqa
    """Copied from https://www.tensorflow.org/tutorials/text/transformer ."""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):  # noqa
        """Split the last dimension into (num_heads, depth). Transpose the
        result such that the shape is (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):  # noqa
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention -> (batch_size, num_heads, seq_len_q, depth)
        # attention_weights -> (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(
            concat_attention
        )  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """Copied from https://www.tensorflow.org/tutorials/text/transformer ."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dff, activation="relu"
            ),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class TransformerLayer(tf.keras.layers.Layer):
    """Copied from https://www.tensorflow.org/tutorials/text/transformer ."""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):  # noqa
        attn_output, _ = self.mha(
            x, x, x, mask
        )  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            x + attn_output
        )  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class InputLayer(tf.keras.layers.Layer):
    """Input layer."""

    def __init__(
        self,
        max_len: int,
        use_duration: bool,
        use_frequency: bool,
        use_onset_hint: bool,
        use_pitch_hint: bool,
        use_pitch_embedding: bool,
        use_time_embedding: bool,
        use_duration_embedding: bool,
        max_beat: int,
        max_duration: int,
        n_tracks: int,
        use_lookahead_mask: bool,
    ):
        super().__init__()
        self.max_len = max_len
        self.use_duration = use_duration
        self.use_frequency = use_frequency
        self.use_onset_hint = use_onset_hint
        self.use_pitch_hint = use_pitch_hint
        self.use_pitch_embedding = use_pitch_embedding
        self.use_time_embedding = use_time_embedding
        self.use_duration_embedding = use_duration_embedding
        self.max_beat = max_beat
        self.max_duration = max_duration
        self.n_tracks = n_tracks
        self.use_lookahead_mask = use_lookahead_mask

        # Embedding
        if use_pitch_embedding:
            self.pitch_embedding = tf.keras.layers.Embedding(
                129, 16, name="pitch_embedding"
            )
        if use_time_embedding:
            self.time_embedding_position = tf.keras.layers.Embedding(
                24, 16, name="time_embedding_position"
            )
            self.time_embedding_beat = tf.keras.layers.Embedding(
                max_beat + 1,
                16,
                weights=[positional_encoding(max_beat + 1, 16)],
                trainable=False,
                name="time_embedding_beat",
            )
        if use_duration and use_duration_embedding:
            self.duration_embedding = tf.keras.layers.Embedding(
                max_duration + 1, 16, name="duration_embedding"
            )

        # Frequency matrix
        if use_frequency:
            frequency_matrix = 440.0 * 2 ** ((np.arange(128) - 69) / 12)
            frequency_matrix = np.pad(frequency_matrix, (1, 0))
            frequency_matrix /= frequency_matrix.max()
            self.frequency_mapping = tf.keras.layers.Embedding(
                129,
                1,
                weights=[np.expand_dims(frequency_matrix, -1)],
                trainable=False,
                name="frequency_mapping",
            )
        else:
            self.frequency_mapping = None

    def call(self, inputs):
        """Apply the layer to the input tensors.

        Parameters
        ----------
        inputs : dict of Tensor
            Input tensors.

            - time : shape=(batch_size, seq_len)
            - pitch : shape=(batch_size, seq_len)
            - duration : shape=(batch_size, seq_len), optional
            - onset_hint : shape=(batch_size, n_tracks), optional
            - pitch_hint : shape=(batch_size, n_tracks), optional

        """
        # Collect input tensors
        seq_len = tf.shape(inputs["time"])[1]
        tensors = []
        if self.use_pitch_embedding:
            tensors.append(self.pitch_embedding(inputs["pitch"]))
        else:
            tensors.append(
                tf.expand_dims(tf.cast(inputs["pitch"], tf.float32), -1)
            )

        if self.use_time_embedding:
            tensors.append(self.time_embedding_position(inputs["time"] % 24))
            tensors.append(
                self.time_embedding_beat(
                    tf.clip_by_value(inputs["time"] // 24, 0, self.max_beat)
                )
            )
        else:
            tensors.append(
                tf.expand_dims(tf.cast(inputs["time"], tf.float32), -1)
            )
        if self.use_duration:
            if self.use_duration_embedding:
                tensors.append(
                    self.duration_embedding(
                        tf.clip_by_value(
                            inputs["duration"], 0, self.max_duration
                        )
                    )
                )
            else:
                tensors.append(
                    tf.expand_dims(tf.cast(inputs["duration"], tf.float32), -1)
                )
        if self.use_frequency:
            tensors.append(self.frequency_mapping(inputs["pitch"]))
        if self.use_onset_hint:
            tensors.append(tf.cast(inputs["onset_hint"], tf.float32))
        if self.use_pitch_hint:
            for i in range(self.n_tracks):
                tensors.append(
                    tf.tile(
                        self.pitch_embedding(
                            tf.expand_dims(inputs["pitch_hint"][..., i], 1)
                        ),
                        (1, seq_len, 1),
                    )
                )

        # Concate features
        tensor_out = tf.concat(tensors, -1)

        # Create mask
        mask = tf.cast(tf.equal(inputs["pitch"], 0), tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        if self.use_lookahead_mask:
            look_ahead_mask = 1 - tf.linalg.band_part(
                tf.ones((seq_len, seq_len)), -1, 0
            )
            look_ahead_mask = tf.tile(
                look_ahead_mask[tf.newaxis, tf.newaxis, :, :],
                [tf.shape(inputs["time"])[0], 1, 1, 1],
            )
            mask = tf.maximum(mask, look_ahead_mask)

        return tensor_out, mask


class Transformer(tf.keras.layers.Layer):
    """A multi-layer Transformer."""

    def __init__(
        self,
        autoregressive: bool,
        max_len: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_feedforward: int,
    ):
        super().__init__()
        self.autoregressive = autoregressive
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_feedforward = d_feedforward

        # Positional embedding
        self.positional_encoding = positional_encoding(max_len, d_model)
        self.positional_encoding = self.positional_encoding[tf.newaxis, :]

        self.dense_in = tf.keras.layers.Dense(d_model)
        self.transformers = [
            TransformerLayer(d_model, n_heads, d_feedforward, rate=0.2)
            for _ in range(n_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, training=False, mask=None):  # noqa
        x = self.dense_in(x)
        x += self.positional_encoding[:, : tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for transformer in self.transformers:
            x = transformer(x, mask=mask)
        return x


class TransformerArranger(tf.keras.layers.Layer):
    """A Transformer-based arranger model."""

    def __init__(
        self,
        max_len: int,
        use_duration: bool,
        use_frequency: bool,
        use_onset_hint: bool,
        use_pitch_hint: bool,
        use_pitch_embedding: bool,
        use_time_embedding: bool,
        use_duration_embedding: bool,
        max_beat: int,
        max_duration: int,
        use_lookahead_mask: bool,
        autoregressive: bool,
        n_tracks: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_feedforward: int,
    ):
        super().__init__()
        self.max_len = max_len
        self.use_duration = use_duration
        self.use_frequency = use_frequency
        self.use_onset_hint = use_onset_hint
        self.use_pitch_hint = use_pitch_hint
        self.use_pitch_embedding = use_pitch_embedding
        self.use_time_embedding = use_time_embedding
        self.use_duration_embedding = use_duration_embedding
        self.max_beat = max_beat
        self.max_duration = max_duration
        self.use_lookahead_mask = use_lookahead_mask
        self.autoregressive = autoregressive
        self.n_tracks = n_tracks
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_feedforward = d_feedforward

        self.input_layer = InputLayer(
            max_len=max_len,
            use_duration=use_duration,
            use_frequency=use_frequency,
            use_onset_hint=use_onset_hint,
            use_pitch_hint=use_pitch_hint,
            use_pitch_embedding=use_pitch_embedding,
            use_time_embedding=use_time_embedding,
            use_duration_embedding=use_duration_embedding,
            max_beat=max_beat,
            max_duration=max_duration,
            n_tracks=n_tracks,
            use_lookahead_mask=use_lookahead_mask,
        )
        self.transformer = Transformer(
            autoregressive=autoregressive,
            max_len=max_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_feedforward=d_feedforward,
        )
        self.dense = tf.keras.layers.Dense(n_tracks + 1)

    def call(self, inputs, training=False, mask=None):  # noqa
        x, mask = self.input_layer(inputs)
        if self.autoregressive:
            x = tf.concat(
                (x, tf.cast(inputs["previous_label"], tf.float32)), -1
            )
        x = self.transformer(x, training=training, mask=mask)
        return self.dense(x)
