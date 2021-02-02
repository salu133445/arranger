"""LSTM model."""
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


class InputLayer(tf.keras.layers.Layer):
    """Input layer."""

    def __init__(
        self,
        max_len: int,
        use_beat_postion: bool,
        use_duration: bool,
        use_frequency: bool,
        use_onset_hint: bool,
        use_pitch_hint: bool,
        use_pitch_embedding: bool,
        use_time_embedding: bool,
        use_beat_embedding: bool,
        use_duration_embedding: bool,
        max_time: int,
        max_beat: int,
        max_duration: int,
        n_tracks: int,
    ):
        assert (
            not use_time_embedding or not use_beat_embedding
        ), "use_time_embedding and use_beat_embedding must not be both True"
        assert (
            not use_beat_postion or not use_time_embedding
        ), "use_time_embedding must be False when use_beat_postion is True"
        assert (
            use_beat_postion or not use_beat_embedding
        ), "use_beat_embedding must be False when use_beat_postion is False"
        super().__init__()
        self.max_len = max_len
        self.use_beat_postion = use_beat_postion
        self.use_duration = use_duration
        self.use_frequency = use_frequency
        self.use_onset_hint = use_onset_hint
        self.use_pitch_hint = use_pitch_hint
        self.use_pitch_embedding = use_pitch_embedding
        self.use_time_embedding = use_time_embedding
        self.use_beat_embedding = use_beat_embedding
        self.use_duration_embedding = use_duration_embedding
        self.max_time = max_time
        self.max_beat = max_beat
        self.max_duration = max_duration
        self.n_tracks = n_tracks

        # Embedding
        if use_pitch_embedding:
            self.pitch_embedding = tf.keras.layers.Embedding(
                129, 16, name="pitch_embedding"
            )
        if use_time_embedding:
            self.time_embedding = tf.keras.layers.Embedding(
                max_time + 1,
                16,
                weights=[positional_encoding(max_time + 1, 16)],
                trainable=False,
                name="time_embedding",
            )
        if use_beat_embedding:
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

        if not self.use_beat_postion:
            if self.use_time_embedding:
                tensors.append(
                    self.time_embedding(
                        tf.clip_by_value(inputs["time"], 0, self.max_time)
                    )
                )
            else:
                tensors.append(
                    tf.expand_dims(tf.cast(inputs["time"], tf.float32), -1)
                )
        elif self.use_beat_embedding:
            tensors.append(self.time_embedding_position(inputs["time"] % 24))
            tensors.append(
                self.time_embedding_beat(
                    tf.clip_by_value(inputs["time"] // 24, 0, self.max_beat)
                )
            )
        else:
            tensors.append(
                tf.expand_dims(tf.cast(inputs["time"] % 24, tf.float32), -1)
            )
            tensors.append(
                tf.expand_dims(
                    tf.cast(
                        tf.clip_by_value(
                            inputs["time"] // 24, 0, self.max_beat
                        ),
                        tf.float32,
                    ),
                    -1,
                )
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
        mask = tf.not_equal(inputs["pitch"], 0)

        return tensor_out, mask


class LSTM(tf.keras.layers.Layer):
    """A multi-layer LSTM."""

    def __init__(self, bidirectional: bool, n_layers: int, n_units: int):
        super().__init__()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.n_units = n_units

        self.lstms = [
            tf.keras.layers.LSTM(n_units, return_sequences=True)
            for _ in range(n_layers)
        ]
        if bidirectional:
            assert (
                n_units % 2 == 0
            ), "`n_units` must be an even number for a bidirectional LSTM"
            self.lstms = [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(n_units // 2, return_sequences=True)
                )
                for _ in range(n_layers)
            ]
        else:
            self.lstms = [
                tf.keras.layers.LSTM(n_units, return_sequences=True)
                for _ in range(n_layers)
            ]
        self.layernorms = [
            tf.keras.layers.LayerNormalization() for _ in range(n_layers)
        ]
        self.dropouts = [tf.keras.layers.Dropout(0.2) for _ in range(n_layers)]

    def call(self, x, training=False, mask=None):  # noqa
        for lstm, dropout, layernorm in zip(
            self.lstms, self.dropouts, self.layernorms
        ):
            x = lstm(x, mask=mask)
            x = layernorm(x)
            x = dropout(x, training=training)
        return x


class LSTMArranger(tf.keras.layers.Layer):
    """An LSTM-based arranger model."""

    def __init__(
        self,
        max_len: int,
        use_beat_postion: bool,
        use_duration: bool,
        use_frequency: bool,
        use_onset_hint: bool,
        use_pitch_hint: bool,
        use_pitch_embedding: bool,
        use_time_embedding: bool,
        use_beat_embedding: bool,
        use_duration_embedding: bool,
        max_time: int,
        max_beat: int,
        max_duration: int,
        autoregressive: bool,
        bidirectional: bool,
        n_tracks: int,
        n_layers: int,
        n_units: int,
    ):
        super().__init__()
        self.max_len = max_len
        self.use_beat_postion = use_beat_postion
        self.use_duration = use_duration
        self.use_frequency = use_frequency
        self.use_onset_hint = use_onset_hint
        self.use_pitch_hint = use_pitch_hint
        self.use_pitch_embedding = use_pitch_embedding
        self.use_time_embedding = use_time_embedding
        self.use_duration_embedding = use_duration_embedding
        self.use_beat_embedding = use_beat_embedding
        self.max_time = max_time
        self.max_beat = max_beat
        self.max_duration = max_duration
        self.autoregressive = autoregressive
        self.bidirectional = bidirectional
        self.n_tracks = n_tracks
        self.n_layers = n_layers
        self.n_units = n_units
        assert not (
            autoregressive and bidirectional
        ), "`autoregressive` and `bidirectional` must not be both True"

        self.input_layer = InputLayer(
            max_len=max_len,
            use_beat_postion=use_beat_postion,
            use_duration=use_duration,
            use_frequency=use_frequency,
            use_onset_hint=use_onset_hint,
            use_pitch_hint=use_pitch_hint,
            use_pitch_embedding=use_pitch_embedding,
            use_time_embedding=use_time_embedding,
            use_beat_embedding=use_beat_embedding,
            use_duration_embedding=use_duration_embedding,
            max_time=max_time,
            max_beat=max_beat,
            max_duration=max_duration,
            n_tracks=n_tracks,
        )
        self.lstm = LSTM(
            bidirectional=bidirectional, n_layers=n_layers, n_units=n_units
        )
        self.dense = tf.keras.layers.Dense(n_tracks + 1)

    def call(self, inputs, training=False, mask=None):  # noqa
        x, mask = self.input_layer(inputs)
        if self.autoregressive:
            x = tf.concat(
                (x, tf.cast(inputs["previous_label"], tf.float32)), -1
            )
        x = self.lstm(x, training=training, mask=mask)
        return self.dense(x)
