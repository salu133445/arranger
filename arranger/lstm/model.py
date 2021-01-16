"""LSTM model."""
import numpy as np
import tensorflow as tf

# pylint:disable=arguments-differ


def get_angles(pos, i, d_model):
    """Copied from https://www.tensorflow.org/tutorials/text/transformer."""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """Copied from https://www.tensorflow.org/tutorials/text/transformer."""
    pos_encoding = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )

    # apply sin to even indices in the array; 2i
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

    return tf.cast(pos_encoding, dtype=tf.float32)


class InputLayer(tf.keras.layers.Layer):
    """Input layer."""

    def __init__(
        self,
        use_duration: bool,
        use_frequency: bool,
        use_previous_label: bool,
        use_onset_hint: bool,
        use_pitch_hint: bool,
        max_beat: int,
        max_duration: int,
        n_tracks: int,
    ):
        super().__init__()
        self.use_duration = use_duration
        self.use_frequency = use_frequency
        self.use_previous_label = use_previous_label
        self.use_onset_hint = use_onset_hint
        self.use_pitch_hint = use_pitch_hint
        self.max_beat = max_beat
        self.max_duration = max_duration
        self.n_tracks = n_tracks

        # Embedding
        self.time_embedding_position = tf.keras.layers.Embedding(
            24, 16, name="time_embedding_position"
        )
        self.time_embedding_beat = tf.keras.layers.Embedding(
            max_beat,
            16,
            weights=[positional_encoding(max_beat, 16)],
            trainable=False,
            name="time_embedding_beat",
        )
        self.pitch_embedding = tf.keras.layers.Embedding(
            129, 16, name="pitch_embedding"
        )
        if use_duration:
            self.duration_embedding = tf.keras.layers.Embedding(
                max_duration + 1, 16, name="duration_embedding"
            )

        # Frequency matrix
        if use_frequency:
            frequency_matrix = 440.0 * 2 ** ((np.arange(128) - 69) / 12)
            frequency_matrix = np.pad(frequency_matrix, (1, 0), "constant")
            frequency_matrix /= frequency_matrix.max()
            self.frequency_mapping = tf.constant(
                frequency_matrix,
                name="frequency_matrix",
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
            - previous_label : shape=(batch_size, seq_len), optional
            - onset_hint : shape=(batch_size, n_tracks), optional
            - pitch_hint : shape=(batch_size, n_tracks), optional

        """
        # Collect input tensors
        seq_len = tf.shape(inputs["time"])[1]
        tensors = [
            self.time_embedding_position(inputs["time"] % 24),
            self.time_embedding_beat(
                tf.clip_by_value(inputs["time"] // 24, 0, self.max_beat)
            ),
            self.pitch_embedding(inputs["pitch"]),
        ]
        if self.use_duration:
            tensors.append(
                self.duration_embedding(
                    tf.clip_by_value(inputs["duration"], 0, self.max_duration)
                )
            )
        if self.use_frequency:
            tensors.append(self.frequency_mapping(inputs["pitch"]))
        if self.use_previous_label:
            tensors.append(tf.expand_dims(inputs["previous_label"], -1))
        if self.use_onset_hint:
            tensors.append(
                tf.tile(
                    tf.expand_dims(inputs["onset_hint"], 1), (1, seq_len, 1)
                )
            )
        if self.use_pitch_hint:
            for i in range(self.n_tracks):
                tensors.append(
                    tf.tile(
                        self.pitch_embedding(inputs["pitch_hint"][..., i]),
                        (1, seq_len, 1),
                    )
                )

        # Concate features
        tensor_out = tf.concat(tensors, -1)
        mask = tf.not_equal(inputs["pitch"], 0)

        return tensor_out, mask


class MultiLayerLSTM(tf.keras.layers.Layer):
    """A multi-layer LSTM."""

    def __init__(self, n_layers: int, n_units: int, bidirectional: bool):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.bidirectional = bidirectional

        # RNN layers
        if bidirectional:
            self.lstms = [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        n_units, return_sequences=True, return_state=True
                    )
                )
                for _ in range(n_layers)
            ]
        else:
            self.lstms = [
                tf.keras.layers.LSTM(
                    n_units, return_sequences=True, return_state=True
                )
                for _ in range(n_layers)
            ]
        self.layernorms = [
            tf.keras.layers.LayerNormalization() for _ in range(n_layers)
        ]
        self.dropouts = [tf.keras.layers.Dropout(0.2) for _ in range(n_layers)]

    def call(self, x, initial_states=None, training=False, mask=None):  # noqa
        final_states = []
        for i, (lstm, dropout, layernorm) in enumerate(
            zip(self.lstms, self.dropouts, self.layernorms)
        ):
            if initial_states is None:
                x, h, c = lstm(x, mask=mask)
            else:
                x, h, c = lstm(x, mask=mask, initial_state=initial_states[i])
                final_states.append((h, c))
            x = layernorm(x)
            x = dropout(x, training=training)
        return x, final_states


class Arranger(tf.keras.layers.Layer):
    """An LSTM-based arranger model."""

    def __init__(
        self,
        use_duration: bool,
        use_frequency: bool,
        use_previous_label: bool,
        use_onset_hint: bool,
        use_pitch_hint: bool,
        max_beat: int,
        max_duration: int,
        n_tracks: int,
        n_layers: int,
        n_units: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.input_layer = InputLayer(
            use_duration=use_duration,
            use_frequency=use_frequency,
            use_previous_label=use_previous_label,
            use_onset_hint=use_onset_hint,
            use_pitch_hint=use_pitch_hint,
            max_beat=max_beat,
            max_duration=max_duration,
            n_tracks=n_tracks,
        )
        self.lstm = MultiLayerLSTM(
            n_layers=n_layers,
            n_units=n_units,
            bidirectional=bidirectional,
        )
        self.dense = tf.keras.layers.Dense(n_tracks + 1)

    def call(self, inputs, training=False, mask=None):  # noqa
        x, mask = self.input_layer(inputs)
        x, final_states = self.lstm(x, training=training, mask=mask)
        return self.dense(x), final_states
