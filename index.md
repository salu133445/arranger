{% include video_player.html id="-KncOGouAh8" %}

Arranger is a project on automatic instrumentation. In a nutshell, we aim to dynamically assign a proper instrument for each note in solo music. Such an automatic instrumentation model could empower a musician to play multiple instruments on a keyboard at the same time. It could also assist a composer in suggesting proper instrumentation for a solo piece.

Here are some sample instrumentation produced by our system (more samples [here](samples)).

| Mixture (input) | Predicted instrumentation |
|:---------------:|:-------------------------:|
| (guitar) | (piano, guitar, bass, strings, brass) |
| {% include audio_player.html filename="assets/audio/cette_annee_la_common_default_drums.mp3" %} | {% include audio_player.html filename="assets/audio/cette_annee_la_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| {% include audio_player.html filename="assets/audio/blame_it_on_the_boogie_common_default_drums.mp3" %} | {% include audio_player.html filename="assets/audio/blame_it_on_the_boogie_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| {% include audio_player.html filename="assets/audio/quando_quando_quando_common_default_drums.mp3" %} | {% include audio_player.html filename="assets/audio/quando_quando_quando_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
