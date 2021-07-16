# Demo

- [Figure 1: _Cette année-là_ by Claude François](#fig1)
- [Figure 2: _String Quartet No. 11 in F minor, Op. 95, Movement 1_ by Ludwig van Beethoven](#fig2)
- [Figure 3: _Wer nur den lieben Gott läßt walten, BWV 434_ by Johann Sebastian Bach](#fig3)
- [Figure 4: _Theme of Universe_ from Miracle Ropit's Adventure in 2100](#fig4)
- [Figure 5: _Blame It On the Boogie_ by The Jacksons](#fig5)
- [Figure 6: _Quando Quando Quando_ by Tony Renis](#fig6)

> All samples are synthesized using FluidSynth with the MuseScore General soundfont.

---

## Figure 1: _Cette année-là_ by Claude François {#fig1}

(Colors: _piano_{:.blue}, _guitar_{:.orange}, _bass_{:.green}, _strings_{:.red}, _brass_{:.purple}.)

- Mixture (input)\\
  ![cette_annee_la_bw](images/cette_annee_la_bw.png){:.score}\\
  {% include audio_player.html filename="cette_annee_la_common_default_drums.mp3" %}

- Ground truth\\
  ![cette_annee_la_truth](images/cette_annee_la_truth.png){:.score}\\
  {% include audio_player.html filename="cette_annee_la_truth_drums.mp3" %}
- Online LSTM prediction\\
  ![cette_annee_la_lstm](images/cette_annee_la_lstm.png){:.score}\\
  {% include audio_player.html filename="cette_annee_la_lstm_default_embedding_onsethint_drums.mp3" %}
- Offline BiLSTM prediction\\
  ![cette_annee_la_bilstm](images/cette_annee_la_bilstm.png){:.score}\\
  {% include audio_player.html filename="cette_annee_la_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %}

---

## Figure 2: _String Quartet No. 11 in F minor, Op. 95, Movement 1_ by Ludwig van Beethoven {#fig2}

(Colors: _first violin_{:.blue}, _second violin_{:.orange}, _viola_{:.green}, _cello_{:.red}.)

![beethoven_op95_score](images/beethoven_op95_score.png){:style="max-width: none;"}

- Mixture (input)\\
  ![beethoven_op95_bw](images/beethoven_op95_bw.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="beethoven_op95_common_default.mp3" %}
- Ground truth\\
  ![beethoven_op95_truth](images/beethoven_op95_truth.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="beethoven_op95_truth.mp3" %}
- Online LSTM prediction\\
  ![beethoven_op95_lstm](images/beethoven_op95_lstm.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="beethoven_op95_lstm_default_embedding_onsethint.mp3" %}
- Offline BiLSTM prediction\\
  ![beethoven_op95_bilstm](images/beethoven_op95_bilstm.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="beethoven_op95_lstm_bidirectional_embedding_onsethint_duration.mp3" %}

---

## Figure 3: _Wer nur den lieben Gott läßt walten, BWV 434_ by Johann Sebastian Bach {#fig3}

(Colors: _soprano_{:.blue}, _alto_{:.orange}, _tenor_{:.green}, _bass_{:.red}.)

![bwv434_score](images/bwv434_score.png){:style="max-width: none;"}

- Mixture (input)\\
  ![bwv434_bw](images/bwv434_bw.png){:.score}\\
  {% include audio_player.html filename="bwv434_common_default.mp3" %}
- Ground truth\\
  ![bwv434_truth](images/bwv434_truth.png){:.score}\\
  {% include audio_player.html filename="bwv434_truth.mp3" %}
- Online LSTM prediction\\
  ![bwv434_lstm](images/bwv434_lstm.png){:.score}\\
  {% include audio_player.html filename="bwv434_lstm_default_embedding_onsethint.mp3" %}
- Offline BiLSTM prediction\\
  ![bwv434_bilstm](images/bwv434_bilstm.png){:.score}\\
  {% include audio_player.html filename="bwv434_lstm_bidirectional_embedding_onsethint_duration.mp3" %}

---

## Figure 4: _Theme of Universe_ from Miracle Ropit's Adventure in 2100 {#fig4}

(Colors: _pulse wave I_{:.blue}, _pulse wave II_{:.orange}, _triangle wave_{:.green}.)

- Mixture (input)\\
  ![miracle_ropits_adventure_in_2100_theme_of_universe_bw](images/miracle_ropits_adventure_in_2100_theme_of_universe_bw.png){:.score}\\
  {% include audio_player.html filename="miracle_ropits_adventure_in_2100_theme_of_universe_common_default.mp3" %}
- Ground truth\\
  ![miracle_ropits_adventure_in_2100_theme_of_universe_truth](images/miracle_ropits_adventure_in_2100_theme_of_universe_truth.png){:.score}\\
  {% include audio_player.html filename="miracle_ropits_adventure_in_2100_theme_of_universe_truth.mp3" %}
- Online LSTM prediction\\
  ![miracle_ropits_adventure_in_2100_theme_of_universe_lstm](images/miracle_ropits_adventure_in_2100_theme_of_universe_lstm.png){:.score}\\
  {% include audio_player.html filename="miracle_ropits_adventure_in_2100_theme_of_universe_lstm_default_embedding_onsethint.mp3" %}
- Offline BiLSTM prediction\\
  ![miracle_ropits_adventure_in_2100_theme_of_universe_bilstm](images/miracle_ropits_adventure_in_2100_theme_of_universe_bilstm.png){:.score}\\
  {% include audio_player.html filename="miracle_ropits_adventure_in_2100_theme_of_universe_lstm_bidirectional_embedding_onsethint_duration.mp3" %}

---

## Figure 5: _Blame It On the Boogie_ by The Jacksons {#fig5}

(Colors: _piano_{:.blue}, _guitar_{:.orange}, _bass_{:.green}, _strings_{:.red}, _brass_{:.purple}.)

- Mixture (input)\\
  ![blame_it_on_the_boogie_bw](images/blame_it_on_the_boogie_bw.png){:.score}\\
  {% include audio_player.html filename="blame_it_on_the_boogie_common_default_drums.mp3" %}
- Ground truth\\
  ![blame_it_on_the_boogie_truth](images/blame_it_on_the_boogie_truth.png){:.score}\\
  {% include audio_player.html filename="blame_it_on_the_boogie_truth_drums.mp3" %}
- Online LSTM prediction\\
  ![blame_it_on_the_boogie_lstm](images/blame_it_on_the_boogie_lstm.png){:.score}\\
  {% include audio_player.html filename="blame_it_on_the_boogie_lstm_default_embedding_onsethint_drums.mp3" %}
- Offline BiLSTM prediction\\
  ![blame_it_on_the_boogie_bilstm](images/blame_it_on_the_boogie_bilstm.png){:.score}\\
  {% include audio_player.html filename="blame_it_on_the_boogie_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %}

---

## Figure 6: _Quando Quando Quando_ by Tony Renis {#fig6}

(Colors: _piano_{:.blue}, _guitar_{:.orange}, _bass_{:.green}, _strings_{:.red}, _brass_{:.purple}.)

- Mixture (input)\\
  ![quando_quando_quando_bw](images/quando_quando_quando_bw.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="quando_quando_quando_common_default_drums.mp3" %}
- Ground truth\\
  ![quando_quando_quando_truth](images/quando_quando_quando_truth.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="quando_quando_quando_truth_drums.mp3" %}
- Online LSTM prediction\\
  ![quando_quando_quando_lstm](images/quando_quando_quando_lstm.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="quando_quando_quando_lstm_default_embedding_drums.mp3" %}
- Offline BiLSTM prediction\\
  ![quando_quando_quando_bilstm](images/quando_quando_quando_bilstm.png){:.score style="min-height: 60px;"}\\
  {% include audio_player.html filename="quando_quando_quando_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %}
