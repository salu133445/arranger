__Towards Automatic Instrumentation by Learning to Separate Parts in Symbolic Multitrack Music__
{:.center .larger}

[Hao-Wen Dong](https://salu133445.github.io/)<sup>1</sup> &emsp;
[Chris Donahue](https://chrisdonahue.com/)<sup>2</sup> &emsp;
[Taylor Berg-Kirkpatrick](https://cseweb.ucsd.edu/~tberg/)<sup>1</sup> &emsp;
[Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/)<sup>1</sup>\\
<sup>1</sup> University of California San Diego &emsp;
<sup>2</sup> Stanford University
{:.center}

{% include icon_link.html text="homepage" icon=site.icons.homepage href="https://salu133445.github.io/arranger/" %} &emsp;
{% include icon_link.html text="paper" icon=site.icons.paper href="https://arxiv.org/pdf/2107.05916.pdf" %} &emsp;
{% include icon_link.html text="video" icon=site.icons.video href="https://youtu.be/-KncOGouAh8" %} &emsp;
{% include icon_link.html text="slides" icon=site.icons.slides href="https://salu133445.github.io/arranger/pdf/arranger_ismir2021_slides.pdf" %} &emsp;
{% include icon_link.html text="code" icon=site.icons.code href="https://github.com/salu133445/arranger" %} &emsp;
{% include icon_link.html text="reviews" icon=site.icons.reviews href="https://salu133445.github.io/arranger/pdf/arranger_ismir2021_reviews.pdf" %}
{:.center}

{% include video_player.html id="-KncOGouAh8" %}

---

## Content

- [Best Samples](#best-samples)
- [Cette année-là_ by Claude François (Figure 1)](#fig1)
- [_String Quartet No. 11 in F minor, Op. 95, Movement 1_ by Ludwig van Beethoven (Figure 2)](#fig2)
- [_Wer nur den lieben Gott läßt walten, BWV 434_ by Johann Sebastian Bach (Figure 3)](#fig3)
- [_Theme of Universe_ from Miracle Ropit's Adventure in 2100 (Figure 4)](#fig4)
- [_Blame It On the Boogie_ by The Jacksons (Figure 5)](#fig5)
- [_Quando Quando Quando_ by Tony Renis (Figure 6)](#fig6)
- [Citation](#citation)

---

## Important Notes

All samples are synthesized using [FluidSynth](https://www.fluidsynth.org/) with the MuseScore General [soundfont](https://musescore.org/en/handbook/3/soundfonts-and-sfz-files).

---

## Best Samples

<div class="table-wrapper" markdown="block">

| Mixture (input) | Predicted instrumentation (output) |
|:-:|:-:|
| (guitar) | (piano, guitar, bass, strings, brass) |
| {% include audio_player.html filename="audio/cette_annee_la_common_default_drums.mp3" %} | {% include audio_player.html filename="audio/cette_annee_la_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| {% include audio_player.html filename="audio/blame_it_on_the_boogie_common_default_drums.mp3" %} | {% include audio_player.html filename="audio/blame_it_on_the_boogie_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| {% include audio_player.html filename="audio/quando_quando_quando_common_default_drums.mp3" %} | {% include audio_player.html filename="audio/quando_quando_quando_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |

</div>

---

## _Cette année-là_ by Claude François (Figure 1) {#fig1}

> Colors: _piano_{:.blue}, _guitar_{:.orange}, _bass_{:.green}, _strings_{:.red}, _brass_{:.purple}.

<div class="table-wrapper" markdown="block">

| Mixture (input) | ![cette_annee_la_bw](images/cette_annee_la_bw.png){:.score} | {% include audio_player.html filename="audio/cette_annee_la_common_default_drums.mp3" %} |
| LSTM | ![cette_annee_la_lstm](images/cette_annee_la_lstm.png){:.score} | {% include audio_player.html filename="audio/cette_annee_la_lstm_default_embedding_onsethint_drums.mp3" %} |
| BiLSTM | ![cette_annee_la_bilstm](images/cette_annee_la_bilstm.png){:.score} | {% include audio_player.html filename="audio/cette_annee_la_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| Ground truth | ![cette_annee_la_truth](images/cette_annee_la_truth.png){:.score} | {% include audio_player.html filename="audio/cette_annee_la_truth_drums.mp3" %} |

</div>

---

## _String Quartet No. 11 in F minor, Op. 95, Movement 1_ by Ludwig van Beethoven (Figure 2) {#fig2}

> Colors: _first violin_{:.blue}, _second violin_{:.orange}, _viola_{:.green}, _cello_{:.red}.

<div class="table-wrapper" markdown="block">

| Mixture (input) | ![beethoven_op95_bw](images/beethoven_op95_bw.png){:.score} | {% include audio_player.html filename="audio/beethoven_op95_common_default.mp3" %} |
| LSTM | ![beethoven_op95_lstm](images/beethoven_op95_lstm.png){:.score} | {% include audio_player.html filename="audio/beethoven_op95_lstm_default_embedding_onsethint.mp3" %} |
| BiLSTM | ![beethoven_op95_bilstm](images/beethoven_op95_bilstm.png){:.score} | {% include audio_player.html filename="audio/beethoven_op95_lstm_bidirectional_embedding_onsethint_duration.mp3" %} |
| Ground truth | ![beethoven_op95_truth](images/beethoven_op95_truth.png){:.score} | {% include audio_player.html filename="audio/beethoven_op95_truth.mp3" %} |

</div>

> Original music score:
>
> ![beethoven_op95_score](images/beethoven_op95_score.png)

---

## _Wer nur den lieben Gott läßt walten, BWV 434_ by Johann Sebastian Bach (Figure 3) {#fig3}

> Colors: _soprano_{:.blue}, _alto_{:.orange}, _tenor_{:.green}, _bass_{:.red}.

<div class="table-wrapper" markdown="block">

| Mixture (input) | ![bwv434_bw](images/bwv434_bw.png){:.score} | {% include audio_player.html filename="audio/bwv434_common_default.mp3" %} |
| LSTM | ![bwv434_lstm](images/bwv434_lstm.png){:.score} | {% include audio_player.html filename="audio/bwv434_lstm_default_embedding_onsethint.mp3" %} |
| BiLSTM | ![bwv434_bilstm](images/bwv434_bilstm.png){:.score} | {% include audio_player.html filename="audio/bwv434_lstm_bidirectional_embedding_onsethint_duration.mp3" %} |
| Ground truth | ![bwv434_truth](images/bwv434_truth.png){:.score} | {% include audio_player.html filename="audio/bwv434_truth.mp3" %} |

</div>

> Original music score:
>
> ![bwv434_score](images/bwv434_score.png)

---

## _Theme of Universe_ from Miracle Ropit's Adventure in 2100 (Figure 4) {#fig4}

> Colors: _pulse wave I_{:.blue}, _pulse wave II_{:.orange}, _triangle wave_{:.green}.

<div class="table-wrapper" markdown="block">

| Mixture (input) | ![miracle_ropits_adventure_in_2100_theme_of_universe_bw](images/miracle_ropits_adventure_in_2100_theme_of_universe_bw.png){:.score} | {% include audio_player.html filename="audio/miracle_ropits_adventure_in_2100_theme_of_universe_common_default.mp3" %} |
| LSTM | ![miracle_ropits_adventure_in_2100_theme_of_universe_lstm](images/miracle_ropits_adventure_in_2100_theme_of_universe_lstm.png){:.score} | {% include audio_player.html filename="audio/miracle_ropits_adventure_in_2100_theme_of_universe_lstm_default_embedding_onsethint.mp3" %} |
| BiLSTM | ![miracle_ropits_adventure_in_2100_theme_of_universe_bilstm](images/miracle_ropits_adventure_in_2100_theme_of_universe_bilstm.png){:.score} | {% include audio_player.html filename="audio/miracle_ropits_adventure_in_2100_theme_of_universe_lstm_bidirectional_embedding_onsethint_duration.mp3" %} |
| Ground truth | ![miracle_ropits_adventure_in_2100_theme_of_universe_truth](images/miracle_ropits_adventure_in_2100_theme_of_universe_truth.png){:.score} | {% include audio_player.html filename="audio/miracle_ropits_adventure_in_2100_theme_of_universe_truth.mp3" %} |

</div>

---

## _Blame It On the Boogie_ by The Jacksons (Figure 5) {#fig5}

> Colors: _piano_{:.blue}, _guitar_{:.orange}, _bass_{:.green}, _strings_{:.red}, _brass_{:.purple}.

<div class="table-wrapper" markdown="block">

| Mixture (input) | ![blame_it_on_the_boogie_bw](images/blame_it_on_the_boogie_bw.png){:.score} | {% include audio_player.html filename="audio/blame_it_on_the_boogie_common_default_drums.mp3" %} |
| LSTM | ![blame_it_on_the_boogie_lstm](images/blame_it_on_the_boogie_lstm.png){:.score} | {% include audio_player.html filename="audio/blame_it_on_the_boogie_lstm_default_embedding_onsethint_drums.mp3" %} |
| BiLSTM | ![blame_it_on_the_boogie_bilstm](images/blame_it_on_the_boogie_bilstm.png){:.score} | {% include audio_player.html filename="audio/blame_it_on_the_boogie_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| Ground truth | ![blame_it_on_the_boogie_truth](images/blame_it_on_the_boogie_truth.png){:.score} | {% include audio_player.html filename="audio/blame_it_on_the_boogie_truth_drums.mp3" %} |

</div>

---

## _Quando Quando Quando_ by Tony Renis (Figure 6) {#fig6}

> Colors: _piano_{:.blue}, _guitar_{:.orange}, _bass_{:.green}, _strings_{:.red}, _brass_{:.purple}.

<div class="table-wrapper" markdown="block">

| Mixture (input) | ![quando_quando_quando_bw](images/quando_quando_quando_bw.png){:.score} | {% include audio_player.html filename="audio/quando_quando_quando_common_default_drums.mp3" %} |
| LSTM | ![quando_quando_quando_lstm](images/quando_quando_quando_lstm.png){:.score} | {% include audio_player.html filename="audio/quando_quando_quando_lstm_default_embedding_drums.mp3" %} |
| BiLSTM | ![quando_quando_quando_bilstm](images/quando_quando_quando_bilstm.png){:.score} | {% include audio_player.html filename="audio/quando_quando_quando_lstm_bidirectional_embedding_onsethint_duration_drums.mp3" %} |
| Ground truth | ![quando_quando_quando_truth](images/quando_quando_quando_truth.png){:.score} | {% include audio_player.html filename="audio/quando_quando_quando_truth_drums.mp3" %} |

</div>

---

## Citation

> Hao-Wen Dong, Chris Donahue, Taylor Berg-Kirkpatrick, and Julian McAuley, "Towards Automatic Instrumentation by Learning to Separate Parts in Symbolic Multitrack Music," Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2021.

```bibtex
@inproceedings{dong2021arranger,
    author = {Hao-Wen Dong and Chris Donahue and Taylor Berg-Kirkpatrick and Julian McAuley},
    title = {Towards Automatic Instrumentation by Learning to Separate Parts in Symbolic Multitrack Music},
    booktitle = {Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
    year = 2021,
}
```
