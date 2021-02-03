#!/bin/bash
set -ex

mkdir -p ~/data/temp/selected/

python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/bwv434_audio.zip -d bach -k 434
python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/beethoven95_audio.zip -d musicnet -k 2494
python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/seicross_audio.zip -d nes -k 290_Seicross_01_02BGM1
python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/miracle_ropit_audio.zip -d nes -k 241_MiracleRopit_sAdventurein2100_05_06ThemeofUniverse
python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/blame_it_on_the_boogie_audio.zip -d lmd -k 5a65323fe3ba1c143a276ca06a0a1a3d
python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/cette_annee_la_audio.zip -d lmd -k aab14a403b02ea384b7a5019042a4372
python3 arranger/collect_audio.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/quando_quando_quando_audio.zip -d lmd -k e12eae3af78be50180ab073aaf91e045

python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/bwv434.zip -d bach -k 434
python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/beethoven95.zip -d musicnet -k 2494
python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/seicross.zip -d nes -k 290_Seicross_01_02BGM1
python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/miracle_ropit.zip -d nes -k 241_MiracleRopit_sAdventurein2100_05_06ThemeofUniverse
python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/blame_it_on_the_boogie.zip -d lmd -k 5a65323fe3ba1c143a276ca06a0a1a3d
python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/cette_annee_la.zip -d lmd -k aab14a403b02ea384b7a5019042a4372
python3 arranger/collect_images.py -i ~/data/arranger/exp/ -o ~/data/temp/selected/quando_quando_quando.zip -d lmd -k e12eae3af78be50180ab073aaf91e045

cd ~/data/temp/
zip selected.zip selected/*.zip
cd -
