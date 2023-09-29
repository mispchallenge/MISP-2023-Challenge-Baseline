#!/usr/bin/env bash
input_folder=/disk3/chime/simulation/gss/exp/gss/test/enhanced/
output_folder=/disk3/chime/simulation/gss/exp/gss/test/enhanced_wav/
python local/flac2wav.py --input_folder $input_folder --output_folder $output_folder