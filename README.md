# Tone prompt generator
# This script generates and plays a sequence of tones based on a CSV file input.
# See examples of csv files in the tone_generator folder for reference.
# The CSV file should contain the following columns: frequency, tempo, note_type
# The script uses the sounddevice library to play the generated tones.
# The generated sequence can also be saved as a .wav file.
# Usage: python tone_generator.py input_filename.csv
# Required libraries: numpy, sounddevice, pandas, scipy
# Install the required libraries using pip:
# python -m pip install numpy sounddevice pandas scipy
