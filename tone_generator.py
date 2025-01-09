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

import sys
import os
import numpy as np
import sounddevice as sd
import pandas as pd
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt

# Define sample rate and maximum volume
sample_rate = 16000
max_volume = 0.0625 # Normalized volume (-12db)
cutoff_freq = 1500  # Low-pass filter cutoff frequency in Hz
decay_ending_gain = 0.125 # Gain at the end of the decay (12.5% of the original volume) 
fade_out_percent = 0.80  # Start fade-out at 80% of the waveform

# Function to calculate note duration based on tempo and note type
def calculate_duration(tempo, note_type='crotchet'):
    beat_duration = 60 / tempo  # Duration of one beat in seconds
    note_durations = {
        'semibreve': beat_duration * 4, # Whole note
        'minim': beat_duration * 2,     # Half note
        'crotchet': beat_duration,      # Quarter note
        'quaver': beat_duration / 2,    # Eighth note
        'semiquaver': beat_duration / 4, # Sixteenth note
        'demisemiquaver': beat_duration / 8, # Thirty-second note
        'hemidemisemiquaver': beat_duration / 16,  # Sixty-fourth note
        'minim_triplet': (beat_duration * 2) / 3,  # Triplet half note
        'crotchet_triplet': beat_duration / 3,     # Triplet quarter note
        'quaver_triplet': beat_duration / 6,       # Triplet eighth note
        'semiquaver_triplet': beat_duration / 12,  # Triplet Sixteenth note
        'demisemiquaver_triplet': beat_duration / 24, # Triplet Thirty-second note
        'hemidemisemiquaver_triplet': beat_duration / 48,  # Triplet Sixty-fourth note      
    }
    return note_durations.get(note_type, beat_duration)  # Default to crotchet if type is unknown

def low_pass_filter_with_decay(data, cutoff, sample_rate, order=2):
    """
    Apply a low-pass Butterworth filter with -6 dB/octave decay.

    Parameters:
        data (np.ndarray): Input waveform.
        cutoff (float): Cutoff frequency in Hz.
        sample_rate (int): Sampling rate in Hz.
        order (int): Filter order (default is 1 for -6 dB/octave).

    Returns:
        np.ndarray: Filtered waveform.
    """
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    # Design the Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Function to generate a tone based on frequency and duration
def generate_tone(frequency, duration, volume=max_volume):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return volume * np.sin(2 * np.pi * frequency * t)

def generate_tone_with_decay(frequency, duration, volume=max_volume, decay_type="exponential"):
    """
    Generate a tone with decaying volume.
    
    Parameters:
        frequency (float): Frequency of the tone in Hz.
        duration (float): Duration of the tone in seconds.
        sample_rate (int): Sampling rate in Hz.
        decay_type (str): Type of decay ("exponential" or "linear").
    
    Returns:
        np.ndarray: Waveform of the decayed tone.
    """
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine wave
    sine_wave = volume * np.sin(2 * np.pi * frequency * t)
    
    # Apply decay
    if decay_type == "exponential":
        decay = decay_ending_gain + (1 -decay_ending_gain) * np.exp(-t / duration)  # Exponential decay to 12.5%
    elif decay_type == "linear":
        decay = 1 - decay_ending_gain * (t / duration)  # Linear decay to 12.5%
    else:
        raise ValueError("Invalid decay_type. Use 'exponential' or 'linear'.")
    
    # Apply the decay to the sine wave
    decayed_wave = sine_wave * decay

    return decayed_wave

def apply_fader(duration, fade_wave):
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
     # Create a fade-out function for the last 20% of the tone
    fade_out_start = int(len(t) * fade_out_percent)  # Start fade-out at 80% of the waveform
    fade_out = np.linspace(1, 0, len(t) - fade_out_start)  # Linear fade to 0
    fade_wave[fade_out_start:] *= fade_out  # Apply fade-out to the last 20%

    return fade_wave

# Load sequence from CSV file
def load_sequence_from_file(filename):
    try:
        # Attempt to read the CSV file with UTF-8 encoding
        df = pd.read_csv(filename, encoding='utf-8-sig')
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 encoding if UTF-8 fails
        df = pd.read_csv(filename, encoding='ISO-8859-1')

     # Ensure the file has the correct columns
    required_columns = ["frequency", "tempo", "note_type"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError("CSV file must contain columns: frequency, tempo, note_type")
    
    # Convert the DataFrame to a list of lists
    sequence = df[["frequency", "tempo", "note_type"]].values.tolist()
    return sequence

# Define the sequence in the format [frequency, tempo, note_type]
'''
sequence = [
    [440, 300, 'crotchet'],  # A4 note, 300 BPM, crotchet (quarter note)
    [494, 1500, 'quaver'],   # B4 note, 1500 BPM, quaver (eighth note)
    [523, 300, 'crotchet'],  # C5 note, 300 BPM, crotchet
    [0, 300, 'quaver'],      # Rest (0 Hz), 300 BPM, quaver
    [587, 300, 'crotchet']   # D5 note, 300 BPM, crotchet
]
'''

#power_off
'''
sequence = [
    [987.77, 300, 'quaver'],  # B5 note, 300 BPM, quaver (eighth note)
    [987.77, 300, 'semiquaver'],  # B5 note, 300 BPM, semiquaver (sixteenth note)  
    [0, 300, 'semiquaver'],   # Rest, 300 BPM, semiquaver (sixteenth note)
    [783.99, 300, 'quaver'],  # G5 note, 300 BPM, quaver (eighth note)
    [783.99, 300, 'semiquaver'],  # G5 note, 300 BPM, semiquaver (sixteenth note)  
    [0, 300, 'semiquaver'],   # Rest, 300 BPM, semiquaver (sixteenth note)
    [587.33, 300, 'quaver'],  # D5 note, 300 BPM, quaver (eighth note)
    [587.33, 300, 'semiquaver'],  # D5 note, 300 BPM, semiquaver (sixteenth note)  
    [0, 300, 'semiquaver'],   # Rest, 300 BPM, semiquaver (sixteenth note)
]
'''

#disconnect
'''
sequence = [
    [783.99, 300, 'minim'],  # G5 note, 300 BPM, crotchet (quarter note)
]
'''

#low_battery
'''
sequence = [
    [1567.98, 300, 'minim'],  # G6 note, 300 BPM, minim (half note) 
    [0, 1500, 'quaver'],   # Rest, 1500 BPM, quaver (eighth note)
    [1567.98, 300, 'minim'],  # G6 note, 300 BPM, minim (half note) 
    [0, 1500, 'quaver'],   # Rest, 1500 BPM, quaver (eighth note)
    [1567.98, 300, 'minim'],  # G6 note, 300 BPM, minim (half note) 
]
'''

#vol_max
'''
sequence = [
    [3951.066, 300, 'quaver'],  # B7 note, 300 BPM, quaver (eighth note) 
    [0, 1500, 'quaver'],   # Rest, 1500 BPM, quaver (eighth note)
    [3951.066, 300, 'quaver'],  # B7 note, 300 BPM, quaver (eighth note) 
]
'''

#vol_min
'''
sequence = [
    [987.77, 300, 'quaver'],  # B5 note, 300 BPM, quaver (eighth note) 
    [0, 1500, 'quaver'],   # Rest, 1500 BPM, quaver (eighth note)
    [987.77, 300, 'quaver'],  # B5 note, 300 BPM, quaver (eighth note) 
]
'''
#mute_on
'''
sequence = [
    [1567.98, 300, 'crotchet'],  # G6 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [783.99, 300, 'crotchet'],  # G5 note, 300 BPM, crotchet (quarter note)
]
'''

#mute_off
'''
sequence = [
    [783.99, 300, 'crotchet'],  # G5 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [1567.98, 300, 'crotchet'],  # G6 note, 300 BPM, crotchet (quarter note)
]
'''

#Active Autopause
'''
sequence = [
    [523.25, 300, 'crotchet'],  # C5 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [659.25, 300, 'crotchet'],  # E5 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [880, 300, 'crotchet'],  # A5 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [987.77, 300, 'crotchet'],  # B5 note, 300 BPM, crotchet (quarter note)
]
'''

#Deactivate Autopause
'''
sequence = [
    [987.77, 300, 'crotchet'],  # B5 note, 300 BPM, crotchet (quarter note)    
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [783.99, 300, 'crotchet'],  # G5 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [587.33, 300, 'crotchet'],  # D5 note, 300 BPM, crotchet (quarter note)
    [0, 300, 'quaver'],   # Rest, 300 BPM, quaver (eighth note)
    [523.25, 300, 'crotchet'],  # C5 note, 300 BPM, crotchet (quarter note)
]
'''

# Main function to generate and play the sequence
def play_save_sequence(sequence, output_wav_file):

    # Generate the full sequence
    full_waveform = np.array([])

    for i, (freq, tempo, note_type) in enumerate(sequence):
        # Calculate duration of each note/rest
        duration = calculate_duration(tempo, note_type)
        
        # Generate either tone or rest
        if freq > 0:
             tone = generate_tone_with_decay(freq, duration)
             # Apply fade-out to the last tone in the sequence
             if i == len(sequence) - 1 or (i < len(sequence) - 1 and sequence[i + 1][0] == 0):
                tone = apply_fader(duration, tone)
        else:
            tone = np.zeros(int(sample_rate * duration))  # Silence for rest
        
        # Append the tone or rest to the full waveform
        full_waveform = np.concatenate((full_waveform, tone))

    # Optional: Add a short silence between tones
    silence = np.zeros(int(sample_rate * 0.05))  # 0.05 seconds of silence
    full_waveform = np.concatenate((silence, full_waveform))
    full_waveform = np.concatenate((full_waveform, silence))

    # apply low pass-filter
    full_waveform = low_pass_filter_with_decay(full_waveform, cutoff_freq, sample_rate)
    # Normalize the waveform to avoid clipping
    full_waveform = np.int16(full_waveform / np.max(np.abs(full_waveform)) * 32767 * max_volume)

    # Play the generated sequence
    sd.play(full_waveform, sample_rate)
    sd.wait()  # Wait until playback is complete
    # Save to a .wav file
    write(output_wav_file, sample_rate, full_waveform)

if __name__ == "__main__":
     # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)
      # Arguments passed
    print("Name of Python script:", sys.argv[0])  

    inputfilename = sys.argv[1]
    # Load sequence from a CSV file
    filename = "tone_sequence.csv"  # Replace with your CSV filename
    sequence = load_sequence_from_file(inputfilename)
    output_wav_file = os.path.splitext(inputfilename)[0] + ".wav"
    
    # Play the sequence
    play_save_sequence(sequence,output_wav_file)