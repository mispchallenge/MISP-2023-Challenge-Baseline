import torch
import torchaudio
import numpy as np

# Load the .pt file containing audio data
#audio_tensor = torch.load("/disk4/cyzhang/misp2023/data/train/clean/S000_R01_S000001_C07_I0_001432-001724.pt")

# Convert audio tensor to numpy array
#audio_np = audio_tensor.numpy()

# Transpose the numpy array to match the shape expected by torchaudio
#audio_np = np.transpose(audio_np)

#audio_tensor_2d = audio_tensor.view(1, -1)

# Save the numpy array as a WAV file
#torchaudio.save("/disk3/chime/jsalt2020_simulate/output_audio.wav", torch.from_numpy(audio_np), sample_rate=16000)

#torchaudio.save("/disk3/chime/jsalt2020_simulate/output_audio.wav", audio_tensor_2d, sample_rate=16000)


import os
import torch
import torchaudio

# Set paths for input and output directories
input_folder = "/disk4/cyzhang/misp2023/data/train/noise/pt/"  # Path to the folder containing .pt files
output_folder = "/disk3/chime/jsalt2020_simulate/misp_noise/"  # Path to the folder where you want to save .wav files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the input folder
pt_files = [f for f in os.listdir(input_folder) if f.endswith(".pt")]

# Convert each .pt file to .wav
for pt_file in pt_files:
    # Load the .pt file
    audio_tensor = torch.load(os.path.join(input_folder, pt_file))

    # Reshape the audio tensor to have two dimensions (1 channel)
    audio_tensor_2d = audio_tensor.view(1, -1)

    # Generate the output .wav file path
    output_wav_path = os.path.join(output_folder, pt_file.replace(".pt", ".wav"))

    # Save the reshaped tensor as a WAV file
    torchaudio.save(output_wav_path, audio_tensor_2d, sample_rate=16000)

print("Conversion completed.")
