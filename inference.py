import os
import torch
import librosa
import look2hear.models
import soundfile as sf
from tqdm.auto import tqdm
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=False, sr=44100)
    print(f'INPUT audio.shape = {audio.shape} | samplerate = {samplerate}')
    #audio = dBgain(audio, -6)
    return torch.from_numpy(audio), samplerate

def save_audio(file_path, audio, samplerate=44100):
    #audio = dBgain(audio, +6)
    sf.write(file_path, audio.T, samplerate, subtype="PCM_16")

def process_chunk(chunk):
    chunk = chunk.unsqueeze(0).cuda()
    with torch.no_grad():
        return model(chunk).squeeze(0).squeeze(0).cpu()

def _getWindowingArray(window_size, fade_size):
    # IMPORTANT NOTE :
    # no fades here in the end, only removing the failed ending of the chunk
    fadein = torch.linspace(1, 1, fade_size)
    fadeout = torch.linspace(0, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def dBgain(audio, volume_gain_dB):
    gain = 10 ** (volume_gain_dB / 20)
    gained_audio = audio * gain 
    return gained_audio


def main(input_wav, output_wav):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    global model
    model = look2hear.models.BaseModel.from_pretrain("/content/Apollo/model/pytorch_model.bin", sr=44100, win=20, feature_dim=256, layer=6).cuda()

    test_data, samplerate = load_audio(input_wav)
    
    C = chunk_size * samplerate  # chunk_size seconds to samples
    N = overlap
    step = C // N
    fade_size = 3 * 44100 # 3 seconds
    print(f"N = {N} | C = {C} | step = {step} | fade_size = {fade_size}")
    
    border = C - step
    
    # handle mono inputs correctly
    if len(test_data.shape) == 1:
        test_data = test_data.unsqueeze(0) 

    # Pad the input if necessary
    if test_data.shape[1] > 2 * border and (border > 0):
        test_data = torch.nn.functional.pad(test_data, (border, border), mode='reflect')

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)

    i = 0
    progress_bar = tqdm(total=test_data.shape[1], desc="Processing audio chunks", leave=False)

    while i < test_data.shape[1]:
        part = test_data[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
            else:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

        out = process_chunk(part)

        window = windowingArray
        if i == 0:  # First audio chunk, no fadein
            window[:fade_size] = 1
        elif i + C >= test_data.shape[1]:  # Last audio chunk, no fadeout
            window[-fade_size:] = 1

        result[..., i:i+length] += out[..., :length] * window[..., :length]
        counter[..., i:i+length] += window[..., :length]

        i += step
        progress_bar.update(step)

    progress_bar.close()

    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    # Remove padding if added earlier
    if test_data.shape[1] > 2 * border and (border > 0):
        final_output = final_output[..., border:-border]

    save_audio(output_wav, final_output, samplerate)
    print(f'Success! Output file saved as {output_wav}')

    # Memory clearing
    model.cpu()
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--in_wav", type=str, required=True, help="Path to input wav file")
    parser.add_argument("--out_wav", type=str, required=True, help="Path to output wav file")
    parser.add_argument("--chunk_size", type=int, help="chunk size value in seconds", default=10)
    parser.add_argument("--overlap", type=int, help="Overlap", default=2)
    args = parser.parse_args()
    
    chunk_size = args.chunk_size
    overlap = args.overlap
    print(f'chunk_size = {chunk_size}, overlap = {overlap}')
    
    main(args.in_wav, args.out_wav)
