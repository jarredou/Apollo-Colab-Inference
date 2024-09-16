import os
import torch
import librosa
import look2hear.models
import soundfile as sf
from tqdm.auto import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

def load_audio(file_path):
    audio, samplerate = librosa.load(file_path, mono=False, sr=44100)
    print(f'INPUT audio.shape = {audio.shape} | samplerate = {samplerate}')
    return torch.from_numpy(audio), samplerate

def save_audio(file_path, audio, samplerate=44100):
    sf.write(file_path, audio.T, samplerate, subtype="PCM_16")

def process_chunk(chunk):
    chunk = chunk.cuda()
    with torch.no_grad():
        return model(chunk)

def main(input_wav, output_wav):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    global model
    model = look2hear.models.BaseModel.from_pretrain("/content/Apollo/model/pytorch_model.bin", sr=44100, win=20, feature_dim=256, layer=6).cuda()

    test_data, samplerate = load_audio(input_wav)
    chunk_length = chunk_size * samplerate  # chunk_size seconds to samples
    num_chunks = (test_data.shape[1] + chunk_length - 1) // chunk_length  # Calculate number of chunks

    processed_chunks = []

    for i in tqdm(range(num_chunks)):
        start = i * chunk_length
        end = min(start + chunk_length, test_data.shape[1])  # Handle last chunk
        chunk = test_data[:, start:end]  # Get the current chunk

        # Process the chunk
        if chunk.shape[0] == 2:  # Stereo
            left_channel = chunk[0].unsqueeze(0).unsqueeze(0).cuda()
            right_channel = chunk[1].unsqueeze(0).unsqueeze(0).cuda()
            out_left = process_chunk(left_channel)
            out_right = process_chunk(right_channel)
            out_stereo = torch.stack((out_left.squeeze(0).squeeze(0).cpu(), out_right.squeeze(0).squeeze(0).cpu()), dim=0)
            processed_chunks.append(out_stereo)
        else:  # Mono
            out = process_chunk(chunk.unsqueeze(0).cuda())
            processed_chunks.append(out.squeeze(0).squeeze(0).cpu())

    # Concatenate all processed chunks
    final_output = torch.cat(processed_chunks, dim=-1)
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
    parser.add_argument("--chunk_size", type=int, help="chunk size value in seconds", default=30)
    args = parser.parse_args()
    print(f'chunk_size = {args.chunk_size}')
    main(args.in_wav, args.out_wav)
