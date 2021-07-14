
import os
import time
import warnings
warnings.filterwarnings("ignore")

import gin
import numpy as np
from scipy.io import wavfile
import torch

from neural_waveshaping_synthesis.data.utils.loudness_extraction import extract_perceptual_loudness
from neural_waveshaping_synthesis.data.utils.mfcc_extraction import extract_mfcc
from neural_waveshaping_synthesis.data.utils.f0_extraction import extract_f0_with_crepe
from neural_waveshaping_synthesis.data.utils.preprocess_audio import preprocess_audio, convert_to_float32_audio, make_monophonic, resample_audio
from neural_waveshaping_synthesis.models.modules.shaping import FastNEWT
from neural_waveshaping_synthesis.models.neural_waveshaping import NeuralWaveshaping
import gradio as gr

torch.hub.download_url_to_file('https://benhayes.net/assets/audio/nws_examples/tt/tt1_in.wav', 'test1.wav')
torch.hub.download_url_to_file('https://benhayes.net/assets/audio/nws_examples/tt/tt2_in.wav', 'test2.wav')
torch.hub.download_url_to_file('https://benhayes.net/assets/audio/nws_examples/tt/tt3_in.wav', 'test3.wav')


try:
  gin.constant("device", "cuda" if torch.cuda.is_available() else "cpu")
except ValueError as err:
  pass

from scipy.io.wavfile import write


gin.parse_config_file("gin/models/newt.gin")
gin.parse_config_file("gin/data/urmp_4second_crepe.gin")

checkpoints = dict(Violin="vn", Flute="fl", Trumpet="tpt")

use_gpu = False 
dev_string = "cuda" if use_gpu else "cpu"
device = torch.device(dev_string)

selected_checkpoint_name = "Violin"
selected_checkpoint = checkpoints[selected_checkpoint_name]

checkpoint_path = os.path.join(
    "checkpoints/nws", selected_checkpoint)
model = NeuralWaveshaping.load_from_checkpoint(
    os.path.join(checkpoint_path, "last.ckpt")).to(device)
original_newt = model.newt
model.eval()
data_mean = np.load(
    os.path.join(checkpoint_path, "data_mean.npy"))
data_std = np.load(
    os.path.join(checkpoint_path, "data_std.npy"))

def inference(wav):
    rate, audio = wavfile.read(wav.name)
    audio = convert_to_float32_audio(make_monophonic(audio))
    audio = resample_audio(audio, rate, model.sample_rate)

    use_full_crepe_model = False 
    with torch.no_grad():
        f0, confidence = extract_f0_with_crepe(
            audio,
            full_model=use_full_crepe_model,
            maximum_frequency=1000)
        loudness = extract_perceptual_loudness(audio)



    octave_shift = 1 
    loudness_scale = 0.5 

 
    loudness_floor = 0 
    loudness_conf_filter = 0 
    pitch_conf_filter = 0 

    pitch_smoothing = 0 
    loudness_smoothing = 0 

    with torch.no_grad():
        f0_filtered = f0 * (confidence > pitch_conf_filter)
        loudness_filtered = loudness * (confidence > loudness_conf_filter)
        f0_shifted = f0_filtered * (2 ** octave_shift)
        loudness_floored = loudness_filtered * (loudness_filtered > loudness_floor) - loudness_floor
        loudness_scaled = loudness_floored * loudness_scale
    
        loud_norm = (loudness_scaled - data_mean[1]) / data_std[1]
    
        f0_t = torch.tensor(f0_shifted, device=device).float()
        loud_norm_t = torch.tensor(loud_norm, device=device).float()

        if pitch_smoothing != 0:
            f0_t = torch.nn.functional.conv1d(
            f0_t.expand(1, 1, -1),
            torch.ones(1, 1, pitch_smoothing * 2 + 1, device=device) /
                (pitch_smoothing * 2 + 1),
            padding=pitch_smoothing
            ).squeeze()
        f0_norm_t = torch.tensor((f0_t.cpu() - data_mean[0]) / data_std[0], device=device).float()

        if loudness_smoothing != 0:
            loud_norm_t = torch.nn.functional.conv1d(
            loud_norm_t.expand(1, 1, -1),
            torch.ones(1, 1, loudness_smoothing * 2 + 1, device=device) /
                (loudness_smoothing * 2 + 1),
            padding=loudness_smoothing
            ).squeeze()
        f0_norm_t = torch.tensor((f0_t.cpu() - data_mean[0]) / data_std[0], device=device).float()
        
        control = torch.stack((f0_norm_t, loud_norm_t), dim=0)

    model.newt = FastNEWT(original_newt)

    with torch.no_grad():
        start_time = time.time()
        out = model(f0_t.expand(1, 1, -1), control.unsqueeze(0))
        run_time = time.time() - start_time
    sample_rates=model.sample_rate
    rtf = (audio.shape[-1] / model.sample_rate) / run_time
    write('test.wav', sample_rates, out.detach().cpu().numpy().T)
    return 'test.wav'

inputs = gr.inputs.Audio(label="input audio", type="file")
outputs =  gr.outputs.Audio(label="output audio", type="file")


title = "neural waveshaping synthesis"
description = "demo for neural waveshaping synthesis: efficient neural audio synthesis in the waveform domain for timbre transfer. To use it, simply add your audio, or click one of the examples to load them. Read more at the links below. Input audio should be in WAV format similar to the example audio below"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2107.05050'>neural waveshaping synthesis</a> | <a href='https://github.com/ben-hayes/neural-waveshaping-synthesis'>Github Repo</a></p>"

examples = [
 ['test1.wav'],
 ['test2.wav'],
 ['test3.wav']
]

gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
