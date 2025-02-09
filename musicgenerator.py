# used google colab
# MusicGen demo

Demo and tutorial created for the youtube channel: https://youtu.be/9-GhKkYA6mA

# @markdown # Setup
!pip install 'torch>=2.0'
!pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft

import math
import IPython
import soundfile as sf
import numpy as np
from tqdm.notebook import tqdm

from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
from audiocraft.utils.notebook import display_audio
import torch
import torchaudio

## Load model
# @markdown ### Select model
# @markdown 1. `facebook/musicgen-small` - 300M transformer decoder. Generates audio at speed 1/2, it takes 2 minutes to generate 1 minute of audio
# @markdown 2. `facebook/musicgen-medium` - 1.5B transformer decoder. Generates audio at speed 1/4
# @markdown 3. `facebook/musicgen-melody` - 1.5B transformer decoder also supporting melody conditioning.
# @markdown 4. `facebook/musicgen-large` - 3.3B transformer decoder. Generates audio at speed 1/4

model_size = 'large' # @param ["small", "medium", "melody", "large"]
USE_DIFFUSION_DECODER = False # @param ["False", "True"] {type:"raw"}
model = MusicGen.get_pretrained(f'facebook/musicgen-{model_size}')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

class ProgressCallback():
    def __init__(self):
        self.tqdm = None
        self.progress = 0

    def __call__(self, progress, total):
        if progress == 1:
            self.tqdm = tqdm(total=total)
            self.progress = 0
        self.tqdm.update(progress - self.progress)
        self.progress = progress
        if progress >= total:
            self.tqdm.close()

model.set_custom_progress_callback(ProgressCallback())
### Define prompts
prompts = [
    'electronic music with futuristic sound, crescendo',
    'electronic music climax, futuristic sound'
]
# @markdown #### Generate audio
# @markdown Select the audio duration in seconds for each prompt and the overlap
# @markdown If using the T4 GPU it should be possible to generate audio of arbitrary duration, but the prompts could not be longer than ~3 minutes

prompt_duration = 30 # @param {type:"integer"}
overlap = 10 # @param {type:"integer"}
output_filepath = 'audio2.mp3' # @param {type:"string"}
display_intermediate_results = True # @param ["False", "True"] {type:"raw"}

model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=int(overlap + prompt_duration),
)

def generate_audio_with_multiple_prompts(prompts, overlap=2, sr=32000, verbose=False):
    torch_audio = generate_audio_with_prompt(prompts[0])[..., :-int(sr*overlap)]
    audio = torch_audio.cpu().numpy().squeeze()
    print(prompts[0])
    for prompt in prompts[1:]:
        if verbose: IPython.display.display(IPython.display.Audio(audio, rate=sr))
        print(prompt)
        torch_audio = continue_audio_with_prompt(torch_audio, prompt, overlap, sr)
        audio = np.concatenate([audio, torch_audio.cpu().numpy().squeeze()[int(sr*overlap):]])
    IPython.display.display(IPython.display.Audio(audio, rate=sr))
    return audio

def generate_audio_with_prompt(prompt):
    output = model.generate(
        descriptions=[prompt],
        progress=True, return_tokens=True
    )
    if USE_DIFFUSION_DECODER:
        audio = mbd.tokens_to_wav(output[1])
    else:
        audio = output[0]
    return audio

def continue_audio_with_prompt(input_audio, prompt, overlap=2, sr=32000):
    output = model.generate_continuation(
        input_audio[..., -int(sr*overlap):],
        sr, [prompt],
        progress=True, return_tokens=True)
    if USE_DIFFUSION_DECODER:
        audio = mbd.tokens_to_wav(output[1])
    else:
        audio = output[0]
    return audio

print(f'{len(prompts)} prompts were given')
estimated_duration = len(prompts)*prompt_duration
print(f'The generated audio will have a total duration of {estimated_duration} seconds')
speed = 0.5 if model_size == 'small' else 0.25
print(f'Using a T4 GPU will take around {estimated_duration/speed:.0f} seconds to generate the audio')
audio = generate_audio_with_multiple_prompts(
    prompts, overlap=overlap, verbose=display_intermediate_results)
sf.write(output_filepath, audio, 32000)
print(f'The audio was saved to {output_filepath}')

## Credits

#This demo builds on: https://github.com/facebookresearch/audiocraft/blob/main/demos/musicgen_demo.ipynb
#From  Guillermo Barbadillo YT channel
#Other useful links:

#- https://github.com/facebookresearch/audiocraft/
#- https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/musicgen.py
#- https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/
#- https://colab.research.google.com/notebooks/forms.ipynb#scrollTo=3jKM6GfzlgpS
#-https://www.youtube.com/watch?v=9-GhKkYA6mA
