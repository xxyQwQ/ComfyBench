---
tags:
- Audio
---

# Audio Noise Reduction (Spectral Subtraction)
## Documentation
- Class name: `SaltAudioNoiseReductionSpectralSubtraction`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node is designed to reduce noise in audio files using spectral subtraction. It aims to enhance audio clarity by removing unwanted noise, making it suitable for improving the quality of recordings or audio tracks.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. It serves as the primary input for noise reduction.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`noise_floor`**
    - A parameter to adjust the intensity of noise reduction. Lower values mean less noise reduction, while higher values increase the noise reduction effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The processed audio data with reduced noise.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioNoiseReductionSpectralSubtraction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "noise_floor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "reduce_noise_spectral"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def reduce_noise_spectral(cls, audio, noise_floor):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        with open(temp_input_path, 'wb') as f:
            f.write(audio)

        command = [
            ffmpeg_path, '-y', '-i', temp_input_path,
            '-af', f'afftdn=nr={noise_floor * 100}',
            temp_output_path
        ]

        subprocess.run(command, check=True)

        with open(temp_output_path, 'rb') as f:
            noise_reduced_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (noise_reduced_audio,)

```
