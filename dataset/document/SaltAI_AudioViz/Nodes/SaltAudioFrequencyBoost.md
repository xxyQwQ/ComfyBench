---
tags:
- Audio
---

# Audio Frequency Boost
## Documentation
- Class name: `SaltAudioFrequencyBoost`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node applies a frequency-specific boost to an audio file using an equalization filter. It enhances or attenuates a specified frequency range of the audio signal, allowing for precise audio frequency manipulation.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. This input is crucial for applying the frequency boost effect to the audio content.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`frequency`**
    - The center frequency to boost, around which the equalization filter is applied. This parameter determines the specific frequency range that will be enhanced or attenuated.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`bandwidth`**
    - Defines the width of the frequency range around the center frequency that will be affected by the boost. It allows for control over how broad or narrow the boosted frequency range is.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`gain_dB`**
    - The amount of gain (in decibels) to apply to the specified frequency range. This determines the intensity of the boost effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The modified audio data after the frequency boost has been applied. It reflects the changes made to the audio's frequency spectrum.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioFrequencyBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "frequency": ("INT", {"min": 20, "max": 20000, "default": 1000}), 
                "bandwidth": ("FLOAT", {"default": 2.0}),
                "gain_dB": ("FLOAT", {"min": -60, "max": 60, "default": 0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "boost_frequency"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def boost_frequency(self, audio, frequency, bandwidth, gain_dB):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input:
            temp_input.write(audio)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        
        eq_filter = f"equalizer=f={frequency}:width_type=o:width={bandwidth}:g={gain_dB}"
        command = [ffmpeg_path, "-y", "-i", temp_input_path, "-af", eq_filter, temp_output_path]

        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(temp_output_path, "rb") as temp_output:
                modified_audio_data = temp_output.read()
                
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
                
            return (modified_audio_data,)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply frequency boost with FFmpeg: {e}")
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            return (audio,)

```
