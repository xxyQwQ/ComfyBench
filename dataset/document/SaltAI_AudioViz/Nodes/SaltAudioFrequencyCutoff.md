---
tags:
- Audio
---

# Audio Frequency Cutoff
## Documentation
- Class name: `SaltAudioFrequencyCutoff`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node applies a frequency cutoff to an audio file, using a specified filter type and cutoff frequency. It leverages FFmpeg to modify the audio signal, either attenuating or completely removing frequencies beyond the specified cutoff point, depending on the filter type used.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. This input is crucial as it represents the audio signal that will undergo the frequency cutoff operation.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`filter_type`**
    - Specifies the type of filter to apply for the frequency cutoff, such as low-pass or high-pass, which determines how frequencies above or below the cutoff point are affected.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`cutoff_frequency`**
    - The frequency threshold at which the audio signal will be cut off. Frequencies beyond this point will be attenuated or removed, depending on the filter type.
    - Comfy dtype: `INT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The modified audio data after the frequency cutoff has been applied, with certain frequencies attenuated or removed based on the filter type and cutoff frequency.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioFrequencyCutoff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "filter_type": (["lowpass", "highpass"], ),
                "cutoff_frequency": ("INT", {"min": 20, "max": 20000, "default": 1000}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_cutoff"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_cutoff(self, audio, filter_type, cutoff_frequency):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input:
            temp_input.write(audio)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        
        filter_command = f"{filter_type}=f={cutoff_frequency}"
        command = [ffmpeg_path, '-y', "-i", temp_input_path, "-af", filter_command, temp_output_path]

        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            with open(temp_output_path, "rb") as temp_output:
                modified_audio_data = temp_output.read()
                
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
                
            return (modified_audio_data,)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply frequency cutoff with FFmpeg: {e}")
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            return (audio,)

```
