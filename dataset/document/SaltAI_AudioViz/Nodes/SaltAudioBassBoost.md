---
tags:
- Audio
---

# Audio Bass Boost
## Documentation
- Class name: `SaltAudioBassBoost`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

The SaltAudioBassBoost node enhances the bass frequencies of an audio input, allowing for a richer and deeper sound experience. It applies a low-pass filter and gain adjustment to boost the bass frequencies up to a specified cutoff frequency.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. This input is crucial for determining the audio content that will undergo bass enhancement.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`cutoff_freq`**
    - The frequency (in Hz) below which the audio's bass will be boosted. It defines the threshold for the low-pass filter, affecting the range of bass frequencies enhanced.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`boost_dB`**
    - The amount of gain (in decibels) to apply to the bass frequencies. This determines the intensity of the bass boost effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The processed audio data with enhanced bass frequencies.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioBassBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "cutoff_freq": ("INT", {"default": 150, "min": 20, "max": 300, "step": 1}),
                "boost_dB": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 24.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "boost_bass"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def boost_bass(self, audio, cutoff_freq, boost_dB):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        
        low_pass_gain = original.low_pass_filter(cutoff_freq).apply_gain(boost_dB)
        if len(low_pass_gain) > len(original):
            low_pass_gain = low_pass_gain[:len(original)]

        boosted = original.overlay(low_pass_gain, position=0)
        
        return (get_buffer(boosted), )

```
