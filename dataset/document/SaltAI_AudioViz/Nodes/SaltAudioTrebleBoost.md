---
tags:
- Audio
---

# Audio Treble Boost
## Documentation
- Class name: `SaltAudioTrebleBoost`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node enhances the treble frequencies of an audio file, allowing for a clearer and more pronounced high-end sound. It applies a high-pass filter at a specified cutoff frequency and then boosts the gain of frequencies above this threshold.
## Input types
### Required
- **`audio`**
    - The audio input to be processed for treble enhancement. It serves as the primary content for treble frequency boosting.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`cutoff_freq`**
    - The cutoff frequency for the high-pass filter, determining the lowest frequency to be boosted in the treble enhancement process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`boost_dB`**
    - The amount of gain, in decibels, to apply to frequencies above the cutoff frequency, enhancing the treble portion of the audio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The processed audio with enhanced treble frequencies.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioTrebleBoost:
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
    FUNCTION = "treble_bass"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def treble_bass(self, audio, cutoff_freq, boost_dB):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        
        high_pass_gain = original.high_pass_filter(cutoff_freq).apply_gain(boost_dB)
        if len(high_pass_gain) > len(original):
            high_pass_gain = high_pass_gain[:len(original)]

        boosted = original.overlay(high_pass_gain, position=0)
        
        return (get_buffer(boosted), )

```
