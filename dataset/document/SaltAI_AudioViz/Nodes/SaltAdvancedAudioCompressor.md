---
tags:
- Audio
---

# Audio Compressor Advanced
## Documentation
- Class name: `SaltAdvancedAudioCompressor`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

This node provides advanced audio compression capabilities, allowing for detailed control over the compression process through parameters such as threshold, ratio, attack, release, and makeup gain. It is designed to modify the dynamic range of audio signals for enhanced sound quality or specific audio effects.
## Input types
### Required
- **`audio`**
    - The raw audio data to be compressed. This input is crucial for the compression process as it directly influences the output quality and effectiveness of the compression.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`threshold_dB`**
    - Sets the threshold level in decibels for the compressor. Audio signals above this level will be compressed, affecting the loudness and dynamic range of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`ratio`**
    - Determines the compression ratio, which affects how much the audio signal is reduced once it crosses the threshold level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attack_ms`**
    - The time in milliseconds it takes for the compressor to apply compression after the audio signal exceeds the threshold. This parameter influences the responsiveness of the compression effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`release_ms`**
    - The time in milliseconds for the compressor to cease the compression effect after the audio signal falls below the threshold level. It affects the smoothness of the audio output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`makeup_gain`**
    - The gain in decibels to be applied after compression, compensating for any loss in volume due to the compression process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The compressed audio data, resulting from the application of the specified compression settings.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAdvancedAudioCompressor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "threshold_dB": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0}),
                "attack_ms": ("INT", {"default": 5, "min": 1, "max": 100}),
                "release_ms": ("INT", {"default": 50, "min": 10, "max": 1000}),
                "makeup_gain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "compress_detailed_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def compress_detailed_audio(self, audio, threshold_dB, ratio, attack_ms, release_ms, makeup_gain):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input:
            temp_input.write(audio)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        attack_sec = max(attack_ms / 1000.0, 0.01)
        release_sec = max(release_ms / 1000.0, 0.01)
        command = [
            'ffmpeg', '-y', '-i', temp_input_path,
            '-af', f'acompressor=threshold={threshold_dB}dB:ratio={ratio}:attack={attack_sec}:release={release_sec}:makeup={makeup_gain}dB',
            temp_output_path
        ]

        subprocess.run(command, check=True)
        
        with open(temp_output_path, 'rb') as temp_output:
            compressed_audio = temp_output.read()

        os.unlink(temp_input_path)
        os.unlink(temp_output_path)

        return (compressed_audio,)

```
