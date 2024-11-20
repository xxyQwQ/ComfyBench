---
tags:
- Audio
---

# Audio Compressor
## Documentation
- Class name: `SaltAudioCompressor`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The SaltAudioCompressor node is designed to compress audio signals, reducing the dynamic range of the audio to ensure a more consistent volume level across the track. It utilizes parameters such as threshold, ratio, attack, and release times to control the compression effect, making it suitable for audio post-production and enhancement tasks.
## Input types
### Required
- **`audio`**
    - The raw audio data to be compressed. This input is crucial as it represents the audio signal that will undergo compression, directly influencing the output quality and characteristics.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`threshold_dB`**
    - The threshold level in decibels (dB) above which compression is applied. This parameter determines the loudness level at which the compressor starts to reduce the gain, playing a key role in the dynamics control of the audio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`ratio`**
    - The compression ratio, indicating how much the audio signal is reduced once it crosses the threshold. It's essential for defining the intensity of the compression effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attack_ms`**
    - The attack time in milliseconds, specifying how quickly the compressor reacts to audio exceeding the threshold. It affects the compressor's responsiveness to sudden loudness changes.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`release_ms`**
    - The release time in milliseconds, defining how quickly the compressor stops affecting the audio after it falls below the threshold. This parameter influences the smoothness of the volume level transitions.
    - Comfy dtype: `INT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The compressed audio data, resulting from the application of dynamic range compression to the input audio. This output showcases the effect of the compression settings on the original audio signal.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioCompressor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "threshold_dB": ("FLOAT", {"default": -20.0, "min": -60.0, "max": 0.0}),
                "ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0}),
                "attack_ms": ("INT", {"default": 50, "min": 0, "max": 1000}),
                "release_ms": ("INT", {"default": 200, "min": 0, "max": 3000}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "compress_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def compress_audio(self, audio, threshold_dB, ratio, attack_ms, release_ms):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        audio_segment.export(temp_input_path, format="wav")
        points = f"-80/-80|-60/-60|{threshold_dB}/{threshold_dB + (20 / ratio)}|20/20"
        
        command = [
            'ffmpeg', '-y', '-i', temp_input_path,
            '-filter_complex',
            f'compand=attacks={attack_ms / 1000.0}:decays={release_ms / 1000.0}:points={points}',
            temp_output_path
        ]

        subprocess.run(command, check=True)
        
        with open(temp_output_path, 'rb') as f:
            compressed_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (compressed_audio,)

```
