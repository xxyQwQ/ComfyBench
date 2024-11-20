---
tags:
- Audio
---

# Audio Pitch Shift
## Documentation
- Class name: `SaltAudioPitchShift`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node applies a pitch shift to an audio input by a specified number of semitones, allowing for audio manipulation that can adjust the pitch without altering the playback speed.
## Input types
### Required
- **`audio`**
    - The raw audio data to be pitch-shifted. This input is crucial for determining the source audio material on which the pitch shift effect will be applied.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`semitones`**
    - The number of semitones by which the audio's pitch is to be shifted. This parameter directly influences the pitch alteration, with positive values raising the pitch and negative values lowering it.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The pitch-shifted audio data, resulting from the application of the specified semitone shift to the input audio.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioPitchShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "semitones": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "shift_pitch"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def shift_pitch(cls, audio, semitones):
        TEMP = tempfile.gettempdir()
        os.makedirs(TEMP, exist_ok=True)

        temp_input_path = tempfile.mktemp(suffix='.wav', dir=TEMP)
        temp_output_path = tempfile.mktemp(suffix='.wav', dir=TEMP)

        with open(temp_input_path, 'wb') as f:
            f.write(audio)

        command = [
            'ffmpeg', '-y', '-i', temp_input_path,
            '-af', f'asetrate=44100*2^({semitones}/12),aresample=44100',
            temp_output_path
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during pitch shifting: {e}")
            os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            raise

        with open(temp_output_path, 'rb') as f:
            pitch_shifted_audio = f.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return (pitch_shifted_audio,)

```
