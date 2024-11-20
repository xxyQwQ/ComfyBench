---
tags:
- Audio
---

# Audio De-esser
## Documentation
- Class name: `SaltAudioDeesser`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

The SaltAudioDeesser node is designed to process audio files by applying a de-esser effect, which reduces or eliminates sibilance (harsh 's' or 'sh' sounds) from the audio. This is achieved through the use of command-line tools to dynamically adjust the intensity, amount, and frequency of the de-essing effect based on user inputs.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. This input is crucial as it is the target for the de-esser effect, aiming to reduce sibilance without affecting the overall quality of the audio.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`intensity`**
    - Defines the intensity of the de-esser effect. Higher values result in more aggressive sibilance reduction.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`amount`**
    - Specifies the amount of sibilance reduction to apply. This controls how much of the detected sibilance is reduced.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`frequency_keep`**
    - The frequency above which sibilance is reduced. This helps in preserving the desired frequencies while reducing unwanted sibilance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The audio data after the de-esser effect has been applied, with reduced sibilance.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioDeesser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "intensity": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 1.0}),
                "amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "frequency_keep": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_deesser"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_deesser(cls, audio, intensity, amount, frequency_keep):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_input, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=TEMP) as temp_output:

                temp_input.write(audio)
                temp_input.flush()

                command = [
                    'ffmpeg', '-y', '-i', temp_input.name,
                    '-af', f'deesser=i={intensity}:m={amount}:f={frequency_keep}:s=o',
                    temp_output.name
                ]

                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                temp_output.flush()

            with open(temp_output.name, 'rb') as f:
                processed_audio = f.read()

        finally:
            # Retry mechanism for deletion
            def safe_delete(file_path):
                for attempt in range(3):
                    try:
                        os.unlink(file_path)
                        break
                    except PermissionError:
                        time.sleep(0.1)

            safe_delete(temp_input.name)
            safe_delete(temp_output.name)

        return (processed_audio, )

```
