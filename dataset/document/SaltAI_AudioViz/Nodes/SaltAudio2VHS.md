---
tags:
- Audio
---

# Audio to VHS Audio
## Documentation
- Class name: `SaltAudio2VHS`
- Category: `SALT/AudioViz/Audio/Util`
- Output node: `False`

The SaltAudio2VHS node is designed to transform audio inputs into a format reminiscent of VHS audio quality, providing a unique auditory experience that emulates the characteristics of vintage video cassette sound.
## Input types
### Required
- **`audio`**
    - The 'audio' input accepts audio data to be processed and converted into VHS audio format, playing a crucial role in determining the final output's quality and authenticity.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
## Output types
- **`vhs_audio`**
    - Comfy dtype: `VHS_AUDIO`
    - The output 'vhs_audio' delivers the transformed audio in VHS format, offering a nostalgic audio experience with the distinct sound quality of a bygone era.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudio2VHS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ), 
            },
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("vhs_audio",)

    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Util"

    def convert(self, audio):
        return (lambda : audio,)

```
