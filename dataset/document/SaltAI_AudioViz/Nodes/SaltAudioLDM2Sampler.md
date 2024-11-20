---
tags:
- Audio
---

# AudioLDM2 Sampler
## Documentation
- Class name: `SaltAudioLDM2Sampler`
- Category: `SALT/AudioViz/Audio/AudioLDM2`
- Output node: `False`

This node is designed for sampling audio using the LDM2 model, providing a specialized approach to generating or processing audio data with advanced machine learning techniques. It focuses on leveraging the capabilities of the LDM2 architecture to offer high-quality audio sampling functionalities.
## Input types
### Required
- **`audioldm2_model`**
    - Specifies the LDM2 model to be used for audio sampling, influencing the quality and characteristics of the generated audio.
    - Comfy dtype: `AUDIOLDM_MODEL`
    - Python dtype: `AudioLDM2Pipeline`
- **`seed`**
    - Determines the random seed for generating audio, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to be used in the sampling process, affecting the detail and quality of the output audio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`guidance_scale`**
    - Controls the strength of the conditioning on the input prompts, influencing the adherence to the specified prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`audio_length_seconds`**
    - Sets the length of the generated audio in seconds.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`num_waveforms`**
    - Determines the number of audio waveforms to generate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`positive_prompt`**
    - The text prompt that encourages certain elements or themes in the generated audio.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_prompt`**
    - The text prompt that discourages certain elements or themes in the generated audio.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The sampled audio output, generated based on the specified inputs and model.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioLDM2Sampler:  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audioldm2_model": ("AUDIOLDM_MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "max": 12.0, "min": 1.0}),
                "audio_length_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_waveforms": ("INT", {"default": 3, "min": 1}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False})
            },
            "optional": {

            },
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio", )

    FUNCTION = "sample"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/AudioLDM2"

    def sample(self, audioldm2_model, **kwargs):
        generator = torch.Generator("cuda").manual_seed(kwargs.get("seed", 0))
        steps = kwargs.get("steps", 200)

        def update_comfy_pbar(step, timestep, latents, **kwargs):
            comfy_pbar.update(1)

        comfy_pbar = ProgressBar(steps)

        audio = audioldm2_model(
            kwargs.get("positive_prompt", "The sound of a hammer hitting a wooden surface."),
            negative_prompt=kwargs.get("negative_prompt", "Low quality."),
            num_inference_steps=steps,
            guidance_scale=kwargs.get("guidance_scale", 3.5),
            audio_length_in_s=kwargs.get("audio_length_seconds", 10.0),
            num_waveforms_per_prompt=kwargs.get("num_waveforms", 3),
            generator=generator,
            output_type="np",
            callback=update_comfy_pbar
        ).audios

        wave_bytes = numpy2wav(audio[0])
        return (wave_bytes, )

```
