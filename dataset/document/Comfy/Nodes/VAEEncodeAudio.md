# VAEEncodeAudio
## Documentation
- Class name: `VAEEncodeAudio`
- Category: `_for_testing/audio`
- Output node: `False`

The VAEEncodeAudio node is designed for encoding audio data into a latent representation using a specified VAE model. It supports audio input of any sample rate by resampling to 44100 Hz if necessary, ensuring compatibility with the VAE's encoding capabilities.
## Input types
### Required
- **`audio`**
    - The 'audio' input is the audio data to be encoded. It includes the waveform and its sample rate, which is resampled to 44100 Hz if not already at this rate, to ensure compatibility with the VAE model's expectations.
    - Comfy dtype: `AUDIO`
    - Python dtype: `Dict[str, Union[torch.Tensor, int]]`
- **`vae`**
    - The 'vae' input specifies the VAE model to be used for encoding the audio data into its latent representation.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is the latent representation of the input audio, encoded by the specified VAE model.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VAEEncodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "_for_testing/audio"

    def encode(self, vae, audio):
        sample_rate = audio["sample_rate"]
        if 44100 != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return ({"samples":t}, )

```
