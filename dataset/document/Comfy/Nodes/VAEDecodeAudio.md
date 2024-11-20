# VAEDecodeAudio
## Documentation
- Class name: `VAEDecodeAudio`
- Category: `_for_testing/audio`
- Output node: `False`

This node is designed for decoding audio data from a latent representation using a specified VAE model. It transforms the latent audio samples back into waveform audio, effectively reconstructing the audio content from its compressed or encoded state.
## Input types
### Required
- **`samples`**
    - The latent representation of audio samples to be decoded. This input is crucial for the reconstruction process, as it contains the compressed information that the VAE model will use to generate the audio waveform.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`vae`**
    - The VAE model used for decoding the latent audio samples. This model is responsible for the transformation process that converts the compressed data back into an audible waveform.
    - Comfy dtype: `VAE`
    - Python dtype: `AudioOobleckVAE`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The decoded audio output, including the waveform and sample rate, reconstructed from the latent samples.
    - Python dtype: `Dict[str, Union[torch.Tensor, int]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"

    CATEGORY = "_for_testing/audio"

    def decode(self, vae, samples):
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        return ({"waveform": audio, "sample_rate": 44100}, )

```
