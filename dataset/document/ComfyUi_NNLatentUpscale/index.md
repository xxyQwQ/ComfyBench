# ComfyUi_NNLatentUpscale
ComfyUi_NNLatentUpscale is a custom node for ComfyUI designed for upscaling latents quickly using a small neural network, avoiding the need to decode and encode with VAE. It is intended for workflows where an initial image is generated at a lower resolution, then the latent is upscaled and fed back into the stable diffusion u-net for a low noise diffusion pass, achieving a significant speed advantage over traditional VAE upscaling methods with minimal quality loss.

## Tags
Latent

## Repo info
- Repo url: `https://github.com/Ttl/ComfyUi_NNLatentUpscale.git`
- Commit hash: `08105da31dbd7a54569661e135835e73bd8064b0`

## Licenses
- **GPL-3.0-only**: `LICENSE`

Visit [licenses page](licenses.md) for the details
