---
tags:
- Image
---

# Stable Video Diffusion
## Documentation
- Class name: `SaltAIStableVideoDiffusion`
- Category: `SALT/API/Stability API`
- Output node: `False`

The SaltAIStableVideoDiffusion node is designed to interface with the Stability AI API for generating videos based on input images. It encapsulates the process of sending an image to the API, managing the request, and processing the returned video data into a sequence of frames. This node leverages advanced video diffusion techniques to create high-quality, stable videos from static images, providing a seamless integration for applications requiring video generation capabilities.
## Input types
### Required
- **`image`**
    - The input image tensor that serves as the basis for video generation. It is crucial for defining the visual content from which the video will be generated, affecting the overall theme and appearance of the output video.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`api_key`**
    - A required parameter for authenticating with the Stability AI API. It ensures authorized access to the video generation service, impacting the node's ability to generate videos.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - An optional parameter to control the randomness of video generation, enhancing the reproducibility of results by allowing for the generation of consistent videos across multiple runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The CFG scale parameter influences the creativity and coherence of the generated video, balancing between fidelity to the input image and the novelty of the generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mbi`**
    - The motion bucket ID parameter guides the motion dynamics in the generated video, affecting how elements within the video move and interact.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`frames`**
    - Comfy dtype: `IMAGE`
    - A sequence of video frames represented as tensors, providing the generated video content in a format suitable for further processing or visualization.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltAIStableVideoDiffusion:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "mbi": ("INT", {"min": 1, "max": 255, "default": 40, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)

    FUNCTION = "generate_video"
    CATEGORY = "SALT/API/Stability API"

    def generate_video(self, image, api_key, seed=0, cfg=2.5, mbi=40):

        if api_key.strip() == "":
            errmsg = "A Stability AI API Key is required for video generaiton."
            logger.error(errmsg)
            raise Exception(errmsg)

        try:
            api = SAIAPI(api_key)

            if image.dim() != 4 and image.size(0) != 1:
                errmsg = "Only one image is allowed for Stability AI Stable Video Generation API."
                logger.error(errmsg)
                raise Exception(errmsg)
            
            image = tensor2pil(image)
            if image.size not in [(1024, 576), (576, 1024), (768, 768)]:
                errmsg = "Image resolution can only be within the following sizes: 1024x576, 576x1024, 768x768"
                logger.error(errmsg)
                raise Exception(errmsg)
            
            frames = api.svd_img2vid(image=image, seed=seed, cfg=cfg, mbi=mbi)
            if frames and len(frames) > 0:
                frame_tensors = [pil2tensor(frame) for frame in frames]
                frame_batch = torch.cat(frame_tensors, dim=0)
            else:
                errmsg = "No frames found from SVD video temp file."
                logger.error(errmsg)
                raise Exception(errmsg)
        except Exception as e:
            logger.error(f"{e}")
            raise e
        
        return (frame_batch, )

```
