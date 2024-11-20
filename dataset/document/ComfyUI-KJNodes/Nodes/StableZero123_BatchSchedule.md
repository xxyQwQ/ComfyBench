---
tags:
- Conditioning
---

# Stable Zero123 Batch Schedule
## Documentation
- Class name: `StableZero123_BatchSchedule`
- Category: `KJNodes/experimental`
- Output node: `False`

This node is designed to manage and schedule batch operations for the StableZero123 model, optimizing the processing of multiple inputs in a batched manner for efficiency and performance.
## Input types
### Required
- **`clip_vision`**
    - Represents the CLIP vision model input, used to provide visual context or features for processing.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `str`
- **`init_image`**
    - Initial image input for the model to process or modify.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`vae`**
    - The variational autoencoder used for encoding or decoding images in the model's processing pipeline.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`width`**
    - Specifies the width of the output image, allowing for customization of the image dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the output image, allowing for customization of the image dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Determines the number of images processed in a single batch, affecting efficiency and performance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation`**
    - Defines the interpolation method used in image processing, affecting the smoothness and quality of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`azimuth_points_string`**
    - A string defining azimuth points for 3D model orientation, used in scheduling the model's view direction.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`elevation_points_string`**
    - A string defining elevation points for 3D model orientation, used in scheduling the model's view direction.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning output from the model, used for enhancing certain features or aspects in the generated content.
    - Python dtype: `Conditioning`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning output from the model, used for suppressing certain features or aspects in the generated content.
    - Python dtype: `Conditioning`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Represents the latent space encoding of the input, capturing the essential features for further processing or generation.
    - Python dtype: `Latent`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StableZero123_BatchSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                              "azimuth_points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                              "elevation_points_string": ("STRING", {"default": "0:(0.0),\n7:(0.0),\n15:(0.0)\n", "multiline": True}),
                             }}
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "KJNodes/experimental"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, azimuth_points_string, elevation_points_string, interpolation):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        def ease_in(t):
            return t * t
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)
        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the azimuth input string into a list of tuples
        azimuth_points = []
        azimuth_points_string = azimuth_points_string.rstrip(',\n')
        for point_str in azimuth_points_string.split(','):
            frame_str, azimuth_str = point_str.split(':')
            frame = int(frame_str.strip())
            azimuth = float(azimuth_str.strip()[1:-1]) 
            azimuth_points.append((frame, azimuth))
        # Sort the points by frame number
        azimuth_points.sort(key=lambda x: x[0])

        # Parse the elevation input string into a list of tuples
        elevation_points = []
        elevation_points_string = elevation_points_string.rstrip(',\n')
        for point_str in elevation_points_string.split(','):
            frame_str, elevation_str = point_str.split(':')
            frame = int(frame_str.strip())
            elevation_val = float(elevation_str.strip()[1:-1]) 
            elevation_points.append((frame, elevation_val))
        # Sort the points by frame number
        elevation_points.sort(key=lambda x: x[0])

        # Index of the next point to interpolate towards
        next_point = 1
        next_elevation_point = 1

        positive_cond_out = []
        positive_pooled_out = []
        negative_cond_out = []
        negative_pooled_out = []
        
        #azimuth interpolation
        for i in range(batch_size):
            # Find the interpolated azimuth for the current frame
            while next_point < len(azimuth_points) and i >= azimuth_points[next_point][0]:
                next_point += 1
            # If next_point is equal to the length of points, we've gone past the last point
            if next_point == len(azimuth_points):
                next_point -= 1  # Set next_point to the last index of points
            prev_point = max(next_point - 1, 0)  # Ensure prev_point is not less than 0

            # Calculate fraction
            if azimuth_points[next_point][0] != azimuth_points[prev_point][0]:  # Prevent division by zero
                fraction = (i - azimuth_points[prev_point][0]) / (azimuth_points[next_point][0] - azimuth_points[prev_point][0])
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                # Use the new interpolate_angle function
                interpolated_azimuth = interpolate_angle(azimuth_points[prev_point][1], azimuth_points[next_point][1], fraction)
            else:
                interpolated_azimuth = azimuth_points[prev_point][1]
            # Interpolate the elevation
            next_elevation_point = 1
            while next_elevation_point < len(elevation_points) and i >= elevation_points[next_elevation_point][0]:
                next_elevation_point += 1
            if next_elevation_point == len(elevation_points):
                next_elevation_point -= 1
            prev_elevation_point = max(next_elevation_point - 1, 0)

            if elevation_points[next_elevation_point][0] != elevation_points[prev_elevation_point][0]:
                fraction = (i - elevation_points[prev_elevation_point][0]) / (elevation_points[next_elevation_point][0] - elevation_points[prev_elevation_point][0])
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                interpolated_elevation = interpolate_angle(elevation_points[prev_elevation_point][1], elevation_points[next_elevation_point][1], fraction)
            else:
                interpolated_elevation = elevation_points[prev_elevation_point][1]

            cam_embeds = camera_embeddings(interpolated_elevation, interpolated_azimuth)
            cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

            positive_pooled_out.append(t)
            positive_cond_out.append(cond)
            negative_pooled_out.append(torch.zeros_like(t))
            negative_cond_out.append(torch.zeros_like(pooled))

        # Concatenate the conditions and pooled outputs
        final_positive_cond = torch.cat(positive_cond_out, dim=0)
        final_positive_pooled = torch.cat(positive_pooled_out, dim=0)
        final_negative_cond = torch.cat(negative_cond_out, dim=0)
        final_negative_pooled = torch.cat(negative_pooled_out, dim=0)

        # Structure the final output
        final_positive = [[final_positive_cond, {"concat_latent_image": final_positive_pooled}]]
        final_negative = [[final_negative_cond, {"concat_latent_image": final_negative_pooled}]]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (final_positive, final_negative, {"samples": latent})

```
