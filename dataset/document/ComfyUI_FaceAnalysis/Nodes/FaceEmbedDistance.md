---
tags:
- Face
---

# Face Embeds Distance
## Documentation
- Class name: `FaceEmbedDistance`
- Category: `FaceAnalysis`
- Output node: `False`

The FaceEmbedDistance node calculates the distance between facial embeddings using cosine or Euclidean (L2) metrics. It supports normalization of embeddings for distance calculation, enabling a versatile approach to measuring facial similarity or dissimilarity.
## Input types
### Required
- **`analysis_models`**
    - Specifies the face analysis models to be used for generating embeddings from the input images. It plays a critical role in the accuracy of the distance measurements.
    - Comfy dtype: `ANALYSIS_MODELS`
    - Python dtype: `object`
- **`reference`**
    - The reference image against which other images are compared. This image is used to generate a facial embedding for similarity comparison.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`image`**
    - The image to be compared against the reference. It is used to generate a facial embedding for calculating the distance to the reference embedding.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`similarity_metric`**
    - Defines the metric used for calculating the distance between embeddings. Options include 'L2_norm', 'cosine', and 'euclidean', affecting the comparison's sensitivity and outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`filter_thresh`**
    - A threshold for filtering out distances above a certain value, enhancing the focus on closer matches.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`filter_best`**
    - Limits the number of results to the best matches below the specified threshold, optimizing the search for similarity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`generate_image_overlay`**
    - A boolean indicating whether to overlay the reference image on top of the comparison image, visually representing the similarity measurement.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The image result of overlaying the reference image on the comparison image, if enabled, providing a visual representation of the similarity.
    - Python dtype: `numpy.ndarray`
- **`distance`**
    - Comfy dtype: `FLOAT`
    - The calculated distance between the facial embeddings of the reference and comparison images, quantifying their similarity.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FaceEmbedDistance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "reference": ("IMAGE", ),
                "image": ("IMAGE", ),
                "similarity_metric": (["L2_norm", "cosine", "euclidean"], ),
                "filter_thresh": ("FLOAT", { "default": 100.0, "min": 0.001, "max": 100.0, "step": 0.001 }),
                "filter_best": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "generate_image_overlay": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "distance")
    FUNCTION = "analize"
    CATEGORY = "FaceAnalysis"

    def analize(self, analysis_models, reference, image, similarity_metric, filter_thresh, filter_best, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        if filter_thresh == 0.0:
            filter_thresh = analysis_models.thresholds[similarity_metric]

        # you can send multiple reference images in which case the embeddings are averaged
        ref = []
        for i in reference:
            ref_emb = analysis_models.get_embeds(np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')))
            if ref_emb is not None:
                ref.append(torch.from_numpy(ref_emb))
        
        if ref == []:
            raise Exception('No face detected in reference image')

        ref = torch.stack(ref)
        ref = np.array(torch.mean(ref, dim=0))

        out = []
        out_dist = []
        
        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB'))

            img = analysis_models.get_embeds(img)

            if img is None: # No face detected
                dist = 100.0
                norm_dist = 0
            else:
                if np.array_equal(ref, img): # Same face
                    dist = 0.0
                    norm_dist = 0.0
                else:
                    if similarity_metric == "L2_norm":
                        #dist = euclidean_distance(ref, img, True)
                        ref = ref / np.linalg.norm(ref)
                        img = img / np.linalg.norm(img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    elif similarity_metric == "cosine":
                        dist = np.float64(1 - np.dot(ref, img) / (np.linalg.norm(ref) * np.linalg.norm(img)))
                        #dist = cos_distance(ref, img)
                    else:
                        #dist = euclidean_distance(ref, img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    
                    norm_dist = min(1.0, 1 / analysis_models.thresholds[similarity_metric] * dist)
           
            if dist <= filter_thresh:
                print(f"\033[96mFace Analysis: value: {dist}, normalized: {norm_dist}\033[0m")

                if generate_image_overlay:
                    tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGBA')
                    txt = Image.new('RGBA', (image.shape[2], txt_height), color=background_color)
                    draw = ImageDraw.Draw(txt)
                    draw.text((0, 0), f"VALUE: {round(dist, 3)} | DIST: {round(norm_dist, 3)}", font=font, fill=(255, 255, 255, 255))
                    composite = Image.new('RGBA', tmp.size)
                    composite.paste(txt, (0, tmp.height - txt.height))
                    composite = Image.alpha_composite(tmp, composite)
                    out.append(T.ToTensor()(composite).permute(1, 2, 0))
                else:
                    out.append(i)

                out_dist.append(dist)

        if not out:
            raise Exception('No image matches the filter criteria.')
    
        out = torch.stack(out)

        # filter out the best matches
        if filter_best > 0:
            filter_best = min(filter_best, len(out))
            out_dist, idx = torch.topk(torch.tensor(out_dist), filter_best, largest=False)
            out = out[idx]
            out_dist = out_dist.cpu().numpy().tolist()
        
        if out.shape[3] > 3:
            out = out[:, :, :, :3]

        return(out, out_dist,)

```
