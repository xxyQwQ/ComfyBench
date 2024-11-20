---
tags:
- LLM
---

# ∞ LLaVA-Next v1 Image Evaluation
## Documentation
- Class name: `LLMLLaVANextImageEvaluator`
- Category: `SALT/Language Toolkit/Tools`
- Output node: `False`

This node is designed to evaluate images using the LLaVA Next V1 model, providing detailed assessments and generating keywords, lists, and documents based on the image content. It aims to facilitate a deeper understanding and categorization of images through advanced image analysis techniques.
## Input types
### Required
- **`lnv1_model`**
    - Specifies the LLaVA Next V1 model to be used for image evaluation, determining the analytical capabilities and accuracy of the assessment.
    - Comfy dtype: `LLAVA_NEXT_V1_MODEL`
    - Python dtype: `str`
- **`images`**
    - The images to be evaluated, serving as the primary input for analysis and categorization by the node.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`max_tokens`**
    - Defines the maximum number of tokens to be used in the evaluation, affecting the depth and detail of the analysis.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`prompt_format`**
    - Determines the format of the prompt used in the evaluation, guiding the model's analysis and output generation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt`**
    - The specific prompt to guide the evaluation, directly influencing the focus and nature of the analysis.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`strings`**
    - Comfy dtype: `STRING`
    - A string representation of the evaluation results, providing a concise summary or keywords.
    - Python dtype: `str`
- **`list`**
    - Comfy dtype: `LIST`
    - A list of structured data or keywords extracted from the image, offering detailed insights.
    - Python dtype: `List[Any]`
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - Generated documents that encapsulate the evaluation findings, providing comprehensive details and context.
    - Python dtype: `List[Document]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMLLaVANextImageEvaluator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lnv1_model": ("LLAVA_NEXT_V1_MODEL",),
                "images": ("IMAGE",),
                "max_tokens": ("INT", {"min": 0, "max": 2048, "default": 48}),
                "prompt_format": ("STRING", {"multiline": True, "dynamicPrompt": False, "default": "[INST] SYSTEM: You are a professional image captioner, follow the directions of the user exactly.\nUSER: <image>\n<prompt>[/INST]"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompt": False, "default": "Describe the image in search engine keyword tags"}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST", "DOCUMENT")
    RETURN_NAMES = ("strings", "list", "documents")
    OUTPUT_IS_LIST =(True, False)

    FUNCTION = "eval_image"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def eval_image(
        self, 
        lnv1_model: LlavaNextV1,
        images: torch.Tensor,  
        max_tokens: int = 48, 
        prompt_format: str = "[INST] SYSTEM: You are a professional image captioner, follow the directions of the user exactly.\nUSER: <image>\n<prompt>[/INST]",
        prompt: str = "Describe the image in search engine keyword tags"
    ):
        results = []
        for image in images:
            pil_image = tensor2pil(image)
            results.append(lnv1_model.eval(pil_image, prompt, prompt_format, max_tokens=max_tokens))
        documents = [Document(text=result, extra_info={"thumbnail": self.b64_thumb(pil_image, (128, 64))}) for result in results]
        logger.info("DOCUMENTS -----")
        logger.data(documents)
        logger.info("END DOCUMENTS -----")
        return (results, results, documents)

    def b64_thumb(self, image, max_size=(128, 128)):
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_string

```
