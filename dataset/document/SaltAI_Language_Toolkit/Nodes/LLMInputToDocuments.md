---
tags:
- LLM
- LlamaIndex
---

# ∞ Input to Documents
## Documentation
- Class name: `LLMInputToDocuments`
- Category: `SALT/Language Toolkit/Documents`
- Output node: `False`

This node is designed to convert various types of input data into a standardized document format, facilitating their processing and analysis in language models. It supports a wide range of input types, including dictionaries, lists, and primitive data types like strings, integers, and floats, and transforms them into a structured document format with optional metadata.
## Input types
### Required
- **`input_data`**
    - The primary input for the node, accepting a wide variety of data types including dictionaries, lists, strings, integers, and floats. This flexibility allows for the processing of diverse data forms into a standardized document format.
    - Comfy dtype: `*`
    - Python dtype: `Union[Dict, List, str, int, float]`
### Optional
- **`extra_info`**
    - Optional metadata in JSON string format that can be attached to the generated documents, providing additional context or information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`concat_input`**
    - A boolean flag indicating whether multiple input items should be concatenated into a single document.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`resize_images`**
    - A boolean flag that determines whether images within the input data should be resized according to specified maximum dimensions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`max_image_width`**
    - The maximum width to which images in the input data can be resized, enhancing uniformity and processing efficiency.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_image_height`**
    - The maximum height to which images in the input data can be resized, ensuring consistency across processed documents.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The output of the node, consisting of standardized documents formatted from the input data, potentially including resized images and attached metadata.
    - Python dtype: `List[Document]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMInputToDocuments:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": (WILDCARD,),
            },
            "optional": {
                "extra_info": ("STRING", {"default": "{}"}),
                "concat_input": ("BOOLEAN", {"default": False}),
                "resize_images": ("BOOLEAN", {"default": True}),
                "max_image_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION}),
                "max_image_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION})
            }
        }

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("documents",)

    FUNCTION = "to_documents"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Documents"

    def to_documents(self, input_data, extra_info="{}", concat_input=False, resize_images=True, max_image_width=1024, max_image_height=1024):
        documents = []

        # Try to parse extra_info
        try:
            extra_info = json.loads(extra_info) if extra_info.strip() else {}
        except json.JSONDecodeError:
            logger.error("Invalid JSON for `extra_info` supplied, defaulting to empty `extra_info` dict.")
            extra_info = {}

        if not isinstance(extra_info, dict):
            logger.error("Failed to decode `extra_info`, defaulting to empty `extra_info` dict.")
            extra_info = {}

        # Dict to string doc
        if isinstance(input_data, dict) and concat_input:
            doc_text = "\n".join(f"{key}: {value}" for key, value in input_data.items())
            documents.append(Document(text=doc_text, metadata={"source_type": "dict", **extra_info}))

        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                doc_text = f"{key}: {value}"
                documents.append(Document(text=doc_text, metadata={"source_type": "dict", **extra_info}))

        # List to string doc
        elif isinstance(input_data, list) and concat_input:
            doc_text = "\n".join(str(item) for item in input_data)
            documents.append(Document(text=doc_text, metadata={"source_type": "list", **extra_info}))

        elif isinstance(input_data, list):
            for item in input_data:
                doc_text = str(item)
                documents.append(Document(text=doc_text, metadata={"source_type": "list", **extra_info}))

        # Primitive to string doc
        elif isinstance(input_data, (str, int, float)):
            documents.append(Document(text=str(input_data), metadata={}))

        elif isinstance(input_data, torch.Tensor):

            temp = folder_paths.get_temp_directory()
            os.makedirs(temp, exist_ok=True)
            output_path = os.path.join(temp, str(uuid.uuid4()))
            os.makedirs(output_path, exist_ok=True)

            images = []
            image_paths = []
            for img in input_data:
                if img.shape[-1] == 3:
                    images.append(self.tensor2pil(img))
                else:
                    images.append(self.mask2pil(img))

            if not images:
                raise ValueError("Invalid image tensor input provided to convert to PIL!")
            
            try:
                for index, pil_image in enumerate(images):
                    if resize_images:
                        pil_image = self.resize_image(pil_image, max_size=(max_image_width, max_image_height))
                    file_prefix = "llm_image_input_"
                    file_ext = ".jpeg"
                    filename = f"{file_prefix}_{index:04d}{file_ext}"
                    image_path = os.path.join(output_path, filename)
                    pil_image.save(image_path, quality=100)
                    image_paths.append(image_path)

                    if os.path.exists(image_path):
                        logger.info(f"[SALT] Saved LLM document image to `{image_path}`")
                    else:
                        logger.error(f"[SALT] Unable to save LLM document image to `{image_path}`")

            except Exception as e:
                raise e
            
            reader = SimpleDirectoryReader(
                input_dir=output_path,
                exclude_hidden=True,
                recursive=False
            )
            
            documents = reader.load_data()

            if not documents:
                raise ValueError("No documents found in the specified directory.")
            
        else:
            raise ValueError(f"LLMInputToDocuments does not support type `{type(input_data).__name__}`. Please provide: dict, list, str, int, float.")

        return (documents,)
    
    def tensor2pil(self, x):
        return Image.fromarray(np.clip(255. * x.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def mask2pil(self, x):
        x = 1. - x
        if x.ndim != 3:
            logger.warning(f"Expected a 3D tensor ([N, H, W]). Got {x.ndim} dimensions.")
            x = x.unsqueeze(0) 
        x_np = x.cpu().numpy()
        if x_np.ndim != 3:
            x_np = np.expand_dims(x_np, axis=0) 
        return Image.fromarray(np.clip(255. * x_np[0, :, :], 0, 255).astype(np.uint8), 'L')
    
    def resize_image(self, image, max_size=(1024, 1024)):
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

```
