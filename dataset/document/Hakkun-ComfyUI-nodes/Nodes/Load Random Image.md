---
tags:
- Image
- Multimedia
---

# Load Random Image
## Documentation
- Class name: `Load Random Image`
- Category: `Hakkun`
- Output node: `False`

This node is designed to load a random image from a specified directory, optionally including subdirectories. It aims to facilitate the random selection and loading of images for various applications, such as data augmentation, testing, or providing variability in input data for machine learning models.
## Input types
### Required
- **`directory`**
    - Specifies the directory from which to load a random image. The node can optionally search through subdirectories if instructed, enhancing its flexibility in handling file structures.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`subdirectories`**
    - Determines whether the node should include subdirectories in its search for an image file, allowing for a more comprehensive search across a nested directory structure.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed`**
    - A seed value for the random number generator, ensuring reproducibility of the random image selection when the same seed is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The randomly selected image, loaded as a tensor, suitable for further processing or analysis in machine learning workflows.
    - Python dtype: `torch.Tensor`
- **`file name`**
    - Comfy dtype: `STRING`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadRandomImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": (TEXT_TYPE, {"default": ""}),
                "subdirectories": (["ignore", "include"],),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = (IMAGE_TYPE,TEXT_TYPE)
    RETURN_NAMES = ("image","file name")
    FUNCTION = "load_images"

    CATEGORY = "Hakkun"

    def load_images(self, directory: str, subdirectories: str, seed):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")

        # Function to recursively find files in subdirectories
        def find_files_in_dir(directory, subdirectories):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            if subdirectories == "include":
                dir_files = []
                for root, dirs, files in os.walk(directory):
                    dir_files += [os.path.join(root, f) for f in files if any(f.lower().endswith(ext) for ext in valid_extensions)]
            else:
                dir_files = [os.path.join(directory, f) for f in os.listdir(directory) if any(f.lower().endswith(ext) for ext in valid_extensions)]
            return dir_files
        
        dir_files = find_files_in_dir(directory, subdirectories)

        if not dir_files:
            raise FileNotFoundError(f"No valid image files in directory '{directory}'.")

        dir_files = sorted(dir_files)
        
        random.seed(seed)
        random_index = random.randint(0, len(dir_files) - 1)
        image_path = dir_files[random_index]

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        file_name = get_file_name_without_extension(image_path)

        return (image ,file_name)

```
