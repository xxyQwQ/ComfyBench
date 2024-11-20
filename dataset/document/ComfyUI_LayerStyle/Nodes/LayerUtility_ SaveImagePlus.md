---
tags:
- Image
- ImageSave
---

# LayerUtility: SaveImage Plus
## Documentation
- Class name: `LayerUtility: SaveImagePlus`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `True`

The SaveImagePlus node is designed to enhance image saving capabilities within a workflow, offering advanced options such as custom file paths, filename prefixes, timestamping, image formatting, quality adjustments, metadata inclusion, and the option to add a blind watermark. It also supports saving workflow configurations as JSON and provides a preview feature, making it a versatile tool for managing output images and related data.
## Input types
### Required
- **`images`**
    - The images to be saved. This parameter is essential for defining the content that will be processed and saved by the node.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`custom_path`**
    - Specifies a custom directory path where the images will be saved. This allows for flexible file management and organization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`filename_prefix`**
    - A prefix added to the filename of the saved images, enabling easy identification and sorting.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`timestamp`**
    - Determines whether and how a timestamp is added to the filename, offering options for none, second, or millisecond precision. This helps in versioning and organizing saved images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Union[NoneType, str]`
- **`format`**
    - The file format for the saved images, with options including PNG and JPG. This choice affects the image's compatibility and quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`quality`**
    - Adjusts the quality of the saved images, applicable for formats that support compression. It allows for balancing between image quality and file size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`meta_data`**
    - A boolean indicating whether to include metadata in the saved images, enhancing the information content of the image files.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blind_watermark`**
    - An optional string to be used as a blind watermark, adding a layer of security or branding to the images without altering their appearance.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`save_workflow_as_json`**
    - When enabled, saves the current workflow configuration as a JSON file alongside the images, useful for documentation or replication purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`preview`**
    - A boolean flag that, when enabled, saves a preview image to a temporary directory, useful for quick checks or as a thumbnail.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`ui`**
    - The node returns a UI component displaying the saved images, providing immediate visual feedback on the output.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveImagePlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "custom_path": ("STRING", {"default": ""}),
                     "filename_prefix": ("STRING", {"default": "comfyui"}),
                     "timestamp": (["None", "second", "millisecond"],),
                     "format": (["png", "jpg"],),
                     "quality": ("INT", {"default": 80, "min": 10, "max": 100, "step": 1}),
                     "meta_data": ("BOOLEAN", {"default": False}),
                     "blind_watermark": ("STRING", {"default": ""}),
                     "save_workflow_as_json": ("BOOLEAN", {"default": False}),
                     "preview": ("BOOLEAN", {"default": True}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_image_plus"
    OUTPUT_NODE = True
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def save_image_plus(self, images, custom_path, filename_prefix, timestamp, format, quality,
                           meta_data, blind_watermark, preview, save_workflow_as_json,
                           prompt=None, extra_pnginfo=None):

        now = datetime.datetime.now()
        custom_path = custom_path.replace("%date", now.strftime("%Y-%m-%d"))
        custom_path = custom_path.replace("%time", now.strftime("%H-%M-%S"))
        filename_prefix = filename_prefix.replace("%date", now.strftime("%Y-%m-%d"))
        filename_prefix = filename_prefix.replace("%time", now.strftime("%H-%M-%S"))
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        temp_sub_dir = generate_random_name('_savepreview_', '_temp', 16)
        temp_dir = os.path.join(folder_paths.get_temp_directory(), temp_sub_dir)
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if blind_watermark != "":
                img_mode = img.mode
                wm_size = watermark_image_size(img)
                import qrcode
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=20,
                    border=1,
                )
                qr.add_data(blind_watermark.encode('utf-8'))
                qr.make(fit=True)
                qr_image = qr.make_image(fill_color="black", back_color="white")
                qr_image = qr_image.resize((wm_size, wm_size), Image.BICUBIC).convert("L")

                y, u, v, _ = image_channel_split(img, mode='YCbCr')
                _u = add_invisibal_watermark(u, qr_image)
                wm_img = image_channel_merge((y, _u, v), mode='YCbCr')

                if img.mode == "RGBA":
                    img = RGB2RGBA(wm_img, img.split()[-1])
                else:
                    img = wm_img.convert(img_mode)

            metadata = None
            if meta_data:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if timestamp == "millisecond":
                file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}'
            elif timestamp == "second":
                file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S")}'
            else:
                file = f'{filename}_{counter:05}'


            preview_filename = ""
            if custom_path != "":
                if not os.path.exists(custom_path):
                    try:
                        os.makedirs(custom_path)
                    except Exception as e:
                        log(f"Error: {NODE_NAME} skipped, because unable to create temporary folder.",
                            message_type='warning')
                        raise FileNotFoundError(f"cannot create custom_path {custom_path}, {e}")

                full_output_folder = os.path.normpath(custom_path)
                # save preview image to temp_dir
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
                try:
                    os.makedirs(temp_dir)
                except Exception as e:
                    print(e)
                    log(f"Error: {NODE_NAME} skipped, because unable to create temporary folder.",
                        message_type='warning')
                try:
                    preview_filename = os.path.join(generate_random_name('saveimage_preview_', '_temp', 16) + '.png')
                    img.save(os.path.join(temp_dir, preview_filename))
                except Exception as e:
                    print(e)
                    log(f"Error: {NODE_NAME} skipped, because unable to create temporary file.", message_type='warning')

            # check if file exists, change filename
            while os.path.isfile(os.path.join(full_output_folder, f"{file}.{format}")):
                counter += 1
                if timestamp == "millisecond":
                    file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]}_{counter:05}'
                elif timestamp == "second":
                    file = f'{filename}_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{counter:05}'
                else:
                    file = f"{filename}_{counter:05}"

            image_file_name = os.path.join(full_output_folder, f"{file}.{format}")
            json_file_name = os.path.join(full_output_folder, f"{file}.json")

            if format == "png":
                img.save(image_file_name, pnginfo=metadata, compress_level= (100 - quality) // 10)
            else:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(image_file_name, quality=quality)
            log(f"{NODE_NAME} -> Saving image to {image_file_name}")

            if save_workflow_as_json:
                try:
                    workflow = (extra_pnginfo or {}).get('workflow')
                    if workflow is None:
                        log('No workflow found, skipping saving of JSON')
                    with open(f'{json_file_name}', 'w') as workflow_file:
                        json.dump(workflow, workflow_file)
                        log(f'Saved workflow to {json_file_name}')
                except Exception as e:
                    log(
                        f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution', message_type="warning")

            if preview:
                if custom_path == "":
                    results.append({
                        "filename": f"{file}.{format}",
                        "subfolder": subfolder,
                        "type": self.type
                    })
                else:
                    results.append({
                        "filename": preview_filename,
                        "subfolder": temp_sub_dir,
                        "type": "temp"
                    })

            counter += 1

        return { "ui": { "images": results } }

```
