---
tags:
- Audio
- SaltNodes
---

# Salt Workflow Output
## Documentation
- Class name: `SaltOutput`
- Category: `SALT/IO`
- Output node: `True`

The SaltOutput node is designed to handle the final output processing within the SaltAI framework, focusing on formatting and logging the output data according to specified parameters. It encapsulates the functionality to dynamically generate output metadata, manage file paths, and support a variety of output types including images, audio, and zip files, ensuring the output is correctly structured and accessible.
## Input types
### Required
- **`output_name`**
    - Specifies the name of the output, which is used to label and identify the output data. It plays a crucial role in organizing and accessing the output effectively.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`output_desc`**
    - Provides a description for the output, offering context or additional details about the nature or content of the output data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`output_type`**
    - Determines the format of the output data, supporting a wide range of types including images, audio, video, and various file formats. This parameter is key to ensuring the output is correctly processed and rendered.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`output_data`**
    - Contains the actual data to be outputted, which can vary widely in type depending on the output_type parameter.
    - Comfy dtype: `*`
    - Python dtype: `Any`
### Optional
- **`video_audio`**
    - Optional audio track for video outputs, allowing for the inclusion of sound in video files.
    - Comfy dtype: `AUDIO`
    - Python dtype: `str`
- **`animation_fps`**
    - Specifies the frames per second for animated outputs, controlling the playback speed of animations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`animation_quality`**
    - Defines the quality level of animations, offering options to balance between performance and visual fidelity.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`ui`**
    - Generates a structured UI dictionary containing metadata and results for the output, facilitating the integration and display of the output data within the SaltAI framework.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": ("STRING", {}),
                "output_desc": ("STRING", {}),
                "output_type": (
                    ["PNG", "JPEG", "GIF", "WEBP", "AVI", "MP4", "WEBM", "MP3", "WAV", "STRING", "ZIP", "ZIP (Audio)", "ZIP (Video)", "Path"],
                ),
                "output_data": (WILDCARD, {}),
            },
            "optional": {
                "video_audio": ("AUDIO", {}),
                "animation_fps": ("INT", {"min": 1, "max": 60, "default": 8}),
                "animation_quality": (["DEFAULT", "HIGH"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "output_subdir": None},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    FUNCTION = "output"
    CATEGORY = "SALT/IO"

    def output(
        self,
        output_name,
        output_desc,
        output_type,
        output_data,
        video_audio=None,
        animation_fps=8,
        animation_quality="DEFAULT",
        unique_id=0,
        output_subdir=None
    ):
        is_asset = False
        asset_id = str(uuid.uuid4())

        # Determine if valid type
        #if not isinstance(output_data, torch.Tensor) and not isinstance(output_data, str) and not isinstance(output_data, bytes) and not is_lambda(output_data):
        #    errmsg = f"Unsupported output_data supplied `{str(type(output_data).__name__)}`. Please provide `IMAGE` (torch.Tensor), `STRING` (str), or `AUDIO` (bytes) input."
        #    logger.error(errmsg)
        #    raise ValueError(errmsg)
        
        # Support VHS audio
        if output_type in ["AVI", "MP4", "WEBM", "MP3", "WAV"]:
            if video_audio is not None and is_lambda(video_audio):
                video_audio = video_audio()
            if is_lambda(output_data):
                output_data = output_data()
        
        if video_audio and not isinstance(video_audio, bytes):
            errmsg = f"Unsupported video_audio supplied `{str(type(output_data).__name__)}. Please provide `AUDIO` (bytes)"
            logger.error(errmsg)
            raise ValueError(errmsg)

        # Is asset? I may have misunderstood this part
        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM", "MP3", "WAV", "ZIP", "ZIP (Audio)", "ZIP (Video)", "Path"]:
            is_asset = True

        # Determine output name, and sanitize if input (for filesystem)
        if output_name.strip() == "":
            output_name = "output_" + str(unique_id)
        else:
            output_name = sanitize_filename(output_name)

        # Create output dir based on uuid4
        subfolder = os.path.join(output_subdir or '', asset_id)
        output_path = os.path.join(folder_paths.get_output_directory(), subfolder)

        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            errmsg = f"Unable to create output directory `{output_path}`"
            logger.warning(errmsg)

        results = []
        if output_type in ("PNG", "JPEG") and isinstance(output_data, torch.Tensor):

            # Save all images in the tensor batch as specified by output_type
            logger.info("Saving image...")
            try:
                for index, img in enumerate(output_data):
                    pil_image = tensor2pil(img)
                    file_prefix = output_name.strip().replace(" ", "_")
                    file_ext = f".{output_type.lower()}"
                    filename = f"{file_prefix}_{index:04d}{file_ext}"
                    image_path = os.path.join(output_path, filename)
                    pil_image.save(image_path, output_type)
                    results.append({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    if os.path.exists(image_path):
                        logger.info(f"Saved image to `{image_path}`")
                    else:
                        errmsg = f"Unable to save image to `{image_path}`"
                        logger.warning(errmsg)

            except Exception as e:
                errmsg = f"Unknown exception {e}"
                logger.error(errmsg)
                raise e

        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM"] and isinstance(output_data, torch.Tensor):

            # Save animation file
            logger.info("Saving animation...")
            filename = os.path.join(output_path, f"{output_name}.{output_type.lower()}")
            animator = ImageAnimator(
                output_data, fps=int(animation_fps), quality=animation_quality
            )
            animator.save_animation(filename, format=output_type, audio=video_audio)
            results.append({
                "filename": os.path.basename(filename),
                "subfolder": subfolder,
                "type": "output"
            })
            if os.path.exists(filename):
                logger.info(f"[SALT] Saved file to `{filename}`")
            else:
                errmsg = f"[SALT] Unable to save file to `{filename}`"
                logger.warning(errmsg)
        elif output_type in ["MP3", "WAV"] and isinstance(output_data, bytes):

            # Save audio file
            logger.info("Saving audio...")
            filename = os.path.join(output_path, f"{output_name}.{output_type.lower()}")

            audio_buffer = io.BytesIO(output_data)
            audio = AudioSegment.from_file(audio_buffer)

            if output_type == "MP3":
                audio.export(filename, format="mp3")
            else:
                audio.export(filename, format="wav")

            results.append({
                "filename": os.path.basename(filename),
                "subfolder": subfolder,
                "type": "output"
            })

            if os.path.exists(filename):
                logger.info(f"Saved file to `{filename}`")
            else:
                errmsg = f"Unable to save file to `{filename}`"
                logger.warning(errmsg)

        elif output_type in ["ZIP", "ZIP (Audio)", "ZIP (Video)", "Path"]:
            zip_filename = os.path.join(output_path, f"{output_name}.zip")
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                if output_type == "ZIP" and isinstance(output_data, torch.Tensor):

                    # Save batch images as ZIP
                    logger.info("Saving image ZIP...")
                    for index, img in enumerate(output_data):
                        pil_image = tensor2pil(img)
                        file_prefix = output_name.strip().replace(" ", "_")
                        file_ext = ".png"  # default to PNG inside ZIP
                        filename = f"{file_prefix}_{index:04d}{file_ext}"
                        image_path = os.path.join(output_path, filename)
                        pil_image.save(image_path, "PNG")
                        zipf.write(image_path, arcname=filename)
                        os.remove(image_path)

                elif output_type == "ZIP" and isinstance(output_data, str):

                    # Save text to ZIP
                    logger.info("Saving document ZIP...")
                    data_filename = f"{output_name}.txt"
                    data_path = os.path.join(output_path, data_filename)
                    with open(data_path, 'wb' if isinstance(output_data, bytes) else 'w') as f:
                        f.write(output_data)
                    zipf.write(data_path, arcname=data_filename)
                    os.remove(data_path)

                elif output_type == "ZIP" and isinstance(output_data, list):

                    # Save each item in the (string) list to ZIP
                    logger.info("Saving documents ZIP...")
                    for index, item in enumerate(output_data):
                        data_filename = f"{output_name}_{index:04d}.txt"
                        data_path = os.path.join(output_path, data_filename)
                        with open(data_path, 'w') as f:
                            f.write(str(item))
                        zipf.write(data_path, arcname=data_filename)
                        os.remove(data_path)

                elif output_type == "ZIP" and isinstance(output_data, dict):

                    # Save dictionary as JSON to ZIP
                    logger.info("Saving JSON ZIP...")
                    data_filename = f"{output_name}.json"
                    data_path = os.path.join(output_path, data_filename)
                    with open(data_path, 'w') as f:
                        json.dump(output_data, f)
                    zipf.write(data_path, arcname=data_filename)
                    os.remove(data_path)

                elif output_type == "ZIP (Audio)" and isinstance(output_data, bytes):

                    # Save audio data to ZIP
                    logger.info("Saving audio ZIP...")
                    data_filename = f"{output_name}.mp3"
                    data_path = os.path.join(output_path, data_filename)
                    audio_buffer = io.BytesIO(output_data)
                    audio = AudioSegment.from_file(audio_buffer)
                    audio.export(data_path, format="mp3")
                    zipf.write(data_path, arcname=data_filename)
                    os.remove(data_path)

                elif output_type == "ZIP (Video)" and isinstance(output_data, torch.Tensor):

                    # Save video data to ZIP
                    logger.info("Saving video ZIP...")
                    filename = f"{output_name}.mp4"
                    video_path = os.path.join(output_path, filename)
                    animator = ImageAnimator(
                        output_data, fps=int(animation_fps), quality=animation_quality
                    )
                    animator.save_animation(video_path, format="MP4", audio=video_audio)
                    zipf.write(video_path, arcname=filename)
                    os.remove(video_path)

                elif output_type == "Path":

                    # Save file or directory as ZIP
                    logger.info("Saving path as ZIP...")
                    if os.path.exists(output_data):
                        if allowed_path(output_data):
                            if os.path.isfile(output_data):
                                zipf.write(output_data, arcname=os.path.basename(output_data))
                            elif os.path.isdir(output_data):
                                for root, dirs, files in os.walk(output_data):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arcname = os.path.relpath(file_path, start=output_data)
                                        zipf.write(file_path, arcname=arcname)
                        else:
                            errmsg = f"The provided path `{output_data}` is not within the allowed directories."
                            logger.error(errmsg)
                            raise ValueError(errmsg)
                    else:
                        errmsg = f"The path specified `{output_data}` does not exist."
                        logger.error(errmsg)
                        raise ValueError(errmsg)
            
            results.append({
                "filename": os.path.basename(zip_filename),
                "subfolder": subfolder,
                "type": "output"
            })
            
            if os.path.exists(zip_filename):
                logger.info(f"Saved ZIP file to `{zip_filename}`")
            else:
                errmsg = f"Unable to save ZIP file to `{zip_filename}`"
                logger.warning(errmsg)

        else:
            # Assume string output
            logger.info("Saving string...")
            if output_type == "STRING":
                results.append(str(output_data))

        # Output Dictionary
        ui = {
            "ui": {
                "salt_metadata": [
                    {
                        "salt_id": unique_id,
                        "salt_reference_uuid": asset_id,
                        "salt_description": output_desc,
                        "salt_asset": is_asset,
                        "salt_file_extension": output_type,
                    }
                ],
                "salt_output": results
            }
        }

        # Assign images for previews of supported types
        if output_type in ["PNG", "GIF", "WEBP", "JPEG"] and results:
            ui["ui"].update({"images": results})

        # Print to log
        logger.info(f"[SaltOutput_{unique_id}] Output:")
        from pprint import pprint

        pprint(ui, indent=4) # Not converting this to logger yet to get the rest done - Daniel

        return ui

```
