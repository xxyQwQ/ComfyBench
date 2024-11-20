---
tags:
- Face
---

# (Down)Load LivePortraitModels
## Documentation
- Class name: `DownloadAndLoadLivePortraitModels`
- Category: `LivePortrait`
- Output node: `False`

This node is responsible for downloading and loading the necessary models for the LivePortrait feature. It ensures that all components required for generating live portraits, such as appearance feature extractors, motion extractors, warping modules, and retargeting networks, are properly initialized and ready for use. The node dynamically adjusts model precision based on the execution environment and manages model dependencies to facilitate seamless live portrait generation.
## Input types
### Required
### Optional
- **`precision`**
    - Specifies the desired precision for model computations, which can be set manually or determined automatically based on the device's capabilities. This affects the overall performance and accuracy of the live portrait generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`live_portrait_pipe`**
    - Comfy dtype: `LIVEPORTRAITPIPE`
    - Returns a fully initialized pipeline configured for live portrait generation, including all necessary models and settings tailored to the specified precision. The specific type returned is a custom class instance that encapsulates the live portrait pipeline.
    - Python dtype: `LivePortraitPipeline`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DownloadAndLoadLivePortraitModels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            },
            "optional": {
                 "precision": (
                    [
                        'auto',
                        'fp16',
                        'fp32',
                    ], {
                        "default": 'auto'
                    }),
            }
        }

    RETURN_TYPES = ("LIVEPORTRAITPIPE",)
    RETURN_NAMES = ("live_portrait_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "LivePortrait"

    def loadmodel(self, precision='auto'):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        if precision == 'auto':
            try:
                if mm.is_device_mps(device):
                    print("LivePortrait using fp32 for MPS")
                    dtype = 'fp32'
                elif mm.should_use_fp16():
                    print("LivePortrait using fp16")
                    dtype = 'fp16'
                else:
                    print("LivePortrait using fp32")
                    dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
        else:
            dtype = precision
            print(f"LivePortrait using {dtype}")

        pbar = comfy.utils.ProgressBar(3)

        download_path = os.path.join(folder_paths.models_dir, "liveportrait")
        model_path = os.path.join(download_path)

        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/LivePortrait_safetensors",
                                local_dir=download_path,
                                local_dir_use_symlinks=False)

        model_config_path = os.path.join(script_directory, 'liveportrait', 'config', 'models.yaml')
        with open(model_config_path, 'r') as file:
            model_config = yaml.safe_load(file)
        
        feature_extractor_path = os.path.join(model_path, 'appearance_feature_extractor.safetensors')
        motion_extractor_path = os.path.join(model_path, 'motion_extractor.safetensors')
        warping_module_path = os.path.join(model_path, 'warping_module.safetensors')
        spade_generator_path = os.path.join(model_path, 'spade_generator.safetensors')
        stitching_retargeting_path = os.path.join(model_path, 'stitching_retargeting_module.safetensors')

        # init F
        model_params = model_config['model_params']['appearance_feature_extractor_params']
        self.appearance_feature_extractor = AppearanceFeatureExtractor(**model_params).to(device)
        self.appearance_feature_extractor.load_state_dict(comfy.utils.load_torch_file(feature_extractor_path))
        self.appearance_feature_extractor.eval()
        print('Load appearance_feature_extractor done.')
        pbar.update(1)
        # init M
        model_params = model_config['model_params']['motion_extractor_params']
        self.motion_extractor = MotionExtractor(**model_params).to(device)
        self.motion_extractor.load_state_dict(comfy.utils.load_torch_file(motion_extractor_path))
        self.motion_extractor.eval()
        print('Load motion_extractor done.')
        pbar.update(1)
        # init W
        model_params = model_config['model_params']['warping_module_params']
        self.warping_module = WarpingNetwork(**model_params).to(device)
        self.warping_module.load_state_dict(comfy.utils.load_torch_file(warping_module_path))
        self.warping_module.eval()
        print('Load warping_module done.')
        pbar.update(1)
        # init G
        model_params = model_config['model_params']['spade_generator_params']
        self.spade_generator = SPADEDecoder(**model_params).to(device)
        self.spade_generator.load_state_dict(comfy.utils.load_torch_file(spade_generator_path))
        self.spade_generator.eval()
        print('Load spade_generator done.')
        pbar.update(1)

        def filter_checkpoint_for_model(checkpoint, prefix):
            """Filter and adjust the checkpoint dictionary for a specific model based on the prefix."""
            # Create a new dictionary where keys are adjusted by removing the prefix and the model name
            filtered_checkpoint = {key.replace(prefix + "_module.", ""): value for key, value in checkpoint.items() if key.startswith(prefix)}
            return filtered_checkpoint

        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = comfy.utils.load_torch_file(stitching_retargeting_path)

        stitcher_prefix = 'retarget_shoulder'
        stitcher_checkpoint = filter_checkpoint_for_model(checkpoint, stitcher_prefix)
        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(stitcher_checkpoint)
        stitcher = stitcher.to(device)
        stitcher.eval()

        lip_prefix = 'retarget_mouth'
        lip_checkpoint = filter_checkpoint_for_model(checkpoint, lip_prefix)
        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(lip_checkpoint)
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        eye_prefix = 'retarget_eye'
        eye_checkpoint = filter_checkpoint_for_model(checkpoint, eye_prefix)
        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(eye_checkpoint)
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()
        print('Load stitching_retargeting_module done.')

        self.stich_retargeting_module = {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }

        pipeline = LivePortraitPipeline(
            self.appearance_feature_extractor,
            self.motion_extractor,
            self.warping_module,
            self.spade_generator,
            self.stich_retargeting_module,
            InferenceConfig(
                device_id=device, 
                flag_use_half_precision = True if dtype == 'fp16' else False
                )
        )

        return (pipeline,)

```
