---
tags:
- Sampling
---

# EasyKSampler (Full)
## Documentation
- Class name: `easy fullkSampler`
- Category: `EasyUse/Sampler`
- Output node: `True`

The `easy fullkSampler` node is designed to facilitate the sampling process in generative models by providing a simplified interface for users. It abstracts the complexities involved in sampling configurations, allowing for easy experimentation with different models, seeds, steps, conditioning, and denoising parameters. This node aims to make the process of generating new samples more accessible and user-friendly, especially for those not deeply familiar with the underlying algorithms.
## Input types
### Required
- **`pipe`**
    - The pipeline configuration for the sampling process, integrating various nodes and operations for a comprehensive workflow.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `str`
- **`steps`**
    - Defines the number of steps to be used in the sampling process. This parameter affects the detail and quality of the generated samples, with a higher number of steps potentially leading to more refined outputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, which influences the adherence of the generated samples to the specified conditions. This parameter allows for fine-tuning the balance between randomness and conditioning in the sampling output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampling algorithm to be used. This choice impacts the sampling behavior and the characteristics of the generated images, offering flexibility in exploring different sampling strategies.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduler for the sampling process, which determines how the sampling parameters are adjusted over time. This affects the progression and quality of the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Sets the denoising strength used in the sampling process. Adjusting this parameter can help in controlling the level of detail and clarity in the generated samples.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`image_output`**
    - Determines the output format for images, including options for file types and quality settings.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`link_id`**
    - A unique identifier for linking the output with other processes or storage locations.
    - Comfy dtype: `INT`
    - Python dtype: `str`
- **`save_prefix`**
    - A prefix added to the filenames of saved images, allowing for organized storage and retrieval.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`seed`**
    - Sets the initial seed for the random number generator used in the sampling process. This ensures reproducibility of results, allowing users to generate the same output given the same set of parameters.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`model`**
    - Specifies the generative model to be used for sampling. This parameter is crucial as it determines the base model from which samples will be generated, affecting the overall quality and characteristics of the output.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`positive`**
    - Specifies positive conditioning to guide the sampling towards desired attributes or features in the generated output. This parameter helps in directing the model to produce samples that match certain criteria.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Specifies negative conditioning to avoid certain attributes or features in the generated output. This parameter is useful for steering the sampling away from undesired characteristics.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent`**
    - Provides an initial latent image to start the sampling from. This can be used for tasks like image editing or continuation, where the starting point is a pre-existing latent representation.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`vae`**
    - Specifies the VAE model to be used in conjunction with the sampling process for tasks like encoding or decoding.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`clip`**
    - Specifies the CLIP model for text or image conditioning, enhancing the relevance of the generated samples to given prompts or images.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`xyPlot`**
    - An optional parameter for plotting or visualizing data related to the sampling process.
    - Comfy dtype: `XYPLOT`
    - Python dtype: `str`
- **`image`**
    - An optional input image for tasks such as image editing or style transfer.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The output pipeline configuration, encapsulating the entire sampling process and its results.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated image or images as a result of the sampling process.
    - Python dtype: `str`
- **`model`**
    - Comfy dtype: `MODEL`
    - The model used in the sampling process, potentially modified or updated through the operation.
    - Python dtype: `str`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning applied during the sampling, influencing the characteristics of the generated output.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning applied to steer the sampling away from certain features or attributes.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output latent representation of the generated sample. This can be used for further processing or conversion into an image, representing the final result of the sampling process.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model involved in the sampling process, if applicable, including any encoding or decoding operations.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model used for conditioning, reflecting its role in shaping the output based on textual or visual prompts.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - The seed value used in the sampling process, indicating the role of randomness and reproducibility in generating the output.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class samplerFull:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS+new_schedulers,),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "image_output": (["Hide", "Preview", "Preview&Choose", "Save", "Hide&Save", "Sender", "Sender&Save"],),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "xyPlot": ("XYPLOT",),
                    "image": ("IMAGE",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT",)
    RETURN_NAMES = ("pipe",  "image", "model", "positive", "negative", "latent", "vae", "clip", "seed",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, seed=None, model=None, positive=None, negative=None, latent=None, vae=None, clip=None, xyPlot=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False, downscale_options=None, image=None):

        samp_model = model if model is not None else pipe["model"]
        samp_positive = positive if positive is not None else pipe["positive"]
        samp_negative = negative if negative is not None else pipe["negative"]
        samp_samples = latent if latent is not None else pipe["samples"]
        samp_vae = vae if vae is not None else pipe["vae"]
        samp_clip = clip if clip is not None else pipe["clip"]

        samp_seed = seed if seed is not None else pipe['seed']

        samp_custom = pipe["loader_settings"]["custom"] if "custom" in pipe["loader_settings"] else None

        steps = steps if steps is not None else pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = cfg if cfg is not None else pipe['loader_settings']['cfg']
        sampler_name = sampler_name if sampler_name is not None else pipe['loader_settings']['sampler_name']
        scheduler = scheduler if scheduler is not None else pipe['loader_settings']['scheduler']
        denoise = denoise if denoise is not None else pipe['loader_settings']['denoise']
        add_noise = pipe['loader_settings']['add_noise'] if 'add_noise' in pipe['loader_settings'] else 'enabled'
        force_full_denoise = pipe['loader_settings']['force_full_denoise'] if 'force_full_denoise' in pipe['loader_settings'] else True

        if image is not None and latent is None:
            samp_samples = {"samples": samp_vae.encode(image[:, :, :, :3])}

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        def downscale_model_unet(samp_model):
            # 获取Unet参数
            if "PatchModelAddDownscale" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                # 自动收缩Unet
                if downscale_options['downscale_factor'] is None:
                    unet_config = samp_model.model.model_config.unet_config
                    if unet_config is not None and "samples" in samp_samples:
                        height = samp_samples['samples'].shape[2] * 8
                        width = samp_samples['samples'].shape[3] * 8
                        context_dim = unet_config.get('context_dim')
                        longer_side = width if width > height else height
                        if context_dim is not None and longer_side > context_dim:
                            width_downscale_factor = float(width / context_dim)
                            height_downscale_factor = float(height / context_dim)
                            if width_downscale_factor > 1.75:
                                log_node_warn("正在收缩模型Unet...")
                                log_node_warn("收缩系数:" + str(width_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], width_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                            elif height_downscale_factor > 1.25:
                                log_node_warn("正在收缩模型Unet...")
                                log_node_warn("收缩系数:" + str(height_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], height_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                else:
                    cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                    log_node_warn("正在收缩模型Unet...")
                    log_node_warn("收缩系数:" + str(downscale_options['downscale_factor']))
                    (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], downscale_options['downscale_factor'], downscale_options['start_percent'], downscale_options['end_percent'], downscale_options['downscale_after_skip'], downscale_options['downscale_method'], downscale_options['upscale_method'])
            return samp_model

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                 samp_negative,
                                 steps, start_step, last_step, cfg, sampler_name, scheduler, denoise,
                                 image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id,
                                 preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise, samp_custom=None):

            # LayerDiffusion
            layerDiffuse = None
            samp_blend_samples = None
            layer_diffusion_method = pipe['loader_settings']['layer_diffusion_method'] if 'layer_diffusion_method' in pipe['loader_settings'] else None
            if layer_diffusion_method is not None:
                layerDiffuse = LayerDiffuse()
                samp_blend_samples = pipe["blend_samples"] if "blend_samples" in pipe else None
                additional_cond = pipe["loader_settings"]['layer_diffusion_cond'] if "layer_diffusion_cond" in pipe[
                    'loader_settings'] else (None, None, None)
                method = layerDiffuse.get_layer_diffusion_method(pipe['loader_settings']['layer_diffusion_method'],
                                                         samp_blend_samples is not None)

                images = pipe["images"] if "images" in pipe else None
                weight = pipe['loader_settings']['layer_diffusion_weight'] if 'layer_diffusion_weight' in pipe[
                    'loader_settings'] else 1.0
                samp_model, samp_positive, samp_negative = layerDiffuse.apply_layer_diffusion(samp_model, method, weight,
                                                                                      samp_samples, samp_blend_samples,
                                                                                      samp_positive, samp_negative,
                                                                                      images, additional_cond)
                resolution = pipe['loader_settings']['resolution'] if 'resolution' in pipe['loader_settings'] else "自定义 X 自定义"
                empty_latent_width = pipe['loader_settings']['empty_latent_width'] if 'empty_latent_width' in pipe['loader_settings'] else 512
                empty_latent_height = pipe['loader_settings']['empty_latent_height'] if 'empty_latent_height' in pipe['loader_settings'] else 512
                batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
                samp_samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size)

            # Downscale Model Unet
            if samp_model is not None and downscale_options is not None:
                samp_model = downscale_model_unet(samp_model)
            # 推理初始时间
            start_time = int(time.time() * 1000)
            # 开始推理
            if samp_custom is not None:
                guider = samp_custom['guider'] if 'guider' in samp_custom else None
                _sampler = samp_custom['sampler'] if 'sampler' in samp_custom else None
                sigmas = samp_custom['sigmas'] if 'sigmas' in samp_custom else None
                noise = samp_custom['noise'] if 'noise' in samp_custom else None
                samp_samples, _ = sampler.custom_advanced_ksampler(noise, guider, _sampler, sigmas, samp_samples)
            elif scheduler == 'align_your_steps':
                model_type = get_sd_version(samp_model)
                if model_type == 'unknown':
                    model_type = 'sdxl'
                    # raise Exception("This Model not supported")
                sigmas, = alignYourStepsScheduler().get_sigmas(model_type.upper(), steps, denoise)
                _sampler = comfy.samplers.sampler_object(sampler_name)
                samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, _sampler, sigmas, samp_positive, samp_negative, samp_samples, disable_noise=disable_noise, preview_latent=preview_latent)
            elif scheduler == 'gits':
                sigmas, = gitsScheduler().get_sigmas(coeff=1.2, steps=steps, denoise=denoise)
                _sampler = comfy.samplers.sampler_object(sampler_name)
                samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, _sampler, sigmas, samp_positive, samp_negative, samp_samples, disable_noise=disable_noise, preview_latent=preview_latent)
            else:
                samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
            # 推理结束时间
            end_time = int(time.time() * 1000)
            latent = samp_samples["samples"]

            # 解码图片
            if tile_size is not None:
                samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
            else:
                samp_images = samp_vae.decode(latent).cpu()

            # LayerDiffusion Decode
            if layerDiffuse is not None:
                new_images, samp_images, alpha = layerDiffuse.layer_diffusion_decode(layer_diffusion_method, latent, samp_blend_samples, samp_images, samp_model)
            else:
                new_images = samp_images
                alpha = None

            # 推理总耗时（包含解码）
            end_decode_time = int(time.time() * 1000)
            spent_time = 'Diffusion:' + str((end_time-start_time)/1000)+'″, VAEDecode:' + str((end_decode_time-end_time)/1000)+'″ '

            results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

            new_pipe = {
                **pipe,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "blend_samples": samp_blend_samples,
                "images": new_images,
                "samp_images": samp_images,
                "alpha": alpha,
                "seed": samp_seed,

                "loader_settings": {
                    **pipe["loader_settings"],
                    "spent_time": spent_time
                }
            }

            del pipe

            if image_output == 'Preview&Choose':
                if my_unique_id not in ChooserMessage.stash:
                    ChooserMessage.stash[my_unique_id] = {}
                my_stash = ChooserMessage.stash[my_unique_id]

                PromptServer.instance.send_sync("easyuse-image-choose", {"id": my_unique_id, "urls": results})
                # wait for selection
                try:
                    selections = ChooserMessage.waitForMessage(my_unique_id, asList=True)
                    samples = samp_samples['samples']
                    samples = [samples[x] for x in selections if x >= 0] if len(selections) > 1 else [samples[0]]
                    new_images = [new_images[x] for x in selections if x >= 0] if len(selections) > 1 else [new_images[0]]
                    samp_images = [samp_images[x] for x in selections if x >= 0] if len(selections) > 1 else [samp_images[0]]
                    new_images = torch.stack(new_images, dim=0)
                    samp_images = torch.stack(samp_images, dim=0)
                    samples = torch.stack(samples, dim=0)
                    samp_samples = {"samples": samples}
                    new_pipe['samples'] = samp_samples
                    new_pipe['loader_settings']['batch_size'] = len(new_images)
                except ChooserCancelled:
                    raise comfy.model_management.InterruptProcessingException()

                new_pipe['images'] = new_images
                new_pipe['samp_images'] = samp_images

                return {"ui": {"images": results},
                        "result": sampler.get_output(new_pipe,)}

            if image_output in ("Hide", "Hide&Save"):
                return {"ui": {},
                    "result": sampler.get_output(new_pipe,)}

            if image_output in ("Sender", "Sender&Save"):
                PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

            if hasattr(ModelPatcher, "original_calculate_weight"):
                ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight

            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe,)}

        def process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom):

            sampleXYplot = easyXYPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id, sampler, easyCache)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                            samp_negative, steps, 0, 10000, cfg,
                                            sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt,
                                            extra_pnginfo, my_unique_id, preview_latent, samp_custom=samp_custom)

            # Downscale Model Unet
            if samp_model is not None and downscale_options is not None:
                samp_model = downscale_model_unet(samp_model)

            blend_samples = pipe['blend_samples'] if "blend_samples" in pipe else None
            layer_diffusion_method = pipe['loader_settings']['layer_diffusion_method'] if 'layer_diffusion_method' in pipe['loader_settings'] else None

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": pipe["loader_settings"]["lora_name"] if "lora_name" in pipe["loader_settings"] else None,
                "lora_model_strength": pipe["loader_settings"]["lora_model_strength"] if "model_strength" in pipe["loader_settings"] else None,
                "lora_clip_strength": pipe["loader_settings"]["lora_clip_strength"] if "clip_strength" in pipe["loader_settings"] else None,
                "lora_stack":  pipe["loader_settings"]["lora_stack"] if "lora_stack" in pipe["loader_settings"] else None,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "seed": samp_seed,
                "images": pipe['images'],

                "model": samp_model, "vae": samp_vae, "clip": samp_clip, "positive_cond": samp_positive,
                "negative_cond": samp_negative,

                "ckpt_name": pipe['loader_settings']['ckpt_name'] if "ckpt_name" in pipe["loader_settings"] else None,
                "vae_name": pipe['loader_settings']['vae_name'] if "vae_name" in pipe["loader_settings"] else None,
                "clip_skip": pipe['loader_settings']['clip_skip'] if "clip_skip" in pipe["loader_settings"] else None,
                "positive": pipe['loader_settings']['positive'] if "positive" in pipe["loader_settings"] else None,
                "positive_token_normalization": pipe['loader_settings']['positive_token_normalization'] if "positive_token_normalization" in pipe["loader_settings"] else None,
                "positive_weight_interpretation": pipe['loader_settings']['positive_weight_interpretation'] if "positive_weight_interpretation" in pipe["loader_settings"] else None,
                "negative": pipe['loader_settings']['negative'] if "negative" in pipe["loader_settings"] else None,
                "negative_token_normalization": pipe['loader_settings']['negative_token_normalization'] if "negative_token_normalization" in pipe["loader_settings"] else None,
                "negative_weight_interpretation": pipe['loader_settings']['negative_weight_interpretation'] if "negative_weight_interpretation" in pipe["loader_settings"] else None,
            }

            if "models" in pipe["loader_settings"]:
                plot_image_vars["models"] = pipe["loader_settings"]["models"]
            if "vae_use" in pipe["loader_settings"]:
                plot_image_vars["vae_use"] = pipe["loader_settings"]["vae_use"]
            if "a1111_prompt_style" in pipe["loader_settings"]:
                plot_image_vars["a1111_prompt_style"] = pipe["loader_settings"]["a1111_prompt_style"]
            if "cnet_stack" in pipe["loader_settings"]:
                plot_image_vars["cnet"] = pipe["loader_settings"]["cnet_stack"]
            if "positive_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["positive_cond_stack"] = pipe["loader_settings"]["positive_cond_stack"]
            if "negative_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["negative_cond_stack"] = pipe["loader_settings"]["negative_cond_stack"]
            if layer_diffusion_method:
                plot_image_vars["layer_diffusion_method"] = layer_diffusion_method
            if "layer_diffusion_weight" in pipe["loader_settings"]:
                plot_image_vars["layer_diffusion_weight"] = pipe['loader_settings']['layer_diffusion_weight']
            if "layer_diffusion_cond" in pipe["loader_settings"]:
                plot_image_vars["layer_diffusion_cond"] = pipe['loader_settings']['layer_diffusion_cond']
            if "empty_samples" in pipe["loader_settings"]:
                plot_image_vars["empty_samples"] = pipe["loader_settings"]['empty_samples']

            latent_image = sampleXYplot.get_latent(pipe["samples"])
            latents_plot = sampleXYplot.get_labels_and_sample(plot_image_vars, latent_image, preview_latent, start_step,
                                                              last_step, force_full_denoise, disable_noise)

            samp_samples = {"samples": latents_plot}

            images, image_list = sampleXYplot.plot_images_and_labels()

            # Generate output_images
            output_images = torch.stack([tensor.squeeze() for tensor in image_list])

            if layer_diffusion_method is not None:
                layerDiffuse = LayerDiffuse()
                new_images, samp_images, alpha = layerDiffuse.layer_diffusion_decode(layer_diffusion_method, latents_plot, blend_samples,
                                                                             output_images, samp_model)
            else:
                new_images = output_images
                samp_images = output_images
                alpha = None

            results = easySave(images, save_prefix, image_output, prompt, extra_pnginfo)

            new_pipe = {
                **pipe,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "blend_samples": blend_samples,
                "samp_images": samp_images,
                "images": new_images,
                "seed": samp_seed,
                "alpha": alpha,

                "loader_settings": pipe["loader_settings"],
            }

            del pipe

            if hasattr(ModelPatcher, "original_calculate_weight"):
                ModelPatcher.calculate_weight = ModelPatcher.original_calculate_weight

            if image_output in ("Hide", "Hide&Save"):
                return sampler.get_output(new_pipe)

            return {"ui": {"images": results}, "result": (sampler.get_output(new_pipe))}

        preview_latent = True
        if image_output in ("Hide", "Hide&Save"):
            preview_latent = False

        xyplot_id = next((x for x in prompt if "XYPlot" in str(prompt[x]["class_type"])), None)
        if xyplot_id is None:
            xyPlot = None
        else:
            xyPlot = pipe["loader_settings"]["xyplot"] if "xyplot" in pipe["loader_settings"] else xyPlot
        if xyPlot is not None:
            return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom)
        else:
            return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, start_step, last_step, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, force_full_denoise, disable_noise, samp_custom)

```
