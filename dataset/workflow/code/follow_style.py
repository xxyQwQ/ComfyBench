# create nodes by instantiation
vaedecode_8 = VAEDecode()
emptylatentimage_5 = EmptyLatentImage(width=768, height=768, batch_size=1)
cliptextencode_7 = CLIPTextEncode(text="""""")
cliptextencode_6 = CLIPTextEncode(text="""a beautiful photograph of an old European city""")
clipvisionencode_13 = CLIPVisionEncode()
unclipconditioning_14 = unCLIPConditioning(strength=1, noise_augmentation=0.1)
unclipcheckpointloader_12 = unCLIPCheckpointLoader(ckpt_name="""sd21-unclip-l.ckpt""")
ksampler_3 = KSampler(seed=52117596413767, control_after_generate="""randomize""", steps=20, cfg=7, sampler_name="""dpmpp_3m_sde_gpu""", scheduler="""sgm_uniform""", denoise=1)
saveimage_9 = SaveImage(filename_prefix="""Result""")
loadimage_15 = LoadImage(image="""budapest.jpg""")

# link nodes by invocation
latent_5 = emptylatentimage_5()
model_12, clip_12, vae_12, clip_vision_12 = unclipcheckpointloader_12()
image_15, mask_15 = loadimage_15()
conditioning_7 = cliptextencode_7(clip=clip_12)
clip_vision_output_13 = clipvisionencode_13(clip_vision=clip_vision_12, image=image_15)
conditioning_6 = cliptextencode_6(clip=clip_12)
conditioning_14 = unclipconditioning_14(conditioning=conditioning_6, clip_vision_output=clip_vision_output_13)
latent_3 = ksampler_3(model=model_12, positive=conditioning_14, negative=conditioning_7, latent_image=latent_5)
image_8 = vaedecode_8(samples=latent_3, vae=vae_12)
result_9 = saveimage_9(images=image_8)