- `IPAdapterBatch`: IPAdapterBatch is designed to enhance image processing capabilities by allowing batch processing of images with advanced image processing adapters. It extends the functionality of IPAdapterAdvanced by enabling the handling of multiple images simultaneously, optimizing the workflow for bulk image manipulation tasks.
    - Inputs:
        - `model` (Required): Specifies the model to be used for image processing, serving as the core component for applying image transformations. Type should be `MODEL`.
        - `ipadapter` (Required): Defines the image processing adapter to be applied to the images, dictating the specific transformations or enhancements to be performed. Type should be `IPADAPTER`.
        - `image` (Required): Represents the images to be processed, allowing for batch processing of multiple images in a single operation. Type should be `IMAGE`.
        - `weight` (Required): Determines the influence of the adapter's effect on the images, with the ability to adjust the intensity of the applied transformations. Type should be `FLOAT`.
        - `weight_type` (Required): Specifies the method for applying weights to the image transformations, influencing the overall effect of the adapter. Type should be `COMBO[STRING]`.
        - `start_at` (Required): Defines the starting point of the adapter's effect, allowing for fine-tuned control over the application of image enhancements. Type should be `FLOAT`.
        - `end_at` (Required): Sets the endpoint of the adapter's effect, enabling precise control over the extent of image transformations. Type should be `FLOAT`.
        - `embeds_scaling` (Required): Controls how embeddings are scaled during the processing, affecting the adaptation of the image features. Type should be `COMBO[STRING]`.
        - `encode_batch_size` (Required): Specifies the batch size for encoding operations, optimizing the processing efficiency for large sets of images. Type should be `INT`.
        - `image_negative` (Optional): Optional parameter for providing negative images to contrast with the primary images, enhancing the adapter's effect. Type should be `IMAGE`.
        - `attn_mask` (Optional): Optional parameter for applying attention masks to the images, directing focus to specific areas during processing. Type should be `MASK`.
        - `clip_vision` (Optional): Optional parameter for integrating CLIP vision features, enriching the image processing with additional contextual understanding. Type should be `CLIP_VISION`.
    - Outputs:
        - `model`: The processed model after applying the image processing adapter, reflecting the transformations or enhancements made to the images. Type should be `MODEL`.