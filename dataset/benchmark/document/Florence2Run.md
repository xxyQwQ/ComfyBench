- `Florence2Run`: The Florence2Run node is designed to process images and text inputs through the Florence2 model, executing a variety of tasks such as image captioning, object detection, and visual question answering. It leverages advanced deep learning techniques to understand and generate descriptions or answers based on the visual and textual context provided.
    - Inputs:
        - `image` (Required): The image to be processed. It is crucial for visual tasks and is used as the primary input for generating outputs based on visual content. Type should be `IMAGE`.
        - `florence2_model` (Required): unknown Type should be `FL2MODEL`.
        - `text_input` (Required): Optional text input that provides context or queries for the model to consider alongside the image. It's essential for tasks requiring textual input like visual question answering. Type should be `STRING`.
        - `task` (Required): Specifies the task to be performed, such as image captioning or object detection, guiding the model's processing and output generation. Type should be `COMBO[STRING]`.
        - `fill_mask` (Required): A parameter used in certain tasks to control how the model fills in masked parts of the image or text. Type should be `BOOLEAN`.
        - `keep_model_loaded` (Optional): Determines whether the model remains loaded after processing, affecting resource utilization and performance for subsequent tasks. Type should be `BOOLEAN`.
        - `max_new_tokens` (Optional): Limits the number of new tokens the model can generate, impacting the length and detail of text outputs. Type should be `INT`.
        - `num_beams` (Optional): Controls the beam search width during text generation, influencing the diversity and quality of the output. Type should be `INT`.
        - `do_sample` (Optional): Enables or disables sampling in text generation, affecting the randomness and variety of the generated text. Type should be `BOOLEAN`.
        - `output_mask_select` (Optional): Allows selection of specific outputs or masks, providing finer control over the output data. Type should be `STRING`.
    - Outputs:
        - `image`: The processed image, potentially modified or annotated based on the task performed. Type should be `IMAGE`.
        - `mask`: A mask generated by the model, applicable for tasks involving segmentation or specific area highlighting. Type should be `MASK`.
        - `caption`: Generated text output by the model, such as captions, descriptions, or answers to queries, based on the task. Type should be `STRING`.
