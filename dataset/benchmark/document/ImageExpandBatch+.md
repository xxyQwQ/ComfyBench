- `ImageExpandBatch+`: This node is designed to facilitate the manipulation of image batches within a graphical interface, specifically focusing on expanding a given batch of images. It abstracts the complexities involved in handling multiple images simultaneously, providing a streamlined approach to either augment the existing batch size or modify the batch in a way that accommodates additional image processing operations.
    - Inputs:
        - `image` (Required): The primary image or batch of images to be expanded. This parameter is the basis for the expansion operation, determining the initial set of images to be modified or augmented. Type should be `IMAGE`.
        - `size` (Required): Specifies the target size for the batch expansion. This could dictate the number of times the image(s) are repeated or the new size of the batch after expansion. Type should be `INT`.
        - `method` (Required): Defines the method of expansion, such as repeating the entire batch, repeating only the first or last image, or expanding the batch size in another specified manner. Type should be `COMBO[STRING]`.
    - Outputs:
        - `image`: The output is an expanded batch of images, modified according to the specified size and method. This facilitates further batch-level image processing or analysis. Type should be `IMAGE`.
