- `ImagePadForOutpaintMasked`: This node is designed to adjust the dimensions of an image for outpainting tasks by applying padding to the specified sides of the image. It extends the functionality of outpainting by allowing for masked areas, enabling selective padding and processing based on the mask provided. This is particularly useful in scenarios where the image composition needs to be carefully controlled or when integrating new elements into existing images without affecting certain regions.
    - Inputs:
        - `image` (Required): The image to be processed and padded. It is the primary input on which the node operates, determining the base for padding adjustments. Type should be `IMAGE`.
        - `left` (Required): The amount of padding to apply to the left side of the image. This parameter directly influences the horizontal expansion of the image. Type should be `INT`.
        - `top` (Required): The amount of padding to apply to the top side of the image. It affects the vertical expansion of the image, particularly on the top edge. Type should be `INT`.
        - `right` (Required): The calculated amount of padding to apply to the right side of the image, based on the target width, current width, and left padding. It adjusts the horizontal dimensions of the image. Type should be `INT`.
        - `bottom` (Required): The amount of padding to apply to the bottom side of the image. This parameter influences the vertical expansion of the image on the bottom edge. Type should be `INT`.
        - `feathering` (Required): The degree of feathering to apply along the edges of the padding. This parameter controls the smoothness of the transition between the original image and the padded areas. Type should be `INT`.
        - `mask` (Optional): A scaled version of the mask that defines areas to be excluded or differently processed during the padding operation. It allows for selective padding adjustments. Type should be `MASK`.
    - Outputs:
        - `image`: The output is an image with applied padding according to the specified parameters and mask, ready for further processing or outpainting tasks. Type should be `IMAGE`.
        - `mask`: The mask output reflects the areas that were selectively processed or excluded during the padding operation, maintaining the integrity of the original masked regions. Type should be `MASK`.