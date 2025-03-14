- `LayerMask_ MaskByColor`: The MaskByColor node is designed to generate masks based on specific color criteria within images. It allows for the creation of layer masks by isolating areas that match a given color, with options to adjust the threshold for color matching, invert the mask, and apply gap fixing to improve mask quality.
    - Inputs:
        - `image` (Required): The input image on which the mask generation is based. This image is analyzed to identify areas that match the specified color criteria. Type should be `IMAGE`.
        - `color` (Required): Specifies the target color for mask generation. This parameter is crucial for determining which areas of the image will be isolated to create the mask. Type should be `COLOR`.
        - `color_in_HEX` (Required): A hexadecimal string representing the target color for mask generation. This provides an alternative method for specifying the color, offering flexibility in defining the color criteria. Type should be `STRING`.
        - `threshold` (Required): Determines the sensitivity of color matching. A lower threshold results in a more inclusive selection of matching colors, while a higher threshold restricts the match to colors more closely resembling the specified target color. Type should be `INT`.
        - `fix_gap` (Required): Activates gap fixing in the mask to close small holes or gaps. This parameter can enhance the mask's quality by ensuring more continuous and coherent masked areas. Type should be `INT`.
        - `fix_threshold` (Required): Sets the threshold for gap fixing, controlling the size of gaps to be filled. This parameter works in conjunction with 'fix_gap' to refine the mask's details and overall appearance. Type should be `FLOAT`.
        - `invert_mask` (Required): When enabled, inverts the generated mask, swapping the masked and unmasked areas. This option allows for flexibility in selecting either the matching areas or their complements. Type should be `BOOLEAN`.
        - `mask` (Optional): unknown Type should be `MASK`.
    - Outputs:
        - `mask`: Returns a tensor of the generated mask, isolating areas matching the specified color criteria, adjusted for threshold, inversion, and gap fixing settings. Type should be `MASK`.
