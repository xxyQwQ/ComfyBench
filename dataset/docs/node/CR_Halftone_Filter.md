- `CR Halftone Filter`: The CR Halftone Filter node applies a halftone effect to images, simulating the look of classic print media by transforming images into a pattern of dots. This node supports various customization options, including dot size, shape, and resolution, as well as color channel separation for more detailed control over the halftone effect.
    - Parameters:
        - `dot_size`: Specifies the size of the dots in the halftone pattern, allowing for control over the granularity of the effect. Type should be `INT`.
        - `dot_shape`: Determines the shape of the dots in the halftone pattern, offering customization of the visual style of the halftone effect. Type should be `COMBO[STRING]`.
        - `resolution`: Controls the resolution of the output image, with options for normal or high resolution, affecting the scale of the halftone effect. Type should be `COMBO[STRING]`.
        - `angle_c`: The angle of the cyan color channel in the CMYK color space, allowing for precise control over the orientation of the halftone dots. Type should be `INT`.
        - `angle_m`: The angle of the magenta color channel in the CMYK color space, enabling adjustment of the halftone dot orientation for this color. Type should be `INT`.
        - `angle_y`: The angle of the yellow color channel in the CMYK color space, used to set the orientation of the halftone dots for this color. Type should be `INT`.
        - `angle_k`: The angle for the black (key) color channel in the CMYK color space, allowing for customization of the halftone dot orientation. Type should be `INT`.
        - `greyscale`: A boolean indicating whether the image should be converted to greyscale before applying the halftone effect, simplifying the color palette. Type should be `BOOLEAN`.
        - `antialias`: A boolean indicating whether antialiasing should be applied, smoothing the edges of the halftone dots for a cleaner appearance. Type should be `BOOLEAN`.
        - `antialias_scale`: Defines the scale factor used during antialiasing to adjust the image size temporarily, enhancing the smoothness of the halftone dots. Type should be `INT`.
        - `border_blending`: A boolean indicating whether border blending should be applied, potentially softening the transition between the halftone dots and the image background. Type should be `BOOLEAN`.
    - Inputs:
        - `image`: The input image to apply the halftone effect to. This node accepts both PIL Images and PyTorch tensors, automatically converting tensors to PIL Images if necessary. Type should be `IMAGE`.
    - Outputs:
        - `IMAGE`: The output image after applying the halftone effect, converted to RGB format regardless of the original color space. Type should be `IMAGE`.
        - `show_help`: A URL string providing additional help and documentation for the CR Halftone Filter node. Type should be `STRING`.