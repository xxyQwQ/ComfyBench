- `Image Seamless Texture`: The node is designed to transform an input image into a seamless texture. It can optionally tile the texture across a specified number of tiles, blending the edges to ensure the texture can be repeated without visible seams. This functionality is particularly useful for creating backgrounds or textures that need to be repeated across larger surfaces without noticeable discontinuities.
    - Inputs:
        - `images` (Required): The images to be transformed into seamless textures. This input is crucial as it directly influences the texture's final appearance and seamless quality. Type should be `IMAGE`.
        - `blending` (Required): Determines the degree of blending at the edges of the texture, affecting how seamlessly the texture tiles. A higher value results in more blending, making the seams less noticeable. Type should be `FLOAT`.
        - `tiled` (Required): A boolean indicating whether the texture should be tiled. If true, the texture is repeated across the specified number of tiles, enhancing the seamless effect. Type should be `COMBO[STRING]`.
        - `tiles` (Required): Specifies the number of tiles the texture is divided into, both horizontally and vertically. This parameter is only relevant if tiling is enabled and directly affects the texture's repetitiveness and seamless nature. Type should be `INT`.
    - Outputs:
        - `images`: The resulting seamless texture, which can be tiled across surfaces without visible seams. This output is essential for applications requiring repetitive background textures or materials. Type should be `IMAGE`.