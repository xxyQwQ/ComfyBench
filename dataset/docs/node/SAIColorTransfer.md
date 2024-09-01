- `SAIColorTransfer`: The SAIColorTransfer node is designed for applying color transfer techniques between images, enabling the modification of the color palette of a target image to match that of a source image. This process is useful for harmonizing the colors between different images or achieving specific aesthetic effects.
    - Parameters:
        - `mode`: Determines the method of color transfer to be applied, such as PDF regraining, mean transfer, or LAB color space transfer, affecting the visual outcome of the color adaptation. Type should be `COMBO[STRING]`.
    - Inputs:
        - `target_images`: Specifies the images whose color palettes are to be modified. This input is crucial for determining the final appearance of the output images. Type should be `IMAGE`.
        - `source_images`: Defines the images that provide the color palette to be transferred to the target images. The choice of source images directly influences the color transformation applied. Type should be `IMAGE`.
    - Outputs:
        - `images`: The resulting images after the color transfer process, showcasing the adapted color palettes. Type should be `IMAGE`.