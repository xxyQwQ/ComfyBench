- `ControlNetApply`: This node applies a control network to a given image and conditioning, adjusting the image's attributes based on the control network's parameters and a specified strength. It enables dynamic modification of image characteristics through control hints, facilitating targeted adjustments without altering the original conditioning structure.
    - Inputs:
        - `conditioning` (Required): The conditioning data to be modified by the control network. It serves as the basis for the control network's adjustments, influencing the final output. Type should be `CONDITIONING`.
        - `control_net` (Required): The control network to be applied. It defines the specific adjustments to be made to the image, based on its trained parameters. Type should be `CONTROL_NET`.
        - `image` (Required): The image to which the control network's adjustments will be applied. It provides the visual context for the control network's operations. Type should be `IMAGE`.
        - `strength` (Required): A scalar value determining the intensity of the control network's adjustments. It allows for fine-tuning the impact of the control network on the image. Type should be `FLOAT`.
    - Outputs:
        - `conditioning`: The modified conditioning data, reflecting the adjustments made by the control network. Type should be `CONDITIONING`.
