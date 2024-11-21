- `ScaledSoftControlNetWeights`: The ScaledSoftControlNetWeights node is designed to dynamically adjust the influence of control weights within a neural network based on a set of parameters. It allows for the customization of weight scaling, enabling fine-tuned control over the network's behavior in response to specific conditions or inputs.
    - Inputs:
        - `base_multiplier` (Required): Defines the base scaling factor for the control weights, serving as the foundational adjustment level before any conditional modifications are applied. Type should be `FLOAT`.
        - `flip_weights` (Required): A boolean flag that, when true, reverses the effect of the control weights, allowing for inverse scaling effects. Type should be `BOOLEAN`.
        - `uncond_multiplier` (Optional): Specifies an unconditional scaling multiplier that is applied regardless of other conditions, offering an additional layer of weight adjustment. Type should be `FLOAT`.
        - `cn_extras` (Optional): A dictionary of extra parameters for further customization of the control weights, providing extended flexibility in their application. Type should be `CN_WEIGHTS_EXTRAS`.
        - `autosize` (Optional): Configures automatic sizing for the control weights, including padding adjustments to accommodate various network configurations. Type should be `ACNAUTOSIZE`.
    - Outputs:
        - `CN_WEIGHTS`: Represents the adjusted control net weights after applying the scaling factors. Type should be `CONTROL_NET_WEIGHTS`.
        - `TK_SHORTCUT`: A shortcut to timestep keyframe data, encapsulating the effects of the adjusted weights on the network's temporal dynamics. Type should be `TIMESTEP_KEYFRAME`.