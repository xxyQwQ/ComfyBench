# Game Of Life
## Documentation
- Class name: `GameOfLife`
- Category: `image/filters`
- Output node: `False`

The GameOfLife node simulates the Conway's Game of Life, a cellular automaton, over a specified number of steps. It evolves the state of a grid based on the rules of survival, death, and birth determined by the neighboring cells.
## Input types
### Required
- **`width`**
    - Specifies the width of the grid for the simulation, affecting the horizontal size of the cellular automaton.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the grid for the simulation, affecting the vertical size of the cellular automaton.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cell_size`**
    - Determines the size of each cell in the grid, influencing the resolution of the simulation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the grid's initial state.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`threshold`**
    - A threshold value to determine the initial state of each cell in the grid, based on random generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - The number of steps to simulate, dictating the duration of the Game of Life evolution.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_start`**
    - An optional parameter to specify the initial state of the grid, allowing for custom starting configurations.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final state of the grid after the specified number of steps, represented as an image.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A binary mask indicating the cells that changed state in the last step of the simulation.
    - Python dtype: `torch.Tensor`
- **`off`**
    - Comfy dtype: `MASK`
    - A mask indicating cells that turned off in the last step of the simulation.
    - Python dtype: `torch.Tensor`
- **`on`**
    - Comfy dtype: `MASK`
    - A mask indicating cells that turned on in the last step of the simulation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GameOfLife:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 32, "min": 8, "max": 1024, "step": 1}),
                "height": ("INT", { "default": 32, "min": 8, "max": 1024, "step": 1}),
                "cell_size": ("INT", { "default": 16, "min": 8, "max": 1024, "step": 8}),
                "seed": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "threshold": ("FLOAT", { "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", { "default": 64, "min": 1, "max": 999999, "step": 1}),
            },
            "optional": {
                "optional_start": ("MASK", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "off", "on")
    FUNCTION = "game"

    CATEGORY = "image/filters"

    def game(self, width, height, cell_size, seed, threshold, steps, optional_start=None):
        F = torch.nn.functional
        
        if optional_start is None:
            # base random initialization
            torch.manual_seed(seed)
            grid = torch.rand(1, 1, height, width)
        else:
            grid = optional_start[0].unsqueeze(0).unsqueeze(0)
            grid = F.interpolate(grid, size=(height, width))
        
        grid = (grid > threshold).type(torch.uint8)
        empty = torch.zeros(1, 1, height, width, dtype=torch.uint8)
        
        # neighbor convolution kernel
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.uint8)
        kernel[0, 0, 1, 1] = 0
        
        game_states = [[], [], []] # grid, turn_off, turn_on
        game_states[0].append(grid.detach().clone())
        game_states[1].append(empty.detach().clone())
        game_states[2].append(empty.detach().clone())
        for step in range(steps - 1):
            new_state = grid.detach().clone()
            neighbors = F.conv2d(F.pad(new_state, pad=(1, 1, 1, 1), mode="circular"), kernel) #, padding="same")
            
            # If a cell is ON and has fewer than two neighbors that are ON, it turns OFF
            new_state[(new_state == 1) == (neighbors < 2)] = 0
            
            # If a cell is ON and has either two or three neighbors that are ON, it remains ON.
            
            # If a cell is ON and has more than three neighbors that are ON, it turns OFF.
            new_state[(new_state == 1) == (neighbors > 3)] = 0
            
            # If a cell is OFF and has exactly three neighbors that are ON, it turns ON.
            new_state[(new_state == 0) == (neighbors == 3)] = 1
            
            turn_off = ((grid - new_state) == 1).type(torch.uint8)
            turn_on = ((new_state - grid) == 1).type(torch.uint8)
            
            game_states[0].append(new_state.detach().clone())
            game_states[1].append(turn_off.detach().clone())
            game_states[2].append(turn_on.detach().clone())
            
            grid = new_state
        
        def postprocess(tensorlist, to_image=False):
            game_anim = torch.cat(tensorlist, dim=0).type(torch.float32)
            game_anim = F.interpolate(game_anim, size=(height * cell_size, width * cell_size))
            game_anim = torch.squeeze(game_anim, dim=1) # BCHW -> BHW
            if to_image:
                game_anim = game_anim.unsqueeze(-1).repeat(1,1,1,3) # BHWC
            return game_anim
        
        image = postprocess(game_states[0], to_image=True)
        mask = postprocess(game_states[0])
        off = postprocess(game_states[1])
        on = postprocess(game_states[2])
        
        return (image, mask, off, on)

```
