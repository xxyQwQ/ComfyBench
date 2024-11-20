---
tags:
- Animation
- CameraControl
---

# Camera Pose Visualizer
## Documentation
- Class name: `CameraPoseVisualizer`
- Category: `KJNodes/misc`
- Output node: `False`

The CameraPoseVisualizer node is designed to visualize camera poses in a 3D plot, either from a provided text file containing camera intrinsics and coordinates or directly from camera control poses. It supports customization of the visualization through parameters such as scale, base value adjustments, and the option to use exact focal lengths. This visualization aids in understanding and analyzing the spatial orientation and field of view of cameras in a given scene.
## Input types
### Required
- **`pose_file_path`**
    - Specifies the path to a text file containing camera poses or can be left empty to use camera control poses directly. It is essential for determining the source of camera poses to visualize.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`base_xval`**
    - A base value for x-axis adjustments in the visualization, allowing for fine-tuning of the camera's position.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zval`**
    - A base value for z-axis adjustments, influencing the depth positioning in the visualization.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`scale`**
    - Scales the entire visualization, affecting the size and spacing of visualized camera poses.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_exact_fx`**
    - Determines whether to use exact focal lengths from the camera poses or a default value, impacting the accuracy of the visualization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`relative_c2w`**
    - Controls whether camera-to-world transformations are considered relative, affecting the positioning and orientation of cameras.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`use_viewer`**
    - Enables or disables the use of an interactive viewer for the visualization, enhancing user interaction.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`cameractrl_poses`**
    - Directly provides camera control poses for visualization, offering an alternative to loading poses from a file.
    - Comfy dtype: `CAMERACTRL_POSES`
    - Python dtype: `list`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Generates a 3D plot image visualizing the camera poses, providing a visual representation of camera orientations and positions.
    - Python dtype: `matplotlib.figure.Figure`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CameraPoseVisualizer:
                
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pose_file_path": ("STRING", {"default": '', "multiline": False}),
            "base_xval": ("FLOAT", {"default": 0.2,"min": 0, "max": 100, "step": 0.01}),
            "zval": ("FLOAT", {"default": 0.3,"min": 0, "max": 100, "step": 0.01}),
            "scale": ("FLOAT", {"default": 1.0,"min": 0.01, "max": 10.0, "step": 0.01}),
            "use_exact_fx": ("BOOLEAN", {"default": False}),
            "relative_c2w": ("BOOLEAN", {"default": True}),
            "use_viewer": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cameractrl_poses": ("CAMERACTRL_POSES", {"default": None}),
            }
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Visualizes the camera poses, from Animatediff-Evolved CameraCtrl Pose  
or a .txt file with RealEstate camera intrinsics and coordinates, in a 3D plot. 
"""
        
    def plot(self, pose_file_path, scale, base_xval, zval, use_exact_fx, relative_c2w, use_viewer, cameractrl_poses=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from torchvision.transforms import ToTensor

        x_min = -2.0 * scale
        x_max = 2.0 * scale
        y_min = -2.0 * scale
        y_max = 2.0 * scale
        z_min = -2.0 * scale
        z_max = 2.0 * scale
        plt.rcParams['text.color'] = '#999999'
        self.fig = plt.figure(figsize=(18, 7))
        self.fig.patch.set_facecolor('#353535')
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_facecolor('#353535') # Set the background color here
        self.ax.grid(color='#999999', linestyle='-', linewidth=0.5)
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        self.ax.set_xlabel('x', color='#999999')
        self.ax.set_ylabel('y', color='#999999')
        self.ax.set_zlabel('z', color='#999999')
        for text in self.ax.get_xticklabels() + self.ax.get_yticklabels() + self.ax.get_zticklabels():
            text.set_color('#999999')
        print('initialize camera pose visualizer')

        if pose_file_path != "":
            with open(pose_file_path, 'r') as f:
                poses = f.readlines()
                w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[1:]]
                fxs = [float(pose.strip().split(' ')[1]) for pose in poses[1:]]
                #print(poses)
        elif cameractrl_poses is not None:
            poses = cameractrl_poses
            w2cs = [np.array(pose[7:]).reshape(3, 4) for pose in cameractrl_poses]
            fxs = [pose[1] for pose in cameractrl_poses]
        else:
            raise ValueError("Please provide either pose_file_path or cameractrl_poses")

        total_frames = len(w2cs)
        transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
        last_row = np.zeros((1, 4))
        last_row[0, -1] = 1.0

        w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
        c2ws = self.get_c2w(w2cs, transform_matrix, relative_c2w)

        for frame_idx, c2w in enumerate(c2ws):
            self.extrinsic2pyramid(c2w, frame_idx / total_frames, hw_ratio=1/1, base_xval=base_xval,
                                    zval=(fxs[frame_idx] if use_exact_fx else zval))

        # Create the colorbar
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=total_frames)
        colorbar = self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical')

        # Change the colorbar label
        colorbar.set_label('Frame', color='#999999') # Change the label and its color

        # Change the tick colors
        colorbar.ax.yaxis.set_tick_params(colors='#999999') # Change the tick color

        # Change the tick frequency
        # Assuming you want to set the ticks at every 10th frame
        ticks = np.arange(0, total_frames, 10)
        colorbar.ax.yaxis.set_ticks(ticks)
        
        plt.title('')
        plt.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        tensor_img = ToTensor()(img)
        buf.close()
        tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
        if use_viewer:
            time.sleep(1)
            plt.show()
        return (tensor_img,)

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=1/1, base_xval=1, zval=3):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        vertex_std = np.array([[0, 0, 0, 1],
                            [base_xval, -base_xval * hw_ratio, zval, 1],
                            [base_xval, base_xval * hw_ratio, zval, 1],
                            [-base_xval, base_xval * hw_ratio, zval, 1],
                            [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.25))

    def customize_legend(self, list_label):
        from matplotlib.patches import Patch
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def get_c2w(self, w2cs, transform_matrix, relative_c2w):
        if relative_c2w:
            target_cam_c2w = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            abs2rel = target_cam_c2w @ w2cs[0]
            ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
        else:
            ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
        ret_poses = [transform_matrix @ x for x in ret_poses]
        return np.array(ret_poses, dtype=np.float32)

```
