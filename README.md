# CLEVR Dataset Generation

This is the code used to generate the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/) as described in the paper:

**[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://cs.stanford.edu/people/jcjohns/clevr/)**
 <br>
 <a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
 <a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
 <a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
 <a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
 <a href='http://larryzitnick.org/'>Larry Zitnick</a>,
 <a href='http://www.rossgirshick.info/'>Ross Girshick</a>
 <br>
 Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)

Code and pretrained models for the baselines used in the paper [can be found here](https://github.com/facebookresearch/clevr-iep).

You can use this code to render synthetic images and compositional questions for those images, like this:

<div align="center">
  <img src="images/example1080.png" width="800px">
</div>

**Q:** How many small spheres are there? <br>
**A:** 2

**Q:**  What number of cubes are small things or red metal objects? <br>
**A:**  2

**Q:** Does the metal sphere have the same color as the metal cylinder? <br>
**A:** Yes

**Q:** Are there more small cylinders than metal things? <br>
**A:** No

**Q:**  There is a cylinder that is on the right side of the large yellow object behind the blue ball; is there a shiny cube in front of it? <br>
**A:**  Yes

If you find this code useful in your research then please cite

```
@inproceedings{johnson2017clevr,
  title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens
          and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
  booktitle={CVPR},
  year={2017}
}
```

All code was developed and tested on OSX and Ubuntu 16.04.

## Step 1: Generating Images
First we render synthetic images using [Blender](https://www.blender.org/), outputting both rendered images as well as a JSON file containing ground-truth scene information for each image.

The scripts have been updated to work with current Blender (tested with 5.2); no `.pth` file or other changes to Blender's bundled Python are needed. You can render some random images like this:

```bash
cd image_generation
blender --background --python render_images.py -- --num_images 10
```

On OSX the `blender` binary is located inside the Blender.app directory; for convenience you may want to
add the following alias to your `~/.zshrc` file:

```bash
alias blender='/Applications/Blender.app/Contents/MacOS/Blender'
```

### Rendering specific scenes
Instead of fully random scenes you can describe the scenes you want in a JSON file and pass it with `--scene_spec`.
Each object may give `shape` (cube, sphere, cylinder), `color` (a name from `data/properties.json` or `[r, g, b]` in 0-255),
`material` (rubber, metal), `size` (large, small), `x` and `y` (ground-plane coordinates, roughly -3 to 3) and `rotation` (degrees).
Any field that is left out is chosen randomly. See [`image_generation/example_scenes.json`](image_generation/example_scenes.json):

```bash
blender --background --python render_images.py -- --scene_spec example_scenes.json --camera_jitter 0
```

### Controlled datasets for VLM analysis
Two controlled datasets for analysing LLaVA-1.5, with objects aligned to its 14 px patch grid:

- `CLEVR_SINGLE_OBJECT`: one object per image, 36 positions x 3 shapes x 8 colors = 864 images at 336x336
- `CLEVR_DUAL_OBJECT`: two objects per image, 9 positions, 3 shapes, red/blue = 1,296 images (2,592 A/B assignments)

Each dataset is made in two steps from a settings file (`output/<name>/settings.json`): a *layout* step (settings that move objects: image and patch size, grid, object size, shapes; computes the placements JSON) and a *render* step (colors, shadows, quality; renders and checks), followed by an export in the CLEVR format to `output/<name>/`. **Start with [`tutorial_generate_datasets.ipynb`](tutorial_generate_datasets.ipynb)**: it replicates the existing datasets and shows how to make a new one. All commands are also in the [single-object README](image_generation/data/vlm_single_object/README.md#generating).
[`visualize_patch_grid.ipynb`](visualize_patch_grid.ipynb) shows how the objects fall into the patch grid.

If you have an NVIDIA GPU with CUDA installed then you can use the GPU to accelerate rendering like this:

```bash
blender --background --python render_images.py -- --num_images 10 --use_gpu 1
```

After this command terminates you should have ten freshly rendered images stored in `output/images` like these:

<div align="center">
  <img src="images/img1.png" width="260px">
  <img src="images/img2.png" width="260px">
  <img src="images/img3.png" width="260px">
  <br>
  <img src="images/img4.png" width="260px">
  <img src="images/img5.png" width="260px">
  <img src="images/img6.png" width="260px">
</div>

The file `output/CLEVR_scenes.json` will contain ground-truth scene information for all newly rendered images.

You can find [more details about image rendering here](image_generation/README.md).

## Step 2: Generating Questions
Next we generate questions, functional programs, and answers for the rendered images generated in the previous step.
This step takes as input the single JSON file containing all ground-truth scene information, and outputs a JSON file 
containing questions, answers, and functional programs for the questions in a single JSON file.

You can generate questions like this:

```bash
cd question_generation
python generate_questions.py
```

The file `output/CLEVR_questions.json` will then contain questions for the generated images.

You can [find more details about question generation here](question_generation/README.md).
