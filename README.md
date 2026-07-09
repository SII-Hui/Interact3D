# Interact3D: Compositional 3D Generation of Interactive Objects

<p align="center">
  <a href="https://arxiv.org/abs/2603.16085">
    <img src="https://img.shields.io/badge/arXiv-2603.16085-b31b1b.svg" alt="arXiv 2603.16085">
  </a>
</p>

![Interact3D Teaser](asset/teaser.png)

**Interact3D** generates collision-aware compositional 3D assets from a natural-language scene prompt. The pipeline first creates a compositional image and isolated object images, reconstructs them with TRELLIS.2, separates object-level guidance with PartField, and finally registers the generated objects into a physically plausible 3D scene.

---

## ✨ What is included

- **Single-sample inference**: prompt → images → GLBs → registered scene.
- **Batch inference**: resumable processing over `data/data.json`.

- **Optional Agent Optimization**: optional VLM-guided retry when registration reports severe residual penetration.
- **Optional Multi-object extension**: sequentially add more objects to an existing registered scene.
- **Optional MuJoCo preview export**: lightweight MJCF scene for manual interaction/inspection.

---

## 🧩 Repository layout

```text
run_image_generation.py          # Stage 1: scene/object image generation via Gemini API
run_trellis2_generation.py       # Stage 2: image-to-3D with TRELLIS.2
run_partfield_segmentation.py    # Stage 3: PartField grouping into guidance meshes
run_registration.py              # Stage 4: registration + collision refinement
run_agent_optimization.py/.sh    # Stage 5: optional retry loop for failed registrations
run_multiobject_composition.sh   # Optional: sequential 3+ object composition
convert_to_mujoco.py             # Optional: MuJoCo MJCF preview exporter
run_batch.sh                     # Batch runner over data/data.json
```

---

## 🚀 Quick start

### 1. Install environments

This repo intentionally reuses the original environments of **PartField** and **TRELLIS.2** instead of forcing a single merged environment. This matches how `run_batch.sh`, `run_agent_optimization.sh`, and `run_multiobject_composition.sh` call the code: Stage 2 runs in the TRELLIS.2 env, while Stage 1/3/4/5 run in the PartField env.

| Env | Default name | Used for |
| --- | --- | --- |
| PartField env | `partfield` | Gemini image stage, PartField segmentation, GeoTransformer registration, Agent Optimization, MuJoCo utility |
| TRELLIS.2 env | `trellis2` | Image-to-3D reconstruction |

Install them following the upstream repos first:

```bash
# PartField: create/install the env as required by the official PartField repo.
git clone https://github.com/nv-tlabs/PartField.git

# TRELLIS.2: create/install the env as required by the official TRELLIS.2 repo.
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
```

Then install this repo's lightweight Python dependencies into the **PartField env**:

```bash
conda run --no-capture-output -n partfield pip install -r requirements.txt
```

If your TRELLIS.2 env does not already include Pillow, install it there too:

```bash
conda run --no-capture-output -n trellis2 pip install Pillow
```

### 2. Install GeoTransformer in the PartField env

GeoTransformer is used by Stage 4 for optional global registration initialization.

```bash
git clone https://github.com/qinzheng93/GeoTransformer.git
cd GeoTransformer
conda run --no-capture-output -n partfield pip install -r requirements.txt
conda run --no-capture-output -n partfield python setup.py build develop
mkdir -p weights
# Put geotransformer-3dmatch.pth.tar under GeoTransformer/weights/.
cd -
```

Expected checkpoint path:

```text
GeoTransformer/weights/geotransformer-3dmatch.pth.tar
```

### 3. Prepare PartField checkpoint and Blender

Place the official PartField checkpoint at:

```text
PartField/model/model_objaverse.ckpt
```

Blender is used by Stage 3 to render part contact sheets. If `blender` is not available, download a local copy and export `BLENDER`:

```bash
mkdir -p third_party
cd third_party
wget https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz
tar -xf blender-4.3.2-linux-x64.tar.xz
cd -
export BLENDER="$PWD/third_party/blender-4.3.2-linux-x64/blender"
$BLENDER -b --version
```

### 4. Configure Gemini API and paths

Set a Gemini API key before running stages that use LLM/VLM/image generation. The shell runners use `conda run` internally, so you do **not** need to manually switch envs during batch/multi-object/agent runs.

```bash
export GEMINI_API_KEY="<your_gemini_api_key>"

export PARTFIELD_ENV=partfield
export TRELLIS2_ENV=trellis2
export TRELLIS2_DIR=TRELLIS.2
export PARTFIELD_DIR=PartField
export GEOTRANSFORMER_DIR=GeoTransformer
export GEOTRANSFORMER_WEIGHTS=GeoTransformer/weights/geotransformer-3dmatch.pth.tar
export BLENDER="$PWD/third_party/blender-4.3.2-linux-x64/blender"

# Optional model overrides used by the shell runners.
export INTERACT3D_GEMINI_TEXT_MODEL=gemini-3.5-flash
export INTERACT3D_GEMINI_VISION_MODEL=gemini-3.5-flash
export INTERACT3D_GEMINI_IMAGE_MODEL=gemini-3-pro-image
```

---

## 🧪 Single-sample pipeline

Use `conda run` to match the stage-specific environments used by the batch scripts.

### Stage 1 — Generate images

```bash
conda run --no-capture-output -n partfield python run_image_generation.py \
  --scene-prompt "A ripe lemon is nestled in a picnic basket, fresh from the market." \
  --outputs-root outputs \
  --category "food" \
  --gemini-text-model "${INTERACT3D_GEMINI_TEXT_MODEL:-gemini-3.5-flash}" \
  --gemini-image-model "${INTERACT3D_GEMINI_IMAGE_MODEL:-gemini-3-pro-image}"
```

The script prints the sample directory as the final stdout line, e.g.:

```text
outputs/food/00001_lemon_picnic_basket
```

Key outputs:

```text
compositional_scene.png
object_a.png
object_b.png
metadata.json
```

### Stage 2 — Reconstruct GLBs with TRELLIS.2

```bash
conda run --no-capture-output -n trellis2 python run_trellis2_generation.py \
  --input-dir <sample_dir> \
  --trellis2-dir TRELLIS.2
```

Key outputs:

```text
compositional_scene.glb
object_a.glb
object_b.glb
```

Existing `.glb` files are skipped unless `--overwrite` is passed.

### Stage 3 — Build guidance meshes with PartField

```bash
conda run --no-capture-output -n partfield python run_partfield_segmentation.py \
  --input-dir <sample_dir> \
  --partfield-dir PartField \
  --blender "$BLENDER" \
  --gemini-vision-model "${INTERACT3D_GEMINI_VISION_MODEL:-gemini-3.5-flash}"
```

Key outputs:

```text
guidance_object_a.glb
guidance_object_b.glb
```

Temporary PartField folders are removed by default. Use `--keep-intermediates` for debugging.

### Stage 4 — Register objects into a scene

```bash
conda run --no-capture-output -n partfield python run_registration.py \
  --input-dir <sample_dir> \
  --geotransformer-dir GeoTransformer \
  --geotransformer-weights GeoTransformer/weights/geotransformer-3dmatch.pth.tar
```

Key output:

```text
registered_scene.glb
```

`metadata.json` records the selected anchor, object transforms, collision refinement details, and `registration.status`. The transform convention is:

```text
p_world = scale * R @ p_local + t
```

---

## 🤖 Optional Agent Optimization

If Stage 4 writes `registration.status = "failure"`, the residual penetration exceeded the configured threshold. Agent Optimization keeps the anchor object fixed, regenerates only the moving object/scene images, then reruns Stage 2-4. For cost control, this step is **manual** and is not invoked by `run_batch.sh`.

```bash
AGENT_OPTIMIZATION_ITERS=2 ./run_agent_optimization.sh <sample_dir>
```


---

## 🏭 Batch inference

`run_batch.sh` is the cost-conscious batch runner used for dataset production. It runs Stage 1-4 only:

```text
image generation -> TRELLIS.2 -> PartField -> registration
```

It does **not** run Agent Optimization automatically. This keeps large-scale generation cheaper; if a sample has `registration.status = "failure"`, inspect it later and optionally launch `run_agent_optimization.sh` on that sample only.

By default, the script reads `DATA_JSON=${DATA_JSON:-data/data.json}`.

```json
{
  "00001": {
    "category": "food",
    "prompt": "A ripe lemon is nestled in a picnic basket, fresh from the market."
  }
}
```

Run a batch:

```bash
DATA_JSON=data/data.json ./run_batch.sh
```


Batch runs are resumable:

```text
outputs/.batch_state/    # completion markers and machine-scoped summaries
outputs/.batch_logs/     # per-item logs
```


## ➕ Multi-object composition

For scenes with three or more objects, start from a completed sample containing `registered_scene.glb`, then add objects sequentially with a tab-separated plan:

```text
napkin	A folded napkin lies beside the picnic basket and lemon, with a visible gap and no intersection.
small jam jar	A small jam jar stands behind the picnic basket, fully supported by a flat base.
```

Run:

```bash
./run_multiobject_composition.sh <sample_dir> additions.tsv
```

Outputs are written under:

```text
<sample_dir>/multiobject/step_XX_<object_name>/
```

Enable per-step Agent Optimization if needed:

```bash
MULTIOBJECT_AGENT_ITERS=2 ./run_multiobject_composition.sh <sample_dir> additions.tsv
```

---

## 🕹️ Optional MuJoCo preview export

To convert a registered two-object sample into a compact MJCF preview scene with separate MuJoCo bodies and preserved visual textures. We can use:

```bash
conda run --no-capture-output -n partfield python convert_to_mujoco.py \
  --input-dir <sample_dir>
```

Outputs:

```text
<sample_dir>/mujoco/scene.xml
<sample_dir>/mujoco/meshes/
```

Default behavior is designed for manual interaction in the MuJoCo viewer:

- zero gravity;
- no floor;
- contacts disabled;
- both objects have `freejoint` and stay still until manipulated.

Useful variants:

```bash
# Only object_a is movable; object_b is fixed.
conda run --no-capture-output -n partfield python convert_to_mujoco.py \
  --input-dir <sample_dir> \
  --dynamic-object object_a

# Physical preview with gravity, floor, and contacts.
conda run --no-capture-output -n partfield python convert_to_mujoco.py \
  --input-dir <sample_dir> \
  --dynamic-object object_a \
  --include-floor \
  --enable-contact \
  --gravity 0 0 -9.81
```

---

## 🖼️ Qualitative results

**Two-part composition**

![Interact3D TwoPart](asset/TwoComp_1.png)

**Multi-object composition**

![Interact3D ManyPart](asset/ManyComp_1.png)

![Interact3D ManyPart](asset/ManyComp_2.png)

---

## 🙏 Acknowledgements

This repository builds on several excellent open-source projects. We sincerely thank their authors and contributors:

- [PartField](https://github.com/nv-tlabs/PartField).
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer).
- [Blender](https://www.blender.org/).
- [trimesh](https://github.com/mikedh/trimesh), [SciPy](https://github.com/scipy/scipy), and [Pillow](https://github.com/python-pillow/Pillow).

Please also follow the licenses and terms of the upstream projects, checkpoints, and external tools when using or redistributing this code.

---

## 📜 Citation

If you find this repository useful, please cite:

```bibtex
@article{shan2026interact3d,
  title={Interact3D: Compositional 3D Generation of Interactive Objects},
  author={Shan, Hui and Luo, Keyang and Li, Ming and Zheng, Sizhe and Fu, Yanwei and Chen, Zhen and Huang, Xiangru},
  journal={arXiv preprint arXiv:2603.16085},
  year={2026}
}
```
