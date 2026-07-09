#!/usr/bin/env bash
# Sequentially extend a completed two-object sample to a 3+ object scene.
#
# Each line in additions.tsv adds exactly one object. The previous registered
# scene becomes the fixed anchor for the next step, while Gemini creates the new
# compositional/object images before Stage 2-4 are rerun.
#
# Usage:
#   ./run_multiobject_composition.sh <base_sample_dir> <additions.tsv>
# additions.tsv format:
#   object_name<TAB>scene prompt for adding this object
set -uo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <base_sample_dir> <additions.tsv>" >&2
  exit 2
fi

BASE_DIR="$1"
PLAN_TSV="$2"

# Repository paths and runtime/model settings; all can be overridden by env vars.
TRELLIS2_DIR="${TRELLIS2_DIR:-TRELLIS.2}"
PARTFIELD_DIR="${PARTFIELD_DIR:-PartField}"
GEOTRANSFORMER_DIR="${GEOTRANSFORMER_DIR:-GeoTransformer}"
GEOTRANSFORMER_WEIGHTS="${GEOTRANSFORMER_WEIGHTS:-$GEOTRANSFORMER_DIR/weights/geotransformer-3dmatch.pth.tar}"
BLENDER="${BLENDER:-$PWD/third_party/blender-4.3.2-linux-x64/blender}"
GEMINI_VISION_MODEL="${GEMINI_VISION_MODEL:-${INTERACT3D_GEMINI_VISION_MODEL:-gemini-3.5-flash}}"
GEMINI_IMAGE_MODEL="${GEMINI_IMAGE_MODEL:-${INTERACT3D_GEMINI_IMAGE_MODEL:-gemini-3-pro-image}}"
PARTFIELD_ENV="${PARTFIELD_ENV:-partfield}"
TRELLIS2_ENV="${TRELLIS2_ENV:-trellis2}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n}"
PARTFIELD_PY=($CONDA_RUN "$PARTFIELD_ENV" python)
TRELLIS2_PY=($CONDA_RUN "$TRELLIS2_ENV" python)
MULTIOBJECT_AGENT_ITERS="${MULTIOBJECT_AGENT_ITERS:-0}"

# Export canonical variables for Python subprocesses launched through conda run.
export INTERACT3D_GEMINI_VISION_MODEL="$GEMINI_VISION_MODEL"
export INTERACT3D_GEMINI_IMAGE_MODEL="$GEMINI_IMAGE_MODEL"
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

log() { printf '[multiobject %s] %s\n' "$(date '+%F %T')" "$*"; }

abs_dir() {
  local dir="$1"
  [ -d "$dir" ] || { echo "Directory not found: $dir" >&2; exit 2; }
  cd "$dir" && pwd
}

# Reuse Python's regex handling to make stable filesystem-safe step names.
slugify() {
  "${PARTFIELD_PY[@]}" - "$1" <<'PY'
import re, sys
text = sys.argv[1].lower()
print(re.sub(r'[^a-z0-9]+', '_', text).strip('_') or 'object')
PY
}

# Query the per-step registration verdict recorded by run_registration.py.
registration_status() {
  "${PARTFIELD_PY[@]}" - "$1/metadata.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding='utf-8')).get('registration', {}).get('status', 'success'))
except Exception:
    print('unknown')
PY
}

# Keep each step directory in the standard two-object layout expected by the
# downstream scripts: object_a is the fixed anchor, object_b is the new object.
cleanup_nonstandard_artifacts() {
  local sample_dir="$1" name path
  for path in "$sample_dir"/*.png "$sample_dir"/*.glb; do
    [ -e "$path" ] || continue
    name="$(basename "$path")"
    case "$name" in
      compositional_scene.png|object_a.png|object_b.png|compositional_scene.glb|object_a.glb|object_b.glb) ;;
      *) log "remove nonstandard artifact: $path"; rm -f "$path" ;;
    esac
  done
}

# Remove stale PartField features before rerunning segmentation for a new step.
clear_partfield_cache() {
  local sample_dir="$1" run_name cache_dir
  run_name="interact3d_$(basename "$sample_dir")"
  cache_dir="$PARTFIELD_DIR/exp_results/partfield_features/$run_name"
  [ -d "$cache_dir" ] && { log "remove stale PartField cache: $cache_dir"; rm -rf "$cache_dir"; }
  rm -rf "$sample_dir/partfield_work" "$sample_dir/partfield_parts" "$sample_dir/partfield_renders"
}

# Run the standard downstream stages for one newly prepared step directory.
# Registration is anchored to object_a so the existing composition stays fixed.
run_stages_2_to_4() {
  local sample_dir="$1" item_log="$2"
  cleanup_nonstandard_artifacts "$sample_dir"
  log "Stage 2 TRELLIS.2 -> $sample_dir"
  "${TRELLIS2_PY[@]}" run_trellis2_generation.py \
    --input-dir "$sample_dir" --trellis2-dir "$TRELLIS2_DIR" </dev/null >>"$item_log" 2>&1 || return 1

  clear_partfield_cache "$sample_dir"
  log "Stage 3 PartField -> $sample_dir"
  "${PARTFIELD_PY[@]}" run_partfield_segmentation.py \
    --input-dir "$sample_dir" --partfield-dir "$PARTFIELD_DIR" \
    --blender "$BLENDER" --gemini-vision-model "$GEMINI_VISION_MODEL" </dev/null >>"$item_log" 2>&1 || return 2

  log "Stage 4 registration -> $sample_dir"
  "${PARTFIELD_PY[@]}" run_registration.py \
    --input-dir "$sample_dir" \
    --geotransformer-dir "$GEOTRANSFORMER_DIR" \
    --geotransformer-weights "$GEOTRANSFORMER_WEIGHTS" \
    --anchor object_a </dev/null >>"$item_log" 2>&1 || return 3

  # run_registration re-exports object_a through trimesh and can lose GLB materials.
  # Rebuild the final scene by keeping object_a.glb byte-for-byte as the visual anchor
  # and applying the relative registration transform only to object_b.
  log "Finalize multi-object scene with preserved anchor materials -> $sample_dir"
  "${PARTFIELD_PY[@]}" run_multiobject_finalize.py --input-dir "$sample_dir" </dev/null >>"$item_log" 2>&1 || return 4
}

# Validate the base sample once before creating any multiobject outputs.
BASE_DIR="$(abs_dir "$BASE_DIR")"
[ -f "$PLAN_TSV" ] || { echo "Plan file not found: $PLAN_TSV" >&2; exit 2; }
[ -f "$BASE_DIR/registered_scene.glb" ] || { echo "Missing base registered_scene.glb: $BASE_DIR" >&2; exit 2; }

# All intermediate step directories, logs, and the manifest live under the base sample.
WORK_ROOT="$BASE_DIR/multiobject"
LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/multiobject_$(date '+%Y%m%d_%H%M%S').log"
CURRENT_GLB="$BASE_DIR/registered_scene.glb"
CURRENT_LABEL="composition from $(basename "$BASE_DIR")"
step=0

# Process additions sequentially because every successful step becomes the next anchor.
while IFS=$'\t' read -r object_name scene_prompt || [ -n "${object_name}${scene_prompt}" ]; do
  [[ -z "${object_name// }" || "${object_name:0:1}" = "#" ]] && continue
  if [ -z "${scene_prompt// }" ]; then
    log "skip malformed line for object=$object_name; missing tab-separated prompt"
    continue
  fi

  step=$((step + 1))
  object_slug="$(slugify "$object_name")"
  step_dir="$WORK_ROOT/step_$(printf '%02d' "$step")_${object_slug}"
  while [ -d "$step_dir" ]; do
    step=$((step + 1))
    step_dir="$WORK_ROOT/step_$(printf '%02d' "$step")_${object_slug}"
  done
  mkdir -p "$step_dir"
  log "Step $step add '$object_name' -> $step_dir"

  if ! "${PARTFIELD_PY[@]}" run_multiobject_step.py \
      --anchor-glb "$CURRENT_GLB" \
      --output-dir "$step_dir" \
      --object-name "$object_name" \
      --scene-prompt "$scene_prompt" \
      --anchor-label "$CURRENT_LABEL" \
      --blender "$BLENDER" \
      --gemini-image-model "$GEMINI_IMAGE_MODEL" </dev/null >>"$RUN_LOG" 2>&1; then
    log "step preparation failed; see $RUN_LOG"
    exit 1
  fi

  run_stages_2_to_4 "$step_dir" "$RUN_LOG"
  stage_status=$?
  if [ "$stage_status" -ne 0 ]; then
    log "Stage 2-4 failed at step $step with code=$stage_status; see $RUN_LOG"
    exit 1
  fi

  status="$(registration_status "$step_dir")"
  if [ "$status" != "success" ] && [ "$MULTIOBJECT_AGENT_ITERS" -gt 0 ]; then
    log "step $step status=$status; run Agent Optimization iters=$MULTIOBJECT_AGENT_ITERS"
    agent_stdout="$LOG_DIR/step_$(printf '%02d' "$step")_agent.stdout"
    AGENT_OPTIMIZATION_ITERS="$MULTIOBJECT_AGENT_ITERS" ./run_agent_optimization.sh "$step_dir" "$MULTIOBJECT_AGENT_ITERS" >"$agent_stdout" 2>>"$RUN_LOG" || {
      cat "$agent_stdout" >>"$RUN_LOG"
      log "Agent Optimization failed at step $step; see $RUN_LOG"
      exit 1
    }
    cat "$agent_stdout" >>"$RUN_LOG"
    step_dir="$(tail -n1 "$agent_stdout")"
    status="$(registration_status "$step_dir")"
  fi

  if [ "$status" != "success" ]; then
    log "step $step finished with status=$status; see $RUN_LOG"
    exit 1
  fi

  CURRENT_GLB="$step_dir/registered_scene.glb"
  CURRENT_LABEL="$CURRENT_LABEL plus $object_name"
  echo -e "$step\t$object_name\t$step_dir" >>"$WORK_ROOT/manifest.tsv"
done <"$PLAN_TSV"

log "SUCCESS final scene -> $CURRENT_GLB"
echo "$CURRENT_GLB"
