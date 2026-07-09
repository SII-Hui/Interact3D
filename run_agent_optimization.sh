#!/usr/bin/env bash
# Run Stage 5 Agent Optimization for one failed sample directory.
#
# The script keeps the registered anchor fixed, asks the Python agent stage to
# regenerate safer Gemini image prompts/images, then reruns Stage 2-4 on each
# retry directory until registration succeeds or the iteration budget is used.
#
# Usage:
#   ./run_agent_optimization.sh <sample_dir> [max_iters]
#   AGENT_OPTIMIZATION_ITERS=2 ./run_agent_optimization.sh <sample_dir>
set -uo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sample_dir> [max_iters]" >&2
  exit 2
fi

INPUT_DIR="$1"
# Positional max_iters takes precedence over the environment variable.
AGENT_OPTIMIZATION_ITERS="${2:-${AGENT_OPTIMIZATION_ITERS:-1}}"

# Repository paths and model/runtime settings; all can be overridden by env vars.
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

# Export canonical variables so Python subprocesses launched via conda run use
# exactly the same Gemini models and HuggingFace cache settings.
export INTERACT3D_GEMINI_VISION_MODEL="$GEMINI_VISION_MODEL"
export INTERACT3D_GEMINI_IMAGE_MODEL="$GEMINI_IMAGE_MODEL"
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

log() { printf '[agent-opt %s] %s\n' "$(date '+%F %T')" "$*"; }

abs_dir() {
  local dir="$1"
  [ -d "$dir" ] || { echo "Directory not found: $dir" >&2; exit 2; }
  cd "$dir" && pwd
}

# Read the registration verdict from metadata.json. Missing/invalid metadata is
# reported as "unknown" so the caller can fail with a clear log message.
registration_status() {
  "${PARTFIELD_PY[@]}" - "$1/metadata.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("registration", {}).get("status", "success"))
except Exception:
    print("unknown")
PY
}

# Preserve the same anchor selected by Stage 5 when rerunning registration.
agent_anchor() {
  "${PARTFIELD_PY[@]}" - "$1/metadata.json" <<'PY'
import json, sys
try:
    meta = json.load(open(sys.argv[1], encoding="utf-8"))
    print(meta.get("agent_optimization", {}).get("anchor_object") or meta.get("registration", {}).get("anchor_object", ""))
except Exception:
    print("")
PY
}

# Keep each retry directory in the standard two-object layout expected by
# TRELLIS.2, PartField, and registration.
cleanup_nonstandard_artifacts() {
  local sample_dir="$1" name
  for path in "$sample_dir"/*.png "$sample_dir"/*.glb; do
    [ -e "$path" ] || continue
    name="$(basename "$path")"
    case "$name" in
      compositional_scene.png|object_a.png|object_b.png|compositional_scene.glb|object_a.glb|object_b.glb) ;;
      *) log "remove nonstandard artifact: $path"; rm -f "$path" ;;
    esac
  done
}

# PartField caches features by run name; remove stale cache/intermediates before
# rerunning Stage 3 on regenerated images/meshes.
clear_partfield_cache() {
  local sample_dir="$1" run_name cache_dir
  run_name="interact3d_$(basename "$sample_dir")"
  cache_dir="$PARTFIELD_DIR/exp_results/partfield_features/$run_name"
  if [ -d "$cache_dir" ]; then
    log "remove stale PartField cache: $cache_dir"
    rm -rf "$cache_dir"
  fi
  rm -rf "$sample_dir/partfield_work" "$sample_dir/partfield_parts" "$sample_dir/partfield_renders"
}

# Rerun the deterministic downstream stages after Gemini regenerates images.
# Return codes identify the failing stage for concise caller-side error handling.
run_stages_2_to_4() {
  local sample_dir="$1" item_log="$2" anchor_arg="$3"
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
  local registration_args=(--input-dir "$sample_dir" --geotransformer-dir "$GEOTRANSFORMER_DIR" --geotransformer-weights "$GEOTRANSFORMER_WEIGHTS")
  [ -n "$anchor_arg" ] && registration_args+=(--anchor "$anchor_arg")
  "${PARTFIELD_PY[@]}" run_registration.py "${registration_args[@]}" </dev/null >>"$item_log" 2>&1 || return 3
}

# All retry outputs and logs stay under the source sample directory.
SOURCE_DIR="$(abs_dir "$INPUT_DIR")"
CURRENT_DIR="$SOURCE_DIR"
LOG_DIR="$SOURCE_DIR/agent_optimization/logs"
mkdir -p "$LOG_DIR"
ITEM_LOG="$LOG_DIR/agent_optimization_$(date '+%Y%m%d_%H%M%S').log"

if [ "$AGENT_OPTIMIZATION_ITERS" -le 0 ]; then
  log "AGENT_OPTIMIZATION_ITERS=$AGENT_OPTIMIZATION_ITERS, nothing to run"
  echo "$CURRENT_DIR"
  exit 0
fi

status="$(registration_status "$CURRENT_DIR")"
log "initial status=$status dir=$CURRENT_DIR"
if [ "$status" = "success" ]; then
  echo "$CURRENT_DIR"
  exit 0
fi

# Only failed registrations are optimized. Unknown statuses fall through to the
# final failure branch instead of being silently retried.
attempt=0
while [ "$status" = "failure" ] && [ "$attempt" -lt "$AGENT_OPTIMIZATION_ITERS" ]; do
  attempt=$((attempt + 1))
  log "Stage 5 attempt=$attempt from $CURRENT_DIR"
  agent_stdout="$LOG_DIR/iter_${attempt}_$(date '+%Y%m%d_%H%M%S').stdout"
  : >"$agent_stdout"

  if ! "${PARTFIELD_PY[@]}" run_agent_optimization.py \
      --input-dir "$CURRENT_DIR" \
      --gemini-vision-model "$GEMINI_VISION_MODEL" \
      --gemini-image-model "$GEMINI_IMAGE_MODEL" \
      --attempt "$attempt" </dev/null >"$agent_stdout" 2>>"$ITEM_LOG"; then
    cat "$agent_stdout" >>"$ITEM_LOG"
    log "Stage 5 prompt/image regeneration failed; see $ITEM_LOG"
    exit 1
  fi
  cat "$agent_stdout" >>"$ITEM_LOG"

  NEXT_DIR="$(tail -n1 "$agent_stdout")"
  if [ -z "$NEXT_DIR" ] || [ ! -d "$NEXT_DIR" ]; then
    log "Stage 5 produced invalid output dir: $NEXT_DIR"
    exit 1
  fi
  CURRENT_DIR="$(abs_dir "$NEXT_DIR")"
  fixed_anchor="$(agent_anchor "$CURRENT_DIR")"
  log "retry_dir=$CURRENT_DIR fixed_anchor=${fixed_anchor:-auto}"

  run_stages_2_to_4 "$CURRENT_DIR" "$ITEM_LOG" "$fixed_anchor"
  stage_status=$?
  if [ "$stage_status" -ne 0 ]; then
    log "Downstream Stage 2-4 failed with code=$stage_status; see $ITEM_LOG"
    exit 1
  fi
  status="$(registration_status "$CURRENT_DIR")"
  log "attempt=$attempt status=$status"
done

if [ "$status" != "success" ]; then
  log "finished without success: status=$status after $attempt attempt(s); see $ITEM_LOG"
  echo "$CURRENT_DIR"
  exit 1
fi

log "SUCCESS -> $CURRENT_DIR/registered_scene.glb"
echo "$CURRENT_DIR"
