#!/usr/bin/env bash
# Batch-run the full Interact3D two-object pipeline:
#   prompt -> Gemini images -> TRELLIS.2 GLBs -> PartField guidance meshes -> registration.
#
# Input:  DATA_JSON, whose entries contain at least `prompt` and optionally `category`.
# Output: one sample directory per item under OUTPUTS_ROOT, plus logs and resume markers.
#
# Resumability:
# - A completed dataset id writes $OUTPUTS_ROOT/.batch_state/<id>.done.
# - Re-running the script skips completed ids and keeps per-item logs in .batch_logs.
#
# Common usage:
#   ./run_batch.sh                                      # run all entries
#   MAX_ITEMS=5 ./run_batch.sh                          # run first 5 eligible entries
#   START_INDEX=501 ./run_batch.sh                      # resume from the 501st entry
#   START_INDEX=1 END_INDEX=500 ./run_batch.sh          # process an inclusive slice
#
# START_INDEX/END_INDEX are 1-based positions in DATA_JSON iteration order. Use
# non-overlapping slices when running on multiple machines.
set -uo pipefail

# ---- User-overridable configuration ----
# Prompt list consumed by entries(); keep it as JSON so item order is stable.
DATA_JSON="${DATA_JSON:-data/data.json}"
# Resolve to an absolute path once so downstream stages receive stable paths,
# regardless of their own working directories.
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
mkdir -p "$OUTPUTS_ROOT"
OUTPUTS_ROOT="$(cd "$OUTPUTS_ROOT" && pwd)"
TRELLIS2_DIR="${TRELLIS2_DIR:-TRELLIS.2}"
PARTFIELD_DIR="${PARTFIELD_DIR:-PartField}"
GEOTRANSFORMER_DIR="${GEOTRANSFORMER_DIR:-GeoTransformer}"
GEOTRANSFORMER_WEIGHTS="${GEOTRANSFORMER_WEIGHTS:-$GEOTRANSFORMER_DIR/weights/geotransformer-3dmatch.pth.tar}"
# Blender is only used by Stage 3 to render PartField contact sheets.
BLENDER="${BLENDER:-$PWD/third_party/blender-4.3.2-linux-x64/blender}"
# Gemini model defaults can be overridden either through short names below or
# through the INTERACT3D_GEMINI_* variables consumed by Python stages.
GEMINI_TEXT_MODEL="${GEMINI_TEXT_MODEL:-${INTERACT3D_GEMINI_TEXT_MODEL:-gemini-3.5-flash}}"
GEMINI_VISION_MODEL="${GEMINI_VISION_MODEL:-${INTERACT3D_GEMINI_VISION_MODEL:-$GEMINI_TEXT_MODEL}}"
GEMINI_IMAGE_MODEL="${GEMINI_IMAGE_MODEL:-${INTERACT3D_GEMINI_IMAGE_MODEL:-gemini-3-pro-image}}"
ANALYSIS_FAILURE_EXIT_CODE="${INTERACT3D_ANALYSIS_FAILURE_EXIT_CODE:-42}"
# Export canonical names so subprocesses launched by conda run see the same models.
export INTERACT3D_GEMINI_TEXT_MODEL="$GEMINI_TEXT_MODEL"
export INTERACT3D_GEMINI_VISION_MODEL="$GEMINI_VISION_MODEL"
export INTERACT3D_GEMINI_IMAGE_MODEL="$GEMINI_IMAGE_MODEL"

# ---- Shared runtime environment inherited by every `conda run` subprocess ----
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Per-stage conda environments. Stage 2 uses the TRELLIS.2 environment; all
# other top-level stages use the PartField environment. `conda run` keeps the
# caller shell untouched, so no manual activate/deactivate is required.
PARTFIELD_ENV="${PARTFIELD_ENV:-partfield}"
TRELLIS2_ENV="${TRELLIS2_ENV:-trellis2}"
CONDA_RUN="${CONDA_RUN:-conda run --no-capture-output -n}"
PARTFIELD_PY=($CONDA_RUN "$PARTFIELD_ENV" python)
TRELLIS2_PY=($CONDA_RUN "$TRELLIS2_ENV" python)

START_INDEX="${START_INDEX:-1}"   # 1-based index of the first entry to process
END_INDEX="${END_INDEX:-0}"       # 1-based index of the last entry (inclusive); 0 = no limit
MAX_ITEMS="${MAX_ITEMS:-0}"       # 0 = no limit

STATE_DIR="$OUTPUTS_ROOT/.batch_state"
LOG_DIR="$OUTPUTS_ROOT/.batch_logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

log() { printf '[batch %s] %s\n' "$(date '+%F %T')" "$*"; }

# Emit "id<TAB>category<TAB>prompt" for every non-empty prompt while preserving
# JSON iteration order. The tab-separated format keeps parsing simple in bash.
entries() {
  "${PARTFIELD_PY[@]}" - "$DATA_JSON" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for key, item in data.items():
    prompt = (item.get("prompt") or "").strip()
    if prompt:
        print(f"{key}\t{(item.get('category') or '').strip()}\t{prompt}")
PY
}

# One-time preflight: Stage 3 needs Blender to render part/contact sheets.
# Failing early is cheaper than letting each item reach PartField and crash on
# missing headless OpenGL/X11 libraries.
preflight_blender() {
  if "$BLENDER" -b --version </dev/null >/dev/null 2>&1; then
    log "preflight: Blender OK ($BLENDER)"
    return 0
  fi
  log "preflight: Blender FAILED to launch: $BLENDER"
  local missing
  missing="$(ldd "$BLENDER" 2>/dev/null | awk '/not found/{print $1}' | sort -u | tr '\n' ' ')"
  [ -n "$missing" ] && log "preflight: missing shared libraries: $missing"
  log "preflight: install the required system libraries, e.g."
  log "  Debian/Ubuntu: apt-get update && apt-get install -y libxkbcommon0 libxrender1 libxi6 libxxf86vm1 libxfixes3 libsm6 libice6 libgl1 libglu1-mesa libxext6 libx11-6"
  log "then verify with: $BLENDER -b --version"
  return 1
}
preflight_blender || exit 1

processed=0; ok=0; skipped=0; failed=0; index=0
ok_ids=(); failed_ids=(); skipped_ids=()
# Read entries through fd 3 instead of stdin. Several Python/conda subprocesses
# also read stdin; isolating the loop input prevents them from consuming it.
while IFS=$'\t' read -r -u 3 id category prompt; do
  index=$((index + 1))
  [ "$index" -lt "$START_INDEX" ] && continue
  [ "$END_INDEX" -gt 0 ] && [ "$index" -gt "$END_INDEX" ] && break
  [ "$MAX_ITEMS" -gt 0 ] && [ "$processed" -ge "$MAX_ITEMS" ] && break
  processed=$((processed + 1))

  marker="$STATE_DIR/$id.done"
  if [ -f "$marker" ]; then
    log "[$id] already complete, skipping"; skipped=$((skipped + 1)); skipped_ids+=("$id"); continue
  fi

  item_log="$LOG_DIR/$id.log"
  : >"$item_log"
  log "[$id] category=$category :: $prompt (log: $item_log)"

  # ---- Stage 1: prompt analysis + Gemini compositional/object images ----
  category_arg=(); [ -n "$category" ] && category_arg=(--category "$category")
  stage1_stdout="$LOG_DIR/$id.stage1.stdout"
  : >"$stage1_stdout"
  "${PARTFIELD_PY[@]}" run_image_generation.py \
      --scene-prompt "$prompt" \
      --outputs-root "$OUTPUTS_ROOT" \
      --gemini-text-model "$GEMINI_TEXT_MODEL" \
      --gemini-image-model "$GEMINI_IMAGE_MODEL" \
      "${category_arg[@]}" </dev/null >"$stage1_stdout" 2>>"$item_log"
  stage1_status=$?
  cat "$stage1_stdout" >>"$item_log"
  if [ "$stage1_status" -eq "$ANALYSIS_FAILURE_EXIT_CODE" ]; then
    log "[$id] stage1 prompt analysis invalid, skipping remaining stages"; skipped=$((skipped + 1)); skipped_ids+=("$id:stage1_analysis"); continue
  fi
  if [ "$stage1_status" -ne 0 ]; then
    log "[$id] stage1 image-generation FAILED"; failed=$((failed + 1)); failed_ids+=("$id:stage1"); continue
  fi
  out_dir=$(tail -n1 "$stage1_stdout")
  if [ -z "$out_dir" ] || [ ! -d "$out_dir" ]; then
    log "[$id] stage1 image-generation FAILED"; failed=$((failed + 1)); failed_ids+=("$id:stage1"); continue
  fi
  out_dir="$(cd "$out_dir" && pwd)"   # absolute path for stages 2-4
  log "[$id] output_dir=$out_dir"

  # ---- Stage 2: TRELLIS.2 image-to-3D (trellis2 conda env) ----
  if ! "${TRELLIS2_PY[@]}" run_trellis2_generation.py \
      --input-dir "$out_dir" --trellis2-dir "$TRELLIS2_DIR" </dev/null >>"$item_log" 2>&1; then
    log "[$id] stage2 TRELLIS.2 FAILED"; failed=$((failed + 1)); failed_ids+=("$id:stage2"); continue
  fi

  # ---- Stage 3: PartField segmentation + Gemini VLM part grouping ----
  if ! "${PARTFIELD_PY[@]}" run_partfield_segmentation.py \
      --input-dir "$out_dir" --partfield-dir "$PARTFIELD_DIR" \
      --blender "$BLENDER" --gemini-vision-model "$GEMINI_VISION_MODEL" </dev/null >>"$item_log" 2>&1; then
    log "[$id] stage3 PartField FAILED"; failed=$((failed + 1)); failed_ids+=("$id:stage3"); continue
  fi

  # ---- Stage 4: registration + collision-aware composition ----
  if ! "${PARTFIELD_PY[@]}" run_registration.py \
      --input-dir "$out_dir" \
      --geotransformer-dir "$GEOTRANSFORMER_DIR" \
      --geotransformer-weights "$GEOTRANSFORMER_WEIGHTS" </dev/null >>"$item_log" 2>&1; then
    log "[$id] stage4 registration FAILED"; failed=$((failed + 1)); failed_ids+=("$id:stage4"); continue
  fi

  touch "$marker"; ok=$((ok + 1)); ok_ids+=("$id")
  log "[$id] COMPLETE -> $out_dir/registered_scene.glb"
done 3< <(entries)

log "finished: processed=$processed ok=$ok skipped=$skipped failed=$failed"
log "ok ids     : ${ok_ids[*]:-<none>}"
log "failed ids : ${failed_ids[*]:-<none>}"       # each entry is <id>:<stage>
log "skipped ids: ${skipped_ids[*]:-<none>}"

# Persist a timestamped, machine-scoped summary so parallel runs don't clobber
# each other's lists (the shared success markers still live in $STATE_DIR).
summary="$STATE_DIR/summary_$(hostname -s 2>/dev/null || echo host)_$(date '+%Y%m%d_%H%M%S').txt"
{
  echo "range=$START_INDEX..${END_INDEX:-end} processed=$processed ok=$ok skipped=$skipped failed=$failed"
  [ "${#ok_ids[@]}" -gt 0 ]      && printf 'ok\t%s\n' "${ok_ids[@]}"
  [ "${#failed_ids[@]}" -gt 0 ]  && printf 'failed\t%s\n' "${failed_ids[@]}"
  [ "${#skipped_ids[@]}" -gt 0 ] && printf 'skipped\t%s\n' "${skipped_ids[@]}"
} >"$summary"
log "summary written to $summary"
