#!/usr/bin/env bash
set -e

DATA_ROOT="data/download"     # 原始帧
OUT_CORE="data/raw/core"      # 输出 mp4
OUT_SEM="data/raw/semantic"
mkdir -p "$OUT_CORE" "$OUT_SEM"

CORE_IDS=("02" "04" "09" "10" "11")
SEM_IDS=("01" "05" "06" "12" "13")

find_seq_dir() {
  local split="$1"  # core | semantic
  local seq_id="$2"
  local dir=""
  for det in FRCNN DPM SDP; do
    cand=$(ls -d "${DATA_ROOT}/${split}/MOT17-${seq_id}-${det}" 2>/dev/null || true)
    if [[ -n "$cand" ]]; then dir="$cand"; break; fi
  done
  echo "$dir"
}

read_fps_or_default() {
  local dir="$1"
  if [[ -f "${dir}/seqinfo.ini" ]]; then
    awk -F= '/frameRate/{gsub(/\r/,"",$2); print $2}' "${dir}/seqinfo.ini"
  else
    echo 30
  fi
}

make_one() {
  local split="$1" out_dir="$2" seq_id="$3"
  local dir; dir=$(find_seq_dir "$split" "$seq_id")
  if [[ -z "$dir" ]]; then echo "!! 未找到 ${split}/MOT17-${seq_id}-*"; return; fi
  local imgdir="${dir}/img1"
  if [[ ! -d "$imgdir" ]]; then echo "!! ${imgdir} 不存在"; return; fi

  local fps; fps=$(read_fps_or_default "$dir")
  local out_mp4="${out_dir}/MOT17-${seq_id}.mp4"
  echo "==> 合成 ${split}: MOT17-${seq_id} -> ${out_mp4} (fps=${fps})"

  # 先尝试小写，再尝试大写
  if compgen -G "${imgdir}/000001.jpg" >/dev/null; then
    ffmpeg -y -r "$fps" -start_number 1 -i "${imgdir}/%06d.jpg" \
      -c:v libx264 -preset faster -crf 22 -pix_fmt yuv420p "$out_mp4"
  elif compgen -G "${imgdir}/000001.JPG" >/dev/null; then
    ffmpeg -y -r "$fps" -start_number 1 -i "${imgdir}/%06d.JPG" \
      -c:v libx264 -preset faster -crf 22 -pix_fmt yuv420p "$out_mp4"
  else
    echo "!! 未找到形如 ${imgdir}/000001.jpg|JPG 的文件"
    return
  fi
}

for id in "${CORE_IDS[@]}"; do make_one "core" "$OUT_CORE" "$id"; done
for id in "${SEM_IDS[@]}"; do make_one "semantic" "$OUT_SEM" "$id"; done

echo "✅ 全部完成：请查看 $OUT_CORE 与 $OUT_SEM"
