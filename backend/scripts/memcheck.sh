#!/usr/bin/env bash
# Memory regression check: run an image the way Render's free tier does (512 MB, 0.1 CPU) and post large photos.
#
#   docker build --target runtime --build-arg RUNTIME=onnx -t mifvsov:onnx .
#   backend/scripts/memcheck.sh mifvsov:onnx
#
# Prints start -> healthy time, idle and peak cgroup memory, and every response code (all must be 200).
set -euo pipefail

IMAGE=${1:?usage: memcheck.sh IMAGE}
PORT=${PORT:-8765}
NAME=memcheck-$$
WORK=$(mktemp -d)
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
cd "$(dirname "$0")/.."

# Phone-sized inputs, under the 5 MB upload limit: a 4032x3024 JPEG and a 2400x1800 PNG.
uv run --quiet python - "$WORK" <<'EOF'
import sys
from PIL import Image
src = Image.open("tests/fixtures/images/batch.jpg").convert("RGB")
src.resize((4032, 3024), Image.Resampling.BICUBIC).save(f"{sys.argv[1]}/large.jpg", quality=90)
src.resize((2400, 1800), Image.Resampling.BICUBIC).save(f"{sys.argv[1]}/large.png", optimize=True)
EOF
ls -l "$WORK"

mem() { docker exec "$NAME" cat "/sys/fs/cgroup/memory.$1" | awk '{printf "%.0f MB", $1 / 1048576}'; }
post() {
    curl -s -o /dev/null -w "%{http_code} %{time_total}s  $(basename "$1")\n" \
        -F "file=@$1;type=$2" "localhost:$PORT/api/detect"
}

start=$(date +%s.%N)
docker run -d --name "$NAME" -p "$PORT:8000" --memory=512m --memory-swap=512m --cpus=0.1 \
    -e MIFVSOV_NUM_THREADS=1 "$IMAGE" >/dev/null
until curl -fs "localhost:$PORT/api/health" >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != true ]; then docker logs "$NAME"; exit 1; fi
    sleep 0.5
done
echo "start -> healthy: $(echo "$(date +%s.%N) - $start" | bc | xargs printf '%.0f') s"
echo "idle memory:      $(mem current)"

echo "--- one of each"
post "$WORK/large.jpg" image/jpeg
post "$WORK/large.png" image/png
echo "--- 10 sequential"
for _ in $(seq 5); do post "$WORK/large.jpg" image/jpeg; post "$WORK/large.png" image/png; done
echo "--- 2 concurrent"
post "$WORK/large.jpg" image/jpeg & post "$WORK/large.png" image/png & wait

echo "peak memory:      $(mem peak)"
echo "OOM kills:        $(docker exec "$NAME" grep oom_kill /sys/fs/cgroup/memory.events | awk '{print $2}')"
