#!/usr/bin/env bash
# Latency check: run an image with a CPU limit and time /api/detect on the batch.jpg fixture.
#
#   docker build --target runtime --build-arg RUNTIME=onnx -t mifvsov:onnx .
#   backend/scripts/latency.sh mifvsov:onnx        # 0.1 CPU, like Render's free tier
#   backend/scripts/latency.sh mifvsov:onnx 1      # one full core
#
# Sends 3 warm-up requests, then RUNS (default 20) sequential ones. Prints the median and p90 of the
# server's inference_ms (preprocess + forward + NMS) and of the whole request as the client sees it.
set -euo pipefail

IMAGE=${1:?usage: latency.sh IMAGE [CPUS]}
CPUS=${2:-0.1}
RUNS=${RUNS:-20}
PORT=${PORT:-8766}
NAME=latency-$$
WORK=$(mktemp -d)
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
cd "$(dirname "$0")/.."
FILE=tests/fixtures/images/batch.jpg

post() {
    curl -s -o "$WORK/out.json" -w "%{time_total}" -F "file=@$FILE;type=image/jpeg" "localhost:$PORT/api/detect"
}

docker run -d --name "$NAME" -p "$PORT:8000" --memory=512m --memory-swap=512m --cpus="$CPUS" \
    -e MIFVSOV_NUM_THREADS=1 "$IMAGE" >/dev/null
until curl -fs "localhost:$PORT/api/health" >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != true ]; then docker logs "$NAME"; exit 1; fi
    sleep 0.5
done

for _ in 1 2 3; do post >/dev/null; done
for _ in $(seq "$RUNS"); do
    total=$(post)
    printf '%s %s\n' "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inference_ms"])' "$WORK/out.json")" \
        "$(echo "$total * 1000" | bc)" >>"$WORK/times"
done

python3 - "$WORK/times" "$IMAGE" "$CPUS" <<'EOF'
import statistics, sys
rows = [tuple(map(float, line.split())) for line in open(sys.argv[1])]
def p90(xs):
    return sorted(xs)[round(0.9 * (len(xs) - 1))]
inf, req = zip(*rows)
print(f"{sys.argv[2]} at {sys.argv[3]} CPU, {len(rows)} runs")
print(f"inference_ms: median {statistics.median(inf):.0f}  p90 {p90(inf):.0f}")
print(f"request ms:   median {statistics.median(req):.0f}  p90 {p90(req):.0f}")
EOF
