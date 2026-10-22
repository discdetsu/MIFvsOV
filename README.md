# MIFvsOV

Minute intestinal fluke (MIF) egg and _Opisthorchis viverrini_ (OV) egg detection and classification in microscopic images.

Upload a micrograph and the YOLOv5 model (`best.pt`, trained in 2022) finds MIF and OV eggs. The browser draws the boxes and lets you filter them by confidence and class, then download an annotated PNG.

**Live demo:** https://mif-vs-ov.onrender.com/ (hosted on Render's free tier, so the first request after it has been idle may take a minute while the app wakes up)

> [!WARNING]
> This is a student research project, not a medical device. Its "MIF" class was trained mostly on the wrong parasite (see [Limitations](#limitations)). Do not use it for diagnosis.

## Background

Liver fluke infection by _Opisthorchis viverrini_ is a major public-health problem in Thailand, especially in the North and Northeast, where it spreads through raw or undercooked freshwater fish. Chronic infection causes cholangiocarcinoma (bile duct cancer), which has a poor prognosis, and the WHO classifies _O. viverrini_ as a Group 1 carcinogen.

Screening is done with the modified Kato-Katz technique: a stool smear is examined under a microscope for eggs. The problem is that OV eggs look almost the same as minute intestinal fluke eggs (heterophyids such as _Haplorchis_ spp.) in both size and shape, so even experts struggle to tell them apart. MIF has no known link to cholangiocarcinoma, so the difference matters clinically. Today the reliable way to separate them is copro-DNA PCR, which takes 3–4 hours, needs a laboratory and costly reagents, and is often unavailable in rural areas.

This project, a Kasetsart University computer science senior project (academic year 2021–22), tests whether a deep learning detector can separate the two eggs from a plain micrograph, as a screening aid where trained microscopists are scarce.

## Model and data

- **Dataset:** 2,053 images ([Eggs.zip](https://drive.google.com/file/d/1eo3-4ODNEwVi85MNfGSvZ3XWetZEgHem/view?usp=sharing)):
  - 53 micrographs from Phramongkutklao Hospital's laboratory (34 OV, 19 MIF), labelled by hand with LabelImg
  - 1,000 OV and 1,000 "MIF" images from the ICIP 2022 Parasitic Egg Detection and Classification challenge[^1], with their COCO boxes converted to YOLO format
- **Split:** 80 / 10 / 10 train / validation / test, at random with scikit-learn.
- **Training:** YOLOv5s from the COCO-pretrained `yolov5s.pt`, 640 px, 10 epochs, on Google Colab Pro. Training notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AQdwrTNCkxA3wdRwyt-6WTWPSt4MeMim?usp=sharing)

Example images batch:<br><br>
<img src="batch.jpg" alt="Example images batch" width="50%"/>

## Results

These are the 2022 numbers, all measured on the **validation** split (see [Limitations](#limitations)).

**Detection** (YOLOv5, boxes):

| mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| ------- | ------------ | --------- | ------ |
| 0.978   | 0.844        | 0.950     | 0.986  |

**Classification** (MIF vs OV, from the detector's confusion matrix), compared with the best two-stage pipeline (crop each egg, then classify the crop with a CNN):

| Model                                    | Evaluated on | Accuracy | Precision | Recall | F1    |
| ---------------------------------------- | ------------ | -------- | --------- | ------ | ----- |
| **YOLOv5s, single-stage (shipped)**      | validation   | 0.990    | 0.982     | 1.000  | 0.991 |
| InceptionV3 + augmentation, two-stage    | validation   | 0.992    | 0.986     | 0.997  | 0.992 |
| InceptionV3 + augmentation, two-stage    | test         | 0.971    | 0.973     | 0.973  | 0.973 |

The two-stage experiments also tried a CNN trained from scratch and InceptionV3 transfer learning, each with and without CLAHE / brightness-contrast / cutout augmentation. All four scored 0.945–0.992 accuracy on validation; augmentation and transfer learning helped. YOLOv5 was shipped because it does detection and classification in one pass, and it beat the two-stage pipeline's test score.

Failure cases seen when testing on new images:

- Faint, translucent eggs are missed, and stool debris that looks like an egg is detected instead.
- One egg sometimes gets two overlapping boxes.
- Some eggs are not detected at all.

## Limitations

- **The "MIF" class is mostly the wrong parasite.** The ICIP 2022 dataset has no minute intestinal fluke class. Its _Fasciolopsis buski_ images were used as MIF by mistake. _F. buski_ is the **giant** intestinal fluke, and its eggs are about 130–150 µm long, roughly five times the length of OV and MIF eggs (about 25–30 µm). So roughly 1,000 of the 1,019 "MIF" images are not MIF, and the model mostly learned to separate _F. buski_ from OV, which is much easier than the real MIF vs OV problem. Only the 19 hospital images show real MIF eggs.
- **The scores don't measure the real task.** Because of the mislabelling, the high scores above say little about how well the model separates real MIF from OV. A real MIF egg may well be classified as OV.
- **Validation set only.** The YOLOv5 numbers come from the validation split, which was also used to choose `best.pt` during training, so they are optimistic. The detector was never scored on the held-out test split. The two-stage model's drop from 0.992 (validation) to 0.971 (test) gives a sense of the gap.
- **Too little real data.** There are only 53 local micrographs, and only a handful of them can be in the validation split. Most images come from one public dataset with its own microscopes, staining and imaging conditions, so performance on other labs' slides is unknown. The labels are morphological, not confirmed by PCR.
- **Short training:** 10 epochs with no hyperparameter search.
- **Two classes only.** Every detection is forced into MIF or OV. Other parasite eggs, artefacts and debris have no "other" class.

To fix this, the model should be retrained on real heterophyid / lecithodendriid egg images (ideally PCR-confirmed), evaluated once on a test split it has never seen, and given an "other" class.

## Architecture

```mermaid
flowchart LR
    subgraph Browser["Browser · SvelteKit single-page app"]
        direction TB
        U([Drop JPEG/PNG]) --> V{Type & size OK?}
        V -- no --> E[Show error]
        R[SVG overlay<br/>colour-coded boxes] --> F[Slider & class toggles<br/>filter client-side]
        F --> D[Download<br/>annotated PNG]
    end

    subgraph Server["FastAPI · Docker container"]
        direction TB
        A["POST /api/detect"] --> C{Valid image?}
        C -- no --> X[413 Too large /<br/>415 Unsupported type]
        C -- yes --> L[EXIF orient<br/>+ letterbox]
        L --> M[YOLOv5 forward<br/>PyTorch or ONNX Runtime]
        M --> N[Non-maximum suppression<br/>confidence ≥ 0.10]
        N --> B[Rescale boxes to<br/>original pixels]
    end

    V -- yes: multipart upload --> A
    B -- JSON detections --> R
```

- The model is loaded once at startup, on PyTorch (`best.pt`) or ONNX Runtime (`best.onnx`); see [Inference runtime](#inference-runtime).
- Pre- and post-processing are plain NumPy, shared by both runtimes.
- The server returns every detection with confidence ≥ 0.10. All further filtering happens in the browser, without another request.

| Path                            | What                                                                       |
| ------------------------------- | -------------------------------------------------------------------------- |
| `backend/app/main.py`           | FastAPI app factory; loads the model in `lifespan`, mounts the SPA         |
| `backend/app/api/routes.py`     | `/api/health`, `/api/model`, `/api/detect`                                 |
| `backend/app/inference/`        | `detector.py` (NumPy pre/post-processing), `backends/` (torch, onnx), `models/` (trimmed YOLOv5 classes) |
| `backend/weights/best.pt`       | the 2022 weights, byte-identical (sha256 `3df91454…5cc9f`)                 |
| `backend/weights/best.onnx`     | exported from `best.pt` by `backend/scripts/export_onnx.py`                |
| `backend/tests/`                | API tests + parity tests against the original 2022 pipeline               |
| `frontend/`                     | SvelteKit app (`adapter-static`)                                           |
| `Dockerfile`, `docker-compose.yml` | multi-stage build: node → python slim                                   |

### API

`POST /api/detect` takes a multipart `file` (JPEG or PNG, ≤ 5 MB) and an optional `?conf=` (default 0.10). It returns:

```json
{
  "image": { "width": 3840, "height": 2160 },
  "detections": [
    { "class_id": 0, "label": "MIF", "confidence": 0.63, "x1": 1654.6, "y1": 739.6, "x2": 1911.1, "y2": 913.2 }
  ],
  "inference_ms": 140.2,
  "model": { "names": ["MIF", "OV"] }
}
```

- Box coordinates are pixels of the original image, after EXIF rotation.
- The client's slider (default 0.25, the old app's threshold) filters the results locally.
- Errors: `413` file too large, `415` not a decodable JPEG/PNG, `422` bad `conf`.
- Interactive docs are at `/docs`.

## Run it

### Docker (production-like)

```sh
docker compose up --build
# open http://localhost:8000
```

The image listens on `$PORT` (default 8000), runs as a non-root user and has a `HEALTHCHECK` on `/api/health`, so it runs as-is on Fly.io, Render, Railway, Hugging Face Spaces (Docker) or any VPS.

### Local development

Requirements: [uv](https://docs.astral.sh/uv/), Node 24 and pnpm (`corepack enable`).

```sh
# backend: http://localhost:8000
cd backend
uv sync
uv run uvicorn app.main:app --reload

# frontend: http://localhost:5173 (proxies /api to :8000)
cd frontend
pnpm install
pnpm dev
```

Or run both in containers with hot reload: `docker compose --profile dev up`.

### Configuration

Settings are read from environment variables with the `MIFVSOV_` prefix (see `backend/app/config.py`):

| Variable                 | Default                 | Meaning                                       |
| ------------------------ | ----------------------- | --------------------------------------------- |
| `MIFVSOV_RUNTIME`        | `torch`                 | `torch` or `onnx`, see below                  |
| `MIFVSOV_DETECT_CONF`    | `0.10`                  | server-side confidence floor                  |
| `MIFVSOV_CONF`           | `0.25`                  | default slider value shown to clients         |
| `MIFVSOV_IOU`            | `0.45`                  | non-maximum suppression IoU threshold         |
| `MIFVSOV_MAX_UPLOAD_MB`  | `5`                     | upload limit                                  |
| `MIFVSOV_NUM_THREADS`    | library default         | torch / onnxruntime intra-op threads          |
| `MIFVSOV_CORS_ORIGINS`   | `[]`                    | JSON list, only needed for cross-origin dev   |

`.env.example` lists every setting. Copy it to `.env` for `docker compose`.

### Inference runtime

The same model runs on either runtime. Both pass the strict 2022 parity tests.

|                                   | `torch` (default)  | `onnx`               |
| --------------------------------- | ------------------ | -------------------- |
| Weights                           | `best.pt`          | `best.onnx`          |
| Docker image (disk / compressed)  | 1.68 GB / 388 MB   | 605 MB / 170 MB      |
| Start → healthy at 0.1 CPU        | 47 s               | 31 s                 |
| Idle memory                       | 254 MB             | 162 MB               |
| Peak memory (see below)           | 366 MB             | 321 MB               |
| Inference, 1 CPU (median / p90)   | 199 / 205 ms       | 158 / 162 ms         |
| Inference, 0.1 CPU (median / p90) | 5.5 / 7.5 s        | 4.3 / 6.4 s          |

#### Choosing a runtime

```sh
# Docker installs only one runtime, chosen at build time (the image sets MIFVSOV_RUNTIME to match)
docker build --target runtime --build-arg RUNTIME=onnx .
RUNTIME=onnx docker compose up --build    # or put RUNTIME=onnx in .env

# Local: `uv sync` installs both, so just set the variable
MIFVSOV_RUNTIME=onnx uv run uvicorn app.main:app
```

If the selected runtime isn't installed, startup fails with a message naming the build arg.

#### Re-exporting after changing `best.pt`

```sh
cd backend
uv run python -m scripts.export_onnx          # export, then compare torch and onnxruntime on the fixtures
uv run python -m scripts.export_onnx --check  # what CI runs: is best.onnx exported from today's best.pt?
```

Commit the new `best.onnx`. Notes on the export:

- It stores the class names, stride and `best.pt`'s sha256 as metadata, so the ONNX image never unpickles `best.pt`.
- Raw outputs match torch within 0.01 px on boxes and 1e-5 on scores.
- It is 28 MB: `best.pt` stores fp16 weights, the export is fp32.

#### Memory check

`backend/scripts/memcheck.sh IMAGE` runs an image with 512 MB and 0.1 CPU. It posts a 4032×3024 JPEG and a 2400×1800 PNG (once each, then 10 in a row, then 2 at once) and reports cgroup memory. The table above shows its results.

#### Latency check

`backend/scripts/latency.sh IMAGE [CPUS]` runs an image with 512 MB, one thread and the given CPU limit (default 0.1). It posts `batch.jpg` (1019×1075) 3 times to warm up, then 20 more times, and reports the median and p90 of `inference_ms` (preprocessing, forward pass and NMS, not upload or decoding). The table above shows its results, measured on an Intel i5-1135G7. At 0.1 CPU the container is throttled in bursts, so those numbers vary more from run to run.

## Tests

```sh
cd backend  && uv run pytest              # API + parity tests, on both runtimes
cd frontend && pnpm check && pnpm lint && pnpm test
cd frontend && pnpm test:e2e              # Playwright, against the real backend
```

Every detector and API test runs on both `torch` and `onnx`.

**Parity with 2022.** `backend/tests/fixtures/parity.json` holds detections from the original pipeline (torch 1.11, `torch.hub.load('yolov5-master', 'custom', ...)`, size 640). The test requires:

- the same classes and count
- box IoU ≥ 0.99
- confidence within ±0.01

JPEG decoding changed between Pillow 8.4 and today's libjpeg-turbo, which alone moves confidences by a few hundredths. So `batch_decoded-pillow8.png` stores the exact pixels the old stack saw for `batch.jpg`, and a separate, looser test covers the modern decoder.

**Other checks:**

- `test_backends.py`: `best.onnx` is exported from the current `best.pt` and matches torch.
- `test_preprocess.py`: the NumPy code against the torch/torchvision code it replaced. Non-maximum suppression is identical; the letterbox differs by at most one level on ≤ 0.0012% of values.

After changing the API schema, run `pnpm gen:api` in `frontend/` to regenerate the TypeScript types from FastAPI's OpenAPI spec.

## Credits

Senior project, Department of Computer Science, Faculty of Science, Kasetsart University. Advisor: Asst. Prof. Dr. Pakaket Wattuya. Co-advisors: Asst. Prof. Dr. Chakrit Watcharopas and Dr. Thammakorn Saethang. Thanks to Phramongkutklao Hospital for the micrographs.

`backend/app/inference/models/` holds a trimmed copy of [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics, from the version used to train `best.pt`, which was licensed under **GPL-3.0** (see `backend/app/inference/models/LICENSE`).

[^1]: Duangdao Palasuwan, Korranat Naruenatthanaset, Thananop Kobchaisawat, Thanarat H Chalidabhongse, Nuntiporn Nunthanasup, Kanyarat Boonpeng, Nantheera Anantrasirichai, January 22, 2022, "Parasitic Egg Detection and Classification in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/vyh8-4h71.
