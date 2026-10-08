# One repository, task / model

READY: yes

## Objective and accepted structure

Refactor the existing public repository in place and merge into main. Root task directories: `asr/gigaam`, `asr/whisper`, `embeddings/e5-small`, `embeddings/user-bge-m3`, `ocr/easyocr`, `ocr/rapid-v5-mobile`. Each model has its own HTTP application, inference backend, settings/queue code, Dockerfile, dependencies and README. No shared inference server or runtime queue and no nested `model-inference` project. Root contains only orchestration/scripts/tests/docs and repository metadata. Generic embedder is superseded by the two explicit embedding services. Preserve the existing GigaAM/EasyOCR CPU inference behavior and idle unload. Preserve the four requested services' model pins, routes, batch limits, CPU/CUDA selection, multiple workers, ports and quality contracts. Model weights remain outside Git. Make first startup prepare missing pinned weights before readiness; cached startup validates and reuses the bundle. Do not silently replace damaged or mismatched existing bundles.

## Owned dependent waves

1. [seq: 1] Primary: inventory, final structure, plan and clean feature branch. Acceptance: real default model identities and retained CPU behavior known; user's task/model structure followed exactly.
2. [seq: 2] API worker: four independent applications/settings/queue/main entrypoints and API/scheduler tests under the four requested model directories. Acceptance: only that model's inference routes, CPU/GPU and workers preserved, preparation runs once before worker processes, no cross-model imports or shared runtime. Verify lightweight HTTP and queue tests.
3. [seq: 2] Backend worker: four independent backend.py/prepare.py/model.lock.json and model backend/preparation tests. Acceptance: original neural operations and exact bundles retained; each preparation handles only its model, valid existing bundle is reused, corruption fails, missing bundle is downloaded/exported. Verify helpers plus real pinned bundle CPU/GPU parity by primary.
4. [seq: 2] Legacy worker: move GigaAM to asr/gigaam and EasyOCR to ocr/easyocr; owns their entire trees. Acceptance: original CPU backends/lifecycle retained, Docker standalone entrypoints and docs work at new path, named API with documented compatibility if needed. Root orchestration remains primary-owned. Verify imports/tests without heavyweight downloads where practical.
5. [seq: 3] Primary: independent Dockerfiles/dependencies and model Compose files for the four new models; single root Compose and CPU/GPU/WSL modes; launch helper accepts task/model paths; root and task/model READMEs, migration and validation docs, CI. Remove old generic embedder and obsolete aggregate subtree after owned code has moved. Validate every Compose mode and model selection, tests and first/cached preparation flows.
6. [seq: 4] Primary: build/deploy on home with persistent weight migration (no redownload of already prepared models), preserve other projects, real HTTP/batches/CPU workers and model outputs. Read-only reviewer: separation of containers/code, runtime independence, startup/parity and docs. Fix material findings, push branch/PR, merge into main (user explicitly requested main), update home and clean user's local checkout by fast-forward if clean. Record actual checks in docs.

## Verification and operational constraints

Use focused unit/acceptance tests and all-model HTTP GPU smoke, real CPU/GPU parity against saved benchmark artifacts, preparation missing/cache/corruption gates, Docker build and Compose profile/device/worker configuration. USER exporter must match the measured opset-17 legacy exporter. OCR recognizer remains Cyrillic mobile; Whisper remains large-v3 with same decode settings and sequential file batching per worker. First-start download may be slow; no readiness until weights are valid. One copy per worker; GPU workers default to one on 16 GB. CPU-only GigaAM/EasyOCR remain explicitly documented CPU alternatives. No auth material reads; actual .env files and weights preserved by file movement only. Migration must not stop unrelated containers or destroy existing caches.

## Handoff, 2026-10-08

Waves 1–5 complete. Public branch `codex/unified-model-services`, draft PR #2.
51 unit/contract tests passed; clean CI passed after adding uvicorn to dev dependencies.
All six CPU/GPU/WSL Compose configurations resolved on home. Read-only review passed
following streamed body-limit fix. Each OpenAPI schema now uses fixed model-specific paths.

Wave 6 remains open: five independent images built, E5 reached image export; GigaAM
and EasyOCR real health/startup passed. Whisper CPU loaded and reached readiness;
verification helper import fixed, final audio inference pending. Existing E5/USER/Rapid
bundles copied without overwrite to separate named volumes; E5 ownership set to 10001.
Whisper copy and remaining ownership/checksum validation need confirmation. Original
bundles and old running deployment were not removed or intentionally stopped.

Home system commands then failed with SIGBUS and SSH became unavailable. Human
was asked to restore host/WSL/SSH. Finish image build and one-at-a-time CPU checks,
Rapid first/cached startup, CPU workers, controlled replacement of the four GPU services,
full saved embedding/OCR quality comparison and Whisper fixture, then mark PR ready,
merge main and fast-forward clean home/local checkouts. Preserve original local ignored
.env/models under asr-gigaam, ocr and embedder when migrating; no auth reads.
Do not claim deployment or quality verification complete until observed.

## Resumed home check, 2026-10-08

After human restored host, all six image tags existed (including E5). Fast-forwarded
home to 1fc6f51, resumed builds from current code. Root lightweight environment
restored locally; 51 tests passed again. Rapid real auto first startup and offline
restart passed with two blank OCR pages and matching weight metadata. E5 real CPU
HTTP inference with WORKERS=2 passed, both worker PIDs observed. All four new
volumes validate successfully via model prepare CLIs; interrupted Whisper copy
repaired from read-only original bundle and owner set to 10001.

During bounded Whisper CPU test and image builds Docker reported unexpected EOF,
then every home exec session returned 255. SSH banner and publickey authorization
work; even /usr/bin/true and uptime fail, SFTP closes immediately. No new model
loads after detecting this failure. Human asked about host restart and Windows
disk space (Linux virtual disk reports plenty of free space). USER/Whisper final
CPU inference, final current-source images, GPU deployment/parity and main merge
remain open. Original GPU services were healthy before this second host failure.

## Source publication gate, latest user objective

Latest explicit goal: finish the code and publish it in GitHub main. Source delivery
can complete independently of unavailable home operations. Preserve and document
all remaining runtime checks; do not claim them done.

Final read-only audit compared all neural operations and four lock entries against
origin/main: pins/operations unchanged; legacy backends byte-identical, USER adds
CUDA DLL preloading. Restored lost AudioDurationExceeded ValueError subclass and
added allocation-limit/decoder-error/HTTP-422 regressions. 54 tests passed twice
(including independent auditor); review verdict GO. Local make config succeeded
for CPU/GPU/WSL using a checksum-verified official Compose 5.5.1 in an isolated
temporary Docker configuration. Publish final source after current-head CI passes,
merge PR #2 into main and safely fast-forward clean local user checkout.

Home exec and SFTP still fail after successful SSH authorization. Final current-source
image execution, USER/Whisper CPU audio inference, and new GPU parity/deployment
are operational follow-up gaps, not claims of this source publication.

## Source delivered and home resumed, 2026-10-08

PR #2 merged into main as d9f552a; the main GitHub Actions run 37834592505
passed. Main's source tree matches the independently audited e7f77a head.
The user's clean local checkout was fast-forwarded; ignored legacy settings,
weights and manual fixtures were preserved or moved to a sibling local backup.

After the user restored home, fast-forwarded its clean checkout to the same main
commit. Existing four GPU containers were healthy, still using the previous
aggregate image. Final E5 build required downloading/installing the large
PyTorch/CUDA layer. SSH dropped again; a subsequent short command succeeded.
Started a detached, sequential six-image build with per-image logs and state in
ignored artifacts so an SSH disconnect alone would not cancel it.

WSL uptime subsequently reset to two minutes. The detached build was alive and
reached E5 dependency installation, but home execution failed again afterwards.
The VPS and its reverse port 2222 remain accessible. No final build/deployment
success was observed. No intentional stop of old or unrelated services occurred.
Requested physical Windows disk space and restart details; root cause remains
unconfirmed. Prepared a detached verification helper, but did not launch it.
Code/main objective is delivered; final independent GPU deployment and parity
remain an explicitly unverified operational follow-up.
