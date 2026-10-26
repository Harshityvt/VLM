# Copies NIM model weights from the old project's Docker volumes into this project's volumes, so the
# first `docker compose up` doesn't re-download them from NGC. The old volumes are mounted read-only and
# the old containers are not touched. Volumes that already hold data are skipped. Safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\seed_volumes.ps1

$old = "ocr_pipeline_nemo_10_16_2026"
$new = "open_source_vlm"
$names = @("nim-cache", "nim-weights", "nim-cache-parse", "nim-cache-embed", "nim-cache-rerank")

foreach ($n in $names) {
    $src = "${old}_$n"
    $dst = "${new}_$n"
    docker volume inspect $src *> $null
    if (-not $?) { Write-Host "skip $n (no $src)"; continue }
    docker volume inspect $dst *> $null
    if (-not $?) {
        # Compose labels, so `docker compose up` adopts the volume instead of warning about it.
        docker volume create --label "com.docker.compose.project=$new" --label "com.docker.compose.volume=$n" $dst | Out-Null
    }
    $files = docker run --rm -v "${dst}:/to:ro" alpine:3.20 sh -c "ls -A /to | head -1"
    if ($files) { Write-Host "skip $n ($dst already has data)"; continue }
    Write-Host "copy $src -> $dst"
    docker run --rm -v "${src}:/from:ro" -v "${dst}:/to" alpine:3.20 sh -c "cp -a /from/. /to/ && du -sh /to"
}
