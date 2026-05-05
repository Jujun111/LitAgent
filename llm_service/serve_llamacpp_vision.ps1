param(
    [string]$ConfigPath = "llm_service\.llamacpp-vision.env"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $name, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
    }
}

$defaultConfig = Join-Path $RepoRoot "llm_service\config.llamacpp-vision.example.env"
Import-DotEnv -Path $defaultConfig
Import-DotEnv -Path (Join-Path $RepoRoot $ConfigPath)

$hostName = $env:LITAGENT_VISION_HOST
$port = $env:LITAGENT_VISION_PORT
$apiKey = $env:LITAGENT_VISION_API_KEY
$alias = $env:LITAGENT_VISION_MODEL_ALIAS
$server = $env:LITAGENT_VISION_SERVER
$hfRepo = $env:LITAGENT_VISION_HF_REPO
$model = $env:LITAGENT_VISION_MODEL
$mmproj = $env:LITAGENT_VISION_MMPROJ
$ctxSize = $env:LITAGENT_VISION_CTX_SIZE
$gpuLayers = $env:LITAGENT_VISION_GPU_LAYERS
$threads = $env:LITAGENT_VISION_THREADS
$imageMaxTokens = $env:LITAGENT_VISION_IMAGE_MAX_TOKENS

if (-not [System.IO.Path]::IsPathRooted($server)) {
    $server = Join-Path $RepoRoot $server
}

if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server executable not found at $server. Download a llama.cpp CUDA release into tools/llama.cpp first."
}

$args = @(
    "--host", $hostName,
    "--port", $port,
    "--api-key", $apiKey,
    "--alias", $alias,
    "--ctx-size", $ctxSize,
    "--n-gpu-layers", $gpuLayers,
    "--threads", $threads,
    "--image-max-tokens", $imageMaxTokens,
    "--jinja"
)

if ($hfRepo) {
    $args += @("--hf-repo", $hfRepo)
} else {
    if (-not $model) {
        throw "Set LITAGENT_VISION_HF_REPO or LITAGENT_VISION_MODEL."
    }
    if (-not [System.IO.Path]::IsPathRooted($model)) {
        $model = Join-Path $RepoRoot $model
    }
    if (-not (Test-Path -LiteralPath $model)) {
        throw "Vision GGUF model not found at $model. Download Qwen3.5-9B-Q4_K_M.gguf into models/ or set LITAGENT_VISION_HF_REPO."
    }
    $args += @("--model", $model)
    if ($mmproj) {
        if (-not [System.IO.Path]::IsPathRooted($mmproj)) {
            $mmproj = Join-Path $RepoRoot $mmproj
        }
        if (-not (Test-Path -LiteralPath $mmproj)) {
            throw "Vision mmproj file not found at $mmproj. Download mmproj-F16.gguf from a Qwen3.5-9B VLM GGUF repo, or set LITAGENT_VISION_HF_REPO to a repo that bundles mmproj."
        }
        $args += @("--mmproj", $mmproj)
    }
}

if ($env:LITAGENT_VISION_EXTRA_ARGS) {
    $args += $env:LITAGENT_VISION_EXTRA_ARGS.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}

Write-Host "Starting LitAgent unified llama.cpp Qwen3.5 VLM microservice"
Write-Host "  server:   $server"
Write-Host "  hf_repo:  $hfRepo"
Write-Host "  alias:    $alias"
Write-Host "  endpoint: http://${hostName}:${port}/v1"
Write-Host "Tip: this server can serve both text synthesis and pixel vision; do not run a second text server on 8GB VRAM."

& $server @args
