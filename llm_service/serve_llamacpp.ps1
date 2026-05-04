param(
    [string]$ConfigPath = "llm_service\.llamacpp.env"
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

$defaultConfig = Join-Path $RepoRoot "llm_service\config.llamacpp.example.env"
Import-DotEnv -Path $defaultConfig
Import-DotEnv -Path (Join-Path $RepoRoot $ConfigPath)

$hostName = $env:LITAGENT_LLAMA_HOST
$port = $env:LITAGENT_LLAMA_PORT
$apiKey = $env:LITAGENT_LLAMA_API_KEY
$alias = $env:LITAGENT_LLAMA_MODEL_ALIAS
$server = $env:LITAGENT_LLAMA_SERVER
$model = $env:LITAGENT_LLAMA_MODEL
$ctxSize = $env:LITAGENT_LLAMA_CTX_SIZE
$gpuLayers = $env:LITAGENT_LLAMA_GPU_LAYERS
$threads = $env:LITAGENT_LLAMA_THREADS

if (-not [System.IO.Path]::IsPathRooted($server)) {
    $server = Join-Path $RepoRoot $server
}
if (-not [System.IO.Path]::IsPathRooted($model)) {
    $model = Join-Path $RepoRoot $model
}

if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server executable not found at $server. Download a llama.cpp CUDA release into tools/llama.cpp first."
}
if (-not (Test-Path -LiteralPath $model)) {
    throw "GGUF model not found at $model. Place Qwen3.5-9B-Q4_K_M.gguf in models/ first."
}

$args = @(
    "--host", $hostName,
    "--port", $port,
    "--api-key", $apiKey,
    "--model", $model,
    "--alias", $alias,
    "--ctx-size", $ctxSize,
    "--n-gpu-layers", $gpuLayers,
    "--threads", $threads,
    "--jinja"
)

if ($env:LITAGENT_LLAMA_EXTRA_ARGS) {
    $args += $env:LITAGENT_LLAMA_EXTRA_ARGS.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}

Write-Host "Starting LitAgent llama.cpp microservice"
Write-Host "  server:   $server"
Write-Host "  model:    $model"
Write-Host "  alias:    $alias"
Write-Host "  endpoint: http://${hostName}:${port}/v1"

& $server @args
