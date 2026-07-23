$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $RepositoryRoot

function Invoke-VerificationStep {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [scriptblock]$Command
    )

    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-VerificationStep "Python compile" { python -m compileall -q src tests }
    Invoke-VerificationStep "Python format" { python -m ruff format --check . }
    Invoke-VerificationStep "Python lint" { python -m ruff check . }
    Invoke-VerificationStep "Python tests and schema validation" { python -m pytest -q }
    Invoke-VerificationStep "Rust format" { cargo fmt --all -- --check }
    Invoke-VerificationStep "Rust check" { cargo check --workspace --all-targets }
    Invoke-VerificationStep "Rust tests" { cargo test --workspace }
    Invoke-VerificationStep "Native GLM-4 process boundary" {
        cargo build -p ams-runtime
        $env:AMS_NATIVE_BINARY = Join-Path $RepositoryRoot "target\\debug\\ams-runtime.exe"
        try {
            python -m pytest -q `
                tests/invariant/test_mini_glm4_forward.py `
                -k "native_bridge_process or native_generation_process or native_worker_process"
            python -m pytest -q tests/integration/test_glm4_native_backend.py
        }
        finally {
            Remove-Item Env:AMS_NATIVE_BINARY -ErrorAction SilentlyContinue
        }
    }
    Invoke-VerificationStep "Rust Clippy" {
        cargo clippy --workspace --all-targets -- -D warnings
    }
}
finally {
    Pop-Location
}
