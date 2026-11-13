param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$JobName = 'unified_report'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    foreach ($tool in @('pdflatex', 'bibtex')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Required command not found: $tool"
        }
    }
    New-Item -ItemType Directory -Path 'build' -Force | Out-Null
    & pdflatex --disable-installer -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "-jobname=$JobName" unified_report.tex
    if ($LASTEXITCODE -ne 0) { throw 'Initial LaTeX pass failed.' }
    & bibtex "build/$JobName"
    if ($LASTEXITCODE -ne 0) { throw 'Bibliography build failed.' }
    for ($pass = 1; $pass -le 2; $pass++) {
        & pdflatex --disable-installer -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "-jobname=$JobName" unified_report.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX reference pass $pass failed." }
    }
    $log = Get-Content -LiteralPath "build/$JobName.log" -Raw
    if ($log -match 'There were undefined references|Citation .* undefined|Reference .* undefined|Rerun to get cross-references right') {
        throw "Unresolved citations or references remain; inspect build/$JobName.log."
    }
    if ($log -match 'Overfull \\[hv]box') {
        Write-Warning 'Overfull content detected; inspect the log and rendered PDF before release.'
    }
    Write-Output "Built: $(Join-Path $PSScriptRoot "build\$JobName.pdf")"
} finally {
    Pop-Location
}
