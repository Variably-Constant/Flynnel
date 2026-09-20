# Read the tail of a file another process is writing.
#
# Get-Content takes a share mode that a live writer refuses, so reading
# a running harness's log throws an IOException. With -ErrorAction
# SilentlyContinue that exception is swallowed and the caller sees an
# empty result, which is indistinguishable from a log with no rows in
# it. That has produced a false "arms=0" and a false "busy_cores=0" in
# this campaign, each of which read as a fact about the run.
#
# Opening with FileShare.ReadWrite reads what is there while the writer
# keeps appending.

param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Last = 20,
    [string]$Match,
    [int]$Retries = 40
)

if (-not (Test-Path $Path)) {
    "NO_SUCH_FILE $Path"
    exit 2
}

# Add-Content opens the file exclusively for the length of one call, so
# the lock is intermittent rather than held for the run. Retrying across
# the gaps reads it; a single attempt reports a busy file as though the
# run had produced nothing.
$text = $null
foreach ($attempt in 1..$Retries) {
    try {
        $fs = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try {
            $reader = New-Object System.IO.StreamReader($fs)
            $text = $reader.ReadToEnd()
        } finally {
            $fs.Close()
        }
        break
    } catch {
        Start-Sleep -Milliseconds 300
    }
}

if ($null -eq $text) {
    "LOCKED_THROUGHOUT $Path was held by its writer for all $Retries attempts; this is a busy file, NOT an empty one"
    exit 3
}

$lines = $text -split "`r?`n"
if ($Match) {
    $lines = @($lines | Where-Object { $_ -match $Match })
}
$lines | Select-Object -Last $Last
