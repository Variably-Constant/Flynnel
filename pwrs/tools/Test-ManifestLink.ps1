# Fetches every URL the gallery manifest advertises, and fails on one
# that does not resolve.
#
# A listing's icon, license and repository links are fetched by whoever
# renders the listing, not by anything in a build. So a manifest can
# name a file that exists only in the working tree, pass every test,
# pack cleanly, and publish a broken image. That is what happened here:
# icon-uri named assets/icon.png on main while the remote carried only
# the three older logos, and nothing inside the build could see it.
#
# Run in the publish dry run. It needs the network, which is why it is
# not in the Pester suites: a unit suite that fails when the network is
# down is a suite people learn to ignore.
#
# A URL that could not be reached at all is reported as UNREACHABLE and
# is not the same answer as one that resolved to a 404. Collapsing them
# would make a flaky connection look like a broken manifest, and a
# broken manifest look like a flaky connection.

[CmdletBinding()]
param(
    # The manifest to read. The crate's own by default.
    [string]$Manifest = (Join-Path (Split-Path $PSScriptRoot -Parent) 'Cargo.toml'),
    # Seconds to wait on each request.
    [int]$TimeoutSec = 30,
    # Treat an unreachable URL as a failure too. Off by default, so a
    # dry run on a machine with no network reports what it could not
    # check rather than failing for the network's sake.
    [switch]$FailOnUnreachable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Manifest)) {
    Write-Error "no manifest at $Manifest"
    exit 2
}

# Hand-parsed rather than through a TOML module: the shape read here is
# three `key = "value"` lines, and a module dependency to read them
# would have to be present wherever the gate runs.
$wanted = @('repository', 'license-uri', 'icon-uri')
$found = @{}
foreach ($line in Get-Content -LiteralPath $Manifest) {
    $trimmed = $line.Trim()
    if ($trimmed.StartsWith('#')) { continue }
    foreach ($key in $wanted) {
        if ($trimmed -match "^$([regex]::Escape($key))\s*=\s*`"(?<url>[^`"]+)`"") {
            $found[$key] = $Matches['url']
        }
    }
}

$missing = @($wanted | Where-Object { -not $found.ContainsKey($_) })
if ($missing.Count -gt 0) {
    # Absent is not the same as broken, and it is still a failure: a
    # listing with no license link is a listing nobody can check.
    Write-Error ("the manifest names no " + ($missing -join ', '))
    exit 2
}

$rows = @()
foreach ($key in $wanted) {
    $url = $found[$key]
    $status = $null
    $state = 'UNREACHABLE'
    $detail = ''
    try {
        $response = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing `
            -TimeoutSec $TimeoutSec
        $status = [int]$response.StatusCode
        $state = if ($status -eq 200) { 'OK' } else { 'BAD' }
    } catch [System.Net.WebException], [Microsoft.PowerShell.Commands.HttpResponseException] {
        # A response that is not 2xx arrives here. The code is the
        # answer; its absence means nothing answered at all.
        $response = $_.Exception.Response
        if ($response -and $response.StatusCode) {
            $status = [int]$response.StatusCode
            $state = 'BAD'
            $detail = $_.Exception.Message
        } else {
            $state = 'UNREACHABLE'
            $detail = $_.Exception.Message
        }
    } catch {
        $state = 'UNREACHABLE'
        $detail = $_.Exception.Message
    }
    $rows += [PSCustomObject]@{
        Key    = $key
        Url    = $url
        Status = $status
        State  = $state
        Detail = $detail
    }
}

$rows | Format-Table Key, State, Status, Url -AutoSize

$bad = @($rows | Where-Object State -eq 'BAD')
$unreachable = @($rows | Where-Object State -eq 'UNREACHABLE')

foreach ($row in $bad) {
    Write-Host ("BROKEN {0} -> {1} answered {2}" -f $row.Key, $row.Url, $row.Status)
}
foreach ($row in $unreachable) {
    Write-Host ("NOT CHECKED {0} -> {1}: {2}" -f $row.Key, $row.Url, $row.Detail)
}

if ($bad.Count -gt 0) {
    Write-Host ("{0} of {1} advertised link(s) do not resolve" -f $bad.Count, $rows.Count)
    exit 1
}
if ($unreachable.Count -gt 0) {
    Write-Host ("{0} of {1} link(s) could not be reached, so they are UNCHECKED rather than good" -f
        $unreachable.Count, $rows.Count)
    if ($FailOnUnreachable) { exit 1 }
    exit 3
}
Write-Host ("all {0} advertised link(s) resolve" -f $rows.Count)
