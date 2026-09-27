# Join two platforms' module folders into one, pack it, and read what
# came out - stopping short of the network.
#
# WHAT THIS DELIBERATELY DOES NOT DO. It does not publish, and it does
# not call the publish cmdlet even to ask it what it would do. Sending
# a package to the gallery is the owner's act and needs their word every
# time, and a version number there cannot be reused, so the dry run ends
# at a package on disk that has been opened and read.
#
# The toolchain's own dry-run subcommand cannot verify this module, and
# that is worth saying plainly rather than leaving someone to reach for
# it: it calls build() fresh and ships that single-platform folder, so
# it never sees a merge result. A module carrying two platforms is
# packed with Compress-PSResource over the merged folder instead.
# Verified against cargo-pwrs 5.9.0's own source; the finding came from
# the agent who shipped the sibling SubEtha module and hit it.
#
# THE MERGE IS STRICT AND ITS REFUSALS ARE THE POINT.
#   - each folder must hold exactly one .psd1
#   - the two manifests must be identical text, character for
#     character, which is what proves both were built from one
#     checkout by one cargo-pwrs
#   - a runtime id already present in the destination is an error and
#     not an overwrite
# Each is checked here before the merge runs, so a mismatch is reported
# as the thing it is rather than as a merge failure with the diff
# hidden.

param(
    # The module folder built on this host, which the other is merged
    # into. Copied first, so a failed merge does not leave the build's
    # own output half-joined.
    [Parameter(Mandatory = $true)][string]$WindowsModule,
    # The module folder built on the Linux guest and copied here.
    [Parameter(Mandatory = $true)][string]$LinuxModule,
    # Where the merged folder and the package are written. Under the
    # crate's own target by default, so the output lands beside the
    # builds it came from and a clean removes it.
    [string]$WorkDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target\pack'),
    [string]$Tag = 'pack',
    # The manifest-link check, which fetches every URL the manifest
    # advertises. Skipped only when the box has no network, and the
    # skip is reported rather than silent.
    [string]$LinkCheck = '',
    [switch]$SkipLinkCheck
)

$ErrorActionPreference = 'Continue'
$log = Join-Path $WorkDir "$Tag.log"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
function Say($text) { Add-Content -Path $log -Value $text -Encoding utf8 }

Set-Content -Path $log -Value ("PACK_START " + (Get-Date -Format 'o')) -Encoding utf8
Say ("HOST " + $PSVersionTable.PSEdition + ' ' + $PSVersionTable.PSVersion)

# Compress-PSResource ships with Microsoft.PowerShell.PSResourceGet,
# which is present under pwsh on this box and absent under Windows
# PowerShell. An ssh session here lands in Windows PowerShell, so a run
# launched without naming the shell reaches the last step and fails
# there, after the merge has already happened. Refusing at the top says
# which shell to use instead.
if ($PSVersionTable.PSEdition -ne 'Core') {
    Say 'WRONG_HOST Compress-PSResource needs pwsh; this is Windows PowerShell'
    Say 'WRONG_HOST_FIX run this under pwsh rather than powershell.exe'
    Say 'PACK_DONE refused'
    exit 1
}

function Get-SingleManifest {
    param([string]$Dir, [string]$Which)
    if (-not (Test-Path $Dir)) {
        Say "MISSING_$Which $Dir"
        return $null
    }
    $found = @(Get-ChildItem -Path $Dir -Filter *.psd1 -File)
    if ($found.Count -ne 1) {
        Say ("BAD_$Which {0} holds {1} psd1 files; a module folder has one" -f $Dir, $found.Count)
        return $null
    }
    $found[0]
}

$winManifest = Get-SingleManifest -Dir $WindowsModule -Which 'WINDOWS'
$linManifest = Get-SingleManifest -Dir $LinuxModule -Which 'LINUX'
if (-not $winManifest -or -not $linManifest) { Say 'PACK_DONE refused'; exit 2 }

Say ("WINDOWS_MANIFEST {0} {1} bytes" -f $winManifest.FullName, $winManifest.Length)
Say ("LINUX_MANIFEST   {0} {1} bytes" -f $linManifest.FullName, $linManifest.Length)

# Byte for byte, not line by line. The merge compares the text and a
# line-ending difference is a text difference, so a comparison that
# normalizes them would pass here and fail there.
$winBytes = [System.IO.File]::ReadAllBytes($winManifest.FullName)
$linBytes = [System.IO.File]::ReadAllBytes($linManifest.FullName)
$same = $winBytes.Length -eq $linBytes.Length
if ($same) {
    for ($i = 0; $i -lt $winBytes.Length; $i++) {
        if ($winBytes[$i] -ne $linBytes[$i]) {
            Say ("MANIFEST_DIFFERS at byte {0}: windows has {1}, linux has {2}" -f $i, $winBytes[$i], $linBytes[$i])
            $same = $false
            break
        }
    }
} else {
    Say ("MANIFEST_DIFFERS in length: windows {0} bytes, linux {1}" -f $winBytes.Length, $linBytes.Length)
}
if (-not $same) {
    # The usual cause is two different cargo-pwrs versions, so name
    # the thing to check rather than only the symptom.
    Say 'MANIFEST_HINT the merge requires identical text; both hosts must run the same cargo-pwrs against the same checkout'
    Say 'PACK_DONE refused'
    exit 3
}
Say 'MANIFEST_IDENTICAL both platforms agree byte for byte'

# The runtime ids each side carries, before anything is copied.
function Get-Rids { param([string]$Dir)
    $r = Join-Path $Dir 'runtimes'
    if (-not (Test-Path $r)) { return @() }
    # The leading comma is load-bearing. A function returning a
    # one-element array hands back the element instead, so a folder
    # with one runtime gave a STRING, and `$winRids + $linRids`
    # concatenated two strings into "win-x64linux-x64" rather than
    # making a two-element array. The merge was correct and the check
    # over it was not.
    , @(Get-ChildItem -Path $r -Directory | ForEach-Object Name)
}
$winRids = Get-Rids $WindowsModule
$linRids = Get-Rids $LinuxModule
Say ("WINDOWS_RIDS " + $(if ($winRids) { $winRids -join ',' } else { 'none' }))
Say ("LINUX_RIDS   " + $(if ($linRids) { $linRids -join ',' } else { 'none' }))
if (-not $winRids -or -not $linRids) {
    Say 'PACK_DONE refused: a module folder with no runtimes carries no native library'
    exit 4
}
$clash = @($winRids | Where-Object { $linRids -contains $_ })
if ($clash) {
    Say ("RID_CLASH both sides carry " + ($clash -join ',') + "; the merge refuses a duplicate rather than overwriting")
    Say 'PACK_DONE refused'
    exit 5
}

# Merge into a copy, so the build's own output stays a single-platform
# folder that can be re-merged after a mistake.
$merged = Join-Path $WorkDir 'merged'
if (Test-Path $merged) { Remove-Item -Recurse -Force $merged }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Copy-Item -Recurse -Path $WindowsModule -Destination $merged
Say "MERGE_INTO $merged"

& cargo pwrs merge $merged $LinuxModule *>&1 | ForEach-Object { Say ("MERGE " + $_) }
$mergeExit = $LASTEXITCODE
Say "MERGE_EXIT=$mergeExit"
if ($mergeExit -ne 0) { Say 'PACK_DONE refused'; exit $mergeExit }

$mergedRids = Get-Rids $merged
Say ("MERGED_RIDS " + ($mergedRids -join ','))
# Each side joined in its own statement. A comparison and a -join in
# one expression bind in an order that compares a string to an array
# and then joins the answer, which is the other half of how the first
# version reported a complete merge as incomplete.
$wantedJoined = ((@($winRids) + @($linRids)) | Sort-Object) -join ','
$gotJoined = (@($mergedRids) | Sort-Object) -join ','
if ($gotJoined -ne $wantedJoined) {
    Say "MERGE_INCOMPLETE expected $wantedJoined and the folder carries $gotJoined"
    Say 'PACK_DONE refused'
    exit 6
}
Say 'MERGE_COMPLETE every rid from both sides is in the merged folder'

# The manifest has to load as a manifest, not merely exist.
$mergedManifest = (Get-ChildItem -Path $merged -Filter *.psd1 -File)[0].FullName
try {
    $m = Test-ModuleManifest -Path $mergedManifest -ErrorAction Stop
    Say ("MANIFEST_OK name={0} version={1} cmdlets={2} aliases={3}" -f
        $m.Name, $m.Version, @($m.ExportedCmdlets.Keys).Count, @($m.ExportedAliases.Keys).Count)
} catch {
    Say "MANIFEST_BAD $_"
    Say 'PACK_DONE refused'
    exit 7
}

# Every URL the manifest freezes at publish. A listing whose icon 404s
# cannot be fixed without burning a version, so this runs before the
# package exists rather than after.
if ($SkipLinkCheck) {
    Say 'LINK_CHECK skipped by request; the manifest URLs are unverified in this run'
} elseif ($LinkCheck -and (Test-Path $LinkCheck)) {
    # Called with no arguments on purpose. Test-ManifestLink reads the
    # crate's Cargo.toml, where the gallery metadata is authored, and
    # defaults to the one beside its own directory - so a copy sitting
    # in a tree checks that tree. Handing it the generated psd1 would
    # be handing it a file it does not parse.
    & $LinkCheck *>&1 | ForEach-Object { Say ("LINK " + $_) }
    Say "LINK_EXIT=$LASTEXITCODE"
} else {
    Say "LINK_CHECK not run: no script at '$LinkCheck'"
}

# The package itself.
$out = Join-Path $WorkDir 'nupkg'
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force -Path $out | Out-Null
try {
    Compress-PSResource -Path $merged -DestinationPath $out -ErrorAction Stop
    Say "COMPRESS_EXIT=0"
} catch {
    Say "COMPRESS_FAILED $_"
    Say 'PACK_DONE refused'
    exit 8
}

$pkg = @(Get-ChildItem -Path $out -Filter *.nupkg -File)
if ($pkg.Count -ne 1) {
    Say ("PACKAGE_COUNT {0} nupkg files; expected one" -f $pkg.Count)
    Say 'PACK_DONE refused'
    exit 9
}
Say ("PACKAGE {0} {1} bytes" -f $pkg[0].Name, $pkg[0].Length)

# Read the package rather than trusting that it was written. A nupkg is
# a zip, and what a listing ships is whatever is inside it - so the two
# native libraries are confirmed by name from the archive's own entry
# table, which is the only place that fact is true.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($pkg[0].FullName)
try {
    $entries = @($zip.Entries | ForEach-Object { $_.FullName })
    Say ("PACKAGE_ENTRIES {0}" -f $entries.Count)
    foreach ($rid in $mergedRids) {
        $hit = @($entries | Where-Object { $_ -like "*runtimes/$rid/*" -or $_ -like "*runtimes\$rid\*" })
        if ($hit.Count -gt 0) {
            Say ("PACKAGE_CARRIES $rid " + ($hit | Select-Object -First 3) -join ' ')
        } else {
            Say "PACKAGE_MISSING $rid is in the merged folder and not in the package"
        }
    }
    foreach ($e in ($entries | Sort-Object)) { Say ("ENTRY " + $e) }
}
finally {
    $zip.Dispose()
}

Say 'PACK_DONE the package exists and has been read; publishing it is a separate, approved act'
exit 0
