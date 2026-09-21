# Shared by every suite: how the module is found and imported, and the
# helpers the assertions lean on.
#
# The runner sets PWRS_MODULE to the built module folder. A suite that
# fell back to an installed copy would test whatever was installed
# rather than what was just built, so an absent variable is a failure
# here and not a reason to go looking.

function Import-FlynnelModule {
    if (Get-Module -Name Flynnel) { return }
    $root = $env:PWRS_MODULE
    if (-not $root) {
        throw 'PWRS_MODULE is not set. The suites import the module the runner built; ' +
              'importing an installed copy would test something else.'
    }
    $manifest = Join-Path $root 'Flynnel.psd1'
    if (-not (Test-Path $manifest)) {
        throw "no manifest at $manifest"
    }
    Import-Module $manifest -Force -ErrorAction Stop
}

# Which host is running, so a result recorded from a suite says where it
# came from. PowerShell 7 and Windows PowerShell differ by a factor of
# three at this boundary, and a figure without its edition cannot be
# compared with one from the other.
function Get-FlynnelTestHost {
    [PSCustomObject]@{
        Edition       = $PSVersionTable.PSEdition
        Version       = $PSVersionTable.PSVersion.ToString()
        Platform      = if ($IsWindows -or $PSVersionTable.PSEdition -eq 'Desktop') { 'Windows' }
                        elseif ($IsLinux) { 'Linux' }
                        else { 'Other' }
        ProcessorCount = [Environment]::ProcessorCount
    }
}

# True when this host has more than one NUMA node, so a suite can say it
# SKIPPED a multi-node assertion rather than passing one it never made.
function Test-FlynnelMultiNode {
    (Get-FlynnelTopology).NodeCount -gt 1
}

# Capturing a warning takes -WarningVariable on its own.
#
# Adding -WarningAction SilentlyContinue captures nothing. The binding
# builds a warning's text only when the engine says the warning stream
# is on, so suppressing the stream suppresses the write itself and the
# variable stays empty. A script calling these cmdlets for real is
# unaffected; it is only a test trying to be quiet that loses the
# thing it came to read.

# Every property of a class the module exports, for the checks that a
# figure which can be unmeasured is nullable rather than a zero-valued
# primitive.
function Get-FlynnelTypeProperty {
    param([Parameter(Mandatory)][string]$TypeName)
    # The generated shell carries a build-identity suffix, so its
    # assembly is Flynnel.Shell.<hash> rather than Flynnel.Shell. An
    # exact match passed only while the build happened not to add one,
    # and then failed every type check in every suite at once with a
    # message naming nothing.
    $shell = [AppDomain]::CurrentDomain.GetAssemblies() |
        Where-Object { $_.GetName().Name -eq 'Flynnel.Shell' -or
                       $_.GetName().Name -like 'Flynnel.Shell.*' } |
        Select-Object -First 1
    if (-not $shell) {
        # Name what is loaded. A bare "not loaded" sends a reader
        # looking for a module that failed to import, when what
        # actually happened is that the assembly is there under a name
        # this did not expect.
        $loaded = @([AppDomain]::CurrentDomain.GetAssemblies() |
            ForEach-Object { $_.GetName().Name } |
            Where-Object { $_ -like '*Flynnel*' })
        $seen = if ($loaded) { $loaded -join ', ' } else { 'none' }
        throw "no Flynnel.Shell assembly is loaded; Flynnel assemblies present: $seen"
    }
    $type = $shell.GetExportedTypes() | Where-Object FullName -eq $TypeName
    if (-not $type) { throw "no exported type named $TypeName in $($shell.GetName().Name)" }
    $type.GetProperties()
}
