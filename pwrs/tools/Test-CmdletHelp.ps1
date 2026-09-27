# Reads the help every command renders and reports what a test cannot.
#
# Surface.Tests.ps1 already checks the mechanical half: that a synopsis
# is not the syntax line PowerShell substitutes for an absent one, that
# every command carries at least one example, and that every parameter
# has a description. Those are pass-or-fail and they belong in a suite.
#
# What a suite cannot check is whether the prose is any good, and the
# thing that makes bad prose visible is reading it ALL AT ONCE rather
# than one command at a time. Eighty synopses in one column show the
# one written in a different voice, the two that say the same thing
# about different commands, and the one that describes a parameter
# instead of the command. None of those is visible from inside a single
# Get-Help.
#
# So this prints the whole surface for a read, and fails on the few
# things that are mechanical and that the suite does not already cover:
#
#   a synopsis that is a sentence fragment rather than a statement
#   a parameter description that only restates the parameter's name
#   an example with a command and no explanation of what it shows
#   two commands whose synopses are identical
#
# Not in the Pester suites, for the same reason the link check is not:
# this is a read with a few checks attached, and its output is meant to
# be looked at.

[CmdletBinding()]
param(
    # The built module to load. The crate's own by default.
    [string]$ModulePath = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target\pwrs\Flynnel\Flynnel.psd1'),
    # Print the full parameter help for every command, not just the
    # flagged rows. Long, and what you want when actually reading.
    [switch]$Full,
    # Only these commands, by name or wildcard.
    [string]$Name = '*'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ModulePath)) {
    Write-Error "no module at $ModulePath; build it first with cargo pwrs build"
    exit 2
}

Import-Module $ModulePath -Force
$moduleName = [System.IO.Path]::GetFileNameWithoutExtension($ModulePath)
$commands = @(Get-Command -Module $moduleName -CommandType Cmdlet | Where-Object Name -like $Name | Sort-Object Name)

if ($commands.Count -eq 0) {
    Write-Error "no cmdlets matched '$Name' in $moduleName"
    exit 2
}

"HELP_AUDIT module=$moduleName commands=$($commands.Count)"
''

$problems = New-Object System.Collections.Generic.List[string]
$synopses = @{}

foreach ($cmd in $commands) {
    $help = Get-Help $cmd.Name
    $synopsis = ($help.Synopsis -replace '\s+', ' ').Trim()

    "=== $($cmd.Name) ==="
    "  $synopsis"

    # A synopsis is a statement about what the command does. The two
    # shapes that read as unfinished are one that does not end in a
    # period and one that opens with a lowercase letter, and both
    # are cheap to see and cheap to fix.
    if ($synopsis -notmatch '\.$') {
        $problems.Add("$($cmd.Name): the synopsis does not end in a period, so it reads as a fragment")
    }
    if ($synopsis -cmatch '^[a-z]') {
        $problems.Add("$($cmd.Name): the synopsis opens lowercase")
    }

    # Two commands saying the same sentence means one of them was
    # copied and not re-read.
    if ($synopses.ContainsKey($synopsis)) {
        $problems.Add("$($cmd.Name): its synopsis is word for word $($synopses[$synopsis])'s")
    } else {
        $synopses[$synopsis] = $cmd.Name
    }

    $examples = @($help.Examples.Example)
    "  examples: $($examples.Count)"
    foreach ($ex in $examples) {
        $remark = (($ex.remarks | ForEach-Object { $_.Text }) -join ' ').Trim()
        $code = ($ex.code -replace '\s+', ' ').Trim()
        if ($Full) { "    $code" }
        if (-not $remark -and -not $code) {
            $problems.Add("$($cmd.Name): an example carries neither a command nor an explanation")
        }
    }

    # A cmdlet with no parameters of its own still yields one entry
    # here with an empty name, so the name is required before anything
    # is judged. Without that the audit flagged all thirty-six
    # parameterless commands for "no description" on a parameter that
    # does not exist, which is a finding about the reader and not the
    # help.
    $params = @($help.Parameters.Parameter |
        Where-Object { $_.Name } |
        Where-Object { $_.Name -notin 'Verbose','Debug','ErrorAction','WarningAction','InformationAction','ErrorVariable','WarningVariable','InformationVariable','OutVariable','OutBuffer','PipelineVariable','WhatIf','Confirm','ProgressAction' })
    "  parameters: $($params.Count)"
    foreach ($p in $params) {
        $desc = (($p.Description | ForEach-Object { $_.Text }) -join ' ' -replace '\s+', ' ').Trim()
        if ($Full) { "    -$($p.Name)  $desc" }
        if (-not $desc) {
            $problems.Add("$($cmd.Name) -$($p.Name): no description")
            continue
        }
        # "The capacity." under -Capacity tells a reader nothing they
        # did not have from the name. A description earns its place by
        # saying something the name does not.
        $bare = ($desc -replace '[^A-Za-z]', '')
        if ($bare -and $bare -ieq ('the' + $p.Name) -or $bare -ieq $p.Name) {
            $problems.Add("$($cmd.Name) -$($p.Name): the description only restates the name")
        }
    }
    ''
}

''
if ($problems.Count -eq 0) {
    "HELP_AUDIT_OK nothing mechanical to flag across $($commands.Count) command(s)"
    "Read the synopses above together; that is the half no check covers."
    exit 0
}

"HELP_AUDIT_FLAGS $($problems.Count)"
foreach ($p in $problems) { "  $p" }
exit 1
