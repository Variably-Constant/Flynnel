# The surface gate: that the module is shaped the way it says it is,
# and that nothing was added to it without being tested.
#
# Every other suite checks what a command does. This one checks that
# the command exists, is named to the house pattern, declares what it
# emits, documents itself, and is reached by some suite. A binding is
# missing a thing and nothing says so: the module imports, every test
# passes, and the gap is invisible until someone needs it. That is the
# defect class this campaign has found nine times, and this file is
# the instrument against it.
#
# The gate reads the module itself rather than a list kept here, so a
# command added tomorrow is in scope tomorrow without anyone
# remembering to add it. The one list this file owns is the bulk-form
# table at the end, which encodes a rule a reader would otherwise have
# to remember.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:Module = Get-Module -Name Flynnel
    $script:Cmdlets = @($script:Module.ExportedCmdlets.Values)
    $script:Aliases = @($script:Module.ExportedAliases.Values)

    # Every other suite's text, for the check that each command is
    # reached by one. Read once rather than per command.
    $script:SuiteText = (Get-ChildItem -Path $PSScriptRoot -Filter '*.Tests.ps1' |
        Where-Object Name -ne 'Surface.Tests.ps1' |
        ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join "`n"

    $script:SuiteFileCount = @(Get-ChildItem -Path $PSScriptRoot -Filter '*.Tests.ps1' |
        Where-Object Name -ne 'Surface.Tests.ps1').Count

    # Every exported type, so the class and enum checks read the
    # assembly rather than a list.
    #
    # The generated shell carries a build-identity suffix, so the name
    # is Flynnel.Shell.<hash>. Matching it exactly left Exported empty
    # and every class and enum check in this file passed over nothing,
    # which is the shape those checks exist to catch.
    $script:Shell = [AppDomain]::CurrentDomain.GetAssemblies() |
        Where-Object { $_.GetName().Name -eq 'Flynnel.Shell' -or
                       $_.GetName().Name -like 'Flynnel.Shell.*' } |
        Select-Object -First 1
    $script:Exported = if ($script:Shell) { $script:Shell.GetExportedTypes() } else { @() }
}

Describe 'the module exports something at all' {
    It 'is loaded' {
        $script:Module | Should -Not -BeNullOrEmpty
    }

    It 'exports cmdlets' {
        # A zero here would make every other assertion in this file
        # pass over an empty set, which is the shape this whole suite
        # exists to refuse.
        $script:Cmdlets.Count | Should -BeGreaterThan 0
    }

    It 'has sibling suites to check against' {
        $script:SuiteFileCount | Should -BeGreaterThan 0
        $script:SuiteText.Length | Should -BeGreaterThan 0
    }

    It 'loaded its assembly' {
        $script:Shell | Should -Not -BeNullOrEmpty
        $script:Exported.Count | Should -BeGreaterThan 0
    }
}

Describe 'naming' {
    It 'gives every cmdlet a Flynnel noun' {
        $wrong = @($script:Cmdlets | Where-Object { $_.Noun -notlike 'Flynnel*' })
        $wrong.Count | Should -Be 0 -Because ("these do not take a Flynnel noun: " +
            (($wrong | ForEach-Object Name) -join ', '))
    }

    It 'uses an approved verb everywhere, bar one named exception' {
        # Sort is not on PowerShell's approved list. Sort-Object is
        # grandfathered, so every user reads Sort- as ordering, and the
        # approved alternatives each say something this cmdlet does not
        # do. The name stands and the analyzer warning is accepted.
        #
        # Named here rather than filtered out of the check, so the
        # exception is one line a reader can see and argue with, and so
        # a second unapproved verb still fails.
        $allowed = @('Sort-FlynnelArray')
        $approved = @(Get-Verb | ForEach-Object Verb)
        $wrong = @($script:Cmdlets | Where-Object {
            $_.Verb -notin $approved -and $_.Name -notin $allowed
        })
        $wrong.Count | Should -Be 0 -Because ("these use an unapproved verb: " +
            (($wrong | ForEach-Object Name) -join ', '))
    }

    It 'still needs the one exception it carries' {
        # An allow-list that has stopped matching anything is a line
        # nobody removes, and the next reader takes it for a rule. This
        # fails when Sort-FlynnelArray goes away, and when Sort joins
        # the approved list and the exception becomes dead.
        $approved = @(Get-Verb | ForEach-Object Verb)
        Get-Command 'Sort-FlynnelArray' -ErrorAction SilentlyContinue |
            Should -Not -BeNullOrEmpty
        $approved | Should -Not -Contain 'Sort'
    }

    It 'gives every cmdlet exactly one Fly alias' {
        $missing = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $expected = $cmdlet.Name -replace '^(\w+)-Flynnel', '$1-Fly'
            if ($script:Aliases.Name -notcontains $expected) { $missing += $cmdlet.Name }
        }
        $missing.Count | Should -Be 0 -Because ("these have no Fly alias: " +
            ($missing -join ', '))
    }

    It 'resolves every alias back to its cmdlet' {
        $wrong = @()
        foreach ($alias in $script:Aliases) {
            $target = $alias.ResolvedCommand
            if (-not $target) { $wrong += "$($alias.Name) resolves to nothing"; continue }
            $expected = $alias.Name -replace '^(\w+)-Fly', '$1-Flynnel'
            if ($target.Name -ne $expected) {
                $wrong += "$($alias.Name) resolves to $($target.Name), not $expected"
            }
        }
        $wrong.Count | Should -Be 0 -Because ($wrong -join '; ')
    }

    It 'has as many aliases as cmdlets' {
        # One each, no more: a stray alias is a name a script can come
        # to depend on that nothing documents.
        $script:Aliases.Count | Should -Be $script:Cmdlets.Count
    }
}

Describe 'what a cmdlet declares' {
    It 'declares an OutputType, or is one of the few that emit nothing' {
        # A cmdlet that emits an object without declaring its type
        # cannot be completed against, and Get-Command cannot say what
        # a pipeline will carry.
        # Update-FlynnelArray is here deliberately: it changes the
        # caller's buffer in place and writing the array back would
        # cost the very crossing it exists to avoid.
        $emitNothing = @('Clear-FlynnelTrace', 'Reset-FlynnelLeafStat',
                         'Reset-FlynnelSpinStats', 'Reset-FlynnelSplitStats',
                         'Start-FlynnelSplitObserver', 'Update-FlynnelArray')
        $missing = @()
        foreach ($cmdlet in $script:Cmdlets) {
            if ($cmdlet.Name -in $emitNothing) { continue }
            $declared = @((Get-Command $cmdlet.Name).OutputType)
            if ($declared.Count -eq 0) { $missing += $cmdlet.Name }
        }
        $missing.Count | Should -Be 0 -Because ("these declare no OutputType: " +
            ($missing -join ', '))
    }
}

Describe 'help' {
    It 'gives every cmdlet a synopsis that is a sentence' {
        # PowerShell fills an absent synopsis with the syntax line, so
        # "not empty" is not enough: a synopsis equal to the name is
        # the absence, wearing the clothes of a presence.
        $bad = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $synopsis = (Get-Help $cmdlet.Name -ErrorAction SilentlyContinue).Synopsis
            if (-not $synopsis) { $bad += "$($cmdlet.Name): none"; continue }
            $text = $synopsis.Trim()
            if ($text -eq $cmdlet.Name) { $bad += "$($cmdlet.Name): just the name"; continue }
            if ($text -like "$($cmdlet.Name) *") {
                $bad += "$($cmdlet.Name): the syntax line"; continue
            }
            # No length floor. What this catches is a synopsis that is
            # absent, which PowerShell fills with the syntax line, and
            # a length
            # threshold would instead fail a short sentence that says
            # everything: "Builds a job plan." is eighteen characters
            # and is not the defect.
            if ($text -notmatch '\s') { $bad += "$($cmdlet.Name): one word" }
        }
        $bad.Count | Should -Be 0 -Because ($bad -join '; ')
    }

    It 'gives every cmdlet at least one example' {
        $bad = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $examples = @((Get-Help $cmdlet.Name -ErrorAction SilentlyContinue).Examples.Example)
            if ($examples.Count -eq 0) { $bad += $cmdlet.Name }
        }
        $bad.Count | Should -Be 0 -Because ("these carry no example: " + ($bad -join ', '))
    }

    It 'describes every parameter' {
        $bad = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $help = Get-Help $cmdlet.Name -ErrorAction SilentlyContinue
            foreach ($parameter in @($help.Parameters.Parameter)) {
                if (-not $parameter) { continue }
                $description = ($parameter.Description | ForEach-Object Text) -join ' '
                if (-not $description -or $description.Trim().Length -lt 10) {
                    $bad += "$($cmdlet.Name) -$($parameter.Name)"
                }
            }
        }
        $bad.Count | Should -Be 0 -Because ("these parameters carry no description: " +
            ($bad -join ', '))
    }

    It 'names only parameters that exist in every example' {
        # The three checks above ask whether help is present. This one
        # asks whether a piece of it is correct, which is the half a
        # structural check cannot reach: an example naming a parameter
        # the cmdlet does not take is wrong in the way a reader finds
        # by running it.
        #
        # Read from the loaded module rather than from the Rust, so
        # what is compared is what PowerShell actually exposes.
        $common = [System.Management.Automation.PSCmdlet]::CommonParameters +
            [System.Management.Automation.PSCmdlet]::OptionalCommonParameters
        $bad = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $declared = @((Get-Command $cmdlet.Name).Parameters.Keys)
            $examples = @((Get-Help $cmdlet.Name -ErrorAction SilentlyContinue).Examples.Example)
            foreach ($example in $examples) {
                $code = "$($example.Code) $(($example.Remarks | ForEach-Object Text) -join ' ')"
                # A parameter belongs to the command it follows, so the
                # line is split at every separator and only the pieces
                # invoking this cmdlet are read. Without that split,
                # -Descending on a trailing Sort-Object is charged to
                # the cmdlet at the head of the line, and a parameter
                # of a nested call in parentheses to its caller.
                foreach ($segment in ($code -split '[|;()]')) {
                    $commands = [regex]::Matches($segment, '\b[A-Z][A-Za-z]*-[A-Za-z]+')
                    if ($commands.Count -eq 0) { continue }
                    if ($commands[0].Value -ne $cmdlet.Name) { continue }
                    # Every Verb-Noun token goes first, or the noun half
                    # of the command reads as a parameter of it. The
                    # verb pattern allows camel case, because
                    # ForEach-Object is a command and a lowercase-only
                    # verb misses it and leaves -Object behind.
                    $stripped = [regex]::Replace($segment, '\b[A-Z][A-Za-z]*-[A-Za-z]+', ' ')
                    foreach ($used in [regex]::Matches($stripped, '-([A-Z][A-Za-z]+)\b')) {
                        $name = $used.Groups[1].Value
                        if ($common -contains $name) { continue }
                        if ($declared -contains $name) { continue }
                        $bad += "$($cmdlet.Name): -$name"
                    }
                }
            }
        }
        $bad.Count | Should -Be 0 -Because ("these examples name a parameter the cmdlet " +
            "does not take: " + ($bad -join ', '))
    }

    It 'names only types that the module exports' {
        # A help page naming a type that does not exist sends the
        # reader to Get-Member on nothing. It is the easiest claim in
        # a doc to get wrong, because a plausible name is written from
        # memory and nothing rejects it: Flynnel.PassRegistryInfo was
        # written here for a class called Flynnel.PassRegistry and was
        # caught only by going and reading the attributes.
        #
        # The dot-then-capital is what separates a type from a file:
        # Flynnel.psd1 does not match, Flynnel.PassRegistry does.
        $known = @($script:Exported | ForEach-Object FullName)
        # The generated assembly carries a build-identity suffix, so
        # its own name is not in the exported-type list.
        $known += 'Flynnel.Shell'
        $bad = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $help = Get-Help $cmdlet.Name -ErrorAction SilentlyContinue
            $text = ($help | Out-String)
            foreach ($hit in [regex]::Matches($text, '\bFlynnel(?:\.[A-Z][A-Za-z0-9]*)+')) {
                if ($known -contains $hit.Value) { continue }
                $bad += "$($cmdlet.Name): $($hit.Value)"
            }
        }
        $bad.Count | Should -Be 0 -Because ("this help names a type the module does not " +
            "export: " + (($bad | Sort-Object -Unique) -join ', '))
    }

    It 'names only cmdlets that the module exports' {
        # Same defect one layer out: help that points at a sibling
        # command by a name the module no longer has. A rename moves
        # the command and leaves every reference to it reading
        # correctly, because prose is not compiled.
        $known = @($script:Cmdlets | ForEach-Object Name)
        $bad = @()
        foreach ($cmdlet in $script:Cmdlets) {
            $help = Get-Help $cmdlet.Name -ErrorAction SilentlyContinue
            $text = ($help | Out-String)
            foreach ($hit in [regex]::Matches($text, '\b[A-Z][A-Za-z]*-Flynnel[A-Za-z]*\b')) {
                if ($known -contains $hit.Value) { continue }
                $bad += "$($cmdlet.Name): $($hit.Value)"
            }
        }
        $bad.Count | Should -Be 0 -Because ("this help names a cmdlet the module does not " +
            "export: " + (($bad | Sort-Object -Unique) -join ', '))
    }
}

Describe 'every command is reached by a suite' {
    It 'names every cmdlet in some other suite' {
        # A command added to the module and not to a suite passes every
        # test in the folder, because no test mentions it. This is the
        # check that makes that fail.
        $untested = @($script:Cmdlets | Where-Object {
            $script:SuiteText -notmatch [regex]::Escape($_.Name)
        } | ForEach-Object Name)
        $untested.Count | Should -Be 0 -Because ("no suite names these: " +
            ($untested -join ', '))
    }

    It 'names every exported class in some other suite' {
        $classes = @($script:Exported | Where-Object {
            $_.IsClass -and $_.FullName -like 'Flynnel.*'
        } | ForEach-Object FullName)
        $classes.Count | Should -BeGreaterThan 0
        $untested = @($classes | Where-Object {
            $short = ($_ -split '\.')[-1]
            $script:SuiteText -notmatch [regex]::Escape($_) -and
            $script:SuiteText -notmatch [regex]::Escape($short)
        })
        $untested.Count | Should -Be 0 -Because ("no suite names these classes: " +
            ($untested -join ', '))
    }

    It 'names every exported enum in some other suite' {
        $enums = @($script:Exported | Where-Object {
            $_.IsEnum -and $_.FullName -like 'Flynnel.*'
        } | ForEach-Object FullName)
        $enums.Count | Should -BeGreaterThan 0
        $untested = @($enums | Where-Object {
            $short = ($_ -split '\.')[-1]
            $script:SuiteText -notmatch [regex]::Escape($_) -and
            $script:SuiteText -notmatch [regex]::Escape($short)
        })
        $untested.Count | Should -Be 0 -Because ("no suite names these enums: " +
            ($untested -join ', '))
    }
}

Describe 'the exported types' {
    It 'gives every enum at least one value' {
        $empty = @()
        foreach ($type in @($script:Exported | Where-Object { $_.IsEnum })) {
            if ([Enum]::GetValues($type).Count -eq 0) { $empty += $type.FullName }
        }
        $empty.Count | Should -Be 0 -Because ("these enums are empty: " + ($empty -join ', '))
    }

    It 'gives every class at least one readable property or method' {
        $bare = @()
        foreach ($type in @($script:Exported | Where-Object {
            $_.IsClass -and $_.FullName -like 'Flynnel.*'
        })) {
            $members = @($type.GetProperties()) + @($type.GetMethods() |
                Where-Object { $_.DeclaringType -eq $type })
            if ($members.Count -eq 0) { $bare += $type.FullName }
        }
        $bare.Count | Should -Be 0 -Because ("these types carry nothing: " + ($bare -join ', '))
    }

    It 'gives every proxy class a Dispose' {
        # A proxy owns a Rust value. Without Dispose the only release
        # is the finalizer, which is not a schedule a script can rely
        # on when the thing held is a pool or a cache reservation.
        # JobPlan is deliberately not here: it is a value, so that a
        # cmdlet can take one as a parameter.
        $proxies = @('Flynnel.CacheReservation', 'Flynnel.IoPool')
        foreach ($name in $proxies) {
            $type = $script:Exported | Where-Object FullName -eq $name
            if (-not $type) {
                Set-ItResult -Skipped -Because "$name is not exported by this build"
                return
            }
            $type.GetMethod('Dispose') | Should -Not -BeNullOrEmpty -Because "$name is a proxy"
        }
    }
}

Describe 'anything over many items crosses in one call' {
    It 'takes a collection rather than one item at a time' {
        # The K_cross rule, made testable rather than remembered. A
        # method call costs 1907 ns in PowerShell 7.6 against 4.5
        # amortized over a thousand, so a cmdlet that can handle many
        # items and takes one is a hundredfold cost a caller cannot
        # avoid. The list is this suite's own, because the rule is
        # about intent and no property of the module carries it.
        $bulk = @{
            'Invoke-FlynnelMap'          = 'InputObject'
            'Invoke-FlynnelZip'          = 'Left'
            'Measure-FlynnelReduce'      = 'InputObject'
            'Get-FlynnelPrefixSum'       = 'InputObject'
            'Get-FlynnelHistogram'       = 'InputObject'
            'Get-FlynnelDotProduct'      = 'Left'
            'Sort-FlynnelArray'          = 'InputObject'
            'Measure-FlynnelFileHash'    = 'Path'
            'Test-FlynnelFileHash'       = 'Path'
            'Search-FlynnelFile'         = 'Path'
            'Measure-FlynnelFileLine'    = 'Path'
            'Measure-FlynnelFileByte'    = 'Path'
            'Get-FlynnelSpread'          = 'Sample'
        }
        $wrong = @()
        foreach ($name in $bulk.Keys) {
            $command = Get-Command $name -ErrorAction SilentlyContinue
            if (-not $command) { $wrong += "$name is not exported"; continue }
            $parameter = $command.Parameters[$bulk[$name]]
            if (-not $parameter) { $wrong += "$name has no -$($bulk[$name])"; continue }
            if (-not $parameter.ParameterType.IsArray) {
                $wrong += "$name -$($bulk[$name]) is $($parameter.ParameterType.Name), not an array"
            }
        }
        $wrong.Count | Should -Be 0 -Because ($wrong -join '; ')
    }

    It 'accepts the whole collection from the pipeline in one binding' {
        # ValueFromPipeline on an array parameter binds the whole
        # collection at once. Without it the binder enumerates and the
        # cmdlet is called per item, which is the cost the rule exists
        # to avoid.
        $fromPipeline = @('Invoke-FlynnelMap', 'Measure-FlynnelReduce',
                          'Get-FlynnelPrefixSum', 'Sort-FlynnelArray',
                          'Measure-FlynnelFileHash')
        $wrong = @()
        foreach ($name in $fromPipeline) {
            $command = Get-Command $name -ErrorAction SilentlyContinue
            if (-not $command) { $wrong += "$name is not exported"; continue }
            $binds = @($command.Parameters.Values | Where-Object {
                $_.Attributes | Where-Object {
                    $_ -is [System.Management.Automation.ParameterAttribute] -and
                    $_.ValueFromPipeline
                }
            })
            if ($binds.Count -eq 0) { $wrong += "$name takes nothing from the pipeline" }
        }
        $wrong.Count | Should -Be 0 -Because ($wrong -join '; ')
    }
}
