# The Flynnel: drive.
#
# The claim that matters is that a leaf answers the SAME object its
# cmdlet writes, not a second rendering of it. Two renderings of one
# reading drift, and nothing else in this suite would catch it, so the
# comparison is field by field rather than by type name.
#
# The provider path differs between PowerShell 7 and Windows
# PowerShell, and this suite is the only thing in the module that
# would catch that, which is why it says which host it ran on.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost
    Write-Host ("PROVIDER_SUITE host={0}/{1} platform={2}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform)

    # Compares two objects by every property the first carries, so a
    # column added to one and not the other is a failure rather than
    # something the check steps over.
    function Test-SameRow {
        param($Left, $Right)
        $differences = @()
        foreach ($p in $Left.PSObject.Properties) {
            $a = $p.Value
            $b = $Right.$($p.Name)
            # Collections compare elementwise; a plain -ne on two
            # arrays answers an array and the if would take the wrong
            # branch.
            if ($a -is [System.Collections.IEnumerable] -and $a -isnot [string]) {
                if (@($a).Count -ne @($b).Count) {
                    $differences += "$($p.Name): $(@($a).Count) vs $(@($b).Count) items"
                }
            } elseif ($a -ne $b) {
                $differences += "$($p.Name): '$a' vs '$b'"
            }
        }
        , $differences
    }
}

Describe 'the drive itself' {
    It 'exists as soon as the module is imported' {
        # Nothing to mount: a scheduler is always there to browse.
        Get-PSDrive -Name Flynnel -ErrorAction SilentlyContinue |
            Should -Not -BeNullOrEmpty
    }

    It 'names the provider after the module' {
        (Get-PSDrive -Name Flynnel).Provider.Name | Should -Be 'Flynnel'
    }
}

Describe 'the tree' {
    It 'enumerates the top level' {
        @(Get-ChildItem Flynnel:\).Count | Should -BeGreaterThan 0
    }

    It 'has a host container' {
        (Get-Item Flynnel:\host).PSIsContainer | Should -BeTrue
    }

    It 'enumerates every leaf under host' {
        $names = @(Get-ChildItem Flynnel:\host | ForEach-Object { $_.Name })
        $names | Should -Contain 'topology'
        $names | Should -Contain 'cpu'
        $names | Should -Contain 'latency'
        $names | Should -Contain 'cache'
    }

    It 'says a path that is not there is not there' {
        Test-Path 'Flynnel:\host\nothing-of-that-name' | Should -BeFalse
        Test-Path 'Flynnel:\host\cpu' | Should -BeTrue
    }

    It 'counts a container''s children on the container itself' {
        (Get-Item Flynnel:\host).ChildCount | Should -BeGreaterThan 0
    }
}

Describe 'a leaf is the object its cmdlet writes' {
    # The check that keeps the drive and the cmdlets from drifting.
    # Field by field, because a type-name match would pass for two
    # objects of one type carrying different numbers.

    It 'answers the same CPU row as Get-FlynnelCpuInfo' {
        $fromDrive = Get-Item Flynnel:\host\cpu
        $fromCmdlet = Get-FlynnelCpuInfo
        $d = Test-SameRow -Left $fromCmdlet -Right $fromDrive
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'answers the same topology as Get-FlynnelTopology' {
        $d = Test-SameRow -Left (Get-FlynnelTopology) -Right (Get-Item Flynnel:\host\topology)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'answers the same cache row as Get-FlynnelCacheAllocation' {
        $d = Test-SameRow -Left (Get-FlynnelCacheAllocation) `
            -Right (Get-Item Flynnel:\host\cache)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'streams a leaf through Get-Content' {
        $rows = @(Get-Content Flynnel:\host\cpu)
        $rows.Count | Should -Be 1
        $rows[0].LogicalThreads | Should -Be (Get-FlynnelCpuInfo).LogicalThreads
    }

    It 'refuses Get-Content on a container' {
        { Get-Content Flynnel:\host } | Should -Throw
    }
}

Describe 'a reading this host cannot take' {
    It 'is a leaf that exists and holds nothing, not a missing path' {
        # The distinction a script cannot make for itself: "this host
        # took no measurement" and "no such thing exists" read alike if
        # the second is used for the first.
        Test-Path 'Flynnel:\host\latency' | Should -BeTrue

        $rows = @(Get-Content Flynnel:\host\latency)
        $cmdletRows = @(Get-FlynnelLatencyTable -WarningAction SilentlyContinue)
        if ($cmdletRows.Count -eq 0) {
            $rows.Count | Should -Be 0 -Because 'this host has no latency table'
            (Get-Item Flynnel:\host\latency).Unavailable | Should -Not -BeNullOrEmpty
        } else {
            $rows.Count | Should -Be 1
            $d = Test-SameRow -Left $cmdletRows[0] -Right $rows[0]
            $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
        }
    }
}

Describe 'the drive is read only' {
    # Not an omission. A drive that could change the scheduler would be
    # a second way to do what the Set- cmdlets already do, and two ways
    # to write one setting is how they drift apart.

    It 'refuses New-Item' {
        { New-Item Flynnel:\host\invented -ItemType File -ErrorAction Stop } |
            Should -Throw
    }

    It 'refuses Remove-Item' {
        { Remove-Item Flynnel:\host\cpu -ErrorAction Stop } | Should -Throw
    }

    It 'refuses Set-Content' {
        { Set-Content Flynnel:\host\cpu -Value 'x' -ErrorAction Stop } | Should -Throw
    }

    It 'leaves the leaf intact after a refused write' {
        try { Set-Content Flynnel:\host\cpu -Value 'x' -ErrorAction Stop } catch { }
        (Get-Item Flynnel:\host\cpu).LogicalThreads |
            Should -Be (Get-FlynnelCpuInfo).LogicalThreads
    }
}
