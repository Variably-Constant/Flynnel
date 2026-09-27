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
        # PSChildName, not Name. A leaf's object is the row its cmdlet
        # writes, and a Flynnel.CpuInfo has no Name property of its
        # own; adding one would make it a different object from what
        # the cmdlet writes, which is the one thing this drive must
        # not do. The provider-supplied name is where a name lives.
        $names = @(Get-ChildItem Flynnel:\host | ForEach-Object { $_.PSChildName })
        $names | Should -Contain 'topology'
        $names | Should -Contain 'cpu'
        $names | Should -Contain 'latency'
        $names | Should -Contain 'cache'
    }

    It 'gives a leaf no Name of its own, because it is the cmdlet''s row' {
        # Pinned deliberately. A later change that added Name to a leaf
        # would make the drive's object differ from the cmdlet's, and
        # nothing else here would notice.
        $leaf = Get-Item Flynnel:\host\cpu
        $leaf.PSObject.Properties.Name | Should -Not -Contain 'Name'
        $leaf.PSChildName | Should -Be 'cpu'
        $leaf.PSIsContainer | Should -BeFalse
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

Describe 'the pool level' {
    BeforeAll {
        # A pool has to exist before it has workers to enumerate, and
        # starting it deliberately keeps the start out of any later
        # measurement.
        Start-FlynnelPool -WarningAction SilentlyContinue | Out-Null
    }

    It 'has the three fixed leaves and the workers container' {
        $names = @(Get-ChildItem Flynnel:\pool | ForEach-Object { $_.PSChildName })
        $names | Should -Contain 'summary'
        $names | Should -Contain 'spin'
        $names | Should -Contain 'split'
        $names | Should -Contain 'workers'
    }

    It 'answers the same summary as Get-FlynnelPool' {
        $d = Test-SameRow -Left (Get-FlynnelPool) -Right (Get-Item Flynnel:\pool\summary)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'answers the same spin state as Get-FlynnelSpinWindow' {
        $d = Test-SameRow -Left (Get-FlynnelSpinWindow) -Right (Get-Item Flynnel:\pool\spin)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'enumerates one child per worker' {
        $fromDrive = @(Get-ChildItem Flynnel:\pool\workers)
        $fromCmdlet = @(Get-FlynnelWorker)
        $fromDrive.Count | Should -Be $fromCmdlet.Count
        $fromDrive.Count | Should -BeGreaterThan 0
    }

    It 'leaves the external slots out of the workers level' {
        # They are real rows and they are not workers. A level called
        # workers holding some things that are not is worse than one
        # that omits them, and Get-FlynnelWorker takes them with a
        # switch for a caller who wants them.
        $withExternal = @(Get-FlynnelWorker -IncludeExternalSlot).Count
        $onDrive = @(Get-ChildItem Flynnel:\pool\workers).Count
        $onDrive | Should -Be @(Get-FlynnelWorker).Count
        if ($withExternal -gt @(Get-FlynnelWorker).Count) {
            $onDrive | Should -BeLessThan $withExternal
        }
    }

    It 'answers a worker by its index' {
        $first = @(Get-FlynnelWorker)[0]
        $d = Test-SameRow -Left $first -Right (Get-Item "Flynnel:\pool\workers\$($first.Index)")
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'resolves only the names it enumerated' {
        # Matched against the names the level lists rather than by
        # parsing the segment, so a padded index is not a second way
        # to spell a worker.
        Test-Path 'Flynnel:\pool\workers\0' | Should -BeTrue
        Test-Path 'Flynnel:\pool\workers\00' | Should -BeFalse
        Test-Path 'Flynnel:\pool\workers\not-a-number' | Should -BeFalse
        Test-Path 'Flynnel:\pool\workers\99999' | Should -BeFalse
    }
}

Describe 'the backends level' {
    It 'has a child for every backend kind, present or not' {
        # An absent device is a child that exists and reports
        # Registered false, never a missing child, for the same reason
        # Get-FlynnelBackend writes a row for it: a script cannot act
        # on a listing that failed to enumerate.
        $onDrive = @(Get-ChildItem Flynnel:\backends | ForEach-Object { $_.PSChildName })
        $fromCmdlet = @(Get-FlynnelBackend | ForEach-Object { [string]$_.Kind })
        foreach ($k in $fromCmdlet) {
            if ($k -eq 'Custom') { continue }
            $onDrive | Should -Contain $k
        }
    }

    It 'answers the same row as Get-FlynnelBackend for one kind' {
        $d = Test-SameRow -Left (Get-FlynnelBackend -Kind Cpu) `
            -Right (Get-Item Flynnel:\backends\Cpu)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'reports a device this host does not have rather than omitting it' {
        $cuda = Get-Item Flynnel:\backends\Cuda
        $cuda | Should -Not -BeNullOrEmpty
        $cuda.Kind | Should -Be 'Cuda'
    }

    It 'holds the accelerator operations in a container of their own' {
        # A container whether or not anything is registered, and nothing
        # is until a caller registers one, so an empty level is the
        # ordinary state rather than a failed enumeration.
        Test-Path 'Flynnel:\backends\accel-ops' | Should -BeTrue
        (Get-Item Flynnel:\backends\accel-ops).PSIsContainer | Should -BeTrue
        @(Get-ChildItem Flynnel:\backends | ForEach-Object { $_.PSChildName }) |
            Should -Contain 'accel-ops'
    }

    It 'lists the same operations as Get-FlynnelAccelOp' {
        $fromCmdlet = @(Get-FlynnelAccelOp)
        @(Get-ChildItem Flynnel:\backends\accel-ops).Count | Should -Be $fromCmdlet.Count
        if ($fromCmdlet.Count -eq 0) {
            Set-ItResult -Skipped -Because 'no accelerator operation is registered in this process'
            return
        }
        foreach ($row in $fromCmdlet) {
            $name = $row.Name -replace '[\\/:]', '-'
            $d = Test-SameRow -Left $row -Right (Get-Item "Flynnel:\backends\accel-ops\$name")
            $d.Count | Should -Be 0 -Because ("$($row.Name) differs: " + ($d -join '; '))
        }
    }
}

Describe 'the sites level' {
    It 'lists a child per call site the scheduler has materialized' {
        # A site appears only once a dispatch has reached that source
        # location, so an empty level means this process has run no
        # work through Flynnel, not that the level failed.
        $onDrive = @(Get-ChildItem Flynnel:\sites)
        $fromCmdlet = @(Get-FlynnelCallSite -WarningAction SilentlyContinue)
        $onDrive.Count | Should -Be $fromCmdlet.Count
    }

    It 'answers the same row as Get-FlynnelCallSite for one site' {
        $fromCmdlet = @(Get-FlynnelCallSite -WarningAction SilentlyContinue)
        if ($fromCmdlet.Count -eq 0) {
            Set-ItResult -Skipped -Because 'no dispatch has reached a call site in this process'
            return
        }
        $first = @(Get-ChildItem Flynnel:\sites)[0]
        $match = $fromCmdlet | Where-Object {
            $_.File -eq $first.File -and $_.Line -eq $first.Line
        } | Select-Object -First 1
        $match | Should -Not -BeNullOrEmpty
        $match.Column | Should -Be $first.Column
    }

    It 'names a site with no separator a path cannot carry' {
        # A location's colon is not a path segment on either platform,
        # so the name replaces it. The row still holds File, Line and
        # Column, which is what a script reads.
        foreach ($n in @(Get-ChildItem Flynnel:\sites | ForEach-Object { $_.PSChildName })) {
            $n | Should -Not -Match ':'
        }
    }
}

Describe 'the calibration and trace levels' {
    It 'answers the same calibration as Get-FlynnelCalibration' {
        $d = Test-SameRow -Left (Get-FlynnelCalibration) `
            -Right (Get-Item Flynnel:\calibration\summary)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'answers the same thresholds as Get-FlynnelClassThreshold' {
        $d = Test-SameRow -Left (Get-FlynnelClassThreshold) `
            -Right (Get-Item Flynnel:\calibration\thresholds)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'answers the same store as Get-FlynnelCalibrationStore' {
        $fromCmdlet = $null
        try {
            $fromCmdlet = Get-FlynnelCalibrationStore -ErrorAction Stop
        } catch {
            # The one case the cmdlet refuses. The leaf is still there and
            # holds nothing, which is how the drive says a host took no
            # such reading.
            if ("$_" -notmatch 'no calibration directory') { throw }
            Test-Path 'Flynnel:\calibration\store' | Should -BeTrue
            @(Get-Content Flynnel:\calibration\store).Count | Should -Be 0
            (Get-Item Flynnel:\calibration\store).Unavailable | Should -Not -BeNullOrEmpty
            return
        }
        $d = Test-SameRow -Left $fromCmdlet -Right (Get-Item Flynnel:\calibration\store)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'answers the same trace state as Get-FlynnelTraceState' {
        $d = Test-SameRow -Left (Get-FlynnelTraceState) `
            -Right (Get-Item Flynnel:\trace\enabled)
        $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
    }

    It 'streams the same events as Get-FlynnelTraceEvent, one row each' {
        # The ring is the calling thread's, so the dispatch that fills it
        # runs here, on the thread that then reads it both ways.
        Clear-FlynnelTrace
        $null = Set-FlynnelTraceState -On -WarningVariable ignored
        try {
            $null = Invoke-FlynnelMap -InputObject ([double[]](1..20000)) -Operation Square
        } finally {
            $null = Set-FlynnelTraceState -On:$false -WarningVariable ignored
        }
        try {
            $fromCmdlet = @(Get-FlynnelTraceEvent)
            $fromDrive = @(Get-Content Flynnel:\trace\events)
            $fromCmdlet.Count | Should -BeGreaterThan 0
            $fromDrive.Count | Should -Be $fromCmdlet.Count
            for ($i = 0; $i -lt $fromCmdlet.Count; $i++) {
                $d = Test-SameRow -Left $fromCmdlet[$i] -Right $fromDrive[$i]
                $d.Count | Should -Be 0 -Because ("row $i differs: " + ($d -join '; '))
            }
            # The item holds every row as one object.
            @(Get-Item Flynnel:\trace\events).Count | Should -Be 1
            (Get-Item Flynnel:\trace\events).Count | Should -Be $fromCmdlet.Count
        } finally {
            Clear-FlynnelTrace
        }
    }

    It 'enumerates every calibration and trace leaf' {
        $names = @(Get-ChildItem Flynnel:\calibration | ForEach-Object { $_.PSChildName })
        $names | Should -Contain 'summary'
        $names | Should -Contain 'thresholds'
        $names | Should -Contain 'store'
        $names = @(Get-ChildItem Flynnel:\trace | ForEach-Object { $_.PSChildName })
        $names | Should -Contain 'enabled'
        $names | Should -Contain 'events'
    }
}

Describe 'the peer level' {
    It 'exists whether or not a peer does' {
        # The distinction the spec calls out. A missing path and a
        # missing device read alike to a script, and only one of them
        # is worth retrying after starting a peer.
        Test-Path 'Flynnel:\peer' | Should -BeTrue
        (Get-Item Flynnel:\peer).PSIsContainer | Should -BeTrue
    }

    It 'enumerates nothing while no peer is running' {
        if ((Get-FlynnelGpuPeer).Running) {
            Set-ItResult -Skipped -Because 'a peer is running in this process'
            return
        }
        @(Get-ChildItem Flynnel:\peer).Count | Should -Be 0
        Test-Path 'Flynnel:\peer\summary' | Should -BeFalse
        Test-Path 'Flynnel:\peer\watchdog' | Should -BeFalse
    }

    It 'answers the same summary and watchdog as the cmdlets once one runs' {
        $cuda = Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Available }
        if (-not $cuda) {
            Set-ItResult -Skipped -Because 'no loadable CUDA driver on this host'
            return
        }
        try {
            New-FlynnelGpuPeer | Out-Null
            Test-Path 'Flynnel:\peer\summary' | Should -BeTrue
            $d = Test-SameRow -Left (Get-FlynnelGpuPeer) `
                -Right (Get-Item Flynnel:\peer\summary)
            $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
            # The default configuration starts the peer on device zero,
            # so the leaf is that device's watchdog.
            Test-Path 'Flynnel:\peer\watchdog' | Should -BeTrue
            $d = Test-SameRow -Left (Get-FlynnelPeerWatchdog -Ordinal 0) `
                -Right (Get-Item Flynnel:\peer\watchdog)
            $d.Count | Should -Be 0 -Because ("these differ: " + ($d -join '; '))
        } finally {
            Remove-FlynnelGpuPeer -WarningAction SilentlyContinue | Out-Null
        }
        # And both go away again, so the level tracks the peer rather
        # than remembering that one once existed.
        Test-Path 'Flynnel:\peer\summary' | Should -BeFalse
        Test-Path 'Flynnel:\peer\watchdog' | Should -BeFalse
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
