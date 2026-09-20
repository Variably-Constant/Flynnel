# The backend family: that every backend gets a row on every host,
# that the three availability questions stay three questions, and that
# a capability nobody registered reads as absent rather than as zero.
#
# What this file does not cover, said first because it is the thing a
# reader is most likely to get wrong. Most of it asserts about a host
# with no accelerator, because that is the common case. The device
# paths - a registered CUDA backend, a bound kernel, a routing decision
# that picks one - are only exercised where a device exists. So the
# suite prints what it found before it asserts anything, and the device
# assertions are skipped by name rather than quietly absent: a passing
# run on a deviceless box must never be read as covering them.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost
    $script:Rows = @(Get-FlynnelBackend)
    $script:Available = @($script:Rows | Where-Object Available | ForEach-Object Name)
    $script:HasDevice = @($script:Rows |
        Where-Object { $_.Available -and $_.Kind -ne 'Cpu' }).Count -gt 0

    Write-Host ("BACKEND_SUITE host={0}/{1} platform={2} cpus={3}" -f
        $script:TestHost.Edition, $script:TestHost.Version,
        $script:TestHost.Platform, $script:TestHost.ProcessorCount)
    Write-Host ("BACKEND_SUITE available={0}" -f
        $(if ($script:Available) { $script:Available -join ',' } else { 'cpu only' }))
    if (-not $script:HasDevice) {
        Write-Host 'BACKEND_SUITE no accelerator on this host: the device assertions are skipped rather than passed'
    }
}

Describe 'the types and enums this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.Backend', 'Flynnel.BackendProbe',
                          'Flynnel.AccelOp', 'Flynnel.AccelTarget') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }

    It 'names every backend kind the taxonomy has' {
        $names = [enum]::GetNames([Flynnel.BackendKind])
        foreach ($kind in 'Cpu', 'Cuda', 'Rocm', 'Metal', 'Tpu', 'Ane',
                          'Wasm', 'SharedMemoryWorker', 'Custom') {
            $names | Should -Contain $kind
        }
    }
}

Describe 'Get-FlynnelBackend' {
    It 'writes a row for every enumerable kind, on every host' {
        # The point of the family: an absent device is a row saying so.
        # A missing row reads exactly like a capability nobody bound,
        # and this module compiles every backend in, so absence is
        # always a runtime fact rather than a build one.
        $kinds = @($script:Rows | ForEach-Object { $_.Kind.ToString() })
        foreach ($kind in 'Cpu', 'Cuda', 'Rocm', 'Metal', 'Tpu', 'Ane',
                          'Wasm', 'SharedMemoryWorker') {
            $kinds | Should -Contain $kind -Because "$kind needs a row even where it is absent"
        }
    }

    It 'always has the CPU registered and available' {
        $cpu = $script:Rows | Where-Object Kind -eq 'Cpu' | Select-Object -First 1
        $cpu | Should -Not -BeNullOrEmpty
        $cpu.Registered | Should -BeTrue
        $cpu.Available | Should -BeTrue
        $cpu.IsSimt | Should -BeFalse -Because 'the CPU backend is not a SIMT device'
    }

    It 'gives the CPU real capabilities rather than zeros' {
        $cpu = $script:Rows | Where-Object Kind -eq 'Cpu' | Select-Object -First 1
        $cpu.CapabilitiesKnown | Should -BeTrue
        $cpu.SimtWidth | Should -Be 1
        $cpu.MaxThreadsInFlight | Should -BeGreaterThan 0
        $cpu.LaunchLatencyNs | Should -BeGreaterThan 0
        $cpu.H2dBandwidthBytesPerSec | Should -Be 0 -Because 'the CPU backend moves nothing across a bus'
    }

    It 'says when the capability columns are not measurements' {
        # Zero SimtWidth on an unregistered backend is the absence of a
        # reading, not a reading of zero. CapabilitiesKnown is the only
        # thing that tells those apart, and without it a script would
        # divide by a width that was never measured.
        foreach ($row in $script:Rows | Where-Object { -not $_.Registered }) {
            $row.CapabilitiesKnown | Should -BeFalse -Because "$($row.Kind) has no implementation to ask"
            $row.SimtWidth | Should -Be 0
            $row.MaxThreadsInFlight | Should -Be 0
            $row.LaunchLatencyNs | Should -Be 0
        }
    }

    It 'keeps Registered, Available and Detected as three columns' {
        foreach ($row in $script:Rows) {
            $row.Registered | Should -BeOfType [bool]
            $row.Available | Should -BeOfType [bool]
            $row.Detected | Should -BeOfType [bool]
        }
    }

    It 'says what each probe looked at' {
        # A false Available a caller cannot argue with is one they have
        # to take on faith.
        foreach ($row in $script:Rows) {
            $row.Probe | Should -Not -BeNullOrEmpty -Because "$($row.Kind) must say how it was decided"
        }
    }

    It 'answers for one kind when asked' {
        $one = Get-FlynnelBackend -Kind Cuda
        @($one).Count | Should -Be 1
        $one.Kind | Should -Be 'Cuda'
    }

    It 'answers to its Fly alias' {
        @(Get-FlyBackend).Count | Should -Be $script:Rows.Count
    }

    It 'throws on nothing, whatever the host has' {
        { Get-FlynnelBackend } | Should -Not -Throw
        { Get-FlynnelBackend -Kind Tpu } | Should -Not -Throw
        { Get-FlynnelBackend -Kind Custom -DeviceId 9999 } | Should -Not -Throw
    }
}

Describe 'Test-FlynnelBackend' {
    It 'agrees with the listing about what is available' {
        # Two cmdlets reading the same probe must not disagree. If they
        # ever do, one of them is caching and the caller cannot tell
        # which answer is current.
        foreach ($probe in Test-FlynnelBackend) {
            $row = $script:Rows | Where-Object Kind -eq $probe.Kind | Select-Object -First 1
            if ($row) {
                $probe.Available | Should -Be $row.Available -Because "$($probe.Kind) must read the same both ways"
            }
        }
    }

    It 'probes one kind when asked' {
        $one = Test-FlynnelBackend -Kind Wasm
        @($one).Count | Should -Be 1
        $one.Kind | Should -Be 'Wasm'
    }

    It 'reports the CPU as available without claiming it probed one' {
        $cpu = Test-FlynnelBackend -Kind Cpu
        $cpu.Available | Should -BeTrue
        $cpu.Probe | Should -Match 'not probed'
    }

    It 'throws on nothing' {
        { Test-FlynnelBackend } | Should -Not -Throw
    }
}

Describe 'Get-FlynnelAccelOp' {
    It 'answers without throwing, whether or not anything is registered' {
        # An empty list is the ordinary state: the crate registers its
        # linear-algebra operations only when a caller asks, so nothing
        # having asked is not the same as the build lacking them.
        { Get-FlynnelAccelOp } | Should -Not -Throw
    }

    It 'gives every listed operation a name and a binding column' {
        foreach ($op in Get-FlynnelAccelOp) {
            $op.Name | Should -Not -BeNullOrEmpty
            $op.HasBinding | Should -BeOfType [bool]
            # HasBinding and the bound list have to agree, or a script
            # branching on one and reading the other gets nothing.
            $op.HasBinding | Should -Be (@($op.BoundKinds).Count -gt 0)
        }
    }

    It 'filters by name' {
        { Get-FlynnelAccelOp -Name 'no-such-operation-anywhere' } | Should -Not -Throw
        @(Get-FlynnelAccelOp -Name 'no-such-operation-anywhere').Count | Should -Be 0
    }
}

Describe 'Get-FlynnelAccelTarget' {
    It 'refuses an unknown operation and says what is registered' {
        # The two mistakes are a misspelt name and an empty registry,
        # and a message that cannot tell them apart sends the reader to
        # the wrong place.
        { Get-FlynnelAccelTarget -Name 'definitely-not-registered' } |
            Should -Throw -ExpectedMessage '*definitely-not-registered*'
    }

    It 'resolves a registered operation without launching it' {
        # The skip is decided HERE and not in a -Skip: expression.
        # Pester evaluates -Skip: during DISCOVERY, before BeforeAll
        # has imported the module, so a -Skip: calling a cmdlet of the
        # module under test throws CommandNotFoundException and the
        # whole file fails discovery. The run then reports a higher
        # pass count with no failures, because the tests that did not
        # run cannot fail - which is the most misleading shape a
        # broken suite can take.
        $ops = @(Get-FlynnelAccelOp)
        if ($ops.Count -eq 0) {
            Set-ItResult -Skipped -Because 'no accelerator operation is registered in this process'
            return
        }
        $op = $ops[0]
        $target = Get-FlynnelAccelTarget -Name $op.Name
        $target.Name | Should -Be $op.Name
        $target.RoutedToBackend | Should -BeOfType [bool]
        if (-not $target.RoutedToBackend) {
            # A false answer is a real one: the CPU implementation runs.
            $target.KernelHandle | Should -Be 0
        }
    }
}

Describe 'what this run did not cover' {
    It 'records whether a device was present, so the skip is visible' {
        # Not an assertion about the host. This exists so the run's own
        # output says which half of the family it exercised, and a
        # passing run on a deviceless box is never mistaken for one
        # that covered the device paths.
        $script:HasDevice | Should -BeOfType [bool]
        if (-not $script:HasDevice) {
            Set-ItResult -Skipped -Because 'no accelerator on this host; the device routing paths were not exercised'
        }
    }
}
