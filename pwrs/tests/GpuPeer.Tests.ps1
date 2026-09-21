# The GPU peer family.
#
# This suite runs on a host with a card and on one without, and the two
# runs assert different things. A deviceless run that quietly passed
# every device assertion would be the worst outcome available here, so
# the device assertions are skipped BY NAME and the suite prints which
# host it found before it asserts anything.
#
# What is bound so far is the watchdog reading, which is the part of
# this family that answers on every host: it says what bounds a piece of
# device work, and on a machine with no card it says that nothing does
# and why.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost

    # What this host actually has, read once and printed, so a reader of
    # the log knows which arm of every skip was taken.
    $script:Watchdog = Get-FlynnelPeerWatchdog
    $script:HasCard = $script:Watchdog.DriverModelKnown

    Write-Host ("GPUPEER_SUITE host={0}/{1} platform={2} model_known={3} model={4} applies={5}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform,
        $script:Watchdog.DriverModelKnown, $script:Watchdog.DriverModel,
        $script:Watchdog.Applies)
    Write-Host ("GPUPEER_BASIS {0}" -f $script:Watchdog.Basis)
}

Describe 'the types and the enum this family exports' {
    It 'shapes the watchdog row the way its cmdlet documents' {
        @(Get-FlynnelTypeProperty -TypeName 'Flynnel.PeerWatchdog').Count |
            Should -BeGreaterThan 0
    }

    It 'names every driver model, including the unreadable one' {
        # Unknown is a value rather than a null, because an unreadable
        # model is a reading and the device is still treated as covered.
        $names = [enum]::GetNames([Flynnel.DriverModel])
        $names | Should -Contain 'Unknown'
        $names | Should -Contain 'Wddm'
        $names | Should -Contain 'Tcc'
        $names | Should -Contain 'Mcdm'
    }

    It 'makes the delay nullable rather than zero-valued' {
        # A zero here would read as a bound of no time at all. The two
        # facts that have to stay apart are "resets after two seconds"
        # and "nothing resets it".
        $p = Get-FlynnelTypeProperty -TypeName 'Flynnel.PeerWatchdog' |
            Where-Object Name -eq 'DelayNs'
        $p.PropertyType.FullName | Should -Match 'Nullable'
    }
}

Describe 'Get-FlynnelPeerWatchdog on any host' {
    It 'answers a row rather than nothing' {
        # An absent device is a row saying so, never a missing row: a
        # script cannot act on a listing that failed to enumerate.
        $script:Watchdog | Should -Not -BeNullOrEmpty
    }

    It 'always says what it read' {
        # The column to quote when a bound looks wrong. It names which
        # of the two reads decided the answer, including the ones that
        # failed.
        $script:Watchdog.Basis | Should -Not -BeNullOrEmpty
        $script:Watchdog.Basis | Should -Match 'driver model'
    }

    It 'echoes the ordinal it was asked about' {
        (Get-FlynnelPeerWatchdog -Ordinal 0).Ordinal | Should -Be 0
        (Get-FlynnelPeerWatchdog -Ordinal 3).Ordinal | Should -Be 3
    }

    It 'answers for an ordinal no device has, rather than throwing' {
        # Sizing work against a device that is not there is a mistake
        # worth an answer, not an exception: the answer is that the read
        # failed and the documented bound was taken.
        $r = Get-FlynnelPeerWatchdog -Ordinal 99
        $r.DriverModelKnown | Should -BeFalse
        $r.DriverModelProblem | Should -Not -BeNullOrEmpty
        $r.DriverModel | Should -Be 'Unknown'
    }

    It 'keeps the delay and the seconds in step' {
        $r = $script:Watchdog
        if ($null -eq $r.DelayNs) {
            $r.DelaySeconds | Should -BeNullOrEmpty
            $r.Applies | Should -BeFalse
        } else {
            $r.Applies | Should -BeTrue
            [math]::Abs($r.DelaySeconds - ($r.DelayNs / 1e9)) | Should -BeLessThan 1e-6
        }
    }

    It 'answers the same reading twice' {
        # The reading is cached for the life of the process, because
        # neither the hardware nor the driver configuration can change
        # under it. Two different answers would mean the cache is keyed
        # wrong.
        $a = Get-FlynnelPeerWatchdog -Ordinal 0
        $b = Get-FlynnelPeerWatchdog -Ordinal 0
        $a.Basis | Should -Be $b.Basis
        $a.DelayNs | Should -Be $b.DelayNs
    }

    It 'names an unreadable model Unknown and says why' {
        $r = $script:Watchdog
        if ($r.DriverModelKnown) {
            $r.DriverModelProblem | Should -BeNullOrEmpty
            $r.DriverModel | Should -Not -Be 'Unknown'
        } else {
            $r.DriverModelProblem | Should -Not -BeNullOrEmpty
            $r.DriverModel | Should -Be 'Unknown'
        }
    }
}

Describe 'Get-FlynnelPeerWatchdog where there is no readable device' {
    It 'treats an unreadable model as covered rather than as absent' {
        # The direction is deliberate and it is the whole safety
        # argument: a watchdog that is present and treated as absent
        # ends in a device reset, while one treated as present only
        # shortens slices. On Windows an unreadable model still takes
        # the documented delay; on a platform with no watchdog at all
        # there is nothing to take.
        $r = Get-FlynnelPeerWatchdog -Ordinal 99
        if ($script:TestHost.Platform -ne 'Windows') {
            Set-ItResult -Skipped -Because 'only Windows has a watchdog to be covered by'
            return
        }
        $r.Applies | Should -BeTrue
        $r.DelaySeconds | Should -BeGreaterThan 0
        $r.Basis | Should -Match 'unreadable'
    }

    It 'says no watchdog is known off Windows' {
        if ($script:TestHost.Platform -eq 'Windows') {
            Set-ItResult -Skipped -Because 'this host is Windows and has TDR'
            return
        }
        $script:Watchdog.Applies | Should -BeFalse
        $script:Watchdog.Basis | Should -Match 'no watchdog is known'
    }
}

Describe 'Get-FlynnelWavePlan' {
    # The cost model alone, so every assertion here holds on a host with
    # no card. The costs are arguments.

    It 'shapes the plan row the way its cmdlet documents' {
        @(Get-FlynnelTypeProperty -TypeName 'Flynnel.WavePlan').Count |
            Should -BeGreaterThan 0
        $names = [enum]::GetNames([Flynnel.Frontier])
        $names | Should -Contain 'Global'
        $names | Should -Contain 'Partition'
    }

    It 'keeps a global frontier when nothing has been observed' {
        # Not a comparison won: there is no imbalance to price a
        # partition against, and a wave run this way records the one the
        # next plan needs. The column says which it was.
        $p = Get-FlynnelWavePlan -Width 32 -BarrierNs 4000 -GenerationNs 90000
        $p.Frontier | Should -Be 'Global'
        $p.ImbalanceSupplied | Should -BeFalse
        $p.CostPerGenerationNs | Should -Be 4000
        $p.GlobalCostNs | Should -Be 4000
        $p.SavingNs | Should -Be 0
    }

    It 'partitions a team whose barrier is dear against a cheap generation' {
        # A barrier costing nearly as much as the generation it guards
        # is the case a partition exists for.
        $p = Get-FlynnelWavePlan -Width 32 -BarrierNs 50000 -GenerationNs 60000 `
            -ImbalancePerMille 1100 -ImbalanceOverGenerations 8 `
            -RebalanceFixedNs 20000 -CopyNsPerId 5 -PendingIds 1000
        $p.Frontier | Should -Be 'Partition'
        $p.CostPerGenerationNs | Should -BeLessThan $p.GlobalCostNs
        $p.SavingNs | Should -BeGreaterThan 0
    }

    It 'keeps a global frontier when the barrier is cheap against the generation' {
        # The other side of the same trade, so the suite is not only
        # testing that the model can say Partition.
        $p = Get-FlynnelWavePlan -Width 32 -BarrierNs 50 -GenerationNs 2000000 `
            -ImbalancePerMille 4000 -ImbalanceOverGenerations 2 `
            -RebalanceFixedNs 500000 -CopyNsPerId 50 -PendingIds 100000
        $p.Frontier | Should -Be 'Global'
        $p.ImbalanceSupplied | Should -BeTrue -Because 'this one was a comparison, not an absence'
    }

    It 'never rebalances a team of one block' {
        # One block has nothing to rebalance with and nothing to idle,
        # so both costs are zero and a barrier would be pure loss.
        $p = Get-FlynnelWavePlan -Width 1 -BarrierNs 4000 -GenerationNs 90000 `
            -ImbalancePerMille 2000 -ImbalanceOverGenerations 4
        $p.Frontier | Should -Be 'Partition'
        $p.RebalancesAtAll | Should -BeFalse
        $p.CostPerGenerationNs | Should -Be 0
    }

    It 'tells a plan that never rebalances from one with no interval to report' {
        # RebalanceEvery is null in both cases and they are different
        # answers, which is why RebalancesAtAll is a column.
        $global = Get-FlynnelWavePlan -Width 32 -BarrierNs 4000 -GenerationNs 90000
        $global.RebalanceEvery | Should -BeNullOrEmpty
        $global.RebalancesAtAll | Should -BeFalse
        $global.Frontier | Should -Be 'Global'
    }

    It 'gives a rebalancing plan a positive interval' {
        $p = Get-FlynnelWavePlan -Width 64 -BarrierNs 50000 -GenerationNs 60000 `
            -ImbalancePerMille 1100 -ImbalanceOverGenerations 8 `
            -RebalanceFixedNs 20000 -CopyNsPerId 5 -PendingIds 1000
        if ($p.RebalancesAtAll) {
            $p.RebalanceEvery | Should -BeGreaterThan 0
        } else {
            $p.RebalanceEvery | Should -BeNullOrEmpty
        }
    }

    It 'echoes the width it planned for' {
        (Get-FlynnelWavePlan -Width 17 -BarrierNs 100 -GenerationNs 1000).Width | Should -Be 17
    }

    It 'refuses half an imbalance' {
        # A ratio with no generation count would quietly become one
        # generation, which prices a divergence as growing far faster
        # than it was seen to.
        { Get-FlynnelWavePlan -Width 32 -BarrierNs 100 -GenerationNs 1000 `
            -ImbalancePerMille 1500 } | Should -Throw -ExpectedMessage '*both*'
        { Get-FlynnelWavePlan -Width 32 -BarrierNs 100 -GenerationNs 1000 `
            -ImbalanceOverGenerations 4 } | Should -Throw -ExpectedMessage '*both*'
    }

    It 'refuses an imbalance over no generations' {
        { Get-FlynnelWavePlan -Width 32 -BarrierNs 100 -GenerationNs 1000 `
            -ImbalancePerMille 1500 -ImbalanceOverGenerations 0 } |
            Should -Throw -ExpectedMessage '*at least one*'
    }

    It 'refuses a width of zero and a cost that is not one' {
        { Get-FlynnelWavePlan -Width 0 -BarrierNs 100 -GenerationNs 1000 } |
            Should -Throw -ExpectedMessage '*above zero*'
        { Get-FlynnelWavePlan -Width 8 -BarrierNs -1 -GenerationNs 1000 } |
            Should -Throw -ExpectedMessage '*at or above zero*'
    }
}

Describe 'New-FlynnelGpuPeerConfig' {
    # A plain settings object, so every assertion holds on any host.

    It 'starts from the crate defaults' {
        $c = New-FlynnelGpuPeerConfig
        $c.Lanes | Should -BeGreaterThan 0
        $c.SlotBytes | Should -BeGreaterThan 0
        $c.SlotsPerLane | Should -BeGreaterThan 0
        $c.QuantumNs | Should -BeGreaterThan 0
    }

    It 'changes only what it was given' {
        $d = New-FlynnelGpuPeerConfig
        $c = New-FlynnelGpuPeerConfig -Lanes 8
        $c.Lanes | Should -Be 8
        $c.SlotBytes | Should -Be $d.SlotBytes
        $c.QuantumNs | Should -Be $d.QuantumNs
    }

    It 'leaves the region path empty unless one is given' {
        # Empty is a per-process file removed when the peer goes. A
        # fixed path is what makes the region attachable by another
        # process, so the two are different intentions.
        (New-FlynnelGpuPeerConfig).RegionPath | Should -BeNullOrEmpty
        (New-FlynnelGpuPeerConfig -RegionPath 'C:\Temp\peer.bin').RegionPath |
            Should -Be 'C:\Temp\peer.bin'
    }

    It 'refuses a per-lane team list that does not cover every lane' {
        # Caught here rather than at init, because init's refusal
        # arrives after a device context has been made and torn down.
        { New-FlynnelGpuPeerConfig -Lanes 4 -LaneTeams @(1, 2) } |
            Should -Throw -ExpectedMessage '*every lane*'
    }

    It 'accepts a per-lane team list that covers every lane' {
        (New-FlynnelGpuPeerConfig -Lanes 3 -LaneTeams @(1, 2, 4)).LaneTeams.Count |
            Should -Be 3
    }

    It 'refuses no lanes at all' {
        { New-FlynnelGpuPeerConfig -Lanes 0 } | Should -Throw -ExpectedMessage '*above zero*'
    }

    It 'has no parameter for user-op source' {
        # The boundary this module does not open: CUDA C from a script,
        # compiled by NVRTC when the peer starts. A parameter appearing
        # here later would be that door opening by accident.
        $names = (Get-Command New-FlynnelGpuPeerConfig).Parameters.Keys
        $names | Should -Not -Contain 'UserOpsCuda'
        $names | Should -Not -Contain 'UserOpsNvrtcOptions'
    }
}

Describe 'the peer lifecycle' {
    It 'says no peer is running before one is started' {
        # A row, not an absent one: a script asking whether a peer
        # exists has to get something it can branch on.
        $p = Get-FlynnelGpuPeer
        $p | Should -Not -BeNullOrEmpty
        $p.Running | Should -BeFalse
    }

    It 'refuses to start without a loadable CUDA driver, and says so' {
        # The refusal path, and the one every deviceless host proves.
        # It has to be a clear error rather than a panic, and it has to
        # come from checking the driver's loadability rather than from
        # calling into the driver and catching what comes back.
        $cuda = Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Available }
        if ($cuda) {
            Set-ItResult -Skipped -Because 'this host has a loadable CUDA driver'
            return
        }
        { New-FlynnelGpuPeer } | Should -Throw -ExpectedMessage '*no loadable CUDA driver*'
    }

    It 'leaves nothing running after a refused start' {
        $cuda = Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Available }
        if ($cuda) {
            Set-ItResult -Skipped -Because 'this host has a loadable CUDA driver'
            return
        }
        try { New-FlynnelGpuPeer } catch { }
        (Get-FlynnelGpuPeer).Running | Should -BeFalse
    }

    It 'removing nothing is not an error' {
        # So a cleanup block does not have to ask first.
        $r = Remove-FlynnelGpuPeer -WarningAction SilentlyContinue
        $r | Should -BeFalse
    }

    It 'warns when there was nothing to remove' {
        Remove-FlynnelGpuPeer -WarningVariable warned | Out-Null
        @($warned).Count | Should -BeGreaterThan 0
    }

    It 'starts, reports and tears down on a host with a device' {
        $cuda = Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Available }
        if (-not $cuda) {
            Set-ItResult -Skipped -Because 'no loadable CUDA driver on this host'
            return
        }
        try {
            $p = New-FlynnelGpuPeer -Config (New-FlynnelGpuPeerConfig -Lanes 2)
            $p.Running | Should -BeTrue
            $p.Lanes | Should -Be 2
            $p.TeamSize | Should -BeGreaterThan 0
            (Get-FlynnelGpuPeer).Running | Should -BeTrue

            # A second peer would contend for the context, the region
            # and the resident kernel, so it is refused rather than
            # quietly made.
            { New-FlynnelGpuPeer } | Should -Throw -ExpectedMessage '*already running*'
        } finally {
            Remove-FlynnelGpuPeer -WarningAction SilentlyContinue | Out-Null
        }
        (Get-FlynnelGpuPeer).Running | Should -BeFalse
    }

    It 'narrows a team wider than the device and says it did' {
        $cuda = Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Available }
        if (-not $cuda) {
            Set-ItResult -Skipped -Because 'no loadable CUDA driver on this host'
            return
        }
        try {
            # Far wider than any device, so the clamp has to fire and
            # the row has to report it rather than only reporting the
            # size that ran.
            $p = New-FlynnelGpuPeer -Config (New-FlynnelGpuPeerConfig -Lanes 1 -BlocksPerLane 4096)
            $p.BlocksPerLaneRequested | Should -Be 4096
            $p.TeamSize | Should -BeLessThan 4096
            $p.TeamNarrowed | Should -BeTrue
        } finally {
            Remove-FlynnelGpuPeer -WarningAction SilentlyContinue | Out-Null
        }
    }
}

Describe 'Get-FlynnelPeerWatchdog where a device is readable' {
    It 'reports the model the device presents' {
        if (-not $script:HasCard) {
            Set-ItResult -Skipped -Because 'no driver model could be read on this host'
            return
        }
        $script:Watchdog.DriverModel | Should -BeIn @('Wddm', 'Tcc', 'Mcdm')
        $script:Watchdog.Basis | Should -Match ('driver model ' + $script:Watchdog.DriverModel)
    }

    It 'bounds work on a covered device and does not on a Tcc one' {
        if (-not $script:HasCard) {
            Set-ItResult -Skipped -Because 'no driver model could be read on this host'
            return
        }
        if ($script:Watchdog.DriverModel -eq 'Tcc') {
            $script:Watchdog.Applies | Should -BeFalse
            $script:Watchdog.Basis | Should -Match 'outside TDR'
        } elseif ($script:TestHost.Platform -eq 'Windows') {
            # Level zero turns detection off, which is a third answer
            # and not the same as an uncovered device.
            if ($script:Watchdog.Applies) {
                $script:Watchdog.Basis | Should -Match 'TdrLevel'
                $script:Watchdog.DelaySeconds | Should -BeGreaterThan 0
            } else {
                $script:Watchdog.Basis | Should -Match 'TdrLevel 0'
            }
        }
    }
}
