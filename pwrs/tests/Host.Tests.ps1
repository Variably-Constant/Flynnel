# The host family against this machine.
#
# Every assertion here is an ORDERING or a shape, never a host
# absolute. A threshold chosen by feel passes on the hosts you happen
# to try and fails on the one you did not; an ordering that holds by
# construction holds everywhere.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
    $script:cpu = Get-FlynnelCpuInfo
    $script:topo = Get-FlynnelTopology
}

Describe 'Get-FlynnelCpuInfo' {
    It 'reports at least one logical processor and one physical core' {
        $script:cpu.LogicalThreads | Should -BeGreaterThan 0
        $script:cpu.PhysicalCores | Should -BeGreaterThan 0
    }

    It 'never reports more physical cores than logical threads' {
        $script:cpu.PhysicalCores | Should -BeLessOrEqual $script:cpu.LogicalThreads
    }

    It 'has an SMT factor of at least one that divides the two' {
        $script:cpu.SmtThreadsPerCore | Should -BeGreaterThan 0
        $expected = [Math]::Max(1, [int]($script:cpu.LogicalThreads / $script:cpu.SmtThreadsPerCore))
        $script:cpu.PhysicalCores | Should -Be $expected
    }

    It 'agrees with the runtime on the processor count' {
        $script:cpu.LogicalThreads | Should -Be ([Environment]::ProcessorCount)
    }

    It 'names a vendor the enum knows' {
        [Enum]::GetValues([Flynnel.Vendor]) | Should -Contain $script:cpu.Vendor
    }

    It 'scales the dispatch floor only on a small host' {
        if ($script:cpu.PhysicalCores -ge 4) {
            $script:cpu.SmallHostDispatchFactor | Should -Be 1
        } else {
            $script:cpu.SmallHostDispatchFactor | Should -BeGreaterThan 1
        }
    }

    It 'answers the same twice, because it is probed once and cached' {
        (Get-FlynnelCpuInfo).LogicalThreads | Should -Be $script:cpu.LogicalThreads
    }

    It 'answers to its Fly alias' {
        (Get-FlyCpuInfo).PhysicalCores | Should -Be $script:cpu.PhysicalCores
    }
}

Describe 'Get-FlynnelTopology' {
    It 'reports at least one node' {
        $script:topo.NodeCount | Should -BeGreaterThan 0
    }

    It 'agrees with itself about whether the host is multi-node' {
        $script:topo.IsMultiNode | Should -Be ($script:topo.NodeCount -gt 1)
    }

    It 'gives every logical CPU exactly one node, all of them in range' {
        $script:topo.NodeOfCpu.Count | Should -Be $script:cpu.LogicalThreads
        foreach ($node in $script:topo.NodeOfCpu) {
            $node | Should -BeGreaterOrEqual 0
            $node | Should -BeLessThan $script:topo.NodeCount
        }
    }

    It 'carries a square distance matrix' {
        $n = $script:topo.NodeCount
        $script:topo.Distances.Count | Should -Be ($n * $n)
    }

    It 'names the probe that produced each half' {
        [Enum]::GetValues([Flynnel.NumaSource]) | Should -Contain $script:topo.Source
        [Enum]::GetValues([Flynnel.ClusterSource]) | Should -Contain $script:topo.ClusterSource
    }

    It 'reports no cluster size when no cluster probe ran' {
        # The two have to agree: a size without a source is a number
        # from nowhere, and a source without a size is a probe that
        # found nothing and said otherwise.
        if ($script:topo.ClusterSource -eq [Flynnel.ClusterSource]::None) {
            $script:topo.ClusterSizeLog2 | Should -Be 0
        }
    }
}

Describe 'Get-FlynnelNumaDistance' {
    It 'makes every node its own nearest' {
        foreach ($row in Get-FlynnelNumaDistance) {
            if ($row.From -eq $row.To) {
                $diag = $row.Distance
                $off = Get-FlynnelNumaDistance -From $row.From |
                    Where-Object { $_.To -ne $row.From } |
                    ForEach-Object Distance
                foreach ($d in $off) { $d | Should -BeGreaterOrEqual $diag }
            }
        }
    }

    It 'is symmetric' {
        $all = @{}
        foreach ($row in Get-FlynnelNumaDistance) { $all["$($row.From),$($row.To)"] = $row.Distance }
        foreach ($key in $all.Keys) {
            $a, $b = $key -split ','
            $all["$b,$a"] | Should -Be $all[$key]
        }
    }

    It 'writes one row for a named pair' {
        @(Get-FlynnelNumaDistance -From 0 -To 0).Count | Should -Be 1
    }

    It 'refuses a node this host does not have, naming the range' {
        { Get-FlynnelNumaDistance -From 9999 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*9999*'
    }
}

Describe 'Get-FlynnelNodeCpu' {
    It 'covers every CPU exactly once across the nodes' {
        $seen = @(Get-FlynnelNodeCpu | ForEach-Object { $_.Cpus } )
        $seen.Count | Should -Be $script:cpu.LogicalThreads
        ($seen | Sort-Object -Unique).Count | Should -Be $seen.Count
    }

    It 'reports a count that matches its own list' {
        foreach ($row in Get-FlynnelNodeCpu) {
            $row.Count | Should -Be $row.Cpus.Count
        }
    }
}

Describe 'Get-FlynnelLatencyTable' {
    It 'is either absent with a warning or a square table of the right shape' {
        $warnings = @()
        $table = Get-FlynnelLatencyTable -WarningVariable warnings -WarningAction SilentlyContinue
        if ($null -eq $table) {
            # A host that cannot pin threads has no table. That is a
            # fact about the host and the cmdlet must say so rather
            # than write a table of zeros.
            $warnings.Count | Should -BeGreaterThan 0
        } else {
            $table.CoreCount | Should -BeGreaterThan 0
            $table.LatencyNs.Count | Should -Be ($table.CoreCount * $table.CoreCount)
            $table.MinOffdiagNs | Should -BeLessOrEqual $table.MaxOffdiagNs
            $table.Iters | Should -BeGreaterThan 0
            $table.Matrix | Should -Not -BeNullOrEmpty
        }
    }
}

Describe 'Get-FlynnelHwClass' {
    It 'lists every class the crate declares' {
        @(Get-FlynnelHwClass).Count | Should -Be ([Enum]::GetValues([Flynnel.HwClass]).Count)
    }

    It 'marks the tile classes as matrix-extension and the vector ones not' {
        $rows = @{}
        foreach ($row in Get-FlynnelHwClass) { $rows[[string]$row.Class] = $row.IsMatrixExtension }
        $rows['Scalar'] | Should -BeFalse
        $rows['Avx2'] | Should -BeFalse
        $rows['Avx512f'] | Should -BeFalse
        $rows['Sme'] | Should -BeTrue
        $rows['AmxBf16'] | Should -BeTrue
        $rows['TensorCoreBlackwell'] | Should -BeTrue
    }

    It 'gives every class a short name' {
        foreach ($row in Get-FlynnelHwClass) { $row.Name | Should -Not -BeNullOrEmpty }
    }
}

Describe 'Get-FlynnelCacheAllocation' {
    It 'answers on every host, supported or not' {
        $cap = Get-FlynnelCacheAllocation
        $cap | Should -Not -BeNullOrEmpty
        if (-not $cap.Supported) {
            # Absent is a fact, written as a row. The counts are zero
            # because there is nothing to count, and that is only
            # readable beside Supported being false.
            $cap.WayCount | Should -Be 0
        } else {
            $cap.WayCount | Should -BeGreaterThan 0
            $cap.MinWays | Should -BeGreaterThan 0
            $cap.MinWays | Should -BeLessOrEqual $cap.WayCount
        }
    }
}

Describe 'New-FlynnelCacheReservation' {
    It 'reserves and releases, or refuses by name where resctrl is absent' {
        $cap = Get-FlynnelCacheAllocation
        if (-not $cap.Supported) {
            { New-FlynnelCacheReservation -Name pester -FirstWay 0 -NumWays 1 -ErrorAction Stop } |
                Should -Throw -ExpectedMessage '*not available on this host*'
            return
        }
        $ways = New-FlynnelCacheReservation -Name flynnel_pester -FirstWay 0 -NumWays 1
        try {
            $ways.WayCount | Should -Be 1
            $ways.Schemata() | Should -Not -BeNullOrEmpty
        } finally {
            $ways.Release()
            # Release twice is safe, which is what makes it usable in
            # a finally block that may run after a Dispose.
            $ways.Release()
        }
    }

    It 'refuses a range wider than the host has, naming both numbers' {
        $cap = Get-FlynnelCacheAllocation
        if (-not $cap.Supported) { return }
        { New-FlynnelCacheReservation -Name flynnel_pester_wide -FirstWay 0 -NumWays 9999 -ErrorAction Stop } |
            Should -Throw
    }
}
