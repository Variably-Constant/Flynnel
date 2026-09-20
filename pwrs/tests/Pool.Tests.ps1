# The pool family: that the arena reports a consistent shape, that
# every dial reads back what it took, and that a counter which has
# measured nothing says so.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
    $script:pool = Start-FlynnelPool
}

Describe 'Start-FlynnelPool and Get-FlynnelPool' {
    It 'reports at least one primary worker' {
        $script:pool.PrimaryWorkers | Should -BeGreaterThan 0
    }

    It 'never has more primaries than workers' {
        $script:pool.PrimaryWorkers | Should -BeLessOrEqual $script:pool.TotalWorkers
    }

    It 'accounts for every worker as a primary or an extension' {
        ($script:pool.PrimaryWorkers + $script:pool.SmtExtensionWorkers) |
            Should -Be $script:pool.TotalWorkers
    }

    It 'agrees with the topology about the node count' {
        $script:pool.NodeCount | Should -BeGreaterThan 0
        $script:pool.IsSingleNode | Should -Be ($script:pool.NodeCount -eq 1)
    }

    It 'is safe to call twice and answers the same' {
        (Start-FlynnelPool).TotalWorkers | Should -Be $script:pool.TotalWorkers
        (Get-FlynnelPool).TotalWorkers | Should -Be $script:pool.TotalWorkers
    }

    It 'says whether the burst ratio has anything behind it' {
        # The ratio starts at 0.5 with nothing pushed, which is not a
        # measurement of a half-burst workload. HasPushed is what
        # tells the two apart, and without it the starting value would
        # be quoted as a reading.
        $script:pool.BurstRatio | Should -BeGreaterOrEqual 0.0
        $script:pool.BurstRatio | Should -BeLessOrEqual 1.0
        $script:pool.HasPushed | Should -BeOfType [bool]
    }

    It 'answers to its Fly alias' {
        (Get-FlyPool).TotalWorkers | Should -Be $script:pool.TotalWorkers
    }
}

Describe 'the types this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.WorkerStat', 'Flynnel.SpinState', 'Flynnel.SplitState') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }
}

Describe 'Get-FlynnelWorker' {
    It 'writes one row per worker, in one call' {
        @(Get-FlynnelWorker).Count | Should -Be $script:pool.TotalWorkers
    }

    It 'leaves the external slots out unless asked, and marks them when asked' {
        # The pool's statistics table runs past its workers into the
        # slots a foreign thread pushes through. Those rows are real
        # and they are not workers, and unmarked they read as a set of
        # permanently idle workers that do not exist.
        $workers = @(Get-FlynnelWorker)
        $all = @(Get-FlynnelWorker -IncludeExternalSlot)
        $all.Count | Should -BeGreaterOrEqual $workers.Count
        ($workers | Where-Object { -not $_.IsWorker }).Count | Should -Be 0
        if ($all.Count -gt $workers.Count) {
            ($all | Where-Object { -not $_.IsWorker }).Count |
                Should -Be ($all.Count - $workers.Count)
        }
    }

    It 'numbers them from zero without a gap' {
        $indices = @(Get-FlynnelWorker | ForEach-Object Index | Sort-Object)
        $indices | Should -Be @(0..($script:pool.TotalWorkers - 1))
    }

    It 'never reports a negative or missing counter' {
        foreach ($w in Get-FlynnelWorker) {
            $w.LocalPops | Should -BeGreaterOrEqual 0
            $w.PeerStealHits | Should -BeGreaterOrEqual 0
            $w.PeerStealMisses | Should -BeGreaterOrEqual 0
            $w.TimesStolenFrom | Should -BeGreaterOrEqual 0
            $w.PushRefusals | Should -BeGreaterOrEqual 0
            $w.BurstRatio | Should -BeGreaterOrEqual 0.0
            $w.BurstRatio | Should -BeLessOrEqual 1.0
        }
    }
}

Describe 'the spin dials' {
    AfterAll {
        # Leave the session as it was found: these are process-wide
        # and a suite that changed one would change every later suite
        # in the same run.
        Set-FlynnelSpinAdaptive -On | Out-Null
    }

    It 'reads back the window it was set to' {
        $set = Set-FlynnelSpinWindow -Rounds 64
        $set.WindowRounds | Should -Be 64
        (Get-FlynnelSpinWindow).WindowRounds | Should -Be 64
    }

    It 'holds a pinned window while the controller is off' {
        Set-FlynnelSpinAdaptive -Off | Out-Null
        $null = Set-FlynnelSpinWindow -Rounds 8
        (Get-FlynnelSpinWindow).WindowRounds | Should -Be 8
    }

    It 'only ever adds to the idle-yield counter until it is reset' {
        $before = (Get-FlynnelSpinWindow).TotalIdleYields
        Start-Sleep -Milliseconds 50
        $after = (Get-FlynnelSpinWindow).TotalIdleYields
        $after | Should -BeGreaterOrEqual $before
    }

    It 'zeroes the counter on reset' {
        (Reset-FlynnelSpinStats).TotalIdleYields | Should -Be 0
    }
}

Describe 'the split dials' {
    It 'reads back the multiplier it was set to, within the clamp' {
        $set = Set-FlynnelSplitMultiplier -Value 4
        $set.Multiplier | Should -Be 4
        (Get-FlynnelSplitMultiplier).Multiplier | Should -Be 4
    }

    It 'clamps out of range and says so rather than accepting it' {
        $warnings = @()
        $set = Set-FlynnelSplitMultiplier -Value 99 -WarningVariable warnings
        $set.Multiplier | Should -BeLessOrEqual 8
        $warnings.Count | Should -BeGreaterThan 0
    }

    It 'reports an unmeasured window as nothing, not as zero' {
        # A window with too few leaves cannot support a spread. Zero
        # would read as a perfectly uniform workload, which is the
        # opposite of what an empty window means.
        $s = Reset-FlynnelSplitStats
        $s.LeafCount | Should -Be 0
        $s.MeanLeafNs | Should -BeNullOrEmpty
        $s.PerItemNs | Should -BeNullOrEmpty
        $s.PerItemCv2PerMille | Should -BeNullOrEmpty
        $s.LeafCv2PerMille | Should -BeNullOrEmpty
    }

    It 'starts the observer once and is safe to call again' {
        $null = Start-FlynnelSplitObserver
        { Start-FlynnelSplitObserver } | Should -Not -Throw
    }
}

Describe 'the IO pool' {
    It 'reports the worker count it was made with' {
        $io = New-FlynnelIoPool -WorkerCount 2
        $io.WorkerCount | Should -Be 2
        $io.Workers() | Should -Be 2
    }

    It 'refuses a pool with no workers' {
        { New-FlynnelIoPool -WorkerCount 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*at least one*'
    }

    It 'says when the process has no global pool rather than writing an empty one' {
        $warnings = @()
        $global = Get-FlynnelIoPool -WarningVariable warnings
        if ($null -eq $global) {
            $warnings.Count | Should -BeGreaterThan 0
        } else {
            $global.WorkerCount | Should -BeGreaterThan 0
        }
    }
}
