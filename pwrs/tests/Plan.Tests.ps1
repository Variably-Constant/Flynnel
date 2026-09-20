# The plan family: that a plan carries what it was told, that the
# builders compose without consuming, and that the one-call snapshot
# agrees with the methods it summarises.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
}

Describe 'New-FlynnelPlan' {
    It 'carries the size and batch it was built with' {
        $p = New-FlynnelPlan -KOuter 8 -BatchSize 100000
        $p.KOuter() | Should -Be 8
        $p.BatchSize() | Should -Be 100000
    }

    It 'records that a profile was named, and applies what it implies' {
        # A plan does not keep the profile: naming one sets the SMT
        # request, the cost estimate and the oversubscription and is
        # dissolved into them. So the readable fact is that one was
        # named, and the evidence it took is in those three.
        $named = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile Streaming
        $unnamed = New-FlynnelPlan -KOuter 8 -BatchSize 100000
        $named.ProfileExplicit() | Should -BeTrue
        $unnamed.ProfileExplicit() | Should -BeFalse
        $named.EffectiveUseSmt() | Should -BeFalse
        (New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile LatencyBound).EffectiveUseSmt() |
            Should -BeTrue
    }

    It 'refuses Bare and Profile together rather than silently preferring one' {
        { New-FlynnelPlan -KOuter 8 -BatchSize 1000 -Bare -Profile Streaming -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*Pass one or neither*'
    }

    It 'refuses a worker count of zero' {
        { New-FlynnelPlan -KOuter 8 -BatchSize 1000 -Workers 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*at least one*'
    }

    It 'warns when only one half of the leaf-width model is given' {
        $warnings = @()
        $null = New-FlynnelPlan -KOuter 8 -BatchSize 1000 -PerItemNs 200 `
            -WarningVariable warnings -WarningAction SilentlyContinue
        $warnings.Count | Should -BeGreaterThan 0
    }

    It 'does not warn when both halves are given' {
        $warnings = @()
        $null = New-FlynnelPlan -KOuter 8 -BatchSize 1000 -PerItemNs 200 -TaskOverheadNs 900 `
            -WarningVariable warnings -WarningAction SilentlyContinue
        $warnings.Count | Should -Be 0
    }

    It 'answers to its Fly alias' {
        (New-FlyPlan -KOuter 4 -BatchSize 16).KOuter() | Should -Be 4
    }
}

Describe 'the builders' {
    It 'answer a fresh plan and leave the original alone' {
        $base = New-FlynnelPlan -KOuter 8 -BatchSize 1000
        $pinned = $base.WithWorkers(3)
        $pinned.CallerPinned() | Should -BeTrue
        $base.CallerPinned() | Should -BeFalse
    }

    It 'compose in a chain' {
        $p = (New-FlynnelPlan -KOuter 8 -BatchSize 1000).WithSmt().WithWorkers(4).WithNumaHint(0)
        $p.CallerPinned() | Should -BeTrue
    }

    It 'carry each value back out' {
        $p = (New-FlynnelPlan -KOuter 8 -BatchSize 1000).
            WithHwClass([Flynnel.HwClass]::Avx2).
            WithVariant([Flynnel.Variant]::Faithful).
            WithLeafShape([Flynnel.LeafShape]::Gather).
            WithDequeTierHint([Flynnel.DequeTier]::IntraCcx).
            WithMailboxRouting($true)
        $p.HwClass() | Should -Be ([Flynnel.HwClass]::Avx2)
        $p.Variant() | Should -Be ([Flynnel.Variant]::Faithful)
        $p.LeafShape() | Should -Be ([Flynnel.LeafShape]::Gather)
        $p.DequeTierHint() | Should -Be ([Flynnel.DequeTier]::IntraCcx)
        $p.UseMailboxRouting() | Should -BeTrue
    }
}

Describe 'the leaf-width model' {
    It 'answers nothing until both its inputs are set' {
        $workers = (Get-FlynnelPool).PrimaryWorkers
        $neither = New-FlynnelPlan -KOuter 8 -BatchSize 100000
        $neither.OptimalChunkCount($workers) | Should -BeNullOrEmpty

        $one = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 `
            -WarningAction SilentlyContinue
        $one.OptimalChunkCount($workers) | Should -BeNullOrEmpty
    }

    It 'answers a count once both are set' {
        $workers = (Get-FlynnelPool).PrimaryWorkers
        $both = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 -TaskOverheadNs 900
        $both.OptimalChunkCount($workers) | Should -BeGreaterThan 0
    }
}

Describe 'the SMT decision' {
    It 'follows the profile, not only the caller' {
        # A latency-bound profile wants the siblings and a streaming
        # one does not, whatever the caller asked. That is the
        # decision the profile table exists to make.
        $latency = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile LatencyBound
        $streaming = (New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile Streaming).WithSmt()
        $latency.EffectiveUseSmt() | Should -BeTrue
        $streaming.EffectiveUseSmt() | Should -BeFalse
    }
}

Describe 'Resolve-FlynnelPlan' {
    BeforeAll {
        $script:plan = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 -TaskOverheadNs 900
        $script:r = Resolve-FlynnelPlan -Plan $script:plan
    }

    It 'agrees field for field with the methods it summarises' {
        # The snapshot is a decomposition of the same state. If it
        # ever disagrees with the methods, one of them is lying and
        # the table would be quoted as though it were not.
        $script:r.KOuter | Should -Be $script:plan.KOuter()
        $script:r.BatchSize | Should -Be $script:plan.BatchSize()
        $script:r.ProfileExplicit | Should -Be $script:plan.ProfileExplicit()
        $script:r.Variant | Should -Be $script:plan.Variant()
        $script:r.HwClass | Should -Be $script:plan.HwClass()
        $script:r.LeafShape | Should -Be $script:plan.LeafShape()
        $script:r.Tier | Should -Be $script:plan.Tier()
        $script:r.CallerPinned | Should -Be $script:plan.CallerPinned()
        $script:r.EffectiveUseSmt | Should -Be $script:plan.EffectiveUseSmt()
        $script:r.ResolvedWorkers | Should -Be $script:plan.ResolvedWorkers()
        $script:r.EffectiveLeavesPerWorker | Should -Be $script:plan.EffectiveLeavesPerWorker()
        $script:r.EstimatedTotalNs | Should -Be $script:plan.EstimatedTotalNs()
        $script:r.Backend | Should -Be $script:plan.BackendName()
    }

    It 'resolves at least one worker and no more than the pool has' {
        $script:r.ResolvedWorkers | Should -BeGreaterThan 0
        $script:r.ResolvedWorkers | Should -BeLessOrEqual (Get-FlynnelPool).TotalWorkers
    }

    It 'takes a plan from the pipeline' {
        (New-FlynnelPlan -KOuter 8 -BatchSize 1000 | Resolve-FlynnelPlan).KOuter | Should -Be 8
    }

    It 'gives the same answer for two plans built the same way' {
        $a = Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 9 -BatchSize 50000)
        $b = Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 9 -BatchSize 50000)
        $a.ResolvedWorkers | Should -Be $b.ResolvedWorkers
        $a.Tier | Should -Be $b.Tier
    }
}

Describe 'Get-FlynnelKBand' {
    It 'never goes down as the size goes up' {
        $order = @{}
        $i = 0
        foreach ($t in [Enum]::GetValues([Flynnel.SchedTier])) { $order[[string]$t] = $i; $i++ }
        $previous = -1
        foreach ($k in 0..14) {
            $rank = $order[[string](Get-FlynnelKBand -KOuter $k)]
            $rank | Should -BeGreaterOrEqual $previous
            $previous = $rank
        }
    }

    It 'puts the smallest sizes inline' {
        Get-FlynnelKBand -KOuter 0 | Should -Be ([Flynnel.SchedTier]::Inline)
    }
}

Describe 'Get-FlynnelDispatchProfile' {
    It 'has a row for every profile' {
        @(Get-FlynnelDispatchProfile).Count |
            Should -Be ([Enum]::GetValues([Flynnel.DispatchProfile]).Count)
    }

    It 'wakes the siblings for exactly the profiles that want them' {
        $rows = @{}
        foreach ($row in Get-FlynnelDispatchProfile) { $rows[[string]$row.Profile] = $row }
        $rows['LatencyBound'].IsLatencyBound | Should -BeTrue
        $rows['MemoryBound'].IsLatencyBound | Should -BeTrue
        $rows['PortBound'].IsLatencyBound | Should -BeFalse
        $rows['Streaming'].IsLatencyBound | Should -BeFalse
    }

    It 'gives Unspecified no cost estimate, rather than a zero one' {
        $rows = @{}
        foreach ($row in Get-FlynnelDispatchProfile) { $rows[[string]$row.Profile] = $row }
        $rows['Unspecified'].DefaultNsPerElem | Should -BeNullOrEmpty
        $rows['PortBound'].DefaultNsPerElem | Should -BeGreaterThan 0
    }
}
