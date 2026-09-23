# The plan family: that a plan carries what it was told, that a change
# answers a copy, and that a plan reaches the cmdlets that take one.
#
# A plan is a value rather than a fluent object, because the binding
# cannot pass an object holding Rust-only state back into a cmdlet. So
# the assertion that matters most is the last group: a plan built here
# and handed to a kernel actually governs that kernel.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
}

Describe 'the types this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        $shape = @{
            'Flynnel.JobPlan'      = @('KOuter', 'BatchSize', 'Profile', 'Workers', 'Shape')
            'Flynnel.ResolvedPlan' = @('Tier', 'ResolvedWorkers', 'CallerPinned',
                                       'EffectiveUseSmt', 'OptimalChunkCount', 'Backend')
            'Flynnel.ProfileRow'   = @('Profile', 'IsLatencyBound', 'DefaultNsPerElem',
                                       'DequeTierHint')
        }
        foreach ($type in $shape.Keys) {
            $properties = @(Get-FlynnelTypeProperty -TypeName $type | ForEach-Object Name)
            foreach ($wanted in $shape[$type]) {
                $properties | Should -Contain $wanted -Because "$type must carry $wanted"
            }
        }
    }

    It 'gives every plan enum at least one value' {
        foreach ($name in 'Flynnel.BisectVariant', 'Flynnel.CooperativeRouting',
                          'Flynnel.VariantRouting', 'Flynnel.WorkloadShape') {
            $type = $name -as [type]
            $type | Should -Not -BeNullOrEmpty -Because "$name must be exported"
            [Enum]::GetValues($type).Count | Should -BeGreaterThan 0
        }
    }

    It 'takes every bisect variant and cooperative routing the enums declare' {
        # A value the enum names and the plan refuses would bind and
        # then fail at whichever kernel first used the plan.
        $base = New-FlynnelPlan -KOuter 8 -BatchSize 1000
        foreach ($v in [Enum]::GetValues([Flynnel.BisectVariant])) {
            (Update-FlynnelPlan -Plan $base -BisectVariant $v).BisectVariant | Should -Be $v
        }
        foreach ($r in [Enum]::GetValues([Flynnel.CooperativeRouting])) {
            (Update-FlynnelPlan -Plan $base -CooperativeRouting $r).CooperativeRouting |
                Should -Be $r
        }
    }
}

Describe 'New-FlynnelPlan' {
    It 'carries the size and batch it was built with' {
        $p = New-FlynnelPlan -KOuter 8 -BatchSize 100000
        $p.KOuter | Should -Be 8
        $p.BatchSize | Should -Be 100000
    }

    It 'records the profile it was given, and nothing when given none' {
        (New-FlynnelPlan -KOuter 8 -BatchSize 1000 -Profile Streaming).Profile |
            Should -Be ([Flynnel.DispatchProfile]::Streaming)
        (New-FlynnelPlan -KOuter 8 -BatchSize 1000).Profile | Should -BeNullOrEmpty
    }

    It 'leaves every option it was not given unset' {
        # An option nobody gave must be null and not a zero, or a plan
        # would silently carry a cost estimate of nought.
        $p = New-FlynnelPlan -KOuter 8 -BatchSize 1000
        foreach ($name in 'Workers', 'NumaHint', 'HwClass', 'Variant', 'LeafShape',
                          'PerItemNs', 'TaskOverheadNs', 'Shape', 'DequeTierHint') {
            $p.$name | Should -BeNullOrEmpty -Because "$name was not given"
        }
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
            -WarningVariable warnings
        $warnings.Count | Should -BeGreaterThan 0
    }

    It 'does not warn when both halves are given' {
        $warnings = @()
        $null = New-FlynnelPlan -KOuter 8 -BatchSize 1000 -PerItemNs 200 -TaskOverheadNs 900 `
            -WarningVariable warnings
        $warnings.Count | Should -Be 0
    }

    It 'answers to its Fly alias' {
        (New-FlyPlan -KOuter 4 -BatchSize 16).KOuter | Should -Be 4
    }
}

Describe '[Flynnel.JobPlan]::new' {
    # The constructor and the cmdlet run the same Rust function, so a
    # plan built either way must agree on every property. One case per
    # arity, each one argument longer than the last.

    It 'builds what New-FlynnelPlan builds from <Name>' -TestCases @(
        @{ Name = 'the two required arguments'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 } }
        @{ Name = 'a profile'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, [Flynnel.DispatchProfile]::Streaming) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile Streaming } }
        @{ Name = 'Bare'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, $null, $true) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Bare } }
        @{ Name = 'a per-item cost alone'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, $null, $false, 200) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 `
                          -WarningAction SilentlyContinue } }
        @{ Name = 'both halves of the leaf-width model'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, $null, $false, 200, 900) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 -TaskOverheadNs 900 } }
        @{ Name = 'a leaf shape'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, $null, $false, $null, $null,
                        [Flynnel.LeafShape]::Gather) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -LeafShape Gather } }
        @{ Name = 'the siblings'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, $null, $false, $null, $null, $null, $true) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Smt } }
        @{ Name = 'a worker count'
           Ctor = { [Flynnel.JobPlan]::new(8, 100000, $null, $false, $null, $null, $null,
                        $false, 3) }
           Cmdlet = { New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Workers 3 } }
    ) {
        $built = & $Ctor
        $built | Should -BeOfType [Flynnel.JobPlan]
        Get-FlynnelPropertyDifference -Left $built -Right (& $Cmdlet) | Should -BeNullOrEmpty
    }

    It 'requires the size and the batch, as the cmdlet does' {
        # No parameterless constructor is left to make a plan of zeros.
        { [Flynnel.JobPlan]::new() } | Should -Throw
        { [Flynnel.JobPlan]::new(8) } | Should -Throw
        foreach ($ctor in [Flynnel.JobPlan].GetConstructors()) {
            @($ctor.GetParameters() | Where-Object { -not $_.IsOptional }).Count |
                Should -Be 2 -Because 'KOuter and BatchSize are the two required arguments'
        }
    }

    It 'refuses Bare and Profile together' {
        { [Flynnel.JobPlan]::new(8, 1000, [Flynnel.DispatchProfile]::Streaming, $true) } |
            Should -Throw -ExpectedMessage '*Pass one or neither*'
    }

    It 'refuses a worker count of zero' {
        { [Flynnel.JobPlan]::new(8, 1000, $null, $false, $null, $null, $null, $false, 0) } |
            Should -Throw -ExpectedMessage '*at least one*'
    }

    It 'builds a per-item cost alone as asked, without a warning' {
        # A constructor has no pipeline to warn on; only the cmdlet says
        # that the leaf-width model then has nothing to solve.
        $out = @(& { [Flynnel.JobPlan]::new(8, 1000, $null, $false, 200) } 3>&1)
        @($out | Where-Object { $_ -is [System.Management.Automation.WarningRecord] }).Count |
            Should -Be 0
        $plan = @($out | Where-Object { $_ -is [Flynnel.JobPlan] })
        $plan.Count | Should -Be 1
        $plan[0].PerItemNs | Should -Be 200
        $plan[0].TaskOverheadNs | Should -BeNullOrEmpty
    }

    It 'governs a kernel it is handed to' {
        $data = 1..2000 | ForEach-Object { [double]$_ }
        $pinned = [Flynnel.JobPlan]::new(10, 2000, $null, $false, $null, $null, $null, $false, 1)
        $got = Invoke-FlynnelMap -InputObject $data -Operation Square -Plan $pinned
        $want = $data | ForEach-Object { $_ * $_ }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }
}

Describe 'Update-FlynnelPlan' {
    It 'answers a copy and leaves the original alone' {
        $base = New-FlynnelPlan -KOuter 8 -BatchSize 1000
        $changed = Update-FlynnelPlan -Plan $base -Workers 3
        $changed.Workers | Should -Be 3
        $base.Workers | Should -BeNullOrEmpty
    }

    It 'takes a plan from the pipeline' {
        $p = New-FlynnelPlan -KOuter 8 -BatchSize 1000 | Update-FlynnelPlan -Smt
        $p.Smt | Should -BeTrue
    }

    It 'leaves an option it was not given as it was' {
        $base = Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -Workers 4 -Variant Faithful
        $next = Update-FlynnelPlan -Plan $base -Workers 6
        $next.Workers | Should -Be 6
        $next.Variant | Should -Be ([Flynnel.Variant]::Faithful)
    }

    It 'carries every value back out' {
        $p = Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -HwClass Avx2 -Variant Faithful -LeafShape Gather `
            -DequeTierHint IntraCcx -MailboxRouting
        $p.HwClass | Should -Be ([Flynnel.HwClass]::Avx2)
        $p.Variant | Should -Be ([Flynnel.Variant]::Faithful)
        $p.LeafShape | Should -Be ([Flynnel.LeafShape]::Gather)
        $p.DequeTierHint | Should -Be ([Flynnel.DequeTier]::IntraCcx)
        $p.MailboxRouting | Should -BeTrue
    }

    It 'composes, each call answering a further copy' {
        $p = New-FlynnelPlan -KOuter 8 -BatchSize 1000 |
            Update-FlynnelPlan -Smt |
            Update-FlynnelPlan -Workers 4 |
            Update-FlynnelPlan -NumaHint 0
        $p.Smt | Should -BeTrue
        $p.Workers | Should -Be 4
        $p.NumaHint | Should -Be 0
    }

    It 'refuses a worker count of zero' {
        { Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -Workers 0 -ErrorAction Stop } | Should -Throw -ExpectedMessage '*at least one*'
    }
}

Describe 'workload shapes' {
    It 'takes a shape that needs no numbers' {
        $p = Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -Shape Streaming
        $p.Shape | Should -Be ([Flynnel.WorkloadShape]::Streaming)
    }

    It 'takes a shape with its numbers' {
        $p = Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -Shape WorkSteal -ShapeConsumers 8 -ShapeBatchSize 64
        $p.ShapeConsumers | Should -Be 8
        $p.ShapeBatchSize | Should -Be 64
    }

    It 'refuses a shape whose numbers are missing rather than using zero' {
        # A shape is a name plus the numbers it needs. A zero consumer
        # count is a number nobody gave, and it would dispatch.
        { Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -Shape WorkSteal -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*ShapeConsumers*'
        { Update-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000) `
            -Shape ProducerFast -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*ShapeBurst*'
    }
}

Describe 'Resolve-FlynnelPlan' {
    It 'carries the plan''s own inputs through unchanged' {
        $plan = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 -TaskOverheadNs 900
        $r = Resolve-FlynnelPlan -Plan $plan
        $r.KOuter | Should -Be $plan.KOuter
        $r.BatchSize | Should -Be $plan.BatchSize
    }

    It 'says a profile was named when one was' {
        (Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000 `
            -Profile Streaming)).ProfileExplicit | Should -BeTrue
        (Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000)).ProfileExplicit |
            Should -BeFalse
    }

    It 'resolves the worker count the caller pinned' {
        $r = Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Workers 3)
        $r.ResolvedWorkers | Should -Be 3
        $r.CallerPinned | Should -BeOfType [bool]
    }

    It 'resolves at least one worker and no more than the pool has' {
        $r = Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 100000)
        $r.ResolvedWorkers | Should -BeGreaterThan 0
        $r.ResolvedWorkers | Should -BeLessOrEqual (Get-FlynnelPool).TotalWorkers
    }

    It 'gives a caller who asks for the siblings the siblings' {
        # The profile decides SMT for a plan that does not say. A plan
        # that does say is the caller's statement and wins.
        (Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 100000 `
            -Profile LatencyBound)).EffectiveUseSmt | Should -BeTrue
        (Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 100000 `
            -Profile Streaming -Smt)).EffectiveUseSmt | Should -BeTrue
    }

    It 'answers nothing for the leaf width until both its inputs are set' {
        (Resolve-FlynnelPlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 100000)).OptimalChunkCount |
            Should -BeNullOrEmpty
        $one = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 `
            -WarningAction SilentlyContinue
        (Resolve-FlynnelPlan -Plan $one).OptimalChunkCount | Should -BeNullOrEmpty
    }

    It 'answers a count once both are set' {
        $both = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -PerItemNs 200 -TaskOverheadNs 900
        (Resolve-FlynnelPlan -Plan $both).OptimalChunkCount | Should -BeGreaterThan 0
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

Describe 'a plan reaches the cmdlets that take one' {
    It 'governs a kernel it is handed to' {
        # The whole reason a plan is a value. A plan that no cmdlet can
        # accept is a plan nobody can use.
        $data = 1..2000 | ForEach-Object { [double]$_ }
        $pinned = New-FlynnelPlan -KOuter 10 -BatchSize 2000 -Workers 1
        $got = Invoke-FlynnelMap -InputObject $data -Operation Square -Plan $pinned
        $want = $data | ForEach-Object { $_ * $_ }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'is reported on the verbose stream by the kernel that ran it' {
        $plan = New-FlynnelPlan -KOuter 10 -BatchSize 1000 -Workers 1
        $verbose = Invoke-FlynnelMap -InputObject (1..1000) -Operation Abs -Plan $plan -Verbose 4>&1 |
            Where-Object { $_ -is [System.Management.Automation.VerboseRecord] }
        ($verbose -join ' ') | Should -Match '1 worker\(s\)'
    }

    It 'refuses a bad shape at the kernel rather than dispatching' {
        $data = 1..100 | ForEach-Object { [double]$_ }
        $bad = New-FlynnelPlan -KOuter 8 -BatchSize 100
        $bad.Shape = [Flynnel.WorkloadShape]::Cooperative
        { Invoke-FlynnelMap -InputObject $data -Operation Square -Plan $bad -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*ShapeCores*'
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
