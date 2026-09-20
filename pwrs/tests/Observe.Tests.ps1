# The observation family: that an instrument which did not read says
# so, rather than reporting a zero.
#
# Every assertion here is about provenance or about physical bounds.
# None is about the size of a measured number, because the suite runs
# on two hosts under whatever load they carry and a threshold on a
# timing figure would fail for the box rather than for the code.
#
# The defect this family exists to avoid has been found nine times in
# this campaign: a figure nobody measured arriving as a zero that a
# reader takes for an answer. So the checks that matter most are the
# ones asserting a property is nullable and is null when unmeasured.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
}

Describe 'the types this family exports' {
    It 'shapes the counter row the way its cmdlet documents' {
        $properties = @(Get-FlynnelTypeProperty -TypeName 'Flynnel.TraceCounters' |
            ForEach-Object Name)
        foreach ($wanted in 'Calls', 'TotalCalls', 'Armed', 'IdleCycles') {
            $properties | Should -Contain $wanted
        }
    }
}

Describe 'Get-FlynnelTraceState' {
    It 'says whether the trace is armed and what arms it' {
        $state = Get-FlynnelTraceState
        $state.IsEnabled | Should -BeOfType [bool]
        $state.EnabledBy | Should -Be 'FLYNNEL_TRACE'
        $state.CountersArmedBy | Should -Be 'FLYNNEL_TRACE_DISPATCH'
    }

    It 'reports the two arming switches separately' {
        # One process can have the event ring without the dispatch
        # counters. A single flag would make a caller who set one
        # believe they had both.
        $state = Get-FlynnelTraceState
        $state.DispatchCountersArmed | Should -BeOfType [bool]
    }

    It 'counts worker flushes' {
        (Get-FlynnelTraceState).WorkerFlushesDone | Should -BeGreaterOrEqual 0
    }
}

Describe 'Get-FlynnelTrace' {
    It 'says the counters are unarmed rather than reporting a quiet run' {
        # An unarmed instrument and an idle scheduler both produce
        # zeros. Armed is what tells them apart, and without it the
        # zeros would be quoted as a measurement.
        $counters = Get-FlynnelTrace -WarningAction SilentlyContinue
        $counters.Armed | Should -BeOfType [bool]
        if (-not $counters.Armed) {
            $counters.Calls | Should -Be 0
        }
    }

    It 'warns when it is read unarmed' {
        $state = Get-FlynnelTraceState
        if ($state.DispatchCountersArmed) {
            Set-ItResult -Skipped -Because 'the dispatch counters are armed in this process'
            return
        }
        $warnings = @()
        $null = Get-FlynnelTrace -WarningVariable warnings
        ($warnings -join ' ') | Should -Match 'not armed'
    }

    It 'does not lose counts when read twice' {
        # The crate's snapshot swaps its counters to zero as it reads,
        # so a second reader would otherwise see none of what the first
        # one took. The totals are what make repeated reads safe.
        $first = Get-FlynnelTrace -WarningAction SilentlyContinue
        $second = Get-FlynnelTrace -WarningAction SilentlyContinue
        $second.TotalCalls | Should -BeGreaterOrEqual $first.TotalCalls
        $second.TotalBodyCycles | Should -BeGreaterOrEqual $first.TotalBodyCycles
        $second.TotalIdleCycles | Should -BeGreaterOrEqual $first.TotalIdleCycles
    }

    It 'keeps the delta no larger than the total' {
        $row = Get-FlynnelTrace -WarningAction SilentlyContinue
        $row.Calls | Should -BeLessOrEqual $row.TotalCalls
        $row.WaitCycles | Should -BeLessOrEqual $row.TotalWaitCycles
    }
}

Describe 'Clear-FlynnelTrace and Request-FlynnelTraceFlush' {
    It 'clears without throwing' {
        { Clear-FlynnelTrace } | Should -Not -Throw
    }

    It 'answers the flush count that stood before the request' {
        $before = (Get-FlynnelTraceState).WorkerFlushesDone
        $answered = Request-FlynnelTraceFlush
        $answered | Should -Be $before
        (Get-FlynnelTraceState).WorkerFlushesDone | Should -BeGreaterOrEqual $answered
    }
}

Describe 'Get-FlynnelLeafStat' {
    It 'gives an unmeasured spread no value, rather than a zero one' {
        foreach ($name in 'MeanLeafNs', 'PerItemNs', 'LeafCv2PerMille', 'PerItemCv2PerMille') {
            $property = Get-FlynnelTypeProperty -TypeName 'Flynnel.LeafStats' |
                Where-Object Name -eq $name
            $property | Should -Not -BeNullOrEmpty -Because "$name must exist"
            $property.PropertyType | Should -Be ([System.Nullable[System.UInt64]]) `
                -Because "$name can be unmeasured and must be nullable, not a zero-valued primitive"
        }
    }

    It 'counts leaves after a declared kernel has run' {
        Reset-FlynnelLeafStat
        $data = 1..100000 | ForEach-Object { [double]$_ }
        $null = Measure-FlynnelReduce -InputObject $data -Operation Sum
        $null = Invoke-FlynnelMap -InputObject $data -Operation Square
        $stats = Get-FlynnelLeafStat
        # The pool may collapse small work inline, in which case no
        # leaf is timed. Either the count rose or it is zero, and a
        # zero count must come with null spreads rather than zeros.
        if ($stats.Count -eq 0) {
            $stats.LeafCv2PerMille | Should -BeNullOrEmpty
            $stats.MeanLeafNs | Should -BeNullOrEmpty
        } else {
            $stats.SumNs | Should -BeGreaterThan 0
        }
    }

    It 'resets to nothing' {
        Reset-FlynnelLeafStat
        $stats = Get-FlynnelLeafStat
        $stats.Count | Should -Be 0
        $stats.SumNs | Should -Be 0
        $stats.MeanLeafNs | Should -BeNullOrEmpty
    }
}

Describe 'Measure-FlynnelOccupancy' {
    It 'answers a share inside physical bounds, or none at all' {
        # A percentage outside 0 to 100 is the instrument, not the box.
        $sample = Measure-FlynnelOccupancy -Seconds 0.25
        if ($null -eq $sample.Percent) {
            $sample.NoReading | Should -Not -BeNullOrEmpty
        } else {
            $sample.Percent | Should -BeGreaterOrEqual 0
            $sample.Percent | Should -BeLessOrEqual 100
            $sample.NoReading | Should -BeNullOrEmpty
        }
    }

    It 'covers a wall window' {
        (Measure-FlynnelOccupancy -Seconds 0.25).WallTicks | Should -BeGreaterThan 0
    }

    It 'names what a thread tick counts on this platform' {
        # Windows answers cycles and the others nanoseconds, so a
        # figure without its unit cannot be compared across hosts.
        (Measure-FlynnelOccupancy -Seconds 0.1).ThreadTickUnit |
            Should -BeIn @('cycles', 'nanoseconds', 'none')
    }

    It 'refuses a window of zero or less' {
        { Measure-FlynnelOccupancy -Seconds 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*above zero*'
    }

    It 'gives an unmeasurable share no value, rather than a zero one' {
        $property = Get-FlynnelTypeProperty -TypeName 'Flynnel.Occupancy' |
            Where-Object Name -eq 'Percent'
        $property.PropertyType | Should -Be ([System.Nullable[System.UInt32]])
    }
}

Describe 'Get-FlynnelThreadTick' {
    It 'answers a tick count or says why it cannot' {
        $row = Get-FlynnelThreadTick
        if ($null -eq $row.ThreadTicks) {
            $row.NoReading | Should -BeIn @([Flynnel.NoReading]::NoClock,
                                            [Flynnel.NoReading]::Unreadable)
        } else {
            $row.ThreadTicks | Should -BeGreaterThan 0
        }
    }

    It 'rises across an interval where it reads at all' {
        $first = (Get-FlynnelThreadTick).ThreadTicks
        if ($null -eq $first) {
            Set-ItResult -Skipped -Because 'this platform has no per-thread clock'
            return
        }
        $spin = 0.0
        for ($i = 1; $i -le 300000; $i++) { $spin += [Math]::Sqrt($i) }
        $second = (Get-FlynnelThreadTick).ThreadTicks
        $second | Should -BeGreaterThan $first
    }
}

Describe 'Get-FlynnelReducePath' {
    It 'names the path after a reduce has run on this thread' {
        $null = Measure-FlynnelReduce -InputObject (1..50000) -Operation Sum
        $path = Get-FlynnelReducePath
        # The crate records this per thread. A cmdlet and the kernels
        # both run on the pipeline thread, so a reduce run just now is
        # visible here.
        $path | Should -BeIn @([Flynnel.ReducePath]::Flat, [Flynnel.ReducePath]::Bisect)
    }
}

Describe 'Get-FlynnelSpread' {
    It 'gives identical samples no spread' {
        (Get-FlynnelSpread -Sample @(100, 100, 100, 100)).SpreadPerMille | Should -Be 0
    }

    It 'gives a wider set more spread than a narrower one' {
        $narrow = (Get-FlynnelSpread -Sample @(100, 101, 102, 103)).SpreadPerMille
        $wide = (Get-FlynnelSpread -Sample @(100, 200, 300, 400)).SpreadPerMille
        $wide | Should -BeGreaterThan $narrow
    }

    It 'does not depend on the order the samples arrive in' {
        $a = Get-FlynnelSpread -Sample @(5, 1, 9, 3, 7)
        $b = Get-FlynnelSpread -Sample @(1, 3, 5, 7, 9)
        $a.SpreadPerMille | Should -Be $b.SpreadPerMille
        $a.Median | Should -Be $b.Median
    }

    It 'reports the order statistics it read' {
        $row = Get-FlynnelSpread -Sample @(5, 1, 9, 3, 7)
        $row.Count | Should -Be 5
        $row.Minimum | Should -Be 1
        $row.Median | Should -Be 5
        $row.Maximum | Should -Be 9
    }
}

Describe 'Get-FlynnelCallSite' {
    BeforeAll {
        # A site exists only once a dispatch has reached that source
        # location, so the suite has to make one before it can read any.
        $script:Work = [double[]](1..4000)
        $null = Invoke-FlynnelMap -InputObject $script:Work -Operation Square
    }

    It 'answers a row per site, each saying where it is' {
        $sites = @(Get-FlynnelCallSite -WarningVariable ignored)
        $sites.Count | Should -BeGreaterThan 0
        foreach ($s in $sites) {
            $s.File | Should -Not -BeNullOrEmpty
            $s.Line | Should -BeGreaterThan 0
        }
    }

    It 'reports an unmeasured figure as nothing rather than zero' {
        # The convention this whole family keeps. A spread of zero is a
        # perfectly uniform workload, which is a reading; no reading at
        # all has to look different from it.
        foreach ($s in @(Get-FlynnelCallSite -WarningVariable ignored)) {
            if ($s.OncoreItems -eq 0) {
                $s.PerItemOncoreCv2PerMille | Should -BeNullOrEmpty
            }
            if ($s.WindowTicks -eq 0) {
                $s.WindowMeanNs | Should -BeNullOrEmpty
                $s.WindowCv2MinPerMille | Should -BeNullOrEmpty
                $s.WindowCv2MaxPerMille | Should -BeNullOrEmpty
            }
        }
    }

    It 'carries the range beside the single window reading' {
        # WindowCv2PerMille is one classifier tick out of thousands and
        # spans the whole range within a run, so a reader judging a run
        # by it alone is reading noise. The extremes are what the row
        # exists to carry.
        foreach ($s in @(Get-FlynnelCallSite -WarningVariable ignored)) {
            if ($null -ne $s.WindowCv2MinPerMille) {
                $s.WindowCv2MaxPerMille | Should -Not -BeNullOrEmpty
                $s.WindowCv2MaxPerMille | Should -BeGreaterOrEqual $s.WindowCv2MinPerMille
            }
        }
    }
}

Describe 'Reset-FlynnelCallSite' {
    BeforeEach {
        $null = Invoke-FlynnelMap -InputObject ([double[]](1..4000)) -Operation Square
    }

    It 'changes nothing under WhatIf' {
        $before = @(Get-FlynnelCallSite -WarningVariable ignored).Count
        $null = Reset-FlynnelCallSite -WhatIf
        @(Get-FlynnelCallSite -WarningVariable ignored).Count | Should -Be $before
    }

    It 'answers how many sites it reset' {
        $held = @(Get-FlynnelCallSite -WarningVariable ignored).Count
        Reset-FlynnelCallSite -Confirm:$false | Should -Be $held
    }

    It 'leaves the sites registered and forgetful, not gone' {
        # A reset clears what a site learned; it must not unregister it.
        # A missing row would read as a location no dispatch ever
        # reached, which is a different fact from one that has been
        # cleared.
        $null = Reset-FlynnelCallSite -Confirm:$false
        $after = @(Get-FlynnelCallSite -WarningVariable ignored)
        $after.Count | Should -BeGreaterThan 0
        foreach ($s in $after) {
            $s.SeedDepthFlips | Should -Be 0
            $s.CollapseOverran | Should -BeFalse
            $s.LearnedClass | Should -BeNullOrEmpty
            $s.RecentOccupancyPct | Should -BeNullOrEmpty
        }
    }
}
