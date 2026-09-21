# The hybrid shapes: the split, the learned placement and the pipeline.
#
# These cmdlets measure, so most of what can be asserted about them is
# structural: the halves add up to the whole, the share is in range, the
# bucket is the logarithm of the count. Two claims are behavioural and
# both are here, because they are the only reasons the shapes exist: the
# halves really overlap, and the placement model really settles.
#
# Neither timing claim is written as a threshold in nanoseconds. A
# figure like that is a statement about the box, and these suites run on
# two editions and three hosts. They are written as a comparison inside
# one call instead, which holds wherever it runs.
#
# No suite here resets the call-site registry. The registry is global to
# the process and Pester runs every file in one, so a reset would
# discard what another suite measured. A cold bucket is obtained by
# picking a size nothing else in this file has used, which works because
# the model keys on log2 of the count.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost

    Write-Host ("HYBRID_SUITE host={0}/{1} platform={2} cpus={3}" -f
        $script:TestHost.Edition, $script:TestHost.Version,
        $script:TestHost.Platform, $script:TestHost.ProcessorCount)

    # Enough work that each half is milliseconds rather than
    # microseconds. The overlap claim compares two measured spans, and
    # at a hundred microseconds a half the thread hand-off is a large
    # enough share of both to make the comparison about the hand-off.
    $script:BigCount = 2000000
    $script:BigReps = 4
}

Describe 'the types and the enum this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.HybridJoin', 'Flynnel.HybridPlacement',
                          'Flynnel.HybridSplit', 'Flynnel.HybridPipeline') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }

    It 'names all three placements' {
        $names = [enum]::GetNames([Flynnel.Placement])
        $names | Should -Contain 'Cpu'
        $names | Should -Contain 'Backend'
        $names | Should -Contain 'Race'
    }
}

Describe 'Measure-FlynnelHybridJoin' {
    It 'divides the count between the two halves and loses none of it' {
        $r = Measure-FlynnelHybridJoin -Count 100000 -Operation Sqrt
        ($r.CpuItems + $r.BackendItems) | Should -Be 100000
    }

    It 'honors the share it was given' {
        $r = Measure-FlynnelHybridJoin -Count 100000 -Operation Sqrt -CpuShare 250
        $r.CpuItems | Should -Be 25000
        $r.BackendItems | Should -Be 75000
    }

    It 'splits evenly when no share is given' {
        $r = Measure-FlynnelHybridJoin -Count 100000 -Operation Sqrt
        $r.CpuItems | Should -Be 50000
    }

    It 'gives one half everything at a share of zero or a thousand' {
        (Measure-FlynnelHybridJoin -Count 1000 -Operation Sqrt -CpuShare 0).CpuItems |
            Should -Be 0
        (Measure-FlynnelHybridJoin -Count 1000 -Operation Sqrt -CpuShare 1000).BackendItems |
            Should -Be 0
    }

    It 'runs the two halves concurrently' {
        # The property the shape exists for. Both halves are sized to
        # tens of milliseconds above, so a total below their sum cannot
        # be timer resolution.
        $r = Measure-FlynnelHybridJoin -Count $script:BigCount -Operation Sqrt `
            -Repetitions $script:BigReps
        $r.CpuNs | Should -BeGreaterThan 0
        $r.BackendNs | Should -BeGreaterThan 0
        $r.TotalNs | Should -BeLessThan ($r.CpuNs + $r.BackendNs) `
            -Because 'the halves overlap, so the call is shorter than one after the other'
    }

    It 'says the backend half ran on the CPU backend when no device is registered' {
        # Not a device reading, and the row has to say so rather than
        # leaving a reader to infer it from a hostname.
        $r = Measure-FlynnelHybridJoin -Count 1000 -Operation Sqrt
        $cuda = Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Registered }
        if ($cuda) {
            Set-ItResult -Skipped -Because 'this host has a registered CUDA backend'
            return
        }
        $r.BackendIsCpu | Should -BeTrue
        $r.Backend | Should -Be 'Cpu'
    }

    It 'answers a finite checksum from each half' {
        $r = Measure-FlynnelHybridJoin -Count 10000 -Operation Sqrt
        [double]::IsFinite($r.CpuChecksum) | Should -BeTrue
        [double]::IsFinite($r.BackendChecksum) | Should -BeTrue
        $r.CpuChecksum | Should -BeGreaterThan 0
    }

    It 'refuses a count of zero' {
        { Measure-FlynnelHybridJoin -Count 0 -Operation Sqrt } |
            Should -Throw -ExpectedMessage '*above zero*'
    }

    It 'refuses a share above a thousand' {
        { Measure-FlynnelHybridJoin -Count 100 -Operation Sqrt -CpuShare 1001 } |
            Should -Throw -ExpectedMessage '*thousand*'
    }

    It 'refuses Clamp without its bounds, before either half starts' {
        # The operand is read on the pipeline thread. If it were read
        # inside the halves, a missing one would be a panic on a backend
        # thread rather than an error record.
        { Measure-FlynnelHybridJoin -Count 100 -Operation Clamp } |
            Should -Throw -ExpectedMessage '*Clamp*'
    }
}

Describe 'Measure-FlynnelHybridPlacement' {
    It 'answers one of the three placements' {
        $r = Measure-FlynnelHybridPlacement -Count 4096 -Operation Sqrt
        [string]$r.Placement | Should -BeIn @('Cpu', 'Backend', 'Race')
    }

    It 'races the first call in a bucket nothing has used' {
        # Racing is the calibration: a cold bucket has no measurement to
        # choose on, so it runs both and times each. A first call that
        # chose a side would be choosing on nothing.
        $r = Measure-FlynnelHybridPlacement -Count 8192 -Operation Sqrt
        $r.Placement | Should -Be 'Race' -Because 'nothing in this file has used bucket 13 yet'
    }

    It 'stops racing once the bucket is warm' {
        # A model that raced every call would never exploit what it
        # learned, and would pay double work forever.
        $seen = 1..12 | ForEach-Object {
            (Measure-FlynnelHybridPlacement -Count 16384 -Operation Sqrt).Placement
        }
        @($seen | Where-Object { $_ -ne 'Race' }).Count |
            Should -BeGreaterThan 0 -Because 'a warm bucket runs one side'
    }

    It 'reports the bucket as the base-two logarithm of the count' {
        (Measure-FlynnelHybridPlacement -Count 1024 -Operation Sqrt).Bucket | Should -Be 10
        (Measure-FlynnelHybridPlacement -Count 1000 -Operation Sqrt).Bucket | Should -Be 9
    }

    It 'echoes the count it was given' {
        (Measure-FlynnelHybridPlacement -Count 777 -Operation Sqrt).Count | Should -Be 777
    }

    It 'answers the same checksum whichever side ran' {
        # The two implementations are the same declared operation, which
        # is the contract the learned model rests on: a placement is a
        # performance decision and never a semantic one. A checksum that
        # moved with the placement would mean the two sides are not the
        # same computation.
        $runs = 1..8 | ForEach-Object {
            Measure-FlynnelHybridPlacement -Count 2048 -Operation Sqrt
        }
        @($runs | Select-Object -ExpandProperty Checksum -Unique).Count | Should -Be 1
    }

    It 'refuses a count of zero' {
        { Measure-FlynnelHybridPlacement -Count 0 -Operation Sqrt } |
            Should -Throw -ExpectedMessage '*above zero*'
    }
}

Describe 'Measure-FlynnelHybridSplit' {
    It 'divides the count between the two sides and loses none of it' {
        $r = Measure-FlynnelHybridSplit -Count 100000 -Operation Sqrt
        ($r.CpuItems + $r.BackendItems) | Should -Be 100000
    }

    It 'falls back to the site-wide share at a size it has no data for' {
        # A size the model has never seen does not start even. It reads
        # the site's overall ratio instead, which is whatever the last
        # calls at other sizes established. Measured here: a first call
        # at this size reported 428, because an earlier call at another
        # size had timed the two sides at 4 and 3 nanoseconds an item.
        #
        # So the claim is the fallback itself: a cold size answers a
        # real share rather than a zero or a refusal.
        $r = Measure-FlynnelHybridSplit -Count 262144 -Operation Sqrt
        $r.CpuSharePerMille | Should -BeGreaterThan 0
        $r.CpuSharePerMille | Should -BeLessThan 1000
        ($r.CpuItems + $r.BackendItems) | Should -Be 262144
    }

    It 'keeps the share inside its range as it learns' {
        $shares = 1..10 | ForEach-Object {
            (Measure-FlynnelHybridSplit -Count 131072 -Operation Sqrt).CpuSharePerMille
        }
        foreach ($s in $shares) {
            $s | Should -BeGreaterOrEqual 0
            $s | Should -BeLessOrEqual 1000
        }
    }

    It 'holds the split near even while both sides cost the same per item' {
        # Measured on zen3: eight calls at one size with the same body
        # on both sides never leave 500. That is the model working, not
        # the model asleep. Each side's clock starts inside its own
        # half, so the backend side's thread hand-off falls outside both
        # readings and the two per-item costs come out the same.
        #
        # A band rather than an exact 500, because the recorded cost is
        # a whole number of nanoseconds and two sides that differ by one
        # of them give 508 rather than 500.
        $shares = 1..8 | ForEach-Object {
            (Measure-FlynnelHybridSplit -Count 32768 -Operation Sqrt -Repetitions 8).CpuSharePerMille
        }
        foreach ($s in $shares) {
            $s | Should -BeGreaterOrEqual 450
            $s | Should -BeLessOrEqual 550
        }
    }

    It 'moves the share when the backend side costs more per item' {
        # The other direction, and the one that shows the model reads
        # its inputs at all. BackendRepetitions makes the backend side
        # several times dearer per item, which is what a split model
        # exists to track, so the share has to move toward the CPU.
        #
        # Repetitions is high on purpose. The model records per-item
        # cost as a whole number of nanoseconds, so at the default of
        # one repetition every side of every call truncates to the same
        # integer and no real difference can be resolved at all.
        $shares = 1..8 | ForEach-Object {
            (Measure-FlynnelHybridSplit -Count 8192 -Operation Exp `
                -Repetitions 16 -BackendRepetitions 64).CpuSharePerMille
        }
        ($shares | Select-Object -Last 1) | Should -BeGreaterThan 550 `
            -Because 'the dearer side should be given fewer items'
    }

    It 'times both sides' {
        $r = Measure-FlynnelHybridSplit -Count 200000 -Operation Sqrt -Repetitions 2
        $r.CpuNs | Should -BeGreaterThan 0
        $r.BackendNs | Should -BeGreaterThan 0
    }

    It 'refuses a count of zero' {
        { Measure-FlynnelHybridSplit -Count 0 -Operation Sqrt } |
            Should -Throw -ExpectedMessage '*above zero*'
    }
}

Describe 'Measure-FlynnelHybridPipeline' {
    It 'produces one result per input' {
        # A stage that dropped work would leave a shorter answer, and a
        # pipeline reporting a throughput over work it silently lost is
        # the worst of the failures available to it.
        $r = Measure-FlynnelHybridPipeline -Count 200 -Width 512
        $r.Inputs | Should -Be 200
        $r.Outputs | Should -Be 200
    }

    It 'echoes the width it was given and defaults when none is' {
        (Measure-FlynnelHybridPipeline -Count 10 -Width 256).Width | Should -Be 256
        (Measure-FlynnelHybridPipeline -Count 10).Width | Should -BeGreaterThan 0
    }

    It 'answers the same checksum for the same arguments' {
        $a = Measure-FlynnelHybridPipeline -Count 50 -Width 128
        $b = Measure-FlynnelHybridPipeline -Count 50 -Width 128
        $a.Checksum | Should -Be $b.Checksum
    }

    It 'moves the checksum when a stage operation changes' {
        # Otherwise a stage could be doing nothing and the suite would
        # not know.
        $a = Measure-FlynnelHybridPipeline -Count 50 -Width 128 -PreOperation Square
        $b = Measure-FlynnelHybridPipeline -Count 50 -Width 128 -PreOperation Sqrt
        $a.Checksum | Should -Not -Be $b.Checksum
    }

    It 'costs less per input over many inputs than over one' {
        # What pipelining buys: the fill is paid once, so the per-input
        # cost falls towards the slowest stage as the run lengthens.
        $one = Measure-FlynnelHybridPipeline -Count 1 -Width 8192
        $many = Measure-FlynnelHybridPipeline -Count 400 -Width 8192
        $many.NsPerInput | Should -BeLessThan $one.NsPerInput `
            -Because 'the fill and the thread spawn are amortized over the run'
    }

    It 'refuses a count of zero' {
        { Measure-FlynnelHybridPipeline -Count 0 } |
            Should -Throw -ExpectedMessage '*above zero*'
    }
}
