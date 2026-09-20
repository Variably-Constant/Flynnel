# The verify chain and the mode region.
#
# The only claim this family makes that matters is that the same bytes
# in the same order give the same root, so that is what most of this
# file is about. The root value itself is deliberately not asserted
# against a constant: it depends on the hasher the build was compiled
# with, and a suite that pinned it would fail on a correct build with
# a different feature set while saying nothing about the property.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost

    function New-Chunk {
        param([int]$Seed, [int]$Length = 64)
        $bytes = [byte[]]::new($Length)
        for ($i = 0; $i -lt $Length; $i++) {
            $bytes[$i] = [byte](($Seed * 131 + $i * 17) % 256)
        }
        , $bytes
    }

    function New-Trace {
        param([int]$Count = 8)
        0..($Count - 1) | ForEach-Object { , (New-Chunk -Seed $_) }
    }

    Write-Host ("VERIFY_SUITE host={0}/{1} platform={2}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform)
}

Describe 'the types and enums this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.VerifyChain', 'Flynnel.VerifyComparison',
                          'Flynnel.MatrixBackend', 'Flynnel.ModeRegionCheck') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }

    It 'names both hashers' {
        [enum]::GetNames([Flynnel.VerifyHasher]) | Should -Contain 'Blake3'
        [enum]::GetNames([Flynnel.VerifyHasher]) | Should -Contain 'FxFallback'
    }
}

Describe 'the same bytes give the same root' {
    It 'roots two chains over identical traces the same' {
        # The claim the family rests on. Everything else here is a way
        # of checking that this one keeps holding.
        $trace = New-Trace
        $a = New-FlynnelVerifyChain
        $b = New-FlynnelVerifyChain
        $a.AddMany($trace) | Out-Null
        $b.AddMany($trace) | Out-Null
        $a.Root() | Should -Be $b.Root()
    }

    It 'roots the same whether the chunks arrive one at a time or in a batch' {
        # AddMany is the batched crossing and Add is the per-chunk one.
        # They must fold the same bytes in the same order, or the two
        # forms are not the same operation.
        $trace = New-Trace
        $one = New-FlynnelVerifyChain
        foreach ($c in $trace) { $one.Add($c) | Out-Null }
        $many = New-FlynnelVerifyChain
        $many.AddMany($trace) | Out-Null
        $one.Root() | Should -Be $many.Root()
    }

    It 'gives a root of sixty-four hex characters' {
        $c = New-FlynnelVerifyChain
        $c.Add((New-Chunk -Seed 1)) | Out-Null
        $c.Root() | Should -Match '^[0-9a-f]{64}$'
    }

    It 'answers the same root when asked twice' {
        # The crate's finalize consumes the hasher and a second call
        # would answer thirty-two zero bytes, which reads as a root
        # and is not one. The binding holds the first answer.
        $c = New-FlynnelVerifyChain
        $c.Add((New-Chunk -Seed 2)) | Out-Null
        $first = $c.Root()
        $c.Root() | Should -Be $first
        $first | Should -Not -Match '^0{64}$'
    }

    It 'refuses a chunk after the root has been taken' {
        $c = New-FlynnelVerifyChain
        $c.Add((New-Chunk -Seed 3)) | Out-Null
        $null = $c.Root()
        { $c.Add((New-Chunk -Seed 4)) } | Should -Throw -ExpectedMessage '*root*'
    }
}

Describe 'a different trace gives a different root' {
    It 'moves the root when one byte changes' {
        $trace = @(New-Trace)
        $altered = @($trace | ForEach-Object { , $_.Clone() })
        $altered[3][0] = [byte](($altered[3][0] + 1) % 256)

        $a = New-FlynnelVerifyChain
        $b = New-FlynnelVerifyChain
        $a.AddMany($trace) | Out-Null
        $b.AddMany($altered) | Out-Null
        $a.Root() | Should -Not -Be $b.Root()
    }

    It 'moves the root when the same chunks are submitted in another order' {
        # Order is what a root means. If a reordering rooted the same,
        # the chain could not tell two traces apart by sequence, which
        # is most of what a trace is.
        $trace = @(New-Trace -Count 4)
        $reversed = @($trace[3], $trace[2], $trace[1], $trace[0])

        $a = New-FlynnelVerifyChain
        $b = New-FlynnelVerifyChain
        $a.AddMany($trace) | Out-Null
        $b.AddMany($reversed) | Out-Null
        $a.Root() | Should -Not -Be $b.Root()
    }
}

Describe 'Compare-FlynnelVerifyChain' {
    It 'says two identical traces agree and names no index' {
        $trace = New-Trace
        $a = New-FlynnelVerifyChain
        $b = New-FlynnelVerifyChain
        $a.AddMany($trace) | Out-Null
        $b.AddMany($trace) | Out-Null

        $r = Compare-FlynnelVerifyChain -Reference $a -Difference $b
        $r.RootsAgree | Should -BeTrue
        $r.HasDivergingIndex | Should -BeFalse
        $r.LengthsDiffer | Should -BeFalse
        $r.ReferenceCount | Should -Be 8
        $r.DifferenceCount | Should -Be 8
    }

    It 'names the first diverging index, and it is the right one' {
        # Built independently: the index is chosen here and the
        # comparison has to find that one rather than any index.
        foreach ($at in 0, 1, 5, 7) {
            $trace = @(New-Trace)
            $altered = @($trace | ForEach-Object { , $_.Clone() })
            $altered[$at][10] = [byte](($altered[$at][10] + 7) % 256)

            $a = New-FlynnelVerifyChain
            $b = New-FlynnelVerifyChain
            $a.AddMany($trace) | Out-Null
            $b.AddMany($altered) | Out-Null

            $r = Compare-FlynnelVerifyChain -Reference $a -Difference $b
            $r.RootsAgree | Should -BeFalse -Because "chunk $at differs"
            $r.HasDivergingIndex | Should -BeTrue
            $r.FirstDivergingIndex | Should -Be $at
        }
    }

    It 'names the FIRST diverging index when several differ' {
        $trace = @(New-Trace)
        $altered = @($trace | ForEach-Object { , $_.Clone() })
        foreach ($at in 2, 4, 6) {
            $altered[$at][0] = [byte](($altered[$at][0] + 3) % 256)
        }
        $a = New-FlynnelVerifyChain
        $b = New-FlynnelVerifyChain
        $a.AddMany($trace) | Out-Null
        $b.AddMany($altered) | Out-Null

        (Compare-FlynnelVerifyChain -Reference $a -Difference $b).FirstDivergingIndex |
            Should -Be 2
    }

    It 'tells a prefix apart from a differing chunk' {
        # Two chains of different lengths that agree on everything
        # they both hold is a different finding from a chunk that
        # differs, and a row that collapsed them would send a reader
        # looking for a corrupted chunk that does not exist.
        $trace = @(New-Trace -Count 6)
        $short = @($trace[0..3])

        $a = New-FlynnelVerifyChain
        $b = New-FlynnelVerifyChain
        $a.AddMany($trace) | Out-Null
        $b.AddMany($short) | Out-Null

        $r = Compare-FlynnelVerifyChain -Reference $a -Difference $b
        $r.RootsAgree | Should -BeFalse
        $r.LengthsDiffer | Should -BeTrue
        $r.HasDivergingIndex | Should -BeFalse -Because 'every shared chunk is the same'
        $r.FirstExtraIndex | Should -Be 4
        $r.ReferenceCount | Should -Be 6
        $r.DifferenceCount | Should -Be 4
    }

    It 'refuses to compare a chain with itself' {
        $a = New-FlynnelVerifyChain
        $a.Add((New-Chunk -Seed 9)) | Out-Null
        { Compare-FlynnelVerifyChain -Reference $a -Difference $a } |
            Should -Throw -ExpectedMessage '*same chain*'
    }

    It 'refuses an object that is not a chain' {
        $a = New-FlynnelVerifyChain
        { Compare-FlynnelVerifyChain -Reference $a -Difference (Get-Date) } | Should -Throw
    }
}

Describe 'the hashers' {
    It 'roots with the fallback when asked' {
        $c = New-FlynnelVerifyChain -Hasher FxFallback
        $c.Hasher | Should -Be 'FxFallback'
        $c.Add((New-Chunk -Seed 5)) | Out-Null
        $c.Root() | Should -Match '^[0-9a-f]{64}$'
    }

    It 'gives the two hashers different roots over the same bytes' {
        $trace = New-Trace
        $blake = New-FlynnelVerifyChain -Hasher Blake3
        $fx = New-FlynnelVerifyChain -Hasher FxFallback
        $blake.AddMany($trace) | Out-Null
        $fx.AddMany($trace) | Out-Null
        $blake.Root() | Should -Not -Be $fx.Root()
    }

    It 'says so when a comparison is between different hashers' {
        # Two hashers over identical bytes root differently, so a
        # disagreement says nothing about the traces. Reporting that
        # as a mismatch would send a reader hunting a corrupted chunk
        # that does not exist - which is the same false-mismatch the
        # chain's own ordering fix removed.
        $trace = New-Trace
        $blake = New-FlynnelVerifyChain -Hasher Blake3
        $fx = New-FlynnelVerifyChain -Hasher FxFallback
        $blake.AddMany($trace) | Out-Null
        $fx.AddMany($trace) | Out-Null

        $r = Compare-FlynnelVerifyChain -Reference $blake -Difference $fx -WarningVariable warned
        $r.HashersDiffer | Should -BeTrue
        $r.RootsAgree | Should -BeFalse
        @($warned).Count | Should -BeGreaterThan 0
        # The index columns are still sound: the per-chunk digests do
        # not depend on what the chain roots with, and these traces
        # are identical.
        $r.HasDivergingIndex | Should -BeFalse
        $r.LengthsDiffer | Should -BeFalse
    }
}

Describe 'Get-FlynnelMatrixBackend' {
    It 'writes a row for the fallback rather than an empty listing' {
        # An empty listing reads as a family that failed to enumerate.
        # A row saying the only backend is the fallback says what is
        # true: the substrate is here and no tile backend implements
        # it yet.
        $rows = @(Get-FlynnelMatrixBackend)
        $rows.Count | Should -BeGreaterThan 0
        $fallback = $rows | Where-Object IsFallback | Select-Object -First 1
        $fallback | Should -Not -BeNullOrEmpty
        $fallback.Available | Should -BeTrue
        $fallback.Note | Should -Not -BeNullOrEmpty
    }
}

Describe 'Test-FlynnelModeRegion' {
    It 'pairs the exit to the enter on a body that returns' {
        (Test-FlynnelModeRegion).NormalReturnPaired | Should -BeTrue
    }

    It 'pairs the exit to the enter on a body that panics' {
        # The guarantee the substrate exists for. A guard that has
        # never seen a panic is untested, so the check runs one.
        $r = Test-FlynnelModeRegion
        $r.PanicCaught | Should -BeTrue -Because 'the panic must be caught, not cross the boundary'
        $r.PanickingBodyPaired | Should -BeTrue -Because 'the region must exit while the panic unwinds'
    }

    It 'counts as many exits as enters' {
        $r = Test-FlynnelModeRegion
        $r.Exits | Should -Be $r.Enters
        $r.Enters | Should -BeGreaterThan 0
    }

    It 'runs a region through the scalar fallback too' {
        (Test-FlynnelModeRegion).FallbackRan | Should -BeTrue
    }

    It 'leaves nothing behind, so it can be run twice' {
        $first = Test-FlynnelModeRegion
        $second = Test-FlynnelModeRegion
        $first.Exits | Should -Be $first.Enters
        $second.Exits | Should -Be $second.Enters
    }
}
