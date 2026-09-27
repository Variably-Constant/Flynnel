# The kernel family: that the work Flynnel's workers did is the work
# that was asked for.
#
# Every assertion compares against an answer computed here, in
# PowerShell, by a different route. A kernel checked against itself
# proves only that it is deterministic, and the defects worth catching
# in a chunked parallel body are exactly the ones that survive being
# run twice: a boundary a match straddles, a chunk whose offset is
# wrong, a per-chunk accumulator that does not combine.
#
# Sizes are chosen so every kernel's input is cut into several blocks.
# The kernels cut an array into blocks of at least 2,048 elements and
# text into blocks of at least 64 KiB, at most 256 either way, from the
# input's length alone, so a kernel tested only on eight elements runs
# in one block and never exercises the combine at all.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    # Big enough that every kernel here is split across chunks on any
    # host the suite runs on.
    $script:N = 50000
    $script:Data = 1..$script:N | ForEach-Object { [double]($_ % 997) - 400.0 }
    $script:Other = 1..$script:N | ForEach-Object { [double](($_ * 7) % 501) - 250.0 }

    $script:Work = Join-Path ([System.IO.Path]::GetTempPath()) "flynnel-kernels-$PID"
    New-Item -ItemType Directory -Path $script:Work -Force | Out-Null

    # The published BLAKE3 root of the empty input. An independent
    # constant rather than a second call to the same code, so the test
    # can tell a working hash from a self-consistent wrong one.
    $script:EmptyBlake3 =
        'af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262'
}

AfterAll {
    if ($script:Work -and (Test-Path $script:Work)) {
        Remove-Item -LiteralPath $script:Work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Describe 'the types this family exports' {
    It 'shapes each row type the way its cmdlet documents' {
        $shape = @{
            'Flynnel.Reduction'    = @('Operation', 'Count', 'Value')
            'Flynnel.HistogramBin' = @('Index', 'Low', 'High', 'Count')
            'Flynnel.Histogram'    = @('Low', 'High', 'Width', 'Counts')
            'Flynnel.FileHash'     = @('Path', 'Hash', 'Bytes')
            'Flynnel.HashCheck'    = @('Path', 'Expected', 'Actual', 'IsMatch')
            'Flynnel.FileMatch'    = @('Path', 'LineNumber', 'Line')
            'Flynnel.FileMeasure'  = @('Path', 'Lines', 'Bytes')
            'Flynnel.TextMatch'    = @('Index', 'LineNumber', 'Line')
            'Flynnel.TextMeasure'  = @('Bytes', 'Lines', 'Words', 'Matches')
            'Flynnel.PrimitiveRun' = @('Primitive', 'Operation', 'Count', 'Sum', 'ElapsedNs')
        }
        foreach ($type in $shape.Keys) {
            $properties = @(Get-FlynnelTypeProperty -TypeName $type | ForEach-Object Name)
            foreach ($wanted in $shape[$type]) {
                $properties | Should -Contain $wanted -Because "$type must carry $wanted"
            }
        }
    }

    It 'gives every operation enum at least one value' {
        foreach ($name in 'Flynnel.MapOp', 'Flynnel.ZipOp', 'Flynnel.ReduceOp',
                          'Flynnel.TextTransform', 'Flynnel.Primitive') {
            $type = $name -as [type]
            $type | Should -Not -BeNullOrEmpty -Because "$name must be exported"
            [Enum]::GetValues($type).Count | Should -BeGreaterThan 0
        }
    }

    It 'covers every map operation the enum declares' {
        # An operation the enum names and no kernel implements would
        # bind and then do nothing recognizable, and nothing else here
        # would report it.
        $x = @(4.0, -9.0, 0.25)
        foreach ($op in [Enum]::GetValues([Flynnel.MapOp])) {
            $extra = switch ([string]$op) {
                'Clamp'  { @{ Min = -1.0; Max = 1.0 } }
                'Scale'  { @{ Factor = 2.0 } }
                'Offset' { @{ Addend = 1.0 } }
                default  { @{} }
            }
            $got = Invoke-FlynnelMap -InputObject $x -Operation $op @extra
            @($got).Count | Should -Be 3 -Because "$op must answer one value per element"
        }
    }

    It 'covers every zip operation the enum declares' {
        $a = @(4.0, -9.0, 0.25)
        $b = @(2.0, 3.0, 0.5)
        foreach ($op in [Enum]::GetValues([Flynnel.ZipOp])) {
            # Not wrapped in @(): a bulk cmdlet answers one array
            # object, so @() would give an array holding that array
            # and count one. Assigning takes the array itself.
            $got = Invoke-FlynnelZip -Left $a -Right $b -Operation $op
            $got.Count | Should -Be 3
        }
    }

    It 'answers one array rather than a stream, for every bulk kernel' {
        # The module's own rule, made testable. A per-item return
        # costs 1712 ns an element in this host; one array costs one
        # crossing. The array is typed, so it is also the fast input
        # path for the next kernel, which is what makes chaining cheap.
        $x = [double[]]@(1, 2, 3, 4)
        foreach ($call in
            { Invoke-FlynnelMap -InputObject $x -Operation Square },
            { Invoke-FlynnelZip -Left $x -Right $x -Operation Add },
            { Get-FlynnelPrefixSum -InputObject $x },
            { Invoke-FlynnelSort -InputObject $x }) {
            $out = @(& $call)
            $out.Count | Should -Be 1 -Because 'the pipeline carries one array, not four doubles'
            $out[0].Count | Should -Be 4
        }
    }
}

Describe 'Invoke-FlynnelMap' {
    It 'squares every element' {
        $got = Invoke-FlynnelMap -InputObject $script:Data -Operation Square
        $want = $script:Data | ForEach-Object { $_ * $_ }
        $got.Count | Should -Be $script:N
        # Compared as a whole rather than element by element: 50000
        # separate Should calls is minutes of Pester, and a single
        # mismatch still fails this.
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'clamps into the range it was given' {
        $got = Invoke-FlynnelMap -InputObject $script:Data -Operation Clamp -Min -10 -Max 10
        ($got | Measure-Object -Minimum).Minimum | Should -BeGreaterOrEqual -10
        ($got | Measure-Object -Maximum).Maximum | Should -BeLessOrEqual 10
        $want = $script:Data | ForEach-Object { [Math]::Max(-10.0, [Math]::Min(10.0, $_)) }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'scales by the factor' {
        $got = Invoke-FlynnelMap -InputObject $script:Data -Operation Scale -Factor 2.5
        $want = $script:Data | ForEach-Object { $_ * 2.5 }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'refuses Clamp without both bounds rather than picking one' {
        { Invoke-FlynnelMap -InputObject 1,2,3 -Operation Clamp -Min 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*both Min and Max*'
    }

    It 'refuses Scale with no Factor' {
        { Invoke-FlynnelMap -InputObject 1,2,3 -Operation Scale -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*needs Factor*'
    }

    It 'answers to its Fly alias' {
        (Invoke-FlyMap -InputObject 3 -Operation Square) | Should -Be 9
    }

    It 'reports the plan it ran under on the verbose stream' {
        $verbose = Invoke-FlynnelMap -InputObject $script:Data -Operation Abs -Verbose 4>&1 |
            Where-Object { $_ -is [System.Management.Automation.VerboseRecord] }
        ($verbose -join ' ') | Should -Match 'worker\(s\)'
    }
}

Describe 'Update-FlynnelArray' {
    It 'changes the caller''s own array' {
        $x = [double[]]@(2.0, 3.0, 4.0)
        Update-FlynnelArray -InputObject $x -Operation Square
        $x | Should -Be @(4.0, 9.0, 16.0)
    }

    It 'writes nothing to the pipeline' {
        # The whole point: the answer is the buffer, not a return.
        $x = [double[]]@(1.0, 2.0)
        $out = @(Update-FlynnelArray -InputObject $x -Operation Negate)
        $out.Count | Should -Be 0
    }

    It 'agrees with the copying form' {
        $source = [double[]](1..500 | ForEach-Object { [double]$_ })
        $copied = Invoke-FlynnelMap -InputObject $source -Operation Sqrt
        $inPlace = [double[]]::new(500)
        [Array]::Copy($source, $inPlace, 500)
        Update-FlynnelArray -InputObject $inPlace -Operation Sqrt
        (Compare-Object $inPlace $copied -SyncWindow 0).Count | Should -Be 0
    }

    It 'refuses an untyped collection rather than quietly copying it' {
        # A silent copy would answer correctly and cost exactly what
        # this cmdlet exists to avoid, and the caller's array would
        # not change, which is worse than an error.
        $boxed = 1..4 | ForEach-Object { [double]$_ }
        { Update-FlynnelArray -InputObject $boxed -Operation Square -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*typed double array*'
    }

    It 'refuses an array of the same width and another type, and leaves it alone' {
        # The case above is refused by the runtime, which declines to
        # pin a boxed collection at all. A long array is blittable and
        # eight bytes wide, so a check on width alone would pin it; what
        # refuses it is the pin's check on the element type. Without
        # that it is read as doubles and written back, and these four
        # values squared come back as zeroes in the caller's own array.
        $longs = [long[]]@(1, 2, 3, 4)
        { Update-FlynnelArray -InputObject $longs -Operation Square -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*typed double array*'
        $longs | Should -Be @(1, 2, 3, 4)
    }

    It 'takes the same operands as the copying form' {
        $x = [double[]]@(-5.0, 0.5, 20.0)
        Update-FlynnelArray -InputObject $x -Operation Clamp -Min 0 -Max 1
        $x | Should -Be @(0.0, 0.5, 1.0)
    }

    It 'refuses Clamp without both bounds' {
        $x = [double[]]@(1.0)
        { Update-FlynnelArray -InputObject $x -Operation Clamp -Min 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*both Min and Max*'
    }
}

Describe 'Invoke-FlynnelZip' {
    It 'adds two arrays elementwise' {
        $got = Invoke-FlynnelZip -Left $script:Data -Right $script:Other -Operation Add
        $want = 0..($script:N - 1) | ForEach-Object { $script:Data[$_] + $script:Other[$_] }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'takes the smaller of each pair' {
        $got = Invoke-FlynnelZip -Left $script:Data -Right $script:Other -Operation Min
        $want = 0..($script:N - 1) |
            ForEach-Object { [Math]::Min($script:Data[$_], $script:Other[$_]) }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'refuses two arrays of different lengths' {
        { Invoke-FlynnelZip -Left 1,2,3 -Right 1,2 -Operation Add -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*same length*'
    }

    It 'leaves the right operand alone' {
        $right = @(1.0, 2.0, 3.0)
        $null = Invoke-FlynnelZip -Left @(9.0, 9.0, 9.0) -Right $right -Operation Add
        $right | Should -Be @(1.0, 2.0, 3.0)
    }
}

Describe 'Measure-FlynnelReduce' {
    It 'sums to what Measure-Object sums to' {
        $got = Measure-FlynnelReduce -InputObject $script:Data -Operation Sum
        $want = ($script:Data | Measure-Object -Sum).Sum
        $got.Count | Should -Be $script:N
        [Math]::Abs($got.Value - $want) | Should -BeLessThan 1e-6
    }

    It 'finds the same smallest and largest' {
        $stats = $script:Data | Measure-Object -Minimum -Maximum
        (Measure-FlynnelReduce -InputObject $script:Data -Operation Min).Value |
            Should -Be $stats.Minimum
        (Measure-FlynnelReduce -InputObject $script:Data -Operation Max).Value |
            Should -Be $stats.Maximum
    }

    It 'means to what Measure-Object averages to' {
        $got = (Measure-FlynnelReduce -InputObject $script:Data -Operation Mean).Value
        $want = ($script:Data | Measure-Object -Average).Average
        [Math]::Abs($got - $want) | Should -BeLessThan 1e-9
    }

    It 'gives the population variance' {
        $mean = ($script:Data | Measure-Object -Average).Average
        $want = ($script:Data | ForEach-Object { ($_ - $mean) * ($_ - $mean) } |
            Measure-Object -Sum).Sum / $script:N
        $got = (Measure-FlynnelReduce -InputObject $script:Data -Operation Variance).Value
        [Math]::Abs($got - $want) / $want | Should -BeLessThan 1e-9
    }

    It 'counts what falls inside the range' {
        $got = (Measure-FlynnelReduce -InputObject $script:Data -Operation CountMatching `
            -Min 0 -Max 100).Value
        $want = ($script:Data | Where-Object { $_ -ge 0 -and $_ -le 100 }).Count
        $got | Should -Be $want
    }

    It 'refuses CountMatching without both bounds' {
        { Measure-FlynnelReduce -InputObject 1,2,3 -Operation CountMatching -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*both Min and Max*'
    }

    It 'gives an unmeasurable reduction no value, rather than a zero one' {
        # The campaign's recurring defect: a figure that was never
        # measured arriving as a zero a reader takes for an answer.
        $value = (Get-FlynnelTypeProperty -TypeName 'Flynnel.Reduction' |
            Where-Object Name -eq 'Value')
        $value.PropertyType | Should -Be ([System.Nullable[double]])
    }
}

Describe 'Get-FlynnelPrefixSum' {
    It 'matches a running total computed here' {
        $got = Get-FlynnelPrefixSum -InputObject $script:Data
        $want = New-Object double[] $script:N
        $acc = 0.0
        for ($i = 0; $i -lt $script:N; $i++) {
            $acc += $script:Data[$i]
            $want[$i] = $acc
        }
        $got.Count | Should -Be $script:N
        # The last element is the total, and it is the one every chunk
        # offset has to have been right for.
        [Math]::Abs($got[-1] - $want[-1]) | Should -BeLessThan 1e-6
        [Math]::Abs($got[0] - $want[0]) | Should -BeLessThan 1e-12
        $mid = [int]($script:N / 2)
        [Math]::Abs($got[$mid] - $want[$mid]) | Should -BeLessThan 1e-6
    }

    It 'never goes down over non-negative input' {
        $positive = 1..2000 | ForEach-Object { [double]$_ }
        $got = Get-FlynnelPrefixSum -InputObject $positive
        $falls = 0
        for ($i = 1; $i -lt $got.Count; $i++) {
            if ($got[$i] -lt $got[$i - 1]) { $falls++ }
        }
        $falls | Should -Be 0
    }
}

Describe 'Get-FlynnelHistogram' {
    It 'bins every element exactly once' {
        $bins = Get-FlynnelHistogram -InputObject $script:Data -Bins 16
        @($bins).Count | Should -Be 16
        ($bins | Measure-Object -Property Count -Sum).Sum | Should -Be $script:N
    }

    It 'agrees with a binning computed here' {
        $bins = Get-FlynnelHistogram -InputObject $script:Data -Bins 8 -Min -400 -Max 600
        $width = (600 - (-400)) / 8
        $want = New-Object 'long[]' 8
        foreach ($x in $script:Data) {
            if ($x -lt -400 -or $x -gt 600) { continue }
            # Floor, not [int]: the cast rounds, and half the values
            # would land one bin high against a kernel that truncates.
            $slot = [Math]::Min(7, [int][Math]::Floor(($x - (-400)) / $width))
            $want[$slot]++
        }
        for ($i = 0; $i -lt 8; $i++) {
            $bins[$i].Count | Should -Be $want[$i]
        }
    }

    It 'covers the range with contiguous bins' {
        $bins = Get-FlynnelHistogram -InputObject $script:Data -Bins 4 -Min 0 -Max 8
        $bins[0].Low | Should -Be 0
        $bins[3].High | Should -Be 8
        for ($i = 1; $i -lt 4; $i++) {
            $bins[$i].Low | Should -Be $bins[$i - 1].High
        }
    }

    It 'refuses zero bins' {
        { Get-FlynnelHistogram -InputObject 1,2,3 -Bins 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*at least one*'
    }

    It 'answers one record for the whole histogram with AsArray' {
        $whole = Get-FlynnelHistogram -InputObject $script:Data -Bins 16 -AsArray
        @($whole).Count | Should -Be 1
        $whole.Counts.Count | Should -Be 16
        ($whole.Counts | Measure-Object -Sum).Sum | Should -Be $script:N
    }

    It 'counts the same bins either way' {
        $bins  = Get-FlynnelHistogram -InputObject $script:Data -Bins 12 -Min -400 -Max 600
        $whole = Get-FlynnelHistogram -InputObject $script:Data -Bins 12 -Min -400 -Max 600 -AsArray
        for ($i = 0; $i -lt 12; $i++) {
            $whole.Counts[$i] | Should -Be $bins[$i].Count
        }
    }

    It 'carries a range that reconstructs the bins it replaced' {
        $bins  = Get-FlynnelHistogram -InputObject $script:Data -Bins 4 -Min 0 -Max 8
        $whole = Get-FlynnelHistogram -InputObject $script:Data -Bins 4 -Min 0 -Max 8 -AsArray
        $whole.Low   | Should -Be 0
        $whole.High  | Should -Be 8
        $whole.Width | Should -Be 2
        for ($i = 0; $i -lt 4; $i++) {
            ($whole.Low + $whole.Width * $i)       | Should -Be $bins[$i].Low
            ($whole.Low + $whole.Width * ($i + 1)) | Should -Be $bins[$i].High
        }
    }

    It 'answers a typed array of counts rather than boxed objects' {
        # The whole point of the shape. An Object[] here would cost the
        # caller 235.8 ns an element to hand back to another kernel,
        # against 0.7 for a typed one.
        $whole = Get-FlynnelHistogram -InputObject $script:Data -Bins 16 -AsArray
        $whole.Counts.GetType().IsArray | Should -BeTrue
        $whole.Counts.GetType().GetElementType().FullName |
            Should -Not -Be 'System.Object'
    }

    It 'refuses an empty array the same way with AsArray as without' {
        # PowerShell will not bind an empty array to a mandatory
        # parameter, so the kernel's own empty-input branch is not
        # reachable from a script at all and there is no "writes
        # nothing" behavior to assert. What is worth holding is that
        # AsArray does not move where that refusal happens.
        { Get-FlynnelHistogram -InputObject ([double[]]@()) -Bins 4 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*empty array*'
        { Get-FlynnelHistogram -InputObject ([double[]]@()) -Bins 4 -AsArray -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*empty array*'
    }
}

Describe 'Get-FlynnelDotProduct' {
    It 'matches a dot product computed here' {
        $got = Get-FlynnelDotProduct -Left $script:Data -Right $script:Other
        $want = 0.0
        for ($i = 0; $i -lt $script:N; $i++) {
            $want += $script:Data[$i] * $script:Other[$i]
        }
        [Math]::Abs($got - $want) / [Math]::Abs($want) | Should -BeLessThan 1e-9
    }

    It 'refuses two arrays of different lengths' {
        { Get-FlynnelDotProduct -Left 1,2,3 -Right 1,2 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*same length*'
    }
}

Describe 'Invoke-FlynnelSort' {
    It 'orders exactly as Sort-Object does' {
        $got = Invoke-FlynnelSort -InputObject $script:Data
        $want = $script:Data | Sort-Object
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'reverses for Descending' {
        $got = Invoke-FlynnelSort -InputObject $script:Data -Descending
        $want = $script:Data | Sort-Object -Descending
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'keeps every element, including the repeats' {
        $got = Invoke-FlynnelSort -InputObject $script:Data
        $got.Count | Should -Be $script:N
        ($got | Measure-Object -Sum).Sum |
            Should -Be ($script:Data | Measure-Object -Sum).Sum
    }

    It 'sorts an odd run count, where the last run has no partner' {
        # A sort cuts runs of at least 4,096 elements, so 13,000 is three
        # runs and the third waits a round for a partner.
        $odd = 1..13000 | ForEach-Object { [double]((7919 * $_) % 9973) }
        $got = Invoke-FlynnelSort -InputObject $odd
        $want = $odd | Sort-Object
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'answers the same under Sort-FlynnelArray and Sort-FlyArray' {
        $want = Invoke-FlynnelSort -InputObject $script:Data -Descending
        foreach ($name in 'Sort-FlynnelArray', 'Sort-FlyArray') {
            $got = & $name -InputObject $script:Data -Descending
            (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0 -Because "$name is an alias"
        }
    }
}

Describe 'Measure-FlynnelFileHash' {
    BeforeAll {
        $script:Empty = Join-Path $script:Work 'empty.bin'
        Set-Content -LiteralPath $script:Empty -Value '' -NoNewline
        $script:A = Join-Path $script:Work 'a.txt'
        Set-Content -LiteralPath $script:A -Value 'hello flynnel' -NoNewline
        $script:ACopy = Join-Path $script:Work 'a-copy.txt'
        Set-Content -LiteralPath $script:ACopy -Value 'hello flynnel' -NoNewline
        $script:B = Join-Path $script:Work 'b.txt'
        Set-Content -LiteralPath $script:B -Value 'hello flynnei' -NoNewline
    }

    It 'gives the published BLAKE3 root for an empty file' {
        $row = Measure-FlynnelFileHash -Path $script:Empty
        $row.Hash | Should -Be $script:EmptyBlake3
        $row.Bytes | Should -Be 0
    }

    It 'gives one row per file, in the order asked for' {
        $rows = @(Measure-FlynnelFileHash -Path @($script:A, $script:B, $script:Empty))
        $rows.Count | Should -Be 3
        $rows[0].Path | Should -Be $script:A
        $rows[1].Path | Should -Be $script:B
        $rows[2].Path | Should -Be $script:Empty
    }

    It 'answers the same roots whichever pool the reads ran on' {
        # The route changes where a file is read, never what it
        # hashes to. Compared against the arena answer on the same
        # files rather than against a stored constant, so the two
        # paths are held to each other.
        $onArena = @(Measure-FlynnelFileHash -Path @($script:A, $script:B, $script:Empty))
        $onPool = @(Measure-FlynnelFileHash -Path @($script:A, $script:B, $script:Empty) -UseIoPool)
        $onPool.Count | Should -Be $onArena.Count
        for ($i = 0; $i -lt $onArena.Count; $i++) {
            $onPool[$i].Path | Should -Be $onArena[$i].Path
            $onPool[$i].Hash | Should -Be $onArena[$i].Hash
            $onPool[$i].Bytes | Should -Be $onArena[$i].Bytes
        }
    }

    It 'says so when it is asked for a pool the process does not have' {
        # A pool exists only when FLYNNEL_SCHED_SMT_AS_IO was set
        # before the first dispatch, so the default session has none
        # and this is the branch a script actually meets. The warning
        # is the whole point: the crate's own submit helper runs the
        # task inline when there is no pool, which would leave the
        # rows identical and the route unreported.
        $warnings = @()
        $rows = @(Measure-FlynnelFileHash -Path $script:A -UseIoPool -WarningVariable +warnings)
        $rows.Count | Should -Be 1
        $rows[0].Hash | Should -Not -BeNullOrEmpty
        if (-not [bool]$env:FLYNNEL_SCHED_SMT_AS_IO) {
            ($warnings -join ' ') | Should -Match 'no IO pool'
        }
    }

    It 'gives identical content the same root and different content a different one' {
        $rows = @(Measure-FlynnelFileHash -Path @($script:A, $script:ACopy, $script:B))
        $rows[0].Hash | Should -Be $rows[1].Hash
        $rows[0].Hash | Should -Not -Be $rows[2].Hash
    }

    It 'reports the byte length it read' {
        (Measure-FlynnelFileHash -Path $script:A).Bytes |
            Should -Be (Get-Item -LiteralPath $script:A).Length
    }

    It 'gives a split file the same root as a streamed one' {
        # The check that matters for the tree path. A file on its own
        # is split across workers, each hashing a subtree at its true
        # input offset; the same file in a list of two is streamed on
        # one thread. The two must agree, and they only can if every
        # offset and every merge is right. A wrong offset gives a
        # well-formed hash that is simply the wrong one, which nothing
        # else here would catch.
        #
        # The size is over the one-megabyte split threshold and has a
        # ragged tail, so the rightmost subtree is short and the
        # recursion cannot be right by accident on a power of two.
        $big = Join-Path $script:Work 'big.bin'
        $bytes = [byte[]]::new(3 * 1MB + 777)
        [System.Random]::new(20260920).NextBytes($bytes)
        [System.IO.File]::WriteAllBytes($big, $bytes)

        $split = Measure-FlynnelFileHash -Path $big
        $streamed = @(Measure-FlynnelFileHash -Path @($big, $script:A)) |
            Where-Object Path -eq $big

        $split.Hash | Should -Be $streamed.Hash
        $split.Bytes | Should -Be $bytes.Length
        $split.Hash.Length | Should -Be 64
    }

    It 'agrees across the split threshold' {
        # Just under the threshold takes the single-hasher path and
        # just over takes the tree. A file that grows by one byte must
        # not change which answer is correct, so both are checked
        # against the streamed path on the same content.
        foreach ($size in (1MB - 1), (1MB + 1)) {
            $path = Join-Path $script:Work "edge-$size.bin"
            $bytes = [byte[]]::new($size)
            [System.Random]::new($size).NextBytes($bytes)
            [System.IO.File]::WriteAllBytes($path, $bytes)
            $alone = (Measure-FlynnelFileHash -Path $path).Hash
            $withOther = (@(Measure-FlynnelFileHash -Path @($path, $script:A)) |
                Where-Object Path -eq $path).Hash
            $alone | Should -Be $withOther -Because "size $size must hash the same either way"
        }
    }

    It 'writes an error for a file it cannot read and keeps going' {
        $missing = Join-Path $script:Work 'not-here.bin'
        $errors = @()
        $rows = @(Measure-FlynnelFileHash -Path @($script:A, $missing, $script:B) `
            -ErrorVariable errors -ErrorAction SilentlyContinue -WarningAction SilentlyContinue)
        $rows.Count | Should -Be 2
        $errors.Count | Should -BeGreaterThan 0
        ($errors -join ' ') | Should -Match 'not-here'
    }

    It 'counts the refusals in a warning' {
        $warnings = @()
        $missing = Join-Path $script:Work 'also-not-here.bin'
        $null = Measure-FlynnelFileHash -Path @($script:A, $missing) `
            -WarningVariable warnings -ErrorAction SilentlyContinue
        ($warnings -join ' ') | Should -Match '1 of 2'
    }
}

Describe 'Test-FlynnelFileHash' {
    BeforeAll {
        $script:Checked = Join-Path $script:Work 'checked.txt'
        Set-Content -LiteralPath $script:Checked -Value 'manifest subject' -NoNewline
        $script:CheckedHash = (Measure-FlynnelFileHash -Path $script:Checked).Hash
    }

    It 'passes a file whose root matches' {
        $row = Test-FlynnelFileHash -Path $script:Checked -Manifest $script:CheckedHash
        $row.IsMatch | Should -BeTrue
        $row.Actual | Should -Be $script:CheckedHash
    }

    It 'ignores the case of the expected root' {
        $row = Test-FlynnelFileHash -Path $script:Checked `
            -Manifest $script:CheckedHash.ToUpperInvariant()
        $row.IsMatch | Should -BeTrue
    }

    It 'fails a file whose root does not match' {
        $wrong = '0' * 64
        $row = Test-FlynnelFileHash -Path $script:Checked -Manifest $wrong
        $row.IsMatch | Should -BeFalse
        $row.Expected | Should -Be $wrong
        $row.Actual | Should -Be $script:CheckedHash
    }

    It 'gives an unreadable file a row saying so, not an absent row' {
        # A manifest check whose failure mode is a missing row cannot be
        # used to decide anything, because the caller counting rows
        # cannot tell a pass from an absence.
        $missing = Join-Path $script:Work 'gone.txt'
        $rows = @(Test-FlynnelFileHash -Path @($script:Checked, $missing) `
            -Manifest @($script:CheckedHash, ('0' * 64)) `
            -ErrorAction SilentlyContinue -WarningAction SilentlyContinue)
        $rows.Count | Should -Be 2
        $gone = $rows | Where-Object Path -eq $missing
        $gone.Actual | Should -BeNullOrEmpty
        $gone.IsMatch | Should -BeFalse
    }

    It 'refuses a manifest of a different length from the path list' {
        { Test-FlynnelFileHash -Path @('a', 'b') -Manifest @('x') -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*exactly one*'
    }
}

Describe 'Search-FlynnelFile' {
    BeforeAll {
        $script:Hay = Join-Path $script:Work 'hay.txt'
        $lines = 1..500 | ForEach-Object {
            if ($_ % 50 -eq 0) { "line $_ NEEDLE here" } else { "line $_ ordinary" }
        }
        Set-Content -LiteralPath $script:Hay -Value $lines
    }

    It 'finds the same lines Select-String finds' {
        $got = @(Search-FlynnelFile -Pattern 'NEEDLE' -Path $script:Hay)
        $want = @(Select-String -LiteralPath $script:Hay -Pattern 'NEEDLE' -SimpleMatch)
        $got.Count | Should -Be $want.Count
        $got[0].LineNumber | Should -Be $want[0].LineNumber
        $got[-1].LineNumber | Should -Be $want[-1].LineNumber
    }

    It 'gives the line without its terminator' {
        $got = @(Search-FlynnelFile -Pattern 'NEEDLE' -Path $script:Hay)
        $got[0].Line | Should -Be 'line 50 NEEDLE here'
    }

    It 'matches case-insensitively only when asked' {
        @(Search-FlynnelFile -Pattern 'needle' -Path $script:Hay).Count | Should -Be 0
        @(Search-FlynnelFile -Pattern 'needle' -Path $script:Hay -IgnoreCase).Count |
            Should -Be 10
    }

    It 'refuses an empty pattern' {
        # PowerShell's own binder refuses it before the cmdlet runs,
        # which is earlier and better than the cmdlet's own check. The
        # assertion is that it is refused, not by whom.
        { Search-FlynnelFile -Pattern '' -Path $script:Hay -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*empty*'
    }
}

Describe 'Measure-FlynnelFileLine and Measure-FlynnelFileByte' {
    BeforeAll {
        $script:ThreeLines = Join-Path $script:Work 'three.txt'
        Set-Content -LiteralPath $script:ThreeLines -Value @('one', 'two', 'three')
        $script:NoNewline = Join-Path $script:Work 'bare.txt'
        Set-Content -LiteralPath $script:NoNewline -Value 'just one' -NoNewline
    }

    It 'counts the lines Get-Content counts' {
        (Measure-FlynnelFileLine -Path $script:ThreeLines).Lines |
            Should -Be (Get-Content -LiteralPath $script:ThreeLines).Count
    }

    It 'counts a final line with no terminator' {
        (Measure-FlynnelFileLine -Path $script:NoNewline).Lines | Should -Be 1
    }

    It 'reports the byte length the file system reports' {
        (Measure-FlynnelFileByte -Path $script:ThreeLines).Bytes |
            Should -Be (Get-Item -LiteralPath $script:ThreeLines).Length
    }

    It 'measures many files in one call' {
        $rows = @(Measure-FlynnelFileByte -Path @($script:ThreeLines, $script:NoNewline))
        $rows.Count | Should -Be 2
    }
}

Describe 'Search-FlynnelText' {
    BeforeAll {
        # About 400 KB, so six blocks, with the pattern on every line and
        # so on either side of every boundary.
        $script:Text = (1..20000 | ForEach-Object { "row $_ value ABCD" }) -join "`n"
    }

    It 'finds every occurrence a .NET scan finds' {
        $got = @(Search-FlynnelText -Text $script:Text -Pattern 'ABCD')
        $got.Count | Should -Be 20000
    }

    It 'reports offsets that really hold the pattern' {
        $got = @(Search-FlynnelText -Text $script:Text -Pattern 'ABCD')
        foreach ($i in 0, 1, 9999, 19999) {
            $at = $got[$i].Index
            $script:Text.Substring($at, 4) | Should -Be 'ABCD'
        }
    }

    It 'reports offsets in ascending order with no repeats' {
        $got = @(Search-FlynnelText -Text $script:Text -Pattern 'ABCD')
        $out_of_order = 0
        for ($i = 1; $i -lt $got.Count; $i++) {
            if ($got[$i].Index -le $got[$i - 1].Index) { $out_of_order++ }
        }
        $out_of_order | Should -Be 0
    }

    It 'numbers lines from one and gives the whole line' {
        $got = @(Search-FlynnelText -Text $script:Text -Pattern 'ABCD')
        $got[0].LineNumber | Should -Be 1
        $got[0].Line | Should -Be 'row 1 value ABCD'
        $got[4].LineNumber | Should -Be 5
    }

    It 'finds a match that straddles a block boundary' {
        # 200,006 bytes is three blocks of 66,669, so a pattern starting
        # at 66,666 crosses the first boundary. It is found only if the
        # block that owns its first byte reads past its own end.
        $big = ('x' * 66666) + 'MIDDLE' + ('y' * 133334)
        $got = @(Search-FlynnelText -Text $big -Pattern 'MIDDLE')
        $got.Count | Should -Be 1
        $got[0].Index | Should -Be 66666
    }
}

Describe 'Measure-FlynnelTextCount' {
    It 'counts bytes, lines and words as .NET does' {
        $text = "alpha beta`ngamma delta epsilon`n"
        $got = Measure-FlynnelTextCount -Text $text
        $got.Bytes | Should -Be ([System.Text.Encoding]::UTF8.GetByteCount($text))
        $got.Lines | Should -Be 2
        $got.Words | Should -Be 5
    }

    It 'counts a final line with no terminator' {
        (Measure-FlynnelTextCount -Text "one`ntwo").Lines | Should -Be 2
    }

    It 'counts words across a block boundary exactly once' {
        $text = (1..40000 | ForEach-Object { "w$_" }) -join ' '
        (Measure-FlynnelTextCount -Text $text).Words | Should -Be 40000
    }

    It 'gives no match count when no pattern was given' {
        (Measure-FlynnelTextCount -Text 'abc').Matches | Should -BeNullOrEmpty
    }

    It 'counts non-overlapping occurrences' {
        # Four overlapping places hold 'aa' in 'aaaaa'; two survive a
        # left-to-right non-overlapping walk.
        (Measure-FlynnelTextCount -Text 'aaaaa' -Pattern 'aa').Matches | Should -Be 2
    }

    It 'counts zero for a pattern that is not there' {
        (Measure-FlynnelTextCount -Text 'abc' -Pattern 'zzz').Matches | Should -Be 0
    }
}

Describe 'Split-FlynnelText' {
    It 'splits exactly as the .NET string does' {
        $text = (1..30000 | ForEach-Object { "field$_" }) -join ','
        $got = Split-FlynnelText -Text $text -Separator ','
        $want = $text.Split(',')
        $got.Count | Should -Be $want.Count
        $got[0] | Should -Be $want[0]
        $got[-1] | Should -Be $want[-1]
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'keeps the empty pieces two separators produce' {
        $got = Split-FlynnelText -Text 'a,,b' -Separator ','
        $got.Count | Should -Be 3
        $got[1] | Should -Be ''
    }

    It 'drops them for NoEmpty' {
        (Split-FlynnelText -Text 'a,,b' -Separator ',' -NoEmpty).Count | Should -Be 2
    }

    It 'splits on a multi-character separator' {
        $got = Split-FlynnelText -Text 'a<->b<->c' -Separator '<->'
        $got | Should -Be @('a', 'b', 'c')
    }

    It 'refuses an empty separator' {
        # Refused by the binder before the cmdlet runs, as with an
        # empty pattern.
        { Split-FlynnelText -Text 'abc' -Separator '' -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*empty*'
    }
}

Describe 'Update-FlynnelText' {
    It 'replaces exactly as the .NET string does' {
        $text = (1..30000 | ForEach-Object { "row $_ OLD" }) -join "`n"
        $got = Update-FlynnelText -Text $text -Operation Replace -Pattern 'OLD' -Replacement 'NEW'
        $got | Should -Be $text.Replace('OLD', 'NEW')
    }

    It 'deletes the pattern when no replacement is given' {
        (Update-FlynnelText -Text 'a-b-c' -Operation Replace -Pattern '-') | Should -Be 'abc'
    }

    It 'replaces a match that straddles a block boundary' {
        # The same construction as the search's: the match crosses the
        # first of three blocks' boundaries.
        $big = ('x' * 66666) + 'MIDDLE' + ('y' * 133334)
        $got = Update-FlynnelText -Text $big -Operation Replace -Pattern 'MIDDLE' -Replacement 'M'
        $got.Length | Should -Be ($big.Length - 5)
        $got | Should -Be $big.Replace('MIDDLE', 'M')
    }

    It 'upper-cases and lower-cases as .NET does' {
        $text = (1..30000 | ForEach-Object { "Mixed Case Row $_" }) -join "`n"
        (Update-FlynnelText -Text $text -Operation ToUpper) | Should -Be $text.ToUpper()
        (Update-FlynnelText -Text $text -Operation ToLower) | Should -Be $text.ToLower()
    }

    It 'does not cut a multi-byte character in half' {
        # Chunking by byte count can land inside a UTF-8 sequence. The
        # answer is checked whole, so a split character shows up as a
        # mismatch rather than as a crash nobody notices.
        $text = ('caf' + [char]0xE9 + ' na' + [char]0xEF + 've ') * 20000
        (Update-FlynnelText -Text $text -Operation ToUpper) | Should -Be $text.ToUpper()
    }

    It 'refuses Replace with no pattern' {
        { Update-FlynnelText -Text 'abc' -Operation Replace -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*needs Pattern*'
    }
}

Describe 'the plan a kernel runs under' {
    It 'takes the caller plan when given one' {
        $plan = New-FlynnelPlan -KOuter 10 -BatchSize $script:N -Workers 1
        $got = Invoke-FlynnelMap -InputObject $script:Data -Operation Square -Plan $plan
        $want = $script:Data | ForEach-Object { $_ * $_ }
        (Compare-Object $got $want -SyncWindow 0).Count | Should -Be 0
    }

    It 'gives the same answer on one worker as on all of them' {
        $pinned = New-FlynnelPlan -KOuter 10 -BatchSize $script:N -Workers 1
        $one = (Measure-FlynnelReduce -InputObject $script:Data -Operation Sum -Plan $pinned).Value
        $many = (Measure-FlynnelReduce -InputObject $script:Data -Operation Sum).Value
        $one | Should -Be $many
    }

    It 'names the pinned worker count on the verbose stream' {
        $plan = New-FlynnelPlan -KOuter 10 -BatchSize 1000 -Workers 1
        $verbose = Invoke-FlynnelMap -InputObject (1..1000) -Operation Abs -Plan $plan -Verbose 4>&1 |
            Where-Object { $_ -is [System.Management.Automation.VerboseRecord] }
        ($verbose -join ' ') | Should -Match '1 worker\(s\)'
    }

    It 'gives the same bits under any plan, because the blocks come from the input' {
        # Non-integer values, so the order of the additions shows in the
        # last bits wherever it changes. The blocks are cut from the
        # length and folded in block order, so no plan can change it.
        $x = [double[]](1..$script:N | ForEach-Object { 1.0 / $_ })
        $pinned = New-FlynnelPlan -KOuter 10 -BatchSize $script:N -Workers 1
        $one = (Measure-FlynnelReduce -InputObject $x -Operation Sum -Plan $pinned).Value
        $many = (Measure-FlynnelReduce -InputObject $x -Operation Sum).Value
        [BitConverter]::DoubleToInt64Bits($one) | Should -Be ([BitConverter]::DoubleToInt64Bits($many))
        $dotOne = Get-FlynnelDotProduct -Left $x -Right $x -Plan $pinned
        $dotMany = Get-FlynnelDotProduct -Left $x -Right $x
        [BitConverter]::DoubleToInt64Bits($dotOne) | Should -Be ([BitConverter]::DoubleToInt64Bits($dotMany))
    }
}

Describe 'the bulk constructors' {
    It 'builds each row class from its columns' {
        $rows = [Flynnel.TextMatch]::FromColumns([uint64[]](0, 7), [uint64[]](1, 2), [string[]]('a', 'b'))
        $rows.Count | Should -Be 2
        $rows[1].GetType().FullName | Should -Be 'Flynnel.TextMatch'
        $rows[1].Index | Should -Be 7
        $rows[1].LineNumber | Should -Be 2
        $rows[1].Line | Should -Be 'b'

        $files = [Flynnel.FileMatch]::FromColumns([string[]]('p', 'q'), [uint64[]](3, 4), [string[]]('x', 'y'))
        $files[0].Path | Should -Be 'p'
        $measures = [Flynnel.FileMeasure]::FromColumns([string[]]('p'), [uint64[]](5), [uint64[]](6))
        $measures[0].Lines | Should -Be 5
        $hashes = [Flynnel.FileHash]::FromColumns([string[]]('p'), [string[]]('ab'), [uint64[]](9))
        $hashes[0].Hash | Should -Be 'ab'
        $bins = [Flynnel.HistogramBin]::FromColumns([uint32[]](0, 1), [double[]](0, 1), [double[]](1, 2), [uint64[]](3, 4))
        $bins[1].High | Should -Be 2
        $bins[1].Count | Should -Be 4
    }

    It 'turns an empty Actual into a null, because a real root is never empty' {
        $checks = [Flynnel.HashCheck]::FromColumns(
            [string[]]('a', 'b'), [string[]]('x', 'y'), [string[]]('x', ''), [bool[]]($true, $false))
        $checks[0].Actual | Should -Be 'x'
        $checks[1].Actual | Should -BeNullOrEmpty
        $checks[1].IsMatch | Should -BeFalse
    }

    It 'refuses columns of different lengths, naming both' {
        { [Flynnel.TextMatch]::FromColumns([uint64[]](0, 1), [uint64[]](1), [string[]]('a')) } |
            Should -Throw -ExpectedMessage '*Index has 2 element(s) and LineNumber has 1*'
    }
}

Describe 'Measure-FlynnelPrimitive' {
    It 'answers the same sum through every primitive' {
        # Squares of integers stay exact in a double at this size, so
        # every split sums to the same value.
        $x = [double[]](1..$script:N)
        $want = 0.0
        foreach ($v in $x) { $want += $v * $v }
        foreach ($primitive in [Enum]::GetValues([Flynnel.Primitive])) {
            $run = Measure-FlynnelPrimitive -InputObject $x -Primitive $primitive
            $run.Primitive | Should -Be $primitive
            $run.Operation | Should -Be ([Flynnel.MapOp]::Square)
            $run.Count | Should -Be $script:N
            $run.Sum | Should -Be $want -Because "$primitive must sum what a serial pass sums"
            $run.ElapsedNs | Should -BeGreaterThan 0
        }
    }

    It 'refuses an operation that needs an operand it does not take' {
        { Measure-FlynnelPrimitive -InputObject 1, 2, 3 -Primitive ForEachChunk -Operation Clamp -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*Clamp needs both Min and Max*'
    }
}
