# The cross-process worker family.
#
# What is bound so far answers without a peer process, so every
# assertion on the cmdlets runs on every host. The routing decision is
# made from the shape and the host; the registry reading is a reading of
# this process. Neither starts a peer. The last block does: it runs the
# crate's four cross-process examples, each of which starts its own peer.
#
# The routing assertions check what the table DECIDES, not only that it
# answers. A suite that accepted any variant would pass for a
# dispatcher that returned the same one for everything, which is the
# failure worth catching in a router.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost
    Write-Host ("CROSSPROC_SUITE host={0}/{1} platform={2}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform)
}

Describe 'the types and the enum this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.DequeVariantInfo', 'Flynnel.CrossProcessRoute',
                          'Flynnel.CrossProcessMeasurement', 'Flynnel.PassRegistry') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }

    It 'names all four deque variants' {
        $names = [enum]::GetNames([Flynnel.DequeVariant])
        $names | Should -Contain 'ChaseLev'
        $names | Should -Contain 'Loh'
        $names | Should -Contain 'Khpd'
        $names | Should -Contain 'Urd'
    }
}

Describe 'Get-FlynnelCrossProcessVariant' {
    It 'lists every variant once' {
        $rows = @(Get-FlynnelCrossProcessVariant)
        $rows.Count | Should -Be 4
        @($rows | Select-Object -ExpandProperty Variant -Unique).Count | Should -Be 4
    }

    It 'names exactly one default' {
        @(Get-FlynnelCrossProcessVariant | Where-Object IsDefault).Count | Should -Be 1
        (Get-FlynnelCrossProcessVariant | Where-Object IsDefault).Variant |
            Should -Be 'ChaseLev'
    }

    It 'gives each variant a label and a positive inline ceiling' {
        foreach ($r in Get-FlynnelCrossProcessVariant) {
            $r.Label | Should -Not -BeNullOrEmpty
            $r.InlineArgsBytes | Should -BeGreaterThan 0
        }
    }

    It 'gives the default the widest inline slot' {
        # The ceiling is what gates a routing, so which variant holds
        # the most is a fact a caller sizing a payload depends on.
        $rows = @(Get-FlynnelCrossProcessVariant)
        $widest = ($rows | Sort-Object InlineArgsBytes -Descending | Select-Object -First 1)
        $widest.Variant | Should -Be 'ChaseLev'
    }
}

Describe 'Get-FlynnelCrossProcessRoute' {
    It 'answers a route for the plain request-and-reply shape' {
        $r = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8
        $r | Should -Not -BeNullOrEmpty
        $r.Label | Should -Not -BeNullOrEmpty
        $r.NDrainThreads | Should -Be 1
        $r.ExpectedBurstSize | Should -Be 1
    }

    It 'echoes the shape it was asked about' {
        $r = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 40 -NDrainThreads 6 `
            -ExpectedBurstSize 32
        $r.ArgsInlineBytes | Should -Be 40
        $r.NDrainThreads | Should -Be 6
        $r.ExpectedBurstSize | Should -Be 32
    }

    It 'routes a payload too wide for the narrow variants to one that holds it' {
        # 40 bytes clears the two 8-byte variants, so a router that
        # chose one of them would be choosing a slot the payload does
        # not fit.
        $r = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 40
        $r.PayloadFits | Should -BeTrue
        $r.InlineArgsBytes | Should -BeGreaterOrEqual 40
    }

    It 'does not give every shape the same answer' {
        # The assertion that catches a router returning a constant. A
        # single drain thread with one item and eight drain threads
        # with a long burst are the two ends of the win zones, so if
        # anything moves the routing, these do.
        $a = (Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8).Variant
        $b = (Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8 -NDrainThreads 8 `
            -ExpectedBurstSize 64).Variant
        $c = (Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8 -ExpectedBurstSize 64).Variant
        @($a, $b, $c) | Select-Object -Unique |
            Should -Not -HaveCount 1 -Because 'the win zones differ by shape'
    }

    It 'says whether a pinned cell or the rule answered' {
        # The table is a small map of overrides over a fixed rule, and
        # a caller reading a routing needs to know which one it got.
        $r = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8
        $r.FromExplicitCell | Should -BeOfType [bool]
        if (-not $r.FromExplicitCell) {
            $r.Variant | Should -Be $r.HeuristicVariant `
                -Because 'with no pinned cell the rule is what answered'
        }
    }

    It 'reports how many cells the table pins' {
        (Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8).ExplicitCells |
            Should -BeGreaterOrEqual 0
    }

    It 'answers the same route twice for the same shape' {
        $a = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8 -NDrainThreads 4
        $b = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8 -NDrainThreads 4
        $a.Variant | Should -Be $b.Variant
    }

    It 'refuses a shape field too large for the byte it travels in' {
        # A wrapped 256 would route as zero and read as a correct
        # answer for a payload that does not exist.
        { Get-FlynnelCrossProcessRoute -ArgsInlineBytes 256 } |
            Should -Throw -ExpectedMessage '*at most 255*'
        { Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8 -KUnified 999 } |
            Should -Throw -ExpectedMessage '*at most 255*'
    }
}

Describe 'Get-FlynnelPassRegistry' {
    It 'answers a count with nothing asked about' {
        $r = Get-FlynnelPassRegistry
        $r.Count | Should -BeGreaterOrEqual 0
    }

    It 'leaves Registered null when nothing was asked about' {
        # Null and false are different answers. A caller reading only
        # the count must not see a false here and take it for an answer
        # about some pass.
        $r = Get-FlynnelPassRegistry
        $r.Registered | Should -BeNullOrEmpty
        $r.Id | Should -BeNullOrEmpty
    }

    It 'hashes a name to the id that travels on the wire' {
        $r = Get-FlynnelPassRegistry -Name 'gemm.f64'
        $r.Name | Should -Be 'gemm.f64'
        $r.Id | Should -Not -BeNullOrEmpty
        $r.Registered | Should -BeOfType [bool]
    }

    It 'hashes the same name to the same id' {
        # The id is what two processes compare, so a hash that moved
        # between calls would make a match impossible to rely on.
        (Get-FlynnelPassRegistry -Name 'pass.one').Id |
            Should -Be (Get-FlynnelPassRegistry -Name 'pass.one').Id
    }

    It 'hashes different names to different ids' {
        (Get-FlynnelPassRegistry -Name 'pass.one').Id |
            Should -Not -Be (Get-FlynnelPassRegistry -Name 'pass.two').Id
    }

    It 'says a pass nothing registered is not registered' {
        # This process registers none, so the honest answer is false
        # with the id it asked about, rather than an empty row.
        $r = Get-FlynnelPassRegistry -Name 'nothing.has.registered.this'
        $r.Registered | Should -BeFalse
        $r.Id | Should -Not -BeNullOrEmpty
    }

    It 'takes an id directly' {
        $r = Get-FlynnelPassRegistry -Id 4242
        $r.Id | Should -Be 4242
        $r.Name | Should -BeNullOrEmpty
        $r.Registered | Should -BeFalse
    }

    It 'refuses a name and an id together' {
        { Get-FlynnelPassRegistry -Name 'a' -Id 1 } |
            Should -Throw -ExpectedMessage '*not both*'
    }
}

Describe 'Measure-FlynnelCrossProcessRouting' {
    # Timing on whatever host runs the suite, so nothing here asserts on
    # the size of a cost: only on which variants answered, that the
    # fastest is the cheapest of them, and that the routing table was
    # left as it was.

    It 'measures every variant once and names one fastest' {
        $rows = @(Measure-FlynnelCrossProcessRouting -ArgsInlineBytes 8 -Iterations 32)
        $rows.Count | Should -Be 4
        @($rows | ForEach-Object { [string]$_.Variant } | Sort-Object -Unique).Count | Should -Be 4
        $fastest = @($rows | Where-Object Fastest)
        $fastest.Count | Should -Be 1
        $fastest[0].NsPerCall | Should -Not -BeNullOrEmpty
        foreach ($row in $rows) {
            if ($null -ne $row.NsPerCall) {
                $row.NsPerCall | Should -BeGreaterThan 0
                $fastest[0].NsPerCall | Should -BeLessOrEqual $row.NsPerCall
            }
            $row.Iterations | Should -Be 32
        }
    }

    It 'gives a variant that cannot carry the arguments no cost, rather than a zero one' {
        # KHPD carries eight argument bytes inline, so sixteen do not fit.
        $rows = @(Measure-FlynnelCrossProcessRouting -ArgsInlineBytes 16 -Iterations 16)
        $khpd = $rows | Where-Object { [string]$_.Variant -eq 'Khpd' }
        $khpd | Should -Not -BeNullOrEmpty
        $khpd.NsPerCall | Should -BeNullOrEmpty
        $khpd.Fastest | Should -BeFalse
    }

    It 'reports the routed variant beside the measurement, and pins nothing' {
        $before = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8
        $rows = @(Measure-FlynnelCrossProcessRouting -ArgsInlineBytes 8 -Iterations 16)
        foreach ($row in $rows) {
            [string]$row.RoutedVariant | Should -Be ([string]$before.Variant)
        }
        $after = Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8
        [string]$after.Variant | Should -Be ([string]$before.Variant)
        $after.FromExplicitCell | Should -Be $before.FromExplicitCell
        $after.ExplicitCells | Should -Be $before.ExplicitCells
    }

    It 'refuses zero iterations' {
        { Measure-FlynnelCrossProcessRouting -ArgsInlineBytes 8 -Iterations 0 } |
            Should -Throw -ExpectedMessage '*at least one*'
    }
}

Describe 'a peer in another process' {
    # Each of the crate's four cross-process examples is a whole round
    # trip over one deque variant: the originator creates the deque and
    # latch files, starts itself again as the worker on the same files,
    # pushes a hundred add jobs, reads every sum back through the latch
    # arena, tells the worker to exit and removes both files. It exits 1
    # when a sum is wrong or the worker failed.
    #
    # They run from the binaries a release build of the crate leaves in
    # target/release/examples, so cargo is not resident while the suite
    # asserts; a host that has not built them skips, saying which build
    # makes them. An originator spins on its latches with no deadline of
    # its own, so each run has one here, and a run that outlives it is
    # stopped with its worker and fails.

    BeforeAll {
        $script:ExampleDir = Join-Path $PSScriptRoot '..\..\target\release\examples'
        $script:ExampleSuffix = ''
        if ($PSVersionTable.PSEdition -eq 'Desktop' -or $IsWindows) { $script:ExampleSuffix = '.exe' }

        # The example's exit code and every line it wrote, or TimedOut
        # when it ran past $Seconds and was stopped with its worker.
        function Invoke-PeerExample {
            param([string]$Path, [int]$Seconds = 60)
            $info = [System.Diagnostics.ProcessStartInfo]::new($Path)
            $info.UseShellExecute = $false
            $info.RedirectStandardOutput = $true
            $info.RedirectStandardError = $true
            $process = [System.Diagnostics.Process]::Start($info)
            $out = $process.StandardOutput.ReadToEndAsync()
            $err = $process.StandardError.ReadToEndAsync()
            $timedOut = -not $process.WaitForExit($Seconds * 1000)
            if ($timedOut) {
                if ($PSVersionTable.PSEdition -eq 'Desktop') {
                    & taskkill.exe /T /F /PID $process.Id | Out-Null
                } else {
                    $process.Kill($true)
                }
                $process.WaitForExit()
            }
            [PSCustomObject]@{
                TimedOut = $timedOut
                ExitCode = $process.ExitCode
                Lines    = @(($out.Result + $err.Result) -split "`r?`n" | Where-Object { $_ })
            }
        }
    }

    It 'runs <Name> to a verified round trip and removes its files' -ForEach @(
        @{ Name = 'chase_lev_mmf_steal' }
        @{ Name = 'khpd_steal' }
        @{ Name = 'loh_steal' }
        @{ Name = 'urd_steal' }
    ) {
        $exe = Join-Path $script:ExampleDir ($Name + $script:ExampleSuffix)
        if (-not (Test-Path $exe)) {
            Set-ItResult -Skipped -Because "$exe is not built; cargo build --release --examples builds it"
            return
        }
        $run = Invoke-PeerExample -Path $exe
        $run.TimedOut | Should -BeFalse -Because ("it ran past its deadline: " + ($run.Lines -join ' | '))
        $run.ExitCode | Should -Be 0 -Because ($run.Lines -join ' | ')
        ($run.Lines | Where-Object { $_ -match '^originator: all 100 results match expected sums' }) |
            Should -Not -BeNullOrEmpty -Because 'every sum came back through the latch arena'
        ($run.Lines | Where-Object { $_ -match '^originator: worker acked exit' }) |
            Should -Not -BeNullOrEmpty -Because 'the worker drained and acknowledged its exit'
        @($run.Lines | Where-Object { $_ -match '^MISMATCH' }).Count | Should -Be 0
        foreach ($role in 'deque', 'latches') {
            $line = $run.Lines | Where-Object { $_ -match "^originator: $role\s+= " } | Select-Object -First 1
            $line | Should -Not -BeNullOrEmpty -Because "the originator names its $role file"
            $path = ($line -replace "^originator: $role\s+= ", '').Trim()
            Test-Path -LiteralPath $path | Should -BeFalse -Because "the originator removes $path"
        }
    }
}
