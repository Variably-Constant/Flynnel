# Racing: several attempts at one piece of work.
#
# Every arm runs the same declared body, so what separates them is the
# host rather than the work. That is deliberate and it is what these
# cmdlets measure, but it also means an assertion about which arm won
# would be an assertion about the scheduler's mood. The claims here
# are structural, plus the two behavioural ones that actually
# distinguish the shapes: a race returns a winner that is one of the
# arms it fired, and an exploration leaves every arm finished.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost
    Write-Host ("RACING_SUITE host={0}/{1} platform={2} cpus={3}" -f
        $script:TestHost.Edition, $script:TestHost.Version,
        $script:TestHost.Platform, $script:TestHost.ProcessorCount)

    # Enough per arm that a cancel signal has somewhere to land. At a
    # few microseconds an arm, every loser finishes before the winner
    # signals and the cancellation path is never exercised.
    $script:Count = 400000
    $script:Reps = 4
}

Describe 'the types this family exports' {
    It 'shapes both rows the way their cmdlets document' {
        foreach ($type in 'Flynnel.RaceOutcome', 'Flynnel.RaceArm') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }
}

Describe 'Measure-FlynnelRaceAny' {
    It 'answers an outcome naming one of the arms it fired' {
        $r = Measure-FlynnelRaceAny -Count 50000 -Attempts 4 -Operation Sqrt
        $r.Attempts | Should -Be 4
        $r.WinnerIndex | Should -BeGreaterOrEqual 0
        $r.WinnerIndex | Should -BeLessThan 4
    }

    It 'waits for every arm, so the slowest is at least the winner' {
        # The crate's join contract, and the row must not pretend
        # otherwise: cancelling a loser stops it spending more, it does
        # not hand the call back early.
        $r = Measure-FlynnelRaceAny -Count 50000 -Attempts 4 -Operation Sqrt
        $r.SlowestArmNs | Should -BeGreaterOrEqual $r.WinnerNs
        $r.TotalNs | Should -BeGreaterOrEqual $r.WinnerNs
    }

    It 'reports a tail ratio at or above one' {
        # The slowest over the winner. Exactly 1.0 is a race that saved
        # nothing, which is a real outcome and not a failure.
        $r = Measure-FlynnelRaceAny -Count 50000 -Attempts 4 -Operation Sqrt
        $r.TailRatio | Should -BeGreaterOrEqual 1.0
    }

    It 'writes one arm per attempt when asked' {
        $rows = @(Measure-FlynnelRaceAny -Count 20000 -Attempts 6 -Operation Sqrt -IncludeArms)
        $arms = @($rows | Where-Object { $null -ne $_.Index })
        $arms.Count | Should -Be 6
        @($arms | Where-Object Won).Count | Should -Be 1
    }

    It 'gives every arm an index exactly once' {
        # An arm that left no row is refused by the cmdlet rather than
        # skipped, so a gap here would have been an error, not a short
        # listing. This checks the listing is complete anyway.
        $arms = @(Measure-FlynnelRaceAny -Count 20000 -Attempts 5 -Operation Sqrt -IncludeArms |
            Where-Object { $null -ne $_.Index })
        @($arms | Select-Object -ExpandProperty Index -Unique).Count | Should -Be 5
    }

    It 'writes only the outcome when arms are not asked for' {
        $rows = @(Measure-FlynnelRaceAny -Count 20000 -Attempts 4 -Operation Sqrt)
        $rows.Count | Should -Be 1
    }

    It 'stops a loser short, or says every arm beat the signal' {
        # Both are real answers and the row tells them apart. A loser
        # that stopped short did fewer items than it was given; one
        # that finished first did all of them. What must not happen is
        # an arm reporting both that it was canceled and that it
        # finished, which would mean the count and the flag disagree.
        $rows = @(Measure-FlynnelRaceAny -Count $script:Count -Attempts 8 `
            -Operation Exp -Repetitions $script:Reps -IncludeArms)
        $outcome = $rows | Where-Object { $null -ne $_.Attempts } | Select-Object -First 1
        $arms = @($rows | Where-Object { $null -ne $_.Index })

        foreach ($a in $arms) {
            if ($a.CancelledEarly) {
                $a.ItemsDone | Should -BeLessThan $script:Count `
                    -Because 'an arm that stopped short did not finish its items'
                $a.Won | Should -BeFalse -Because 'the winner is never the one canceled'
            } else {
                $a.ItemsDone | Should -Be $script:Count
            }
        }
        $outcome.CancelledEarly | Should -Be @($arms | Where-Object CancelledEarly).Count
    }

    It 'refuses a race of none' {
        { Measure-FlynnelRaceAny -Count 1000 -Attempts 0 -Operation Sqrt } |
            Should -Throw -ExpectedMessage '*above zero*'
    }

    It 'refuses a count of zero' {
        { Measure-FlynnelRaceAny -Count 0 -Operation Sqrt } |
            Should -Throw -ExpectedMessage '*above zero*'
    }
}

Describe 'Measure-FlynnelExploreSelect' {
    It 'runs every arm to completion' {
        # The difference from a race, and the reason this shape exists:
        # a slow explorer that finds the best answer is the point, so
        # nothing is canceled and every arm did all its items.
        $rows = @(Measure-FlynnelExploreSelect -Count 50000 -Attempts 5 `
            -Operation Sqrt -IncludeArms)
        $arms = @($rows | Where-Object { $null -ne $_.Index })
        $arms.Count | Should -Be 5
        foreach ($a in $arms) {
            $a.ItemsDone | Should -Be 50000
            $a.CancelledEarly | Should -BeFalse
        }
    }

    It 'reports no cancellation, because the shape cancels nothing' {
        (Measure-FlynnelExploreSelect -Count 20000 -Attempts 4 -Operation Sqrt).CancelledEarly |
            Should -Be 0
    }

    It 'picks the fastest arm' {
        $rows = @(Measure-FlynnelExploreSelect -Count 50000 -Attempts 6 `
            -Operation Sqrt -IncludeArms)
        $outcome = $rows | Where-Object { $null -ne $_.Attempts } | Select-Object -First 1
        $arms = @($rows | Where-Object { $null -ne $_.Index })
        $fastest = ($arms | Sort-Object ElapsedNs | Select-Object -First 1)
        $outcome.WinnerNs | Should -Be $fastest.ElapsedNs
    }

    It 'gives every arm the same checksum, because every arm ran the same body' {
        # If these diverged the arms would not be interchangeable and
        # neither shape would mean what it says.
        $arms = @(Measure-FlynnelExploreSelect -Count 20000 -Attempts 4 `
            -Operation Sqrt -IncludeArms | Where-Object { $null -ne $_.Index })
        @($arms | Select-Object -ExpandProperty Checksum -Unique).Count | Should -Be 1
    }

    It 'refuses an exploration of none' {
        { Measure-FlynnelExploreSelect -Count 1000 -Attempts 0 -Operation Sqrt } |
            Should -Throw -ExpectedMessage '*above zero*'
    }
}
