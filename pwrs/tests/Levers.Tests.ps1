# The lever family: that every switch has a row, that a set which did
# not take says so, and that the allowed width is a number the host
# agrees with.
#
# What this file cannot test, said first. A lever latches on its first
# read and holds for the life of the process, and this suite shares a
# process with every suite that ran before it - each of which has
# dispatched, and dispatching reads levers. So by the time anything
# here runs, the latching levers are already resolved and the case
# "a set before the first read takes effect" is not reachable from
# inside a shared session. It is skipped by name below rather than
# quietly omitted, and the case that is reachable, a set that does not
# take, is asserted rather than assumed.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    # Reading resolves. This is deliberate and it happens here rather
    # than inside a test, so every assertion below runs against levers
    # in the same state and no test depends on running first.
    $script:Levers = @(Get-FlynnelLever)
    $script:Width = Get-FlynnelAllowedWidth
    $script:TestHost = Get-FlynnelTestHost

    Write-Host ("LEVER_SUITE host={0}/{1} platform={2} width={3}/{4}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform,
        $script:Width.Width, $script:Width.LogicalProcessors)
}

Describe 'the types and enums this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.Lever', 'Flynnel.AllowedWidth', 'Flynnel.ServePolicyState') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }

    It 'names every serve policy the crate has' {
        $names = [enum]::GetNames([Flynnel.ServePolicy])
        foreach ($p in 'Spread', 'Any', 'Occupancy', 'SpreadAt') {
            $names | Should -Contain $p
        }
    }
}

Describe 'Get-FlynnelLever' {
    It 'writes a row for every switch the crate reports' {
        # The list the crate's own describe prints, which is what a
        # reader of its logs will have seen, and the switch it leaves
        # out, which a script can still set.
        $names = @($script:Levers | ForEach-Object Name)
        foreach ($lever in 'oncore_spread', 'batch_weight', 'smt_window', 'allowed_width',
                           'calibration_refusal', 'latch_monitor', 'spin_monitor', 'join_park',
                           'join_park_oversubscribed', 'slot_park_now', 'spin_adaptive',
                           'spin_window', 'serve_policy', 'occupancy_floor',
                           'wasm_local_store') {
            $names | Should -Contain $lever
        }
    }

    It 'gives every row a name and a value' {
        foreach ($row in $script:Levers) {
            $row.Name | Should -Not -BeNullOrEmpty
            $row.Value | Should -Not -BeNullOrEmpty
        }
    }

    It 'says which levers latch and which do not' {
        # The spin pair has real setters and can move at any time; the
        # rest resolve once. A row that got this wrong would tell a
        # caller a set will take when it cannot.
        $latching = @($script:Levers | Where-Object LatchesOnFirstRead | ForEach-Object Name)
        $free = @($script:Levers | Where-Object { -not $_.LatchesOnFirstRead } | ForEach-Object Name)
        $latching | Should -Contain 'oncore_spread'
        $latching | Should -Contain 'serve_policy'
        $free | Should -Contain 'spin_adaptive'
        $free | Should -Contain 'spin_window'
    }

    It 'gives the levers that latch an environment variable and the others none' {
        foreach ($row in $script:Levers | Where-Object LatchesOnFirstRead) {
            $row.Variable | Should -Match '^FLYNNEL_' -Because "$($row.Name) is set through the environment"
        }
        foreach ($row in $script:Levers | Where-Object { -not $_.LatchesOnFirstRead }) {
            $row.Variable | Should -BeNullOrEmpty -Because "$($row.Name) has its own setter"
        }
    }

    It 'carries a measured price on each switch that defaults on' {
        # A switch that ships on has to say what it costs. A switch that
        # nothing has measured yet carries no price, and that is honest
        # rather than an omission.
        foreach ($name in 'oncore_spread', 'allowed_width', 'calibration_refusal',
                          'join_park_oversubscribed') {
            $row = $script:Levers | Where-Object Name -eq $name
            $row.Default | Should -Be 'true'
            $row.Price | Should -Not -BeNullOrEmpty -Because "$name ships on and must say what it costs"
        }
        foreach ($name in 'batch_weight', 'smt_window', 'latch_monitor', 'spin_monitor',
                          'join_park', 'slot_park_now', 'wasm_local_store') {
            $row = $script:Levers | Where-Object Name -eq $name
            $row.Default | Should -Be 'false'
        }
    }

    It 'says why each switch that was measured and kept off stays off' {
        # The row is where a script reads why a switch it might reach for
        # is off, so a switch a measurement turned down carries it.
        foreach ($name in 'join_park', 'slot_park_now', 'wasm_local_store') {
            $row = $script:Levers | Where-Object Name -eq $name
            $row.Price | Should -Not -BeNullOrEmpty -Because "$name was measured and is off"
        }
    }

    It 'answers for one lever when asked' {
        $one = Get-FlynnelLever -Name oncore_spread
        @($one).Count | Should -Be 1
        $one.Name | Should -Be 'oncore_spread'
    }

    It 'refuses an unknown lever and names the ones there are' {
        { Get-FlynnelLever -Name 'no_such_lever' } |
            Should -Throw -ExpectedMessage '*oncore_spread*'
    }

    It 'answers to its Fly alias' {
        @(Get-FlyLever).Count | Should -Be $script:Levers.Count
    }
}

Describe 'Set-FlynnelLever' {
    It 'refuses a lever that is not set through the environment' {
        # spin_adaptive has a real setter. Pointing a caller at the
        # environment for it would be pointing them at nothing.
        { Set-FlynnelLever -Name spin_adaptive -Value on } |
            Should -Throw -ExpectedMessage '*own setter*'
    }

    It 'refuses an unknown lever and names the ones there are' {
        { Set-FlynnelLever -Name 'no_such_lever' -Value on } |
            Should -Throw -ExpectedMessage '*oncore_spread*'
    }

    It 'warns and reports the old value when the lever has already resolved' {
        # This is the case the family exists for. BeforeAll read every
        # lever, so oncore_spread is resolved for certain; setting it
        # to the opposite must change the variable and leave the value
        # alone, and must say so rather than returning a row that looks
        # like a successful set.
        $before = Get-FlynnelLever -Name oncore_spread
        $opposite = if ($before.Value -eq 'true') { 'off' } else { 'on' }

        $after = Set-FlynnelLever -Name oncore_spread -Value $opposite -WarningVariable warned

        $after.Value | Should -Be $before.Value -Because 'a resolved lever does not move'
        $after.VariableValue | Should -Be $opposite -Because 'the variable was written even though it did not take'
        $after.EffectiveMatchesVariable | Should -BeFalse
        @($warned).Count | Should -BeGreaterThan 0 -Because 'a set that did not take must say so'
        ($warned -join ' ') | Should -Match 'oncore_spread'
    }

    It 'takes effect on a lever nothing has read' {
        # Not reachable from inside a shared session: every latching
        # lever was resolved by BeforeAll, and by the suites that ran
        # before this one, each of which dispatched. A test that set a
        # lever and asserted it took would be asserting about a process
        # that does not exist here.
        Set-ItResult -Skipped -Because 'every latching lever is already resolved in this process; the take-effect path needs a fresh process and is covered by the crate test that starts one'
    }
}

Describe 'Get-FlynnelAllowedWidth' {
    It 'reports at least one CPU' {
        $script:Width.Width | Should -BeGreaterThan 0
    }

    It 'never reports more than the machine has' {
        if ($script:Width.LogicalProcessors -gt 0) {
            $script:Width.Width | Should -BeLessOrEqual $script:Width.LogicalProcessors
        } else {
            Set-ItResult -Skipped -Because "this host would not report its processor count: $($script:Width.LogicalProcessorsProblem)"
        }
    }

    It 'says whether the process is narrowed, and agrees with the two numbers' {
        $expected = ($script:Width.LogicalProcessors -gt 0) -and
                    ($script:Width.Width -lt $script:Width.LogicalProcessors)
        $script:Width.IsNarrowed | Should -Be $expected
    }

    It 'reports the width whether or not the lever is on' {
        # The lever decides whether a plan is capped by the width, not
        # whether the width is read. A row that returned nothing with
        # the lever off would hide the number a caller came for.
        $script:Width.LeverOn | Should -BeOfType [bool]
        $script:Width.Width | Should -BeGreaterThan 0
    }

    It 'agrees with the host about the machine' {
        if ($script:Width.LogicalProcessors -gt 0) {
            $script:Width.LogicalProcessors | Should -Be ([Environment]::ProcessorCount)
        } else {
            Set-ItResult -Skipped -Because 'the processor count was not readable'
        }
    }
}

Describe 'Get-FlynnelServePolicy' {
    It 'reports a policy and says whether it carries a bound' {
        $p = Get-FlynnelServePolicy
        $p.Policy | Should -BeIn @('Spread', 'Any', 'Occupancy', 'SpreadAt')
        $p.HasBound | Should -BeOfType [bool]
        if (-not $p.HasBound) {
            # Zero with HasBound false is the absence of a bound, not a
            # bound of zero, which would admit nothing.
            $p.SpreadBoundPerMille | Should -Be 0
        } else {
            $p.SpreadBoundPerMille | Should -BeGreaterThan 0
        }
    }

    It 'names the variable that sets it' {
        (Get-FlynnelServePolicy).Variable | Should -Be 'FLYNNEL_SERVE_POLICY'
    }
}

Describe 'Get-FlynnelOccupancyFloor' {
    It 'reports the floor as a lever row' {
        $f = Get-FlynnelOccupancyFloor
        $f.Name | Should -Be 'occupancy_floor'
        [int]$f.Value | Should -BeGreaterThan 0
        $f.Default | Should -Be '900'
    }
}
