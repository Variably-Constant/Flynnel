# The GPU peer family.
#
# This suite runs on a host with a card and on one without, and the two
# runs assert different things. A deviceless run that quietly passed
# every device assertion would be the worst outcome available here, so
# the device assertions are skipped BY NAME and the suite prints which
# host it found before it asserts anything.
#
# What is bound so far is the watchdog reading, which is the part of
# this family that answers on every host: it says what bounds a piece of
# device work, and on a machine with no card it says that nothing does
# and why.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost

    # What this host actually has, read once and printed, so a reader of
    # the log knows which arm of every skip was taken.
    $script:Watchdog = Get-FlynnelPeerWatchdog
    $script:HasCard = $script:Watchdog.DriverModelKnown

    Write-Host ("GPUPEER_SUITE host={0}/{1} platform={2} model_known={3} model={4} applies={5}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform,
        $script:Watchdog.DriverModelKnown, $script:Watchdog.DriverModel,
        $script:Watchdog.Applies)
    Write-Host ("GPUPEER_BASIS {0}" -f $script:Watchdog.Basis)
}

Describe 'the types and the enum this family exports' {
    It 'shapes the watchdog row the way its cmdlet documents' {
        @(Get-FlynnelTypeProperty -TypeName 'Flynnel.PeerWatchdog').Count |
            Should -BeGreaterThan 0
    }

    It 'names every driver model, including the unreadable one' {
        # Unknown is a value rather than a null, because an unreadable
        # model is a reading and the device is still treated as covered.
        $names = [enum]::GetNames([Flynnel.DriverModel])
        $names | Should -Contain 'Unknown'
        $names | Should -Contain 'Wddm'
        $names | Should -Contain 'Tcc'
        $names | Should -Contain 'Mcdm'
    }

    It 'makes the delay nullable rather than zero-valued' {
        # A zero here would read as a bound of no time at all. The two
        # facts that have to stay apart are "resets after two seconds"
        # and "nothing resets it".
        $p = Get-FlynnelTypeProperty -TypeName 'Flynnel.PeerWatchdog' |
            Where-Object Name -eq 'DelayNs'
        $p.PropertyType.FullName | Should -Match 'Nullable'
    }
}

Describe 'Get-FlynnelPeerWatchdog on any host' {
    It 'answers a row rather than nothing' {
        # An absent device is a row saying so, never a missing row: a
        # script cannot act on a listing that failed to enumerate.
        $script:Watchdog | Should -Not -BeNullOrEmpty
    }

    It 'always says what it read' {
        # The column to quote when a bound looks wrong. It names which
        # of the two reads decided the answer, including the ones that
        # failed.
        $script:Watchdog.Basis | Should -Not -BeNullOrEmpty
        $script:Watchdog.Basis | Should -Match 'driver model'
    }

    It 'echoes the ordinal it was asked about' {
        (Get-FlynnelPeerWatchdog -Ordinal 0).Ordinal | Should -Be 0
        (Get-FlynnelPeerWatchdog -Ordinal 3).Ordinal | Should -Be 3
    }

    It 'answers for an ordinal no device has, rather than throwing' {
        # Sizing work against a device that is not there is a mistake
        # worth an answer, not an exception: the answer is that the read
        # failed and the documented bound was taken.
        $r = Get-FlynnelPeerWatchdog -Ordinal 99
        $r.DriverModelKnown | Should -BeFalse
        $r.DriverModelProblem | Should -Not -BeNullOrEmpty
        $r.DriverModel | Should -Be 'Unknown'
    }

    It 'keeps the delay and the seconds in step' {
        $r = $script:Watchdog
        if ($null -eq $r.DelayNs) {
            $r.DelaySeconds | Should -BeNullOrEmpty
            $r.Applies | Should -BeFalse
        } else {
            $r.Applies | Should -BeTrue
            [math]::Abs($r.DelaySeconds - ($r.DelayNs / 1e9)) | Should -BeLessThan 1e-6
        }
    }

    It 'answers the same reading twice' {
        # The reading is cached for the life of the process, because
        # neither the hardware nor the driver configuration can change
        # under it. Two different answers would mean the cache is keyed
        # wrong.
        $a = Get-FlynnelPeerWatchdog -Ordinal 0
        $b = Get-FlynnelPeerWatchdog -Ordinal 0
        $a.Basis | Should -Be $b.Basis
        $a.DelayNs | Should -Be $b.DelayNs
    }

    It 'names an unreadable model Unknown and says why' {
        $r = $script:Watchdog
        if ($r.DriverModelKnown) {
            $r.DriverModelProblem | Should -BeNullOrEmpty
            $r.DriverModel | Should -Not -Be 'Unknown'
        } else {
            $r.DriverModelProblem | Should -Not -BeNullOrEmpty
            $r.DriverModel | Should -Be 'Unknown'
        }
    }
}

Describe 'Get-FlynnelPeerWatchdog where there is no readable device' {
    It 'treats an unreadable model as covered rather than as absent' {
        # The direction is deliberate and it is the whole safety
        # argument: a watchdog that is present and treated as absent
        # ends in a device reset, while one treated as present only
        # shortens slices. On Windows an unreadable model still takes
        # the documented delay; on a platform with no watchdog at all
        # there is nothing to take.
        $r = Get-FlynnelPeerWatchdog -Ordinal 99
        if ($script:TestHost.Platform -ne 'Windows') {
            Set-ItResult -Skipped -Because 'only Windows has a watchdog to be covered by'
            return
        }
        $r.Applies | Should -BeTrue
        $r.DelaySeconds | Should -BeGreaterThan 0
        $r.Basis | Should -Match 'unreadable'
    }

    It 'says no watchdog is known off Windows' {
        if ($script:TestHost.Platform -eq 'Windows') {
            Set-ItResult -Skipped -Because 'this host is Windows and has TDR'
            return
        }
        $script:Watchdog.Applies | Should -BeFalse
        $script:Watchdog.Basis | Should -Match 'no watchdog is known'
    }
}

Describe 'Get-FlynnelPeerWatchdog where a device is readable' {
    It 'reports the model the device presents' {
        if (-not $script:HasCard) {
            Set-ItResult -Skipped -Because 'no driver model could be read on this host'
            return
        }
        $script:Watchdog.DriverModel | Should -BeIn @('Wddm', 'Tcc', 'Mcdm')
        $script:Watchdog.Basis | Should -Match ('driver model ' + $script:Watchdog.DriverModel)
    }

    It 'bounds work on a covered device and does not on a Tcc one' {
        if (-not $script:HasCard) {
            Set-ItResult -Skipped -Because 'no driver model could be read on this host'
            return
        }
        if ($script:Watchdog.DriverModel -eq 'Tcc') {
            $script:Watchdog.Applies | Should -BeFalse
            $script:Watchdog.Basis | Should -Match 'outside TDR'
        } elseif ($script:TestHost.Platform -eq 'Windows') {
            # Level zero turns detection off, which is a third answer
            # and not the same as an uncovered device.
            if ($script:Watchdog.Applies) {
                $script:Watchdog.Basis | Should -Match 'TdrLevel'
                $script:Watchdog.DelaySeconds | Should -BeGreaterThan 0
            } else {
                $script:Watchdog.Basis | Should -Match 'TdrLevel 0'
            }
        }
    }
}
