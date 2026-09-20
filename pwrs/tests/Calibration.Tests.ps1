# The calibration family: that every figure says where it came from,
# and that a default is never mistaken for a measurement.
#
# No assertion here is on the size of a measured number. These suites
# run on two hosts under whatever load they carry, and a threshold on a
# dispatch cost would fail for the box rather than for the code. What
# is asserted is provenance, ordering, and that a setter's read-back
# agrees with what was set.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
}

Describe 'the types this family exports' {
    It 'shapes each type the way its cmdlet documents' {
        $shape = @{
            'Flynnel.ThresholdCalibration' = @('JoinNs', 'FineGrainNs', 'MeasuredAt')
            'Flynnel.KGatingResult'        = @('PerSlotNs', 'CounterOnlyNs', 'Winner')
        }
        foreach ($type in $shape.Keys) {
            $properties = @(Get-FlynnelTypeProperty -TypeName $type | ForEach-Object Name)
            foreach ($wanted in $shape[$type]) {
                $properties | Should -Contain $wanted -Because "$type must carry $wanted"
            }
        }
    }
}

Describe 'Get-FlynnelCalibration' {
    It 'answers before any calibration has been run' {
        # A getter that needs a measurement first is a getter a script
        # cannot open with.
        { Get-FlynnelCalibration } | Should -Not -Throw
    }

    It 'answers every figure in one call' {
        $c = Get-FlynnelCalibration
        foreach ($name in 'DispatchCostNs', 'CollapseThresholdNs', 'JecWakeThresholdNs',
                          'FineGrainNs', 'PortHeavyNs', 'MemoryLatencyNs',
                          'Cv2LowPerMille', 'Cv2HighPerMille', 'TrivialReduceCycles',
                          'KGating', 'SeedHysteresis', 'ActiveProfile', 'ActiveClass',
                          'HostSource', 'ClassSource') {
            $c.PSObject.Properties.Name | Should -Contain $name
        }
    }

    It 'agrees field for field with the cmdlets it summarizes' {
        # The one-call form is a decomposition of the same state. If it
        # disagrees with the per-figure cmdlets, one of them is lying
        # and the table would be quoted as though it were not.
        $c = Get-FlynnelCalibration
        $t = Get-FlynnelClassThreshold
        $h = Get-FlynnelHostDispatch
        $c.FineGrainNs | Should -Be $t.FineGrainNs
        $c.PortHeavyNs | Should -Be $t.PortHeavyNs
        $c.MemoryLatencyNs | Should -Be $t.MemoryLatencyNs
        $c.Cv2LowPerMille | Should -Be $t.Cv2LowPerMille
        $c.Cv2HighPerMille | Should -Be $t.Cv2HighPerMille
        $c.TrivialReduceCycles | Should -Be $t.TrivialReduceCycles
        $c.ClassSource | Should -Be $t.Source
        $c.DispatchCostNs | Should -Be $h.DispatchCostNs
        $c.CollapseThresholdNs | Should -Be $h.CollapseThresholdNs
        $c.JecWakeThresholdNs | Should -Be $h.JecWakeThresholdNs
        $c.HostSource | Should -Be $h.Source
    }

    It 'names a source for every group of figures' {
        $c = Get-FlynnelCalibration
        $valid = @([Flynnel.Source]::Default, [Flynnel.Source]::Measured,
                   [Flynnel.Source]::Stored, [Flynnel.Source]::Unattributed)
        $c.HostSource | Should -BeIn $valid
        $c.ClassSource | Should -BeIn $valid
    }
}

Describe 'Get-FlynnelClassThreshold' {
    It 'orders its boundaries the way the classifier reads them' {
        # FineGrain below PortHeavy below MemoryLatency, and the low
        # spread bound below the high one. An inversion would make one
        # of the classes unreachable.
        $t = Get-FlynnelClassThreshold
        $t.FineGrainNs | Should -BeLessThan $t.PortHeavyNs
        $t.PortHeavyNs | Should -BeLessThan $t.MemoryLatencyNs
        $t.Cv2LowPerMille | Should -BeLessThan $t.Cv2HighPerMille
    }

    It 'gives a never-measured figure no timestamp, rather than the epoch' {
        $t = Get-FlynnelClassThreshold
        if ($t.Source -ne [Flynnel.Source]::Measured) {
            $t.MeasuredAt | Should -BeNullOrEmpty
        }
    }

    It 'keeps the timestamp nullable' {
        $property = Get-FlynnelTypeProperty -TypeName 'Flynnel.ClassThresholds' |
            Where-Object Name -eq 'MeasuredAt'
        $property.PropertyType | Should -Be ([System.Nullable[System.UInt64]])
    }
}

Describe 'Get-FlynnelHostDispatch' {
    It 'distinguishes a measured collapse threshold from the shipped one' {
        # A default and a measurement that happen to agree must still
        # be distinguishable, which is what the crate's own
        # measured-or-not answer is for.
        $h = Get-FlynnelHostDispatch
        $property = Get-FlynnelTypeProperty -TypeName 'Flynnel.HostDispatch' |
            Where-Object Name -eq 'MeasuredCollapseThresholdNs'
        $property.PropertyType | Should -Be ([System.Nullable[System.UInt64]])
        if ($null -eq $h.MeasuredCollapseThresholdNs) {
            $h.Source | Should -Be ([Flynnel.Source]::Default)
        }
    }

    It 'says whether the call that produced the row paid for the measurement' {
        # The crate measures on the first call in a process, whoever
        # makes it. A caller timing this cmdlet needs to know which
        # call they got.
        # The first call is made for its side effect: whoever pays,
        # pays once, and the assertion is about the call after it.
        $null = Get-FlynnelHostDispatch
        $second = Get-FlynnelHostDispatch
        $second.ThisCallMeasured | Should -BeFalse
        $second.MeasuredCollapseThresholdNs | Should -Not -BeNullOrEmpty
    }

    It 'gives the same figures on two reads with no measurement between' {
        $a = Get-FlynnelHostDispatch
        $b = Get-FlynnelHostDispatch
        $a.DispatchCostNs | Should -Be $b.DispatchCostNs
        $a.CollapseThresholdNs | Should -Be $b.CollapseThresholdNs
        $a.JecWakeThresholdNs | Should -Be $b.JecWakeThresholdNs
    }
}

Describe 'Measure-FlynnelHostDispatch' {
    It 'flips the source from Default to Measured and sets the timestamp' {
        $before = Get-FlynnelHostDispatch
        $result = Measure-FlynnelHostDispatch
        $result.Source | Should -Be ([Flynnel.Source]::Measured)
        $result.MeasuredAt | Should -Not -BeNullOrEmpty
        $result.ThisCallMeasured | Should -BeTrue
        # And the getter agrees afterwards, rather than the provenance
        # living only in the measuring call's own answer.
        $after = Get-FlynnelHostDispatch
        $after.Source | Should -Be ([Flynnel.Source]::Measured)
        $after.MeasuredAt | Should -Be $result.MeasuredAt
        $before.MeasuredAt | Should -Not -Be $after.MeasuredAt -Because 'a measurement happened'
    }

    It 'stamps a plausible time' {
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $result = Measure-FlynnelHostDispatch
        [Math]::Abs($result.MeasuredAt - $now) | Should -BeLessThan 300
    }
}

Describe 'Measure-FlynnelClassThreshold' {
    It 'answers every figure it measured and stamps when' {
        $result = Measure-FlynnelClassThreshold
        $result.JoinNs | Should -BeGreaterThan 0
        $result.FineGrainNs | Should -BeGreaterThan 0
        $result.MemoryLatencyNs | Should -BeGreaterThan 0
        $result.TrivialReduceCycles | Should -BeGreaterThan 0
        $result.MeasuredAt | Should -BeGreaterThan 0
    }

    It 'flips the threshold source to Measured' {
        $null = Measure-FlynnelClassThreshold
        $t = Get-FlynnelClassThreshold
        $t.Source | Should -Be ([Flynnel.Source]::Measured)
        $t.MeasuredAt | Should -Not -BeNullOrEmpty
    }

    It 'installs boundaries that still order correctly' {
        # A calibration that inverts two boundaries makes one class
        # unreachable, and nothing else would report it.
        $null = Measure-FlynnelClassThreshold
        $t = Get-FlynnelClassThreshold
        $t.FineGrainNs | Should -BeLessThan $t.PortHeavyNs
        $t.PortHeavyNs | Should -BeLessThan $t.MemoryLatencyNs
        $t.Cv2LowPerMille | Should -BeLessThan $t.Cv2HighPerMille
    }

    It 'writes nothing for Background, and says why' {
        # The crate's spawned form answers no handle and no result, so
        # a row here would be invented rather than measured.
        $warnings = @()
        $output = Measure-FlynnelClassThreshold -Background `
            -WarningVariable warnings
        $output | Should -BeNullOrEmpty
        ($warnings -join ' ') | Should -Match 'no handle'
    }
}

Describe 'Measure-FlynnelKGating' {
    It 'times both schemes and names a winner' {
        $result = Measure-FlynnelKGating
        $result.PerSlotNs | Should -BeGreaterThan 0
        $result.CounterOnlyNs | Should -BeGreaterThan 0
        $result.Winner | Should -BeIn @([Flynnel.KGating]::CounterOnly,
                                        [Flynnel.KGating]::PerSlot)
    }

    It 'picks the cheaper of the two it timed' {
        # The winner is the whole point of the measurement. A winner
        # that is not the cheaper arm means the comparison inverted.
        $result = Measure-FlynnelKGating
        if ($result.Winner -eq [Flynnel.KGating]::PerSlot) {
            $result.PerSlotNs | Should -BeLessOrEqual $result.CounterOnlyNs
        } else {
            $result.CounterOnlyNs | Should -BeLessOrEqual $result.PerSlotNs
        }
    }

    It 'marks itself measured and stamps when' {
        $result = Measure-FlynnelKGating
        $result.Source | Should -Be ([Flynnel.Source]::Measured)
        $result.MeasuredAt | Should -BeGreaterThan 0
    }
}

Describe 'Get-FlynnelKGating' {
    It 'resolves to one of the two real schemes' {
        # Auto is a request, not a resolution. A resolved reading of
        # Auto would mean the calibration never ran and nobody noticed.
        Get-FlynnelKGating | Should -BeIn @([Flynnel.KGating]::CounterOnly,
                                            [Flynnel.KGating]::PerSlot)
    }
}

Describe 'Get-FlynnelSeedHysteresis and Set-FlynnelSeedHysteresis' {
    AfterEach {
        # Left as found, so one test's arm is not the next one's
        # starting condition.
        $null = Set-FlynnelSeedHysteresis -On
    }

    It 'reads without changing what it read' {
        $first = Get-FlynnelSeedHysteresis
        $second = Get-FlynnelSeedHysteresis
        $second | Should -Be $first
    }

    It 'answers the value it replaced' {
        $null = Set-FlynnelSeedHysteresis -On
        $was = Set-FlynnelSeedHysteresis -On:$false
        $was | Should -BeTrue
        Get-FlynnelSeedHysteresis | Should -BeFalse
    }

    It 'takes, and reads back as what it was set to' {
        $null = Set-FlynnelSeedHysteresis -On:$false
        Get-FlynnelSeedHysteresis | Should -BeFalse
        $null = Set-FlynnelSeedHysteresis -On
        Get-FlynnelSeedHysteresis | Should -BeTrue
    }
}

Describe 'Get-FlynnelWorkloadClass' {
    It 'classifies a leaf below the fine-grain boundary as FineGrain' {
        # Taken from the live boundary rather than a constant, so a
        # calibration that moved it does not fail this for the wrong
        # reason.
        $boundary = (Get-FlynnelClassThreshold).FineGrainNs
        Get-FlynnelWorkloadClass -MeanNs ([Math]::Max(1, $boundary - 1)) -Cv2PerMille 0 |
            Should -Be ([Flynnel.WorkloadClass]::FineGrain)
    }

    It 'never answers the same class for every input' {
        # A classifier that answers one class whatever it is given has
        # stopped classifying, and nothing else here would notice.
        $seen = @{}
        foreach ($mean in 1, 100, 1000, 100000) {
            foreach ($cv in 0, 100, 5000) {
                $seen[[string](Get-FlynnelWorkloadClass -MeanNs $mean -Cv2PerMille $cv)] = $true
            }
        }
        $seen.Keys.Count | Should -BeGreaterThan 1
    }

    It 'gives the same answer twice for the same input' {
        Get-FlynnelWorkloadClass -MeanNs 300 -Cv2PerMille 40 |
            Should -Be (Get-FlynnelWorkloadClass -MeanNs 300 -Cv2PerMille 40)
    }

    It 'refuses a mean of zero' {
        { Get-FlynnelWorkloadClass -MeanNs 0 -Cv2PerMille 0 -ErrorAction Stop } |
            Should -Throw -ExpectedMessage '*above zero*'
    }
}

Describe 'Get-FlynnelHostStamp' {
    It 'describes this host' {
        $stamp = Get-FlynnelHostStamp
        $stamp.TotalWorkers | Should -BeGreaterThan 0
        $stamp.PrimaryWorkers | Should -BeGreaterThan 0
        $stamp.PrimaryWorkers | Should -BeLessOrEqual $stamp.TotalWorkers
        $stamp.Hash | Should -Not -Be 0
        $stamp.Canonical | Should -Not -BeNullOrEmpty
    }

    It 'gives the same stamp twice' {
        (Get-FlynnelHostStamp).Hash | Should -Be (Get-FlynnelHostStamp).Hash
    }

    It 'names the layout version the build understands' {
        (Get-FlynnelHostStamp).LayoutVersion | Should -BeGreaterThan 0
    }
}

Describe 'Get-FlynnelCalibrationStore' {
    It 'treats an absent table as a row, not a failure' {
        # A host that has never calibrated has no table. That is the
        # normal first state and a cmdlet that threw for it would be
        # unusable on a fresh box.
        $store = Get-FlynnelCalibrationStore -ErrorAction SilentlyContinue
        if ($null -eq $store) {
            Set-ItResult -Skipped -Because 'no calibration directory is configured on this host'
            return
        }
        $store.Path | Should -Not -BeNullOrEmpty
        $store.ProcessStampHash | Should -Be (Get-FlynnelHostStamp).Hash
    }

    It 'separates a failed read from an empty table' {
        $store = Get-FlynnelCalibrationStore -ErrorAction SilentlyContinue
        if ($null -eq $store) {
            Set-ItResult -Skipped -Because 'no calibration directory is configured on this host'
            return
        }
        if (-not $store.Exists) {
            $store.ReadSettled | Should -BeFalse
            $store.CpuSamples | Should -BeNullOrEmpty
        } elseif ($store.ReadSettled) {
            $store.CpuSamples | Should -Not -BeNullOrEmpty
        }
    }

    It 'says whether the stamps agree rather than leaving it to be inferred' {
        $store = Get-FlynnelCalibrationStore -ErrorAction SilentlyContinue
        if ($null -eq $store -or -not $store.Exists) {
            Set-ItResult -Skipped -Because 'no table exists on this host'
            return
        }
        $store.StampMatches | Should -Be ($store.StampHash -eq $store.ProcessStampHash)
    }
}

Describe 'Get-FlynnelCpuCalibration' {
    It 'gives an unpublished record no timestamp, rather than the epoch' {
        $record = Get-FlynnelCpuCalibration -WarningAction SilentlyContinue `
            -ErrorAction SilentlyContinue
        if ($null -eq $record) {
            Set-ItResult -Skipped -Because 'no stored record on this host'
            return
        }
        if ($record.Samples -eq 0) {
            $record.MeasuredAt | Should -BeNullOrEmpty
            $record.SpreadPerMille | Should -BeNullOrEmpty
            $record.IsTrustworthy | Should -BeFalse
        } else {
            $record.MeasuredAt | Should -BeGreaterThan 0
        }
    }

    It 'marks its provenance as stored' {
        $record = Get-FlynnelCpuCalibration -WarningAction SilentlyContinue `
            -ErrorAction SilentlyContinue
        if ($null -eq $record) {
            Set-ItResult -Skipped -Because 'no stored record on this host'
            return
        }
        $record.Source | Should -Be ([Flynnel.Source]::Stored)
    }
}

Describe 'Get-FlynnelAccelCalibration' {
    It 'answers without throwing on a host with no device' {
        { Get-FlynnelAccelCalibration -WarningAction SilentlyContinue `
            -ErrorAction SilentlyContinue } | Should -Not -Throw
    }

    It 'names an unrecognized device kind rather than calling it none' {
        # A record written by a newer build must not read as "no
        # device", which is a different fact.
        $rows = @(Get-FlynnelAccelCalibration -WarningAction SilentlyContinue `
            -ErrorAction SilentlyContinue)
        foreach ($row in $rows) {
            if ($row.Kind -eq [Flynnel.AccelKind]::Unknown) {
                $row.KindRaw | Should -Not -Be 0
            }
        }
    }

    It 'carries the whole wave record rather than its width alone' {
        # Asserted against the type, not a row: no device here means no
        # stored wave costs to read, and a binding that keeps one field
        # of an eight-field record drops seven with nothing to show it.
        $names = @([Flynnel.AccelCalibration].GetProperties() | ForEach-Object Name)
        foreach ($field in @(
            'WaveWidth', 'WaveBarrierNs', 'WaveFixedNs', 'WaveSegmentPs',
            'WaveRebalanceFixedNs', 'WaveCopyPsPerId', 'WaveSkewNs',
            'WaveGenerationNs')) {
            $names | Should -Contain $field
        }
    }

    It 'nulls every wave figure together, or none of them' {
        # The crate marks "no wave costs" with a zero width, and this
        # module turns that into null. A row with a width and no timings,
        # or timings and no width, would be a half-read record.
        $rows = @(Get-FlynnelAccelCalibration -WarningAction SilentlyContinue `
            -ErrorAction SilentlyContinue)
        foreach ($row in $rows) {
            $figures = @($row.WaveBarrierNs, $row.WaveFixedNs, $row.WaveSegmentPs,
                         $row.WaveRebalanceFixedNs, $row.WaveCopyPsPerId,
                         $row.WaveSkewNs, $row.WaveGenerationNs)
            $present = @($figures | Where-Object { $null -ne $_ }).Count
            if ($null -eq $row.WaveWidth) {
                $present | Should -Be 0
            } else {
                $present | Should -Be 7
            }
        }
    }
}
