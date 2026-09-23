# The size suffixes that the byte-size and large-count parameters read.
#
# pwsh 7 binds a string such as '2MB' by itself and Windows PowerShell
# 5.1 does not, so the checks that matter most are the ones that run on
# 5.1. Every expected value in the table is what pwsh 7.6.6 bound for the
# same text, and on 7 the suite asks 7 again rather than trusting the
# table.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    $script:TestHost = Get-FlynnelTestHost
    Write-Host ("SUFFIX_SUITE host={0}/{1} platform={2}" -f
        $script:TestHost.Edition, $script:TestHost.Version, $script:TestHost.Platform)
}

Describe 'which parameters read a size suffix' {
    It 'is the byte sizes and large counts the owner named, and nothing else' {
        # A transform names its cmdlet and parameter by string, and one
        # naming a cmdlet or parameter that does not exist attaches to
        # nothing and reports nothing. So the list is read back from the
        # loaded commands rather than from the declarations.
        $carrying = @(Get-Command -Module Flynnel -CommandType Cmdlet | ForEach-Object {
            $cmd = $_
            foreach ($p in $cmd.Parameters.Values) {
                $reads = @($p.Attributes | Where-Object {
                    $null -ne $_.GetType().BaseType -and $_.GetType().BaseType.FullName -eq 'Pwrs.TransformBase'
                })
                if ($reads.Count -gt 0) { "$($cmd.Name) -$($p.Name)" }
            }
        } | Sort-Object)
        $expected = @(
            'Measure-FlynnelExploreSelect -Count'
            'Measure-FlynnelHybridJoin -Count'
            'Measure-FlynnelHybridPipeline -Count'
            'Measure-FlynnelHybridPlacement -Count'
            'Measure-FlynnelHybridSplit -Count'
            'Measure-FlynnelRaceAny -Count'
            'New-FlynnelComposedMpmc -Capacity'
            'New-FlynnelComposedMpsc -Capacity'
            'New-FlynnelGpuPeerConfig -SlotBytes'
            'New-FlynnelGpuPeerConfig -SlotsPerLane'
            'New-FlynnelGpuPeerConfig -VramBlockBytes'
            'New-FlynnelInjector -Capacity'
            'New-FlynnelMpscRing -Capacity'
            'New-FlynnelNotifyRing -Capacity'
            'New-FlynnelPlan -BatchSize'
            'New-FlynnelRing -Capacity'
            'New-FlynnelSpscRing -Capacity'
            'Update-FlynnelPlan -ShapeBatchSize'
        ) | Sort-Object
        $carrying | Should -Be $expected
    }
}

Describe 'the forms a size suffix reads' {
    It 'binds <Text> as <Expected>, as pwsh 7 does' -TestCases @(
        @{ Text = '2MB'; Expected = 2097152 }
        @{ Text = '64kb'; Expected = 65536 }
        @{ Text = '2Mb'; Expected = 2097152 }
        @{ Text = ' 4KB '; Expected = 4096 }
        @{ Text = '+4KB'; Expected = 4096 }
        @{ Text = '.5MB'; Expected = 524288 }
        @{ Text = '1.KB'; Expected = 1024 }
        @{ Text = '1.5KB'; Expected = 1536 }
        @{ Text = '1.1KB'; Expected = 1126 }
        @{ Text = '1e3KB'; Expected = 1024000 }
        @{ Text = '1GB'; Expected = 1073741824 }
    ) {
        param($Text, $Expected)
        (New-FlynnelGpuPeerConfig -SlotBytes $Text).SlotBytes | Should -Be $Expected
        if ($script:TestHost.Edition -eq 'Core') {
            # pwsh 7 read without this module, as the reference.
            [long]$Text | Should -Be $Expected
        }
    }

    It 'binds a size read from text, which is how a size reaches a script from a file' {
        $row = 'SlotBytes,SlotsPerLane,VramBlockBytes', '4KB,1KB,1MB' | ConvertFrom-Csv
        $config = New-FlynnelGpuPeerConfig -SlotBytes $row.SlotBytes -SlotsPerLane $row.SlotsPerLane `
            -VramBlockBytes $row.VramBlockBytes
        $config.SlotBytes | Should -Be 4096
        $config.SlotsPerLane | Should -Be 1024
        $config.VramBlockBytes | Should -Be 1048576
    }

    It 'leaves a plain number to the binder' {
        (New-FlynnelGpuPeerConfig -SlotBytes 4096).SlotBytes | Should -Be 4096
        (New-FlynnelGpuPeerConfig -SlotBytes '4096').SlotBytes | Should -Be 4096
        (New-FlynnelGpuPeerConfig -SlotBytes 4KB).SlotBytes | Should -Be 4096
        (New-FlynnelGpuPeerConfig -SlotBytes '0x1000').SlotBytes | Should -Be 4096
    }

    It 'leaves the binder to refuse what it refuses' {
        # A space inside, a suffix pwsh 7 does not have, a negative into
        # an unsigned parameter, and a value past what the parameter
        # holds. Each is refused by the binder on both editions.
        { New-FlynnelGpuPeerConfig -SlotBytes '2 MB' } | Should -Throw
        { New-FlynnelGpuPeerConfig -SlotBytes '2MiB' } | Should -Throw
        { New-FlynnelGpuPeerConfig -SlotBytes '-1KB' } | Should -Throw
        { New-FlynnelGpuPeerConfig -SlotBytes '5GB' } | Should -Throw
        { New-FlynnelGpuPeerConfig -SlotBytes 'lots' } | Should -Throw
    }
}

Describe 'the parameters that read one' {
    It 'sizes a ring from a suffix' {
        $ring = New-FlynnelRing -Capacity '64KB'
        try { $ring.Capacity | Should -Be 65536 } finally { $ring.Dispose() }
    }

    It 'sizes every ring shape from a suffix' {
        $made = @(
            @(New-FlynnelSpscRing -Capacity '1KB')
            @(New-FlynnelMpscRing -Capacity '1KB' -Producers 1)
            @(New-FlynnelComposedMpsc -Capacity '1KB' -Producers 1)
            @(New-FlynnelComposedMpmc -Capacity '1KB' -Producers 1 -Consumers 1)
            @(New-FlynnelNotifyRing -Capacity '1KB' -Consumers 1)
        )
        try {
            @($made | ForEach-Object { $_.Capacity } | Sort-Object -Unique) | Should -Be @(1024)
        } finally {
            foreach ($o in $made) { $o.Dispose() }
        }
    }

    It 'sizes an injector from a suffix' {
        $q = New-FlynnelInjector -Capacity '64KB'
        try { $q.Capacity | Should -Be 65536 } finally { $q.Dispose() }
    }

    It 'sizes a plan batch and a work-steal batch from a suffix' {
        $plan = New-FlynnelPlan -KOuter 8 -BatchSize '1MB'
        $plan.BatchSize | Should -Be 1048576
        $stealing = Update-FlynnelPlan -Plan $plan -Shape WorkSteal -ShapeConsumers 8 -ShapeBatchSize '64KB'
        $stealing.ShapeBatchSize | Should -Be 65536
    }
}
