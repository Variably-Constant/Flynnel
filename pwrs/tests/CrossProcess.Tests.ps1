# The cross-process worker family.
#
# What is bound so far answers without a peer process, so every
# assertion here runs on every host. The routing decision is made from
# the shape and the host; the registry reading is a reading of this
# process. Neither starts a peer.
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
                          'Flynnel.PassRegistry') {
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
