# The native entry points: that Get-FlynnelNativeEntry hands out an
# address and an ABI version another native library can call, in the
# types that library reads them as.
#
# What the entry does when called is covered in Rust, in native.rs,
# because a script cannot call a native function pointer. What only a
# script can check is the hand-off: that the object comes through the
# engine at all, and that its numbers keep the widths the ABI needs.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule
}

Describe 'Get-FlynnelNativeEntry' {
    BeforeAll {
        $script:Entry = Get-FlynnelNativeEntry
    }

    It 'answers one Flynnel.NativeEntry' {
        @($script:Entry).Count | Should -Be 1
        $script:Entry.GetType().FullName | Should -Be 'Flynnel.NativeEntry'
    }

    It 'reports ABI version 1 as a UInt32' {
        # The caller branches on this, so it has to arrive as the width
        # the native side wrote rather than widened.
        $script:Entry.AbiVersion | Should -Be 1
        $script:Entry.AbiVersion.GetType().FullName | Should -Be 'System.UInt32'
    }

    It 'reports the kernels revision this build carries as a UInt32' {
        # A library linking the crate compares its own kernels revision
        # with this one before it runs a kernel's blocks on this pool.
        $script:Entry.KernelsRevision | Should -BeGreaterOrEqual 1
        $script:Entry.KernelsRevision.GetType().FullName | Should -Be 'System.UInt32'
    }

    It 'reports the run-chunks entry as a nonzero UInt64' {
        # An address wider than 32 bits is the ordinary case on a 64-bit
        # process, so anything narrower than UInt64 would truncate it.
        $script:Entry.RunChunksV1 | Should -Not -Be 0
        $script:Entry.RunChunksV1.GetType().FullName | Should -Be 'System.UInt64'
    }

    It 'says the run-chunks entry takes a site key, as a Boolean' {
        # The entry's signature changed under the same name and ABI
        # version, so this flag is what a caller checks before calling:
        # a module without it, or with it false, has the entry that takes
        # no key, and calling one form as the other jumps through the
        # wrong argument.
        $script:Entry.SiteKey | Should -BeTrue
        $script:Entry.SiteKey.GetType().FullName | Should -Be 'System.Boolean'
    }

    It 'answers the same address twice in one session' {
        # One loaded copy of the library, one address. A caller that asks
        # per dispatch sees a change only when a reload loads a new copy.
        (Get-FlynnelNativeEntry).RunChunksV1 | Should -Be $script:Entry.RunChunksV1
    }

    It 'reports whether the pool has started, and a started pool as started' {
        # Read without starting the pool, so the first answer depends on
        # what ran before it in this session; once Start-FlynnelPool has
        # run it can only be true.
        $script:Entry.PoolStarted | Should -BeOfType [bool]
        $null = Start-FlynnelPool
        (Get-FlynnelNativeEntry).PoolStarted | Should -BeTrue
    }

    It 'reports the plan entry as a nonzero UInt64 and says it has one, as a Boolean' {
        # A caller checks PlanHandle before calling the plan entry, whose
        # address is a separate symbol from the one RunChunksV1 names.
        $script:Entry.PlanHandle | Should -BeTrue
        $script:Entry.PlanHandle.GetType().FullName | Should -Be 'System.Boolean'
        $script:Entry.RunChunksPlanV1 | Should -Not -Be 0
        $script:Entry.RunChunksPlanV1.GetType().FullName | Should -Be 'System.UInt64'
        $script:Entry.RunChunksPlanV1 | Should -Not -Be $script:Entry.RunChunksV1
    }
}

Describe 'New-FlynnelNativePlan' {
    It 'writes a Flynnel.NativePlan carrying a nonzero UInt64 handle and the plan it names' {
        $native = New-FlynnelPlan -KOuter 8 -BatchSize 100000 | New-FlynnelNativePlan
        $native.GetType().FullName | Should -Be 'Flynnel.NativePlan'
        $native.Handle | Should -Not -Be 0
        $native.Handle.GetType().FullName | Should -Be 'System.UInt64'
        $native.KOuter | Should -Be 8
        $native.BatchSize | Should -Be 100000
        $native.Dispose()
    }

    It 'gives two plans two handles' {
        $first = New-FlynnelNativePlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000)
        $second = New-FlynnelNativePlan -Plan (New-FlynnelPlan -KOuter 8 -BatchSize 1000)
        $second.Handle | Should -Not -Be $first.Handle
        $first.Dispose()
        $second.Dispose()
    }
}
