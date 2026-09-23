# The ring family: that what goes in comes out in order, that a full
# ring refuses without losing the item, that an empty pop says which
# kind of nothing it found, and that bytes survive a round trip as
# bytes.
#
# Every comparison of a payload is byte by byte. A round trip that
# mangles one byte in a thousand passes a string comparison of the
# first line, and the whole reason these rings carry a byte[] rather
# than a string is that a caller's payload is not text.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Common.ps1')
    Import-FlynnelModule

    # A payload with every byte value in it, so a conversion that
    # mistakes bytes for text loses something a comparison can see.
    function New-Payload {
        param([int]$Seed, [int]$Length = 64)
        $bytes = [byte[]]::new($Length)
        for ($i = 0; $i -lt $Length; $i++) {
            $bytes[$i] = [byte](($Seed * 31 + $i * 7) % 256)
        }
        , $bytes
    }

    function Should-MatchBytes {
        param([byte[]]$Actual, [byte[]]$Expected, [string]$What)
        $Actual.Length | Should -Be $Expected.Length -Because "$What must round-trip whole"
        for ($i = 0; $i -lt $Expected.Length; $i++) {
            if ($Actual[$i] -ne $Expected[$i]) {
                throw "$What differs at byte ${i}: got $($Actual[$i]), expected $($Expected[$i])"
            }
        }
    }
}

Describe 'the types and enums this family exports' {
    It 'shapes each copied type the way its cmdlet documents' {
        foreach ($type in 'Flynnel.PushOutcome', 'Flynnel.PopOutcome', 'Flynnel.RingStat') {
            @(Get-FlynnelTypeProperty -TypeName $type).Count |
                Should -BeGreaterThan 0 -Because "$type must carry something"
        }
    }

    It 'gives every handle class an Id, because that is what reaches its ring' {
        # Send-FlynnelItem and Receive-FlynnelItem find the Rust handle
        # by the object's Id and by nothing else, so a class that lost
        # the property would be unreachable from the pipeline forms
        # while every method on it still worked.
        $handles = @(
            'Flynnel.Ring'
            'Flynnel.SpscProducer'
            'Flynnel.SpscConsumer'
            'Flynnel.MpscProducer'
            'Flynnel.MpscConsumer'
            'Flynnel.ComposedConsumer'
            'Flynnel.GridProducer'
            'Flynnel.GridConsumer'
            'Flynnel.Injector'
            'Flynnel.NotifySender'
            'Flynnel.NotifyReceiver'
        )
        foreach ($type in $handles) {
            # The helper answers "<clr type> <name>" per property, so
            # the name is matched at the end rather than compared whole.
            $props = @(Get-FlynnelTypeProperty -TypeName $type)
            @($props | Where-Object { $_ -match '\bId$' }).Count |
                Should -Be 1 -Because "$type is reached by its Id"
        }
    }

    It 'leaves Dispose to the generated class rather than declaring one' {
        # A declared Dispose would hide the inherited member and a
        # script's using block would free nothing. IsDisposed beside it
        # is the generated pair, and its presence is what says the
        # inherited one is there to do the freeing.
        @(Get-FlynnelTypeProperty -TypeName 'Flynnel.Ring' |
            Where-Object { $_ -match '\bIsDisposed$' }).Count | Should -Be 1
    }

    It 'names every push and pop outcome rather than returning a null' {
        [enum]::GetNames([Flynnel.PushKind]) | Should -Contain 'Ok'
        [enum]::GetNames([Flynnel.PushKind]) | Should -Contain 'Full'
        [enum]::GetNames([Flynnel.PushKind]) | Should -Contain 'Closed'
        [enum]::GetNames([Flynnel.PopKind]) | Should -Contain 'Ok'
        [enum]::GetNames([Flynnel.PopKind]) | Should -Contain 'Empty'
        [enum]::GetNames([Flynnel.PopKind]) | Should -Contain 'Retry'
        [enum]::GetNames([Flynnel.PopKind]) | Should -Contain 'Closed'
    }

    It 'names the three sides a handle can be' {
        [enum]::GetNames([Flynnel.RingRole]) | Should -Be @('Ring', 'Producer', 'Consumer')
    }
}

Describe 'New-FlynnelRing' {
    It 'rounds the capacity up to a power of two and says so' {
        $ring = New-FlynnelRing -Capacity 1000
        # 1000 is not a power of two, so the ring has more slots than
        # were asked for and Capacity reports what it has.
        $ring.Capacity | Should -Be 1024
        $ring.Role | Should -Be 'Ring'
    }

    It 'refuses a capacity that describes no ring' {
        { New-FlynnelRing -Capacity 0 } | Should -Throw
        { New-FlynnelRing -Capacity -8 } | Should -Throw
    }

    It 'answers to its Fly alias' {
        (New-FlyRing -Capacity 16).Capacity | Should -Be 16
    }

    It 'gives every handle its own id' {
        $a = New-FlynnelRing -Capacity 8
        $b = New-FlynnelRing -Capacity 8
        $a.Id | Should -Not -Be $b.Id
    }
}

Describe 'what goes in comes out, in order and byte for byte' {
    It 'round-trips one item' {
        $ring = New-FlynnelRing -Capacity 8
        $sent = New-Payload -Seed 1
        $push = $ring.Push($sent)
        $push.Accepted | Should -BeTrue
        $push.Kind | Should -Be 'Ok'
        $pop = $ring.Pop()
        $pop.GotItem | Should -BeTrue
        Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the single item'
    }

    It 'round-trips a batch in order' {
        $ring = New-FlynnelRing -Capacity 64
        $sent = 0..15 | ForEach-Object { , (New-Payload -Seed $_) }
        $outcomes = $ring.PushMany($sent)
        @($outcomes).Count | Should -Be 16
        @($outcomes | Where-Object { -not $_.Accepted }).Count | Should -Be 0

        $got = $ring.PopMany(16)
        @($got).Count | Should -Be 16
        for ($i = 0; $i -lt 16; $i++) {
            Should-MatchBytes -Actual $got[$i] -Expected $sent[$i] -What "item $i"
        }
    }

    It 'keeps a zero-length item distinct from no item' {
        # An empty byte[] is a real item. GotItem is what says so; the
        # payload alone reads the same as an empty pop.
        $ring = New-FlynnelRing -Capacity 4
        $ring.Push([byte[]]::new(0)).Accepted | Should -BeTrue
        $pop = $ring.Pop()
        $pop.GotItem | Should -BeTrue
        $pop.Item.Length | Should -Be 0

        $empty = $ring.Pop()
        $empty.GotItem | Should -BeFalse
        $empty.Kind | Should -Be 'Empty'
        $empty.Item.Length | Should -Be 0
    }
}

Describe 'a full ring refuses and hands the item back' {
    It 'returns the refused item rather than dropping it' {
        $ring = New-FlynnelRing -Capacity 2
        $ring.Push((New-Payload -Seed 1 -Length 8)).Accepted | Should -BeTrue
        $ring.Push((New-Payload -Seed 2 -Length 8)).Accepted | Should -BeTrue

        $rejected = New-Payload -Seed 3 -Length 8
        $full = $ring.Push($rejected)
        $full.Accepted | Should -BeFalse
        $full.Kind | Should -Be 'Full'
        Should-MatchBytes -Actual $full.Item -Expected $rejected -What 'the refused item'
    }

    It 'stops a batch at the first refusal rather than skipping past it' {
        # Skipping would deliver a later item ahead of an earlier one,
        # which is not something an ordered ring may do. The count of
        # outcomes is how far it got.
        $ring = New-FlynnelRing -Capacity 2
        $batch = 1..6 | ForEach-Object { , (New-Payload -Seed $_ -Length 8) }
        $outcomes = @($ring.PushMany($batch))

        $outcomes.Count | Should -Be 3
        $outcomes[0].Accepted | Should -BeTrue
        $outcomes[1].Accepted | Should -BeTrue
        $outcomes[2].Accepted | Should -BeFalse
        Should-MatchBytes -Actual $outcomes[2].Item -Expected $batch[2] -What 'the item that did not fit'
    }

    It 'counts the refusal against the pushes' {
        $ring = New-FlynnelRing -Capacity 2
        $null = $ring.Push((New-Payload -Seed 1 -Length 4))
        $null = $ring.Push((New-Payload -Seed 2 -Length 4))
        $null = $ring.Push((New-Payload -Seed 3 -Length 4))
        $stat = $ring.Stat()
        $stat.Pushed | Should -Be 2
        $stat.FullRefusals | Should -Be 1
    }
}

Describe 'an empty pop reports which nothing it found' {
    It 'answers Empty rather than a null' {
        $ring = New-FlynnelRing -Capacity 4
        $pop = $ring.Pop()
        $pop | Should -Not -BeNullOrEmpty
        $pop.Kind | Should -Be 'Empty'
        $pop.GotItem | Should -BeFalse
    }

    It 'returns fewer than Count when the ring runs out' {
        $ring = New-FlynnelRing -Capacity 8
        $null = $ring.Push((New-Payload -Seed 1 -Length 4))
        @($ring.PopMany(10)).Count | Should -Be 1
    }

    It 'refuses a Count that asks for nothing' {
        # Zero is a bug in whatever expression computed the count, and
        # an empty array would hide it.
        $ring = New-FlynnelRing -Capacity 4
        { $ring.PopMany(0) } | Should -Throw
    }
}

Describe 'Get and Set on the stat row' {
    It 'says whether the depth column means anything' {
        # A ring reads its own depth. An SPSC producer has no reader
        # for the ring behind it, so a zero there is not an empty ring.
        $ring = New-FlynnelRing -Capacity 8
        $ring.Stat().DepthKnown | Should -BeTrue

        $p, $c = New-FlynnelSpscRing -Capacity 8
        $p.Stat().DepthKnown | Should -BeFalse
    }

    It 'tracks depth as items go in and out' {
        $ring = New-FlynnelRing -Capacity 8
        $ring.Stat().Depth | Should -Be 0
        $ring.Stat().IsEmpty | Should -BeTrue
        $null = $ring.Push((New-Payload -Seed 1 -Length 4))
        $ring.Stat().Depth | Should -Be 1
        $ring.Stat().IsEmpty | Should -BeFalse
        $null = $ring.Pop()
        $ring.Stat().Depth | Should -Be 0
    }

    It 'counts an empty pop against the pops' {
        $ring = New-FlynnelRing -Capacity 4
        $null = $ring.Pop()
        $null = $ring.Pop()
        $stat = $ring.Stat()
        $stat.Popped | Should -Be 0
        $stat.EmptyPops | Should -Be 2
    }

    It 'names the handle the row came from' {
        $ring = New-FlynnelRing -Capacity 4
        $ring.Stat().Id | Should -Be $ring.Id
    }
}

Describe 'New-FlynnelSpscRing' {
    It 'writes the producer and then the consumer as two objects' {
        $p, $c = New-FlynnelSpscRing -Capacity 16
        $p.Role | Should -Be 'Producer'
        $c.Role | Should -Be 'Consumer'
        $p.Id | Should -Not -Be $c.Id
    }

    It 'round-trips a batch in order' {
        $p, $c = New-FlynnelSpscRing -Capacity 64
        $sent = 0..9 | ForEach-Object { , (New-Payload -Seed ($_ + 100)) }
        @($p.PushMany($sent) | Where-Object { -not $_.Accepted }).Count | Should -Be 0
        $got = @($c.PopMany(10))
        $got.Count | Should -Be 10
        for ($i = 0; $i -lt 10; $i++) {
            Should-MatchBytes -Actual $got[$i] -Expected $sent[$i] -What "spsc item $i"
        }
    }

    It 'refuses a pop on the producer and a push on the consumer' {
        # The two ends are separate objects precisely so this is a type
        # error in the script rather than a silent nothing.
        $p, $c = New-FlynnelSpscRing -Capacity 8
        $p.PSObject.Methods.Name | Should -Not -Contain 'Pop'
        $c.PSObject.Methods.Name | Should -Not -Contain 'Push'
    }
}

Describe 'New-FlynnelMpscRing' {
    It 'writes the consumer first and then every producer' {
        $c, $producers = New-FlynnelMpscRing -Capacity 64 -Producers 4
        $c.Role | Should -Be 'Consumer'
        @($producers).Count | Should -Be 4
        @($producers | ForEach-Object { $_.Index }) | Should -Be @(0, 1, 2, 3)
    }

    It 'lets the consumer see every producer''s items' {
        $c, $producers = New-FlynnelMpscRing -Capacity 64 -Producers 3
        $expected = @{}
        foreach ($p in $producers) {
            $payload = New-Payload -Seed ([int]$p.Index + 200) -Length 16
            $expected[[int]$p.Index] = $payload
            $p.Push($payload).Accepted | Should -BeTrue
        }

        $got = @($c.PopMany(3))
        $got.Count | Should -Be 3
        # The order between producers is not defined, so each item is
        # matched to the producer whose payload it is rather than to a
        # position.
        foreach ($item in $got) {
            $matched = $false
            foreach ($key in $expected.Keys) {
                if ($item.Length -eq $expected[$key].Length -and $item[0] -eq $expected[$key][0]) {
                    Should-MatchBytes -Actual $item -Expected $expected[$key] -What "producer $key"
                    $matched = $true
                    break
                }
            }
            $matched | Should -BeTrue -Because 'every item must be one a producer sent'
        }
    }

    It 'refuses a producer count of zero rather than clamping it to one' {
        { New-FlynnelMpscRing -Capacity 8 -Producers 0 } | Should -Throw
    }
}

Describe 'New-FlynnelComposedMpsc' {
    It 'gives each producer its own ring' {
        $c, $producers = New-FlynnelComposedMpsc -Capacity 8 -Producers 4
        $c.RingCount | Should -Be 4
        @($producers).Count | Should -Be 4
    }

    It 'lets one producer fill its ring without blocking another' {
        # The whole point of the composed shape: no producer contends
        # with another, so a full ring on one says nothing about the
        # rest.
        $c, $producers = New-FlynnelComposedMpsc -Capacity 2 -Producers 2
        $first = $producers[0]
        $second = $producers[1]
        $null = $first.Push((New-Payload -Seed 1 -Length 4))
        $null = $first.Push((New-Payload -Seed 2 -Length 4))
        $first.Push((New-Payload -Seed 3 -Length 4)).Kind | Should -Be 'Full'
        $second.Push((New-Payload -Seed 4 -Length 4)).Accepted | Should -BeTrue
    }

    It 'reads every ring before saying it is empty' {
        $c, $producers = New-FlynnelComposedMpsc -Capacity 4 -Producers 3
        $sent = New-Payload -Seed 7 -Length 8
        # Only the last producer has anything, so an Empty from the
        # consumer would mean it stopped at the first ring it looked at.
        $producers[2].Push($sent).Accepted | Should -BeTrue
        $pop = $c.Pop()
        $pop.GotItem | Should -BeTrue
        Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the item on the third ring'
    }
}

Describe 'New-FlynnelComposedMpmc' {
    It 'writes the producers and then the consumers, each naming its role' {
        $all = @(New-FlynnelComposedMpmc -Capacity 8 -Producers 3 -Consumers 2)
        $prod = @($all | Where-Object Role -eq 'Producer')
        $cons = @($all | Where-Object Role -eq 'Consumer')
        $prod.Count | Should -Be 3
        $cons.Count | Should -Be 2
        $prod[0].ConsumerCount | Should -Be 2
        $cons[0].ProducerCount | Should -Be 3
    }

    It 'delivers what a producer sends to some consumer' {
        $all = @(New-FlynnelComposedMpmc -Capacity 8 -Producers 1 -Consumers 2)
        $p = @($all | Where-Object Role -eq 'Producer')[0]
        $cons = @($all | Where-Object Role -eq 'Consumer')
        $sent = New-Payload -Seed 42 -Length 32
        $p.Push($sent).Accepted | Should -BeTrue

        $found = $null
        foreach ($c in $cons) {
            $pop = $c.Pop()
            if ($pop.GotItem) { $found = $pop.Item; break }
        }
        $found | Should -Not -BeNullOrEmpty -Because 'the item went to one of the columns'
        Should-MatchBytes -Actual $found -Expected $sent -What 'the grid item'
    }
}

Describe 'New-FlynnelInjector' {
    It 'takes the crate default when no capacity is named' {
        (New-FlynnelInjector).Capacity | Should -Be 4096
    }

    It 'round-trips items under the steal protocol' {
        $q = New-FlynnelInjector -Capacity 16
        $sent = New-Payload -Seed 9 -Length 24
        $q.Push($sent).Accepted | Should -BeTrue
        $pop = $q.Pop()
        $pop.Kind | Should -Be 'Ok'
        Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the injected item'
    }

    It 'answers Empty and never Retry, because its pop loops internally' {
        $q = New-FlynnelInjector -Capacity 8
        $q.Pop().Kind | Should -Be 'Empty'
    }
}

Describe 'the Ring and Injector constructors' {
    # Each New- cmdlet builds its object through the class's constructor,
    # so these check that the constructor a script calls directly makes
    # the object the cmdlet makes, and refuses what the cmdlet refuses.
    It 'makes the ring New-FlynnelRing makes' {
        $made = [Flynnel.Ring]::new(1000)
        $cmdlet = New-FlynnelRing -Capacity 1000
        try {
            $made | Should -BeOfType [Flynnel.Ring]
            $made.Capacity | Should -Be $cmdlet.Capacity
            $made.Role | Should -Be $cmdlet.Role
            $made.Id | Should -Not -Be $cmdlet.Id -Because 'each route registers a ring of its own'
        } finally {
            $made.Dispose()
            $cmdlet.Dispose()
        }
    }

    It 'makes a ring whose items round-trip byte for byte' {
        $ring = [Flynnel.Ring]::new(8)
        try {
            $sent = New-Payload -Seed 21
            $ring.Push($sent).Accepted | Should -BeTrue
            $pop = $ring.Pop()
            $pop.GotItem | Should -BeTrue
            Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the constructed ring''s item'
        } finally {
            $ring.Dispose()
        }
    }

    It 'refuses the capacities New-FlynnelRing refuses' {
        { [Flynnel.Ring]::new(0) } | Should -Throw -ExpectedMessage '*describes no ring*'
        { [Flynnel.Ring]::new(-8) } | Should -Throw -ExpectedMessage '*describes no ring*'
    }

    It 'frees the ring on Dispose' {
        $ring = [Flynnel.Ring]::new(8)
        $ring.Dispose()
        $ring.IsDisposed | Should -BeTrue
        { $ring.Pop() } | Should -Throw
    }

    It 'makes an injector at the crate default when no capacity is given' {
        $made = [Flynnel.Injector]::new()
        $cmdlet = New-FlynnelInjector
        try {
            $made | Should -BeOfType [Flynnel.Injector]
            $made.Capacity | Should -Be $cmdlet.Capacity
            $made.Capacity | Should -Be 4096
        } finally {
            $made.Dispose()
            $cmdlet.Dispose()
        }
    }

    It 'makes the injector New-FlynnelInjector makes for a named capacity' {
        $made = [Flynnel.Injector]::new(16)
        $cmdlet = New-FlynnelInjector -Capacity 16
        try {
            $made.Capacity | Should -Be $cmdlet.Capacity
            $made.Role | Should -Be $cmdlet.Role
            $sent = New-Payload -Seed 22 -Length 24
            $made.Push($sent).Accepted | Should -BeTrue
            $pop = $made.Pop()
            $pop.Kind | Should -Be 'Ok'
            Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the constructed injector''s item'
        } finally {
            $made.Dispose()
            $cmdlet.Dispose()
        }
    }

    It 'refuses an injector capacity that describes no queue' {
        { [Flynnel.Injector]::new(0) } | Should -Throw -ExpectedMessage '*describes no ring*'
    }
}

Describe 'the Flynnel.Rings factory' {
    # The New- cmdlet for each shape that comes as several objects builds
    # through the same function as its static here. These check that a
    # static returns what its cmdlet writes, in the same order, and
    # refuses what the cmdlet refuses. An object's shape here is its
    # type, role and capacity, which every handle class carries.
    It 'returns the SPSC ends New-FlynnelSpscRing writes, producer first' {
        $made = @([Flynnel.Rings]::Spsc(8))
        $cmdlet = @(New-FlynnelSpscRing -Capacity 8)
        try {
            $shape = @($made | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $shape | Should -Be @($cmdlet | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            @($made | ForEach-Object { $_.GetType().FullName }) |
                Should -Be @('Flynnel.SpscProducer', 'Flynnel.SpscConsumer')
            $p, $c = $made
            $sent = New-Payload -Seed 31
            $p.Push($sent).Accepted | Should -BeTrue
            $pop = $c.Pop()
            $pop.GotItem | Should -BeTrue
            Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the factory SPSC item'
        } finally {
            foreach ($o in $made + $cmdlet) { $o.Dispose() }
        }
    }

    It 'returns the MPSC consumer and producers New-FlynnelMpscRing writes' {
        $made = @([Flynnel.Rings]::Mpsc(64, 3))
        $cmdlet = @(New-FlynnelMpscRing -Capacity 64 -Producers 3)
        try {
            $made.Count | Should -Be 4
            $shape = @($made | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $shape | Should -Be @($cmdlet | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $made[0] | Should -BeOfType [Flynnel.MpscConsumer]
            @($made[1..3] | ForEach-Object { $_.Index }) | Should -Be @(0, 1, 2)
            $sent = New-Payload -Seed 32
            $made[2].Push($sent).Accepted | Should -BeTrue
            $pop = $made[0].Pop()
            $pop.GotItem | Should -BeTrue
            Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the factory MPSC item'
        } finally {
            foreach ($o in $made + $cmdlet) { $o.Dispose() }
        }
    }

    It 'returns the composed MPSC New-FlynnelComposedMpsc writes, consumer first' {
        $made = @([Flynnel.Rings]::ComposedMpsc(16, 2))
        $cmdlet = @(New-FlynnelComposedMpsc -Capacity 16 -Producers 2)
        try {
            $shape = @($made | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $shape | Should -Be @($cmdlet | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $made[0] | Should -BeOfType [Flynnel.ComposedConsumer]
            $made[0].RingCount | Should -Be 2
            @($made[1..2] | ForEach-Object { $_.GetType().FullName }) |
                Should -Be @('Flynnel.SpscProducer', 'Flynnel.SpscProducer')
            $sent = New-Payload -Seed 33
            $made[1].Push($sent).Accepted | Should -BeTrue
            $pop = $made[0].Pop()
            $pop.GotItem | Should -BeTrue
            Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the factory composed item'
        } finally {
            foreach ($o in $made + $cmdlet) { $o.Dispose() }
        }
    }

    It 'returns the grid New-FlynnelComposedMpmc writes, producers first' {
        $made = @([Flynnel.Rings]::ComposedMpmc(16, 2, 3))
        $cmdlet = @(New-FlynnelComposedMpmc -Capacity 16 -Producers 2 -Consumers 3)
        try {
            $made.Count | Should -Be 5
            $shape = @($made | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $shape | Should -Be @($cmdlet | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            @($made[0..1] | ForEach-Object { $_.ConsumerCount }) | Should -Be @(3, 3)
            @($made[2..4] | ForEach-Object { $_.ProducerCount }) | Should -Be @(2, 2, 2)
        } finally {
            foreach ($o in $made + $cmdlet) { $o.Dispose() }
        }
    }

    It 'returns the notify hub New-FlynnelNotifyRing writes, sender first' {
        $made = @([Flynnel.Rings]::Notify(16, 2))
        $cmdlet = @(New-FlynnelNotifyRing -Capacity 16 -Consumers 2)
        try {
            $shape = @($made | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $shape | Should -Be @($cmdlet | ForEach-Object { "$($_.GetType().FullName) $($_.Role) $($_.Capacity)" })
            $made[0] | Should -BeOfType [Flynnel.NotifySender]
            $made[0].ConsumerCount | Should -Be 2
            @($made[1..2] | ForEach-Object { $_.Index }) | Should -Be @(0, 1)
            $sent = New-Payload -Seed 34
            $made[0].Push($sent).Accepted | Should -BeTrue
            $pop = $made[1].Pop()
            if (-not $pop.GotItem) { $pop = $made[2].Pop() }
            $pop.GotItem | Should -BeTrue -Because 'the item went to one of the receivers'
            Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the factory notify item'
            $made[0].Shutdown()
        } finally {
            foreach ($o in $made + $cmdlet) { $o.Dispose() }
        }
    }

    It 'refuses what the cmdlets refuse' {
        { [Flynnel.Rings]::Spsc(0) } | Should -Throw -ExpectedMessage '*describes no ring*'
        { [Flynnel.Rings]::Mpsc(8, 0) } | Should -Throw -ExpectedMessage '*Producers must be at least one*'
        { [Flynnel.Rings]::ComposedMpsc(8, 0) } | Should -Throw -ExpectedMessage '*Producers must be at least one*'
        { [Flynnel.Rings]::ComposedMpmc(8, 1, 0) } | Should -Throw -ExpectedMessage '*Consumers must be at least one*'
        { [Flynnel.Rings]::Notify(8, 0) } | Should -Throw -ExpectedMessage '*Consumers must be at least one*'
    }

    It 'has no public constructor, because it only carries statics' {
        @([Flynnel.Rings].GetConstructors()).Count | Should -Be 0
        { [Flynnel.Rings]::new() } | Should -Throw
    }
}

Describe 'New-FlynnelNotifyRing' {
    It 'writes the sender first and then one receiver per consumer slot' {
        $s, $receivers = New-FlynnelNotifyRing -Capacity 16 -Consumers 2
        $s.Role | Should -Be 'Producer'
        $s.ConsumerCount | Should -Be 2
        @($receivers).Count | Should -Be 2
        @($receivers | ForEach-Object { $_.Index }) | Should -Be @(0, 1)
    }

    It 'round-trips an item from the sender to a receiver' {
        $s, $receivers = New-FlynnelNotifyRing -Capacity 16 -Consumers 1
        $sent = New-Payload -Seed 11 -Length 48
        $s.Push($sent).Accepted | Should -BeTrue
        $pop = $receivers.Pop()
        $pop.GotItem | Should -BeTrue
        Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the notified item'
    }

    It 'answers Closed on a send after shutdown, and hands the item back' {
        $s, $receivers = New-FlynnelNotifyRing -Capacity 16 -Consumers 1
        $s.Shutdown()
        $rejected = New-Payload -Seed 12 -Length 8
        $after = $s.Push($rejected)
        $after.Accepted | Should -BeFalse
        $after.Kind | Should -Be 'Closed'
        Should-MatchBytes -Actual $after.Item -Expected $rejected -What 'the item a closed hub refused'
    }

    It 'is safe to shut down twice' {
        $s, $receivers = New-FlynnelNotifyRing -Capacity 8 -Consumers 1
        $s.Shutdown()
        { $s.Shutdown() } | Should -Not -Throw
    }
}

Describe 'Send-FlynnelItem and Receive-FlynnelItem' {
    It 'pushes pipeline items into a ring' {
        $ring = New-FlynnelRing -Capacity 64
        $sent = 0..4 | ForEach-Object { , (New-Payload -Seed ($_ + 300) -Length 12) }
        $rejected = @($sent | Send-FlynnelItem -To $ring)
        $rejected.Count | Should -Be 0
        $ring.Stat().Depth | Should -Be 5
    }

    It 'writes a refused item back rather than dropping it' {
        $ring = New-FlynnelRing -Capacity 2
        $sent = 0..3 | ForEach-Object { , (New-Payload -Seed ($_ + 400) -Length 12) }
        $rejected = @($sent | Send-FlynnelItem -To $ring)
        # Two fit and two came back, so nothing was lost and a retry is
        # possible with exactly what the ring refused.
        $rejected.Count | Should -Be 2
        Should-MatchBytes -Actual $rejected[0] -Expected $sent[2] -What 'the first item back'
        Should-MatchBytes -Actual $rejected[1] -Expected $sent[3] -What 'the second item back'
    }

    It 'reads items out until the ring empties' {
        $ring = New-FlynnelRing -Capacity 64
        $sent = 0..4 | ForEach-Object { , (New-Payload -Seed ($_ + 500) -Length 12) }
        $null = $ring.PushMany($sent)
        $got = @(Receive-FlynnelItem -From $ring)
        $got.Count | Should -Be 5
        for ($i = 0; $i -lt 5; $i++) {
            Should-MatchBytes -Actual $got[$i] -Expected $sent[$i] -What "received item $i"
        }
    }

    It 'stops at Count' {
        $ring = New-FlynnelRing -Capacity 64
        $null = $ring.PushMany(@(0..9 | ForEach-Object { , (New-Payload -Seed $_ -Length 8) }))
        @(Receive-FlynnelItem -From $ring -Count 4).Count | Should -Be 4
        $ring.Stat().Depth | Should -Be 6
    }

    It 'refuses an object that is not a ring' {
        # The item is a real byte[] so the only thing left to object to
        # is -To. A string here would fail to bind to the byte[]
        # parameter and the test would pass without the ring check ever
        # running, which is a test asserting the wrong thing.
        $item = New-Payload -Seed 1 -Length 8
        { , $item | Send-FlynnelItem -To (Get-Date) } | Should -Throw
        { Receive-FlynnelItem -From 42 } | Should -Throw
    }

    It 'refuses a push at a pop side and says how much did not go in' {
        # A push refused at the wrong side is a per-item error, and so
        # non-terminating like any one failed pipeline item; the test asks
        # for it to stop rather than borrowing the runner's preference.
        $p, $c = New-FlynnelSpscRing -Capacity 8
        { , (New-Payload -Seed 1 -Length 8) | Send-FlynnelItem -To $c -ErrorAction Stop } | Should -Throw
    }

    It 'answers to its Fly aliases' {
        $ring = New-FlynnelRing -Capacity 8
        $null = , (New-Payload -Seed 1 -Length 4) | Send-FlyItem -To $ring
        @(Receive-FlyItem -From $ring).Count | Should -Be 1
    }
}

Describe 'disposing' {
    It 'frees the handle, so a later call on the same object is refused' {
        # Dispose is inherited from the generated class and nothing here
        # declares one; it drops the object, which drops the guard that
        # holds the table entry. A method after that has no handle to
        # find, which is what proves the ring was actually freed rather
        # than the object merely detached.
        $ring = New-FlynnelRing -Capacity 8
        $ring.Dispose()
        { $ring.Push((New-Payload -Seed 1 -Length 4)) } | Should -Throw
    }

    It 'is safe to dispose twice' {
        $ring = New-FlynnelRing -Capacity 8
        $ring.Dispose()
        { $ring.Dispose() } | Should -Not -Throw
    }

    It 'leaves another handle on the same ring working' {
        # Disposing one end of an SPSC ring must not take the other
        # with it: they are separate handles on separate table entries.
        $p, $c = New-FlynnelSpscRing -Capacity 8
        $sent = New-Payload -Seed 5 -Length 8
        $p.Push($sent).Accepted | Should -BeTrue
        $p.Dispose()
        $pop = $c.Pop()
        $pop.GotItem | Should -BeTrue
        Should-MatchBytes -Actual $pop.Item -Expected $sent -What 'the item a disposed producer left'
    }
}
