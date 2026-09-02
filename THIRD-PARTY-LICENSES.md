# Third-Party Licenses

Flynnel is MIT-licensed (see [LICENSE](LICENSE)). A few Flynnel source files
contain code derived from third-party projects. Each such project's license
text and copyright notice is reproduced below in full, as those licenses
require.

---

## rayon-core 1.13.0

- **Upstream project**: [rayon-rs/rayon](https://github.com/rayon-rs/rayon)
- **Source tag**: `rayon-core-v1.13.0`
- **Upstream URL**: <https://github.com/rayon-rs/rayon/tree/rayon-core-v1.13.0/rayon-core>
- **Upstream license**: MIT OR Apache-2.0 (dual-licensed; consumer picks)
- **License chosen for this redistribution**: MIT

### Flynnel files derived from rayon-core 1.13.0

| Flynnel file                  | Derived from upstream                                                                  | Nature of derivation                                                                                                                                       |
|-------------------------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/sched/jec_sleep.rs`      | `rayon-core/src/sleep/counters.rs` and `rayon-core/src/sleep/mod.rs`                   | Verbatim port of the JEC (Jobs Event Counter) sleep protocol. The counter packing, four-phase consumer state machine, and producer wake heuristic are unchanged; type names and integration glue were adapted to Flynnel's worker layout. |
| `src/sched/job.rs`            | `rayon-core/src/job.rs`                                                                | The `JobRef` two-word vtable shape and the `StackJob` pattern are adapted from rayon-core. Tag bytes (`k_outer`, `numa_hint`, `variant`) and the `JobResult` panic-capture wrapper are Flynnel additions.                                  |
| `src/sched/latch.rs`          | `rayon-core/src/latch.rs`                                                              | The four-state `CoreLatch` machine (UNSET → SLEEPY → SLEEPING → SET) and the `Latch::set(*const Self)` self-invalidation pattern are adapted from rayon-core.                                                                              |

### MIT license text (rayon-core 1.13.0)

```
Copyright (c) 2010 The Rust Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

Verbatim from <https://github.com/rayon-rs/rayon/blob/rayon-core-v1.13.0/rayon-core/LICENSE-MIT>.
