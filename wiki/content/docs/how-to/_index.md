---
title: How-to
weight: 2
sidebar:
  open: true
---

Task-oriented recipes. Each page solves a concrete problem and assumes you already know the basics from [Tutorials](../tutorials/).

{{< cards >}}
  {{< card link="How-To-Write-A-Backend/" title="How To Write A Backend" subtitle="Implement your own `DispatchBackend` and register it with the Flynnel registry. Walks through the CPU, CUDA, and ROCm shapes." >}}
  {{< card link="Reference-Backends-CUDA-And-TPU/" title="Reference Backends: CUDA, TPU, and WASM" subtitle="The three in-process reference backends Flynnel ships - feature gates, runtime requirements, and graceful-degradation behavior. CUDA / TPU / WASM in one page." >}}
  {{< card link="How-To-Use-The-Shared-Memory-Worker-Backend/" title="How To Use The Shared-Memory Worker Backend" subtitle="Dispatch handler invocations from an originator process into one or more peer worker processes over a lock-free MMF ring. ~800 ns per call." >}}
{{< /cards >}}
