# ggml-mmq: llama.cpp's mmq.cuh with a per-expert pointer table

`mmq.cuh` is `ggml/src/ggml-cuda/mmq.cuh` of llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (the ggml
revision this repository pins, `third_party/ggml/VERSION.txt`; MIT, see `LICENSE`) with `patches/ggml-mmq-xptrs.diff`
applied: `mmq_args` gains `x_ptrs`, a device table of one weight base per channel (expert), so the prompt path's
grouped expert products read each expert's gate/up weights where they already are instead of a gathered copy.

The build never modifies your llama.cpp checkout. CMake copies this file, together with the ggml-cuda files that
include `mmq.cuh` by a relative path (`mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `quantize.cuh`, `quantize.cu` and the
`template-instances/mmq-instance-*.cu` it compiles), into `<build>/strata_mmq_src`, so every translation unit sees this
one `mmq.cuh`. It checks that the checkout's own `mmq.cuh` is the pinned one (unpatched or already patched) and warns
otherwise.

SHA-256: pinned original `50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b`, this file
`358038838e59692b986976e3eb46d6168a9c96453f9e7909085ad2334f804f72`.
