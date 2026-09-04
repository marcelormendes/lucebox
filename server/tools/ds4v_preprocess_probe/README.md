# DeepSeek-V4 vision preprocessing probe

This standalone CPU target verifies the reusable decoded-RGB preprocessing unit without configuring the server or a GPU SDK. It implements the fixed source recipe only: patch size 14, downsample ratio 3, maximum 384 layout tokens, minimum 147456 planned pixels, maximum wide-image ratio 8, and mean/std 0.5 normalization.

Generate reference fixtures on a machine containing the original parent source and its Python environment:

```sh
python generate_reference_fixtures.py \
  --source /path/to/DeepSeek-V4-Flash-Vision-Uncensored \
  --output /tmp/ds4v-preprocess-fixtures
```

Build and run the probe:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
./build/ds4v_preprocess_probe --fixtures /tmp/ds4v-preprocess-fixtures
```

The fixture check compares resized RGB bytes, BF16 patch words, dimensions, all five grounded start-position layouts, permutations, spans, and a second deterministic run. The built-in self-test covers invalid fixed config, invalid and oversized decoded dimensions without allocating them, wrong RGB byte counts, output bounds, layout budget overflow, and absolute-position overflow.

JPEG and PNG decoding are a separate gate. This target accepts already decoded interleaved RGB bytes and has no image-codec or Python runtime dependency.

The resampler follows Pillow 12.3.0 `src/libImaging/Resample.c`, including signed 22-bit fixed-point bicubic coefficients and the tall-image vertical-first path.
