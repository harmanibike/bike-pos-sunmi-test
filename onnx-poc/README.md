# Tata ONNX Android PoC

Status: isolated experimental Android application; not integrated into the Tata app.

## Boundaries

- Uses ONNX Runtime Android 1.32.0, pinned to an explicit version.
- Opens only a caller-supplied local model file.
- Contains no network permission, model downloader, cloud inference, or training corpus.
- Is a separate `:onnx-poc` application module, not a dependency of `:app`.
- Does not replace the current llama.cpp/GGUF compatibility path.
- Does not establish production suitability or device performance.

## Automated validation

The Android instrumentation test uses a tiny synthetic ONNX Identity model generated during CI by `tools/model/create_onnx_android_fixture.py`. The fixture is a software test artifact, not a language model or training data.

The emulator test checks:
1. ONNX Runtime initializes on Android.
2. A model is opened from a local file copied from the APK test assets.
3. Input and output names match the fixture.
4. A float tensor is run through the model and the output matches the input.

This proves basic Android runtime loading and inference only. It does not validate Tata Transformer model export, tokenizer integration, dynamic sequence shapes, model parity, offline behavior on physical hardware, or performance.

## Next gates

1. Export a tiny Tata Transformer checkpoint and compare PyTorch vs ONNX Runtime Python.
2. Run the same model on Android and compare logits.
3. Verify dynamic batch/sequence dimensions and INT64 token input.
4. Measure APK/AAR size, initialization time, peak memory, and token latency on target hardware.
5. Decide whether the runtime is suitable before any integration into `:app`.

No production model or training data is included in this PoC.
