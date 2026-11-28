# Rencana proof-of-concept ONNX Runtime Android Tata

**Status: PoC Android terisolasi sudah dibuat; pengujian unit lulus pada commit `4b8bee1873370de486bce7a1a5f1f702d87d02c4`. Pengujian instrumentasi emulator belum memiliki hasil akhir yang terverifikasi. Belum ada integrasi ke aplikasi utama.**

## Keputusan sementara

Kandidat runtime CPU Android dipatok pada artefak Maven:

```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")
```

Versi ini digunakan dalam modul PoC terpisah. Ini belum merupakan keputusan runtime produksi.

Sumber pemeriksaan:
- Dokumentasi instalasi resmi ONNX Runtime: https://onnxruntime.ai/docs/install/
- Paket Android di Maven Central: https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android

## Implementasi yang sudah ada

- Modul aplikasi Android mandiri `:onnx-poc`, terpisah dari `:app`.
- Dependency ONNX Runtime Android dipatok pada versi `1.32.0`.
- `TataOnnxRuntimeProbe` membuka model dari file lokal dan menyediakan nama input/output serta eksekusi tensor.
- Tidak ada izin `INTERNET` di manifest PoC.
- Fixture ONNX Identity sintetis untuk menguji pemuatan dan eksekusi runtime.
- Fixture Transformer Tata sintetis (bukan model produksi dan bukan hasil pelatihan corpus) dengan referensi logits dari PyTorch.
- Instrumentation test menguji dua bentuk input dinamis, `1×4` dan `2×5`, lalu membandingkan logits Android dengan referensi PyTorch.
- CI sudah melewati pembuatan fixture, instalasi NDK/CMake, dan Android unit tests pada commit yang disebutkan di atas.

## Batasan keselamatan dan ruang lingkup

- Tidak mengubah `:app`, jalur GGUF/llama.cpp, chat engine, atau APK utama.
- Tidak menambahkan izin INTERNET, download model otomatis, telemetry, atau cloud inference.
- Model ONNX dibaca dari file lokal yang disediakan penguji.
- Gunakan model sintetis kecil untuk tes perangkat lunak; jangan menganggapnya mewakili kualitas model Tata.
- Pertahankan `minSdk 26` pada PoC.
- Jangan mengintegrasikan dependency atau mengganti runtime aplikasi utama sebelum semua gerbang uji lulus dan ada persetujuan.

## Tahap uji dan status

1. **Dependency resolution:** dependency sudah dideklarasikan dan proses build/unit test sudah berjalan di CI; ukuran final artefak masih perlu dicatat.
2. **Runtime initialization:** ada implementasi pembukaan session; validasi emulator menunggu hasil instrumentasi.
3. **Local model load:** ada tes model sintetis lokal; hasil emulator menunggu.
4. **Tensor contract:** tes memeriksa input INT64 `input_ids` dan output `logits`; hasil instrumentasi menunggu.
5. **Numerical parity:** tes membandingkan logits Android dengan golden output PyTorch untuk `1×4` dan `2×5`; belum boleh dinyatakan lulus sebelum CI selesai.
6. **Dynamic shapes:** tercakup dalam tes instrumentasi yang sama; hasil menunggu.
7. **Offline assertion:** manifest tidak meminta izin INTERNET; pengujian ini belum setara dengan audit jaringan menyeluruh pada perangkat.
8. **Resource measurement:** ukuran APK, RAM puncak, waktu inisialisasi, latency inferensi, dan stabilitas belum diukur.
9. **Physical-device gate:** belum diuji pada HP fisik.
10. **Decision gate:** jangan integrasikan ke aplikasi utama sampai seluruh tes dan pengukuran selesai.

## Risiko yang harus diuji

- Operator atau pola graph dari exporter PyTorch mungkin tidak didukung pada runtime Android.
- Hasil numerik dapat berbeda karena implementasi kernel dan presisi.
- Dynamic sequence, causal mask, dan dimensi vocabulary harus konsisten.
- AAR membawa native libraries; ukuran dan ABI harus diverifikasi dari artefak nyata.
- Resource native harus ditutup sesuai lifecycle.
- Keberhasilan di emulator bukan bukti performa pada perangkat fisik.

## Kriteria lulus

PoC dinyatakan lulus hanya jika model lokal dapat dimuat dan dijalankan di Android, bentuk/dtype input-output sesuai kontrak, logits cocok dengan referensi dalam toleransi, perilaku offline terverifikasi, dan pengukuran ukuran/RAM/latency tersedia. Jika salah satu gagal, jangan integrasikan runtime ke aplikasi utama.

## Kondisi yang belum terpenuhi

- Belum ada model ONNX Tata produksi yang diekspor dari checkpoint hasil pelatihan nyata.
- Hasil akhir instrumentasi emulator belum terverifikasi.
- Belum ada hasil pengujian perangkat fisik.
- Belum ada pengukuran ukuran, RAM, waktu inisialisasi, dan latency.
- Belum ada keputusan akhir untuk runtime produksi.
