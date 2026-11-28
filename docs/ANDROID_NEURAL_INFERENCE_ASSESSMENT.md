# Penilaian jalur inferensi neural network Tata di Android

**Status: ekspor ONNX eksperimental, tokenizer Kotlin, dan modul PoC ONNX Runtime Android terisolasi sudah tersedia. Android unit tests lulus pada CI; tes instrumentasi emulator dan pengujian perangkat fisik belum terverifikasi. Belum ada integrasi ke aplikasi utama.**

## Temuan dari kode saat ini

| Area | Kondisi yang terverifikasi |
|---|---|
| Android utama | `minSdk 26`, `targetSdk 34`, Java/Kotlin JVM 17 |
| ABI native aplikasi utama | Hanya `arm64-v8a` yang dikonfigurasi |
| Native build | CMake 3.22.1; C++17 |
| Runtime native aplikasi utama | `llama.cpp` pada commit yang dipatok di CMake |
| Format model jalur llama.cpp | GGUF |
| Model Tata dari training | Checkpoint PyTorch `.pt` dan tokenizer Character-BPE JSON |
| Ekspor bobot Tata | Eksperimental melalui `tools/model/export_transformer_onnx.py`; diuji pada fixture sintetis |
| Tokenizer Kotlin Character-BPE | Parser JSON tersedia; unit test parity dengan fixture sintetis Python tersedia |
| Modul ONNX Android | PoC aplikasi mandiri `:onnx-poc`, dependency ONNX Runtime Android 1.32.0 |
| Tes ONNX Android | Unit tests lulus; instrumentasi emulator masih menunggu hasil akhir |
| Integrasi aplikasi utama | Belum dilakukan dan tidak boleh diasumsikan selesai |

### Kesimpulan penting

Runtime `llama.cpp` yang sudah ada adalah jalur kompatibilitas GGUF. Ia **tidak otomatis dapat memuat checkpoint PyTorch Tata**. Mengubah ekstensi `.pt` menjadi `.gguf` juga tidak mengubah format atau arsitektur bobot.

Modul `:onnx-poc` merupakan eksperimen terisolasi. Ia membuka model dari file lokal, tidak meminta izin INTERNET, dan menguji fixture Identity serta Transformer sintetis. Ini belum membuktikan model produksi Tata siap dijalankan di Android.

## Arah evaluasi runtime

Kandidat yang sedang diuji adalah ekspor model Tata ke ONNX lalu menjalankannya melalui ONNX Runtime Android CPU. Ini belum merupakan keputusan runtime produksi.

Urutan pembuktian:
1. Ekspor checkpoint hasil training yang sah ke ONNX dan validasi manifest/checksum.
2. Buktikan kesetaraan numerik PyTorch dan ONNX Runtime untuk batch dan sequence berbeda.
3. Perluas parity tokenizer Kotlin/Python pada kumpulan uji Bahasa Indonesia.
4. Jalankan inferensi autoregresif lokal tanpa jaringan.
5. Ukur ukuran APK/AAR, RAM puncak, waktu token pertama, token/detik, panas, dan baterai.
6. Uji pada perangkat Android fisik dengan ABI dan versi Android yang benar-benar didukung.
7. Baru pertimbangkan integrasi ke aplikasi utama setelah seluruh gerbang lulus dan mendapat persetujuan.

## Batasan dan gerbang yang masih terbuka

- Belum ada korpus produksi berizin yang lolos review; training model produksi belum boleh dimulai.
- Checkpoint CI berasal dari data sintetis dan hanya menguji pipeline, bukan kualitas model.
- Belum ada bobot Tata produksi terlatih untuk diekspor.
- Tes instrumentasi emulator untuk ONNX Runtime Android belum memiliki hasil akhir terverifikasi.
- Belum ada pengujian runtime model Tata pada perangkat fisik.
- Belum ada pengukuran performa inferensi pada perangkat fisik.
- `minSdk 26` berarti konfigurasi proyek menargetkan Android 8.0 ke atas; jangan menyatakan dukungan Android 6/7 tanpa perubahan dan pengujian.
- Jalur model Tata yang baru harus tetap tanpa permintaan jaringan dan diuji offline.

## Kriteria selesai

Tahap inferensi Android baru dapat disebut selesai jika ekspor dari checkpoint yang sah, parity tokenizer pada kumpulan uji yang memadai, kesetaraan numerik, runtime offline, validasi paket, dan uji performa perangkat fisik semuanya lulus. Sampai itu terjadi, mesin aturan Kotlin dan jalur GGUF yang ada tidak boleh disebut sebagai neural network Tata hasil training sendiri.

## Eksperimen ekspor ONNX yang tersedia di branch

Skrip `tools/model/export_transformer_onnx.py` menyediakan ekspor eksperimental dari checkpoint `best_validation`. Skrip memuat checkpoint memakai `weights_only=True`, memeriksa format/role, asal bobot, ukuran vocabulary, hash tokenizer, dan konsistensi hash register sumber train/validation. Graph diekspor dengan ONNX opset 17 dan disertai manifest checksum.

Contoh pemakaian setelah tersedia checkpoint dan tokenizer hasil pipeline training yang sah:

```bash
python -m pip install torch onnx
python -m tools.model.export_transformer_onnx \
  --checkpoint runs/tata.best.pt \
  --tokenizer artifacts/tokenizer.json \
  --output artifacts/tata-experimental.onnx
```

Manifest berstatus `experimental_not_android_validated`. Ekspor yang lolos ONNX checker belum membuktikan kesetaraan numerik dengan PyTorch, kompatibilitas ONNX Runtime Android, atau kelayakan performa perangkat. Jangan masukkan artefak ini ke APK produksi sebelum seluruh gerbang tersebut lulus.

## Dokumen terkait

- `docs/ONNX_ANDROID_POC_PLAN.md`: rincian modul PoC, dependency, instrumentasi, dan gerbang validasi.
- `docs/OWN_NEURAL_NETWORK_ROADMAP.md`: roadmap model neural network Tata.
