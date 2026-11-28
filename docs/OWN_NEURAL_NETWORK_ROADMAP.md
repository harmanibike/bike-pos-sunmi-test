# Fondasi Neural Network Tata AI (Rencana Teknis)

**Status: rancangan dan spesifikasi awal; belum ada neural network Tata yang selesai dilatih.**

Dokumen ini memisahkan generator berbasis aturan yang sudah ada dari model neural network yang kelak dilatih sendiri. Tidak boleh menyebut generator Kotlin sebagai model yang belajar atau sebagai neural network.

## Prinsip tetap
- Tidak memakai API inferensi AI, API key, layanan AI berbayar, atau server AI cloud.
- Model akhir ditujukan untuk inferensi lokal/offline. Proses pelatihan dilakukan pada komputer lokal yang memadai, bukan dipaksakan ke HP Android.
- Tidak menjadikan GGUF atau bobot model pihak ketiga sebagai model final Tata. GGUF yang sudah ada tetap jalur kompatibilitas opsional dan terpisah.
- Kode aplikasi menentukan kemampuan; korpus dan bobot model menentukan pola bahasa yang dipelajari; basis pengetahuan aktif tetap terpisah.
- Data percakapan pribadi dan memori pengguna dikecualikan dari korpus secara default. Tidak boleh mengambil chat atau Learning Lab sebagai data pelatihan tanpa izin pelatihan yang spesifik, terpisah, dan dapat dicabut.
- Setiap sumber data harus memiliki asal-usul, lisensi/izin, tanggal perolehan, dan aturan penggunaan yang dapat diaudit.

## Tahapan pengembangan

### Tahap 0 — tata kelola data (sebelum mengumpulkan korpus)
1. Gunakan hanya teks domain publik yang lisensinya mengizinkan penggunaan untuk pelatihan, materi milik sendiri, atau kontribusi dengan izin eksplisit.
2. Catat URL/nama sumber, pemilik, lisensi, tanggal akses, bahasa, dan batas penggunaan.
3. Buang data pribadi, nomor telepon, alamat, kredensial, dan identitas yang tidak diperlukan.
4. Sediakan daftar sumber yang dilarang dan mekanisme menghapus sumber beserta turunan dataset-nya.
5. Pisahkan izin untuk menyimpan koreksi sebagai pengetahuan, membagikan kontribusi, dan menggunakan contoh untuk pelatihan. Ketiganya tidak saling menggantikan.

### Tahap 1 — format korpus dan validasi
Format sumber kanonis adalah JSONL UTF-8, satu objek JSON per baris. Contoh skema konseptual:

```json
{
  "schema_version": 1,
  "example_id": "id-stabil-universal",
  "source": {
    "source_id": "sumber-001",
    "title": "Materi milik sendiri",
    "license": "owned_or_explicitly_permitted",
    "uri": null,
    "accessed_at": "2026-11-28"
  },
  "language": "id",
  "task": "instruction_response",
  "messages": [
    {"role": "user", "content": "Jelaskan ..."},
    {"role": "assistant", "content": " ... "}
  ],
  "privacy_review": "passed",
  "quality_review": "pending",
  "split": "unassigned"
}
```

Nilai contoh di atas hanya contoh struktur, bukan data pelatihan yang sudah dikumpulkan. Validator wajib menolak:
- JSON rusak, baris kosong, role tidak dikenal, isi kosong, dan contoh terlalu panjang;
- sumber tanpa lisensi/izin yang jelas;
- data yang mengandung rahasia atau informasi pribadi yang tidak relevan;
- duplikat persis maupun duplikat nyaris sama;
- contoh yang melanggar kebijakan keselamatan atau memuat instruksi berbahaya tanpa konteks mitigasi.

Setiap perubahan korpus menghasilkan manifest versi baru, jumlah contoh, hash file, daftar sumber, dan laporan validasi.

### Tahap 2 — pembersihan dan pembagian dataset
- Normalisasi Unicode dan spasi tanpa mengubah makna atau ejaan istilah.
- Deteksi bahasa; dataset tahap awal fokus Bahasa Indonesia.
- Dedup dilakukan sebelum pembagian data agar contoh serupa tidak bocor ke train dan test.
- Pembagian awal yang dapat diuji: 90% train, 5% validation, 5% test. Pembagian deterministik dilakukan berdasarkan kelompok `source_id` agar materi dari satu sumber tidak bocor ke split lain. Proporsi dihitung pada tingkat kelompok sumber sehingga persentase jumlah contoh dapat sedikit berbeda.
- Test set dikunci; jangan digunakan untuk menyetel model.
- Simpan provenance setiap contoh agar hasil dapat ditelusuri ke sumbernya.

### Tahap 3 — tokenizer Bahasa Indonesia
- Eksperimen awal Character-BPE mandiri tersedia di `tools/corpus/train_tokenizer.py`, tanpa paket pihak ketiga, API, atau jaringan. Ini prototipe riset, bukan tokenizer produksi dan belum setara tokenizer subword modern.
- Contoh awal yang seluruhnya sintetis tersedia di `tools/corpus/examples/synthetic_sample.jsonl`. Delapan contoh ini hanya fixture uji; status kualitasnya **pending**, sehingga loader pelatihan wajib mengabaikannya sampai ditinjau.
- Loader tokenizer hanya menerima contoh yang lolos tinjauan privasi dan kualitas serta sudah ditetapkan ke split `train`. Contoh `unassigned`, validation, dan test tidak boleh ikut melatih tokenizer.
- Mulai dengan eksperimen tokenizer subword BPE atau Unigram yang dilatih dari korpus berizin Tata sendiri.
- Ukuran kosakata awal 8.000 token adalah hipotesis eksperimen, bukan keputusan final; bandingkan dengan 4.000 dan 16.000 berdasarkan efisiensi token, kualitas ejaan, angka, imbuhan, nama, dan ukuran embedding.
- Simpan tokenizer, konfigurasi normalisasi, versi korpus, dan hash dalam satu paket versi. Penanda peran dan newline dipertahankan sebagai batas giliran yang sama pada tokenizer, training, prompt generasi, dan evaluator.
- Uji kata berimbuhan, kata ulang, singkatan, nama lokal, angka rupiah, tanda baca, campuran Indonesia/Inggris, serta teks suara yang umum.
- Tes awal mencakup round-trip Unicode, determinisme, token tidak dikenal, penyimpanan atomik, validasi urutan merge, simbol yang harus sudah tersedia, larangan token hasil merge duplikat, kecocokan ukuran vocabulary, dan pemisahan data yang belum ditinjau atau held-out. Loader pada tokenizer/training/evaluator juga menjalankan validasi korpus sebelum membaca data.
- Jalankan smoke test setelah dataset sintetis ditinjau: `python tools/corpus/validate_corpus.py tools/corpus/examples/synthetic_sample.jsonl` lalu `python tools/corpus/train_tokenizer.py tools/corpus/examples/synthetic_sample.jsonl --source-register docs/dataset_source_register.csv --output /tmp/tata-tokenizer.json`.

### Tahap 4 — eksperimen model pertama
- Arsitektur riset awal: decoder-only Transformer kecil yang dilatih dari bobot acak, bukan fine-tuning bobot model lain.
- Mulai dari model sangat kecil untuk membuktikan pipeline end-to-end; skala parameter ditentukan setelah ukuran korpus dan kemampuan perangkat pelatihan diketahui.
- Catat parameter model, jumlah layer, hidden size, attention heads, context length, optimizer, learning rate, batch size, seed, jumlah token pelatihan, dan perangkat.
- Model kecil pertama hanya membuktikan pipeline dan kemampuan dasar memprediksi token. Jangan mengharapkan kemampuan asisten umum setara model besar.
- Pelatihan memerlukan komputer dengan RAM/VRAM dan ruang penyimpanan memadai. HP Android target bukan perangkat pelatihan.

### Tahap 5 — evaluasi sebelum integrasi Android
Wajib melaporkan:
- validation loss dan perplexity;
- kualitas jawaban pada kumpulan pertanyaan Bahasa Indonesia yang ditahan;
- kepatuhan instruksi, relevansi, konsistensi, dan kemampuan mengatakan tidak tahu;
- halusinasi fakta, kebocoran data, bias, dan keluaran berbahaya;
- ukuran bobot, ukuran tokenizer, waktu inferensi, RAM puncak, panas, dan baterai pada perangkat target.

Tidak ada klaim model siap digunakan sebelum melewati evaluasi dan uji perangkat fisik.

### Tahap 6 — format bobot dan runtime lokal
- Bobot Tata harus memiliki format paket dan versi milik proyek, dengan manifest berisi arsitektur, tokenizer hash, versi korpus, konfigurasi, quantization, checksum, dan lisensi sumber.
- Model tidak harus dibundel dalam APK; impor lokal melalui pemilih file dapat mengurangi ukuran APK dan memberi pengguna kendali.
- Runtime Android akan dipilih setelah model pertama terbukti. Runtime komputasi bukan model pihak ketiga, tetapi setiap dependensi harus ditinjau lisensi, ukuran, ABI, dan dukungan Android.
- Sediakan pemeriksaan checksum, batas ukuran, validasi arsitektur, ruang penyimpanan, pembatalan impor, dan penghapusan aman.
- Model yang tidak kompatibel tidak boleh dimuat atau menimpa model aktif.

## Gerbang persetujuan
1. Data: sumber dan izin tervalidasi.
2. Korpus: lolos privasi, deduplikasi, dan pemeriksaan kualitas.
3. Tokenizer: evaluasi tokenisasi selesai.
4. Model: konfigurasi dan pelatihan tercatat.
5. Evaluasi: test set dan uji keselamatan selesai.
6. Android: inferensi offline diuji pada perangkat fisik.

## Langkah implementasi berikutnya
Validator korpus JSONL tersedia di `tools/corpus/validate_corpus.py`; generator manifest di `tools/corpus/build_manifest.py`; pembagi dataset deterministik tersedia di `tools/corpus/split_corpus.py`. Pembagian mengelompokkan berdasarkan `source_id`, hanya menerima contoh yang lolos tinjauan privasi dan kualitas, dan menolak dataset dengan kurang dari tiga sumber yang disetujui. Tes mencakup determinisme, pencegahan kebocoran sumber, dan catatan pending yang tetap unassigned. Fixture sintetis masih berstatus tinjauan kualitas pending, sehingga belum menjadi data latih. Spesifikasi baseline Transformer v0 ada di `docs/TRANSFORMER_V0_SPEC.md`; estimator konfigurasi tersedia di `tools/model/transformer_config.py` dan menghitung baseline 1.350.400 parameter dengan tied embeddings. Implementasi decoder-only Transformer dan loop training lokal kini tersedia di `tools/model/train_transformer.py`; keduanya tetap fondasi eksperimen, bukan model produksi yang sudah terlatih.


### Implementasi mesin training awal

Loop pelatihan lokal tersedia di `tools/model/train_transformer.py`, dengan loader data terpisah `tools/model/training_data.py`. Loader hanya mengizinkan split train/validation dan contoh yang lulus tinjauan privasi serta kualitas; split test tidak dapat dimuat. Model decoder-only dibuat dari bobot acak dan dilatih dengan next-token cross-entropy, AdamW, gradient clipping, validasi loss/perplexity, serta checkpoint atomik. Hash train/validation/tokenizer/register sumber dicatat dan diperiksa saat resume. Ukuran vocabulary diambil dari tokenizer aktual dan hitungan parameter runtime dicocokkan dengan estimator. CI menjalankan tes runtime serta simulasi training/resume singkat memakai data sintetis terkontrol; CI tidak melatih model produksi dari korpus nyata. Implementasi membutuhkan PyTorch 2.x di komputer lokal. Belum ada korpus produksi yang lolos review dan belum ada bobot model terlatih. Langkah berikutnya adalah memperoleh sumber data berizin, menyelesaikan review manusia, lalu menjalankan eksperimen kecil pada komputer pelatihan.


## Evaluasi model

Evaluator held-out tersedia di `tools/model/evaluate_transformer.py`, dengan test set yang terpisah dari training dan validation. Rubrik penilaian manual ada di `docs/MODEL_EVALUATION_RUBRIC.md`. Perplexity tidak cukup untuk menyatakan model layak; evaluasi manusia, audit privasi/keamanan, dan pengujian perangkat target wajib dilakukan sebelum integrasi Android.


### Evaluasi tokenizer sebelum model training

Tool `tools/corpus/evaluate_tokenizer.py` menyediakan probe tetap untuk imbuhan Indonesia, reduplikasi, format rupiah, tanggal/jam, tanda baca, campuran bahasa, nama lokal, dan karakter emoji. Laporan hanya menyimpan ID probe dan metrik agregat (unknown rate, token-per-character, dan round-trip), bukan menyalin isi teks uji. Probe hanya untuk evaluasi dan tidak boleh digabung ke data training. Tes memastikan teks khusus tidak bocor ke JSON laporan. Rasio token lebih kecil bukan bukti model lebih pintar atau lebih akurat.


### Konteks pilot dan sampling validation

Trainer mendukung `--context-length` dengan default 256. Konteks lebih pendek dapat dipakai untuk eksperimen pipeline pada contoh yang lebih pendek, tetapi hasilnya tidak boleh dibandingkan langsung dengan model berkonteks 256 tanpa mencatat konfigurasi. Validation memilih jendela yang tersebar merata dari seluruh kumpulan jendela penuh (maksimal 32 per evaluasi), bukan hanya jendela awal. Resume checkpoint tetap terikat pada konfigurasi yang sama.


### Pemisahan checkpoint terbaik dan checkpoint resume

Trainer kini memisahkan dua keluaran: file yang ditentukan melalui `--output` adalah checkpoint terakhir untuk melanjutkan training, sedangkan nama dengan sisipan `.best` sebelum ekstensi menyimpan bobot dengan validation loss terendah. Checkpoint resume juga membawa snapshot bobot terbaik dan step-nya sehingga status best tidak hilang ketika training dilanjutkan. Penulisan kedua file memakai mekanisme atomik. Train loss, validation loss, dan gradient norm yang bukan angka finite menghentikan training agar checkpoint rusak tidak diteruskan. Evaluator dapat membaca checkpoint best karena format model dan metadata utamanya tetap kompatibel. Fitur ini diuji melalui tes helper/path, pemuatan aman `weights_only=True`, round-trip checkpoint yang mencakup state RNG, serta tes integrasi CI yang menjalankan dua langkah training, menyimpan checkpoint, memuatnya kembali, lalu melanjutkan hingga langkah keempat. Tes tersebut memakai data sintetis terkontrol dan bukan bukti training pada korpus produksi. Training produksi tetap menunggu korpus berizin yang telah disetujui.


### Early stopping dan stabilitas training

Trainer mendukung `--early-stopping-patience` (default 10 evaluasi berturut-turut tanpa perbaikan; nilai 0 menonaktifkan). Saat patience tercapai, checkpoint resume terakhir tetap ditulis sebelum proses berhenti. Counter patience disimpan sehingga resume melanjutkan hitungan secara konsisten. Keluaran model terbaik disimpan terpisah berdasarkan validation loss terendah. Train loss, validation loss, dan gradient norm yang tidak finite menghentikan proses. Ini mengurangi pemborosan komputasi dan membantu menghindari melanjutkan training yang tidak stabil, tetapi tidak menjamin model bebas overfitting; test set dan evaluasi manusia tetap diperlukan.


Training log juga menampilkan rata-rata train loss sejak evaluasi terakhir dan selisih validation loss minus train loss. Selisih ini hanya indikator diagnostik untuk ditinjau bersama kurva beberapa evaluasi; ia bukan bukti otomatis overfitting maupun ukuran kualitas jawaban.


### Audit kesiapan dataset tanpa membocorkan isi

Tool `tools/corpus/audit_dataset.py` merangkum status review, jumlah contoh dan kelompok sumber, distribusi train/validation/test/unassigned, lisensi, bahasa, serta task. Laporan tidak mencetak isi pesan. Gerbang minimum tiga sumber disetujui dan split train/validation/test terisi hanya membuat dataset berstatus `ready_for_pilot_review`; ini bukan klaim dataset representatif. Peringatan diberikan bila sumber kurang dari 20 atau contoh disetujui kurang dari 1.000. Dataset sintetis saat ini tetap pending dan tidak boleh dilatih.


### Statistik panjang dan kecocokan konteks

Audit dataset juga menghitung distribusi karakter per contoh. Jika file tokenizer disediakan, audit menghitung distribusi token per contoh, median, persentil ke-90, rata-rata, nilai minimum/maksimum, dan jumlah contoh yang lebih panjang dari context length eksperimen. Statistik bersifat agregat dan tidak menampilkan teks mentah. Contoh yang melebihi context length tidak otomatis dihapus; trainer membentuk jendela konteks. Perlu pemeriksaan manual agar pemotongan jendela tidak menghilangkan giliran dialog yang penting.


### Dataset quality and balance diagnostics
The dataset audit reports aggregate character/token lengths by split, task distribution per split, largest-source concentration, short user prompts, short assistant responses, long examples, and groups of identical answers repeated at least three times. These heuristics only flag material for human review; they do not establish semantic correctness or detect every near-duplicate. Raw conversation text is never included in the audit report. Training remains blocked until approved and properly split data exists.

Exact normalized prompt hashes are compared across approved examples. Repeated prompts and conflicting answers are flagged for human review. This does not detect paraphrases or semantic similarity and must not be presented as a semantic duplicate detector.


### Deteksi kandidat pertanyaan mirip secara leksikal
Audit kini membandingkan trigram karakter menggunakan cosine similarity untuk menemukan kandidat pertanyaan yang mirip, termasuk variasi kecil/typo. Hanya contoh yang lulus review privasi dan kualitas pada split train/validation/test yang dibandingkan. Laporan hanya berisi jumlah pasangan, skor maksimum agregat, dan jumlah konflik jawaban; tidak menampilkan teks. Ini heuristik kemiripan bentuk tulisan, bukan pemahaman semantik atau deteksi parafrasa yang andal. Semua kandidat wajib ditinjau manusia dan tidak dihapus otomatis.


### Antrean peninjauan kandidat data
Audit kini menyertakan antrean kandidat pertanyaan mirip berisi pasangan `example_id`, skor kemiripan leksikal, dan penanda apakah jawaban berbeda. Isi pesan mentah tidak disertakan. Peninjau harus membuka sumber korpus lokal, memeriksa konteks dan lisensi, lalu memutuskan secara manual; laporan ini tidak mengubah status, mengoreksi, atau menghapus contoh secara otomatis. Kandidat hasil trigram adalah petunjuk, bukan keputusan semantik.


### Tinjauan manusia dan salinan korpus
Tool `tools/corpus/review_corpus.py` menerapkan keputusan eksplisit `approve`, `reject`, atau `correct` dari file JSON keputusan ke salinan korpus baru. Contoh penggunaan:

```bash
python tools/corpus/review_corpus.py corpus.jsonl --decisions review-decisions.json --output corpus-reviewed.jsonl
```

File keputusan menggunakan objek `decisions` berisi `example_id`, `action`, dan `reviewer`; aksi `correct` juga wajib menyertakan array `messages` yang telah diperbaiki. Output yang sudah ada tidak pernah ditimpa dan file sumber tidak diubah. Persetujuan kualitas hanya diizinkan jika tinjauan privasi berstatus `passed`. Penolakan memberi status kualitas `rejected`, yang tidak dapat masuk loader training. Koreksi mengembalikan privasi dan kualitas ke `pending` serta split ke `unassigned`; setelah itu contoh harus ditinjau ulang dan dibagi ulang. Status pending dapat disimpan oleh validator, tetapi tidak dapat masuk tokenizer atau training karena loader hanya menerima status passed. Catatan riwayat menyimpan reviewer dan jenis keputusan, bukan teks pribadi atau komentar bebas.

Contoh template keputusan ada di `tools/corpus/examples/review_decisions.example.json`. Untuk koreksi, gunakan `"action": "correct"` dan sertakan `"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`. Untuk penolakan gunakan `"action": "reject"`. Ganti ID contoh dan identitas reviewer dengan nilai yang benar; jangan menaruh data pribadi dalam metadata reviewer.


### Membuat antrean tinjauan lokal
Sebelum menerapkan keputusan, buat daftar ID yang perlu diperiksa tanpa menyalin isi percakapan:

```bash
python tools/corpus/build_review_queue.py corpus.jsonl --output review-queue.json
```

Antrean mencakup contoh dengan status privasi/kualitas pending serta kandidat pertanyaan mirip dari audit leksikal. Untuk setiap item tersedia ID, sumber, lisensi, split, alasan, dan tautan ID ke contoh terkait; teks mentah tidak diekspor. Buka korpus lokal berdasarkan `example_id`, periksa sumber dan konteks, lalu isi file keputusan berdasarkan template sebelum menjalankan `review_corpus.py`.


### Gerbang izin sumber terhubung ke tokenizer dan trainer

- Register sumber CSV berada di `docs/dataset_source_register.csv`; format dan kolomnya dijelaskan di `docs/DATASET_SOURCE_REGISTER_TEMPLATE.md`.
- Validasi register dapat dijalankan dengan `python tools/corpus/validate_source_register.py docs/dataset_source_register.csv`. Template kosong memang akan gagal karena belum ada sumber yang ditinjau.
- Tokenizer kini memerlukan argumen `--source-register` dan hanya mengambil contoh train yang lolos review serta berasal dari `source_id` berstatus approved dengan `training_permission=yes`.
- Loader Transformer juga memerlukan path register; setiap sumber contoh train/validation harus ditemukan dalam register approved. Judul dan URI kanonis pada JSONL harus cocok dengan register.
- Hash register dicatat pada metadata dataset dan diperiksa saat melanjutkan checkpoint. Perubahan izin atau isi register membuat resume ditolak agar checkpoint tidak dilanjutkan dengan provenance berbeda.
- Perintah tokenizer dan training harus selalu menyertakan register:
  `python tools/corpus/train_tokenizer.py <corpus.jsonl> --source-register <sources.csv> --output <tokenizer.json>`
  `python tools/model/train_transformer.py --corpus <corpus.jsonl> --tokenizer <tokenizer.json> --source-register <sources.csv> --output <checkpoint.pt>`
- Pemeriksaan ini adalah gerbang teknis berbasis metadata. Ia tidak membuktikan secara hukum bahwa dokumen izin benar atau lisensi ditafsirkan tepat; bukti tetap harus diperiksa manusia.


Generator manifest kini juga memerlukan `--source-register` dan menolak sumber yang tidak terdaftar, belum approved, atau metadata judul/URI-nya tidak cocok. Manifest mencatat URI bukti izin, declared license, izin training, dan status penggunaan komersial untuk setiap sumber yang dipakai.

## Audit jalur inferensi Android

Penilaian struktur Android, format checkpoint, tokenizer, dan kandidat runtime dicatat di [`docs/ANDROID_NEURAL_INFERENCE_ASSESSMENT.md`](ANDROID_NEURAL_INFERENCE_ASSESSMENT.md). Runtime GGUF/llama.cpp yang sudah ada tetap jalur kompatibilitas terpisah; ia tidak memuat checkpoint PyTorch Tata secara langsung. Ekspor ONNX dan ONNX Runtime Android baru sebatas kandidat proof-of-concept, belum dipilih atau diimplementasikan. Integrasi menunggu ekspor valid, parity tokenizer, kesetaraan numerik, dan uji perangkat fisik.
