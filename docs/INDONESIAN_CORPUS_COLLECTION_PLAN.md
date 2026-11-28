# Rencana Pengumpulan Korpus Bahasa Indonesia Tata AI

**Status: rencana kerja; belum ada korpus produksi yang dikumpulkan atau disetujui.**

Dokumen ini menjadi panduan pengumpulan data orisinal/berizin. Angka target adalah target pilot internal, bukan klaim bahwa dataset sudah tersedia atau cukup untuk asisten umum.

Untuk pemeriksaan setiap sumber sebelum diterima, ikuti [Checklist Penerimaan Sumber Data](CORPUS_SOURCE_INTAKE_CHECKLIST.md). Checklist ini melengkapi register dan validator; bukan pengganti pemeriksaan manusia.

## 1. Tujuan pilot

Buktikan pipeline data → tokenizer → training kecil → evaluasi tertahan dengan data yang dapat diaudit. Pilot tidak ditujukan untuk menghasilkan asisten setara model besar.

Urutan prioritas:
1. Percakapan dasar Bahasa Indonesia: sapaan, klarifikasi, ucapan terima kasih, dan penjelasan sederhana.
2. Instruksi dan prosedur aman: langkah kerja, daftar periksa, dan penjelasan sebab-akibat.
3. Pengetahuan umum yang stabil: konsep sains dasar, matematika, bahasa, dan literasi digital.
4. Kemampuan dialog: pertanyaan lanjutan, konteks beberapa giliran, koreksi salah paham, dan ringkasan.
5. Ketidakpastian dan keselamatan: mengakui tidak tahu, meminta konteks, menolak permintaan berbahaya, dan memberi alternatif aman.

Jangan mulai dari scraping web secara massal. Utamakan materi yang ditulis khusus untuk Tata, materi milik proyek, kontribusi sukarela dengan izin tertulis, atau sumber yang lisensinya secara jelas mengizinkan pelatihan model.

## 2. Target pilot yang disarankan

Target minimum untuk menguji pipeline, bukan untuk menyatakan kualitas asisten:
- 3.000–5.000 pasangan percakapan orisinal/berizin yang sudah lolos review.
- Sedikitnya 20 kelompok sumber/penulis independen yang sah; jangan memecah satu dokumen menjadi banyak source_id untuk mengejar angka.
- Setiap domain prioritas memiliki contoh di train dan contoh evaluasi yang berasal dari kelompok sumber berbeda.
- Validation dan test dikunci berdasarkan kelompok sumber sebelum tokenizer/training.
- Target awal distribusi tugas: 30% penjelasan/QA, 25% instruksi, 20% dialog multi-giliran, 10% penalaran sederhana, 10% ringkasan, 5% ketidakpastian/keselamatan. Distribusi boleh disesuaikan berdasarkan hasil audit, bukan dengan menduplikasi contoh.

Jika belum mencapai target, jalankan eksperimen pipeline dengan ukuran kecil dan beri label **pilot terbatas**. Jangan menaikkan status kesiapan hanya karena skrip berhasil berjalan.

## 3. Prosedur satu sumber

1. Catat sumber pada `DATASET_SOURCE_REGISTER_TEMPLATE.md`.
2. Simpan bukti izin/lisensi dan tanggal akses di penyimpanan proyek yang dapat diaudit.
3. Periksa apakah izin secara eksplisit mencakup text/data mining dan pelatihan model; lisensi yang ambigu berarti `pending`.
4. Jangan mengambil konten di balik login, paywall, atau kontrol akses; patuhi ketentuan dan batas akses sumber.
5. Hilangkan data pribadi, rahasia, kredensial, dan percakapan privat.
6. Ubah materi menjadi contoh dialog orisinal yang mempertahankan fakta yang diizinkan, bukan menyalin ekspresi panjang sumber.
7. Beri ID contoh stabil dan source_id yang benar. Simpan provenance dan batas penggunaan.
8. Lakukan review privasi dan review kualitas secara terpisah oleh manusia.
9. Jalankan validator, audit duplikasi, audit kualitas, lalu split berdasarkan source_id.
10. Buat manifest dan hash versi dataset; jangan menimpa dataset yang sudah dirilis.

## 4. Rubrik review contoh

Nilai setiap dimensi 0–2:
- Ketepatan: 0 salah, 1 sebagian benar/perlu konteks, 2 benar dan dapat diperiksa.
- Relevansi: 0 tidak menjawab, 1 sebagian, 2 menjawab pertanyaan.
- Kejelasan Bahasa Indonesia: 0 sulit dipahami, 1 kurang alami, 2 jelas dan alami.
- Kelengkapan: 0 tidak membantu, 1 ada bagian penting hilang, 2 cukup untuk konteksnya.
- Keselamatan/privasi: 0 berbahaya atau membocorkan data, 1 perlu redaksi/konteks, 2 aman dan tidak mengandung data pribadi.

Saran keputusan:
- **passed**: total minimal 8/10 dan nilai keselamatan/privasi 2; tidak ada fakta berisiko yang belum diverifikasi.
- **pending**: perlu verifikasi fakta, penyuntingan, atau keputusan lisensi.
- **rejected**: informasi salah/menyesatkan, duplikat, privasi bermasalah, izin tidak ada, atau konten tidak aman.

Nilai numerik adalah panduan reviewer, bukan pengganti penilaian ahli. Untuk medis, hukum, finansial, keselamatan fisik, dan informasi yang cepat berubah, jangan masukkan sebagai jawaban otoritatif tanpa sumber ahli dan konteks yang sesuai.

## 5. Pembagian dan pencegahan kebocoran

- Deduplikasi persis dan near-duplicate dilakukan sebelum split.
- Kelompokkan semua contoh dari sumber/dokumen/penulis yang sama ke satu split.
- Gunakan train untuk tokenizer dan training saja.
- Gunakan validation untuk pemilihan konfigurasi/early stopping.
- Simpan test set tertutup untuk evaluasi akhir; jangan memilih hyperparameter dari hasil test.
- Catat jumlah contoh, token, sumber, bahasa, tugas, lisensi, dan hash untuk tiap split.

Proporsi 90/5/5 pada roadmap adalah titik awal, bukan kewajiban jika jumlah source group terlalu sedikit. Dengan kelompok sumber terbatas, lebih baik menunda test set yang representatif daripada membuat klaim evaluasi yang menyesatkan.

## 6. Kriteria sebelum training pertama

Training hanya boleh dimulai setelah:
- register sumber dan bukti izin lengkap;
- semua contoh yang akan dipakai berstatus privacy_review=passed dan quality_review=passed;
- validator dan audit dataset lulus tanpa error;
- duplikasi/near-duplicate ditinjau;
- train/validation/test terpisah pada tingkat source_id;
- tokenizer hanya dilatih dari train;
- manifest, hash, konfigurasi, dan keputusan reviewer tersimpan;
- tidak ada data chat pribadi atau Learning Lab yang ikut terambil.

## 7. Status Tata saat dokumen dibuat

- Fixture sintetis repositori: delapan contoh untuk tes software; bukan korpus produksi.
- Contoh fixture masih pending dan tidak boleh dipakai training.
- Korpus produksi berizin: belum tersedia.
- Tokenizer produksi dari korpus Tata: belum dibuat.
- Checkpoint model terlatih: belum tersedia.
- Integrasi model neural ke Android: belum dilakukan.

**Tindakan berikut:** kumpulkan materi orisinal/berizin, masukkan register sumber, lalu lakukan review manusia. Jangan mengarang provenance, lisensi, atau status review untuk membuat pipeline tampak siap.
