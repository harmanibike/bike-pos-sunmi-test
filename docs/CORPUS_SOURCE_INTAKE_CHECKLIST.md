# Checklist Penerimaan Sumber Data Tata AI

**Status: prosedur; belum ada sumber produksi yang dinyatakan disetujui oleh dokumen ini.**

Gunakan checklist ini untuk setiap sumber sebelum kontennya masuk ke korpus. Validator otomatis hanya memeriksa kelengkapan dan format metadata; ia tidak dapat membuktikan bahwa izin benar-benar sah atau bahwa suatu sumber bebas dari data pribadi.

## A. Identitas dan asal sumber

- [ ] Sumber berasal dari pemilik/penerbit/pembuat yang dapat diidentifikasi.
- [ ] Judul, pemilik, URL kanonis, dan tanggal akses dicatat di `docs/dataset_source_register.csv`.
- [ ] Bukti izin/lisensi disimpan sebagai salinan atau arsip yang dapat diaudit, bukan hanya tautan yang mungkin berubah.
- [ ] Versi, tanggal berlaku, dan bagian ketentuan yang relevan dicatat.
- [ ] Pengambilan tidak melewati login, paywall, CAPTCHA, rate limit, atau kontrol akses.

## B. Hak penggunaan untuk pelatihan

- [ ] Bukti secara jelas mengizinkan penggunaan teks/data untuk pelatihan atau pengembangan model AI/ML.
- [ ] Hak membuat transformasi, ringkasan, atau contoh dialog turunan diperiksa.
- [ ] Kewajiban atribusi dicatat dan dapat dipenuhi.
- [ ] Batas penggunaan komersial dan redistribusi dicatat terpisah.
- [ ] Ketentuan tidak jelas, bertentangan, atau tidak menyebut pelatihan model: set `training_permission=unclear` dan `decision=pending`.
- [ ] Jangan menafsirkan izin membaca, mengunduh, atau menggunakan untuk riset sebagai izin training.

**Aturan keputusan:** sumber hanya boleh berstatus `approved` jika bukti yang ditinjau secara nyata mendukung training_permission=yes. Persetujuan training tidak otomatis berarti penggunaan komersial atau redistribusi juga diizinkan.

## C. Privasi dan keamanan

- [ ] Tidak ada percakapan privat, riwayat chat pengguna, data akun, atau materi yang dikumpulkan tanpa persetujuan khusus untuk training.
- [ ] Tidak ada nama lengkap, alamat, nomor telepon, email, identitas pemerintah, lokasi pribadi, data anak, atau informasi sensitif yang tidak diperlukan.
- [ ] Tidak ada kata sandi, token, API key, private key, atau rahasia operasional.
- [ ] Data pribadi pihak ketiga dihapus atau dianonimkan secara memadai sebelum contoh dibuat.
- [ ] Pemeriksaan manual dilakukan; regex/validator otomatis hanya bantuan dan tidak menjamin semua data pribadi terdeteksi.
- [ ] Hasil pemeriksaan dan reviewer dicatat. Jika ragu, status privasi tetap pending dan contoh tidak boleh masuk training.

## D. Penyusunan contoh

- [ ] Buat contoh dialog orisinal; jangan menyalin paragraf panjang atau meniru ekspresi khas sumber.
- [ ] Fakta yang dipertahankan memang boleh digunakan dan dapat diverifikasi.
- [ ] Setiap contoh memiliki example_id stabil dan source_id yang benar.
- [ ] Metadata lisensi, judul, URI, dan source_id cocok dengan register.
- [ ] Tidak ada contoh duplikat, spam, promosi terselubung, instruksi berbahaya, atau jawaban yang mengarang fakta.
- [ ] Data yang belum jelas haknya tetap di luar korpus aktif.

## E. Review dan persetujuan

- [ ] Review privasi selesai dan `privacy_review=passed`.
- [ ] Review kualitas terpisah selesai dan `quality_review=passed`.
- [ ] Keputusan, reviewer, dan alasan dapat ditelusuri melalui review history.
- [ ] Contoh yang dikoreksi kembali ke status pending dan split unassigned; lakukan review ulang.
- [ ] Sumber yang ditolak atau pending tidak boleh dipakai melatih tokenizer/model.
- [ ] Near-duplicate dan jawaban yang saling bertentangan diperiksa manual.

## F. Gate sebelum tokenizer dan training

- [ ] Register sumber lolos `validate_source_register.py`.
- [ ] Korpus lolos `validate_corpus.py` dan audit dataset.
- [ ] Semua sumber yang dipakai memiliki decision=approved dan training_permission=yes.
- [ ] Semua contoh yang dipakai memiliki privacy_review=passed dan quality_review=passed.
- [ ] Split train/validation/test dilakukan berdasarkan source_id; satu sumber tidak boleh bocor lintas split.
- [ ] Tokenizer hanya dilatih dari split train.
- [ ] Test set dikunci dan tidak dipakai memilih konfigurasi.
- [ ] Manifest, hash korpus, register, tokenizer, konfigurasi, dan checkpoint disimpan untuk reproduksibilitas.

## G. Aturan khusus Tata

- Data percakapan pengguna dan Learning Lab tidak digunakan untuk training secara default. Persetujuan menyimpan pengetahuan bukan persetujuan melatih model.
- Jangan mengisi dataset dengan fixture sintetis repositori. Fixture hanya untuk pengujian software.
- Jangan membuat source_id, lisensi, bukti izin, reviewer, atau keputusan persetujuan fiktif.
- Bila jumlah sumber yang sah belum mencukupi, tunda training produksi. Eksperimen pipeline sintetis harus tetap diberi label tes/pilot terbatas.
- Seluruh proses pengumpulan, review, tokenizer, dan training berjalan lokal/offline. Jangan mengunggah korpus privat ke layanan eksternal.
