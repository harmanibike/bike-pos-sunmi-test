# Register Sumber Dataset Tata AI

**Status:** template kosong. Belum ada sumber eksternal yang disetujui atau ditambahkan ke korpus.

Gunakan satu baris untuk setiap sumber yang akan dipertimbangkan. Jangan memasukkan teks percakapan, data pribadi, atau materi berhak cipta ke register ini. Register ini mencatat bukti asal-usul dan izin; ia tidak menggantikan pemeriksaan manusia atau nasihat hukum.

## Kolom wajib

| Kolom | Isi |
|---|---|
| source_id | ID unik dan stabil, misalnya `id-wiki-001` |
| title | Nama sumber atau koleksi |
| owner | Pemilik/penerbit atau pembuat materi |
| canonical_uri | URL resmi atau pengenal sumber |
| accessed_at | Tanggal akses dalam format YYYY-MM-DD |
| declared_license | Lisensi persis sebagaimana dinyatakan pemilik |
| license_evidence_uri | Tautan ke halaman lisensi/ketentuan yang berlaku |
| training_permission | `yes`, `no`, atau `unclear` |
| commercial_use | `yes`, `no`, atau `unclear` |
| derivative_permission | `yes`, `no`, atau `unclear` |
| attribution_required | Keterangan atribusi yang diwajibkan |
| redistribution_allowed | `yes`, `no`, atau `unclear` |
| privacy_screen | Ringkasan pemeriksaan data pribadi/rahasia |
| reviewer | Identitas peninjau proyek, bukan data pribadi pemilik sumber |
| decision | `pending`, `approved`, atau `rejected` |
| decision_notes | Alasan singkat dan batas penggunaan |

## Gerbang sebelum sumber disetujui

1. Pastikan sumber benar-benar berasal dari pemilik/penerbit yang disebutkan.
2. Simpan bukti lisensi atau izin yang berlaku pada tanggal pengambilan.
3. Pastikan izin mencakup penggunaan untuk pelatihan model; izin membaca atau mengunduh saja tidak cukup.
4. Periksa syarat atribusi, penggunaan komersial, pembuatan turunan, dan pembagian hasil.
5. Periksa ketentuan situs, batas akses, dan larangan pengambilan otomatis. Jangan melewati login, paywall, rate limit, atau kontrol akses.
6. Singkirkan data pribadi, kredensial, percakapan privat, dan materi sensitif yang tidak diperlukan.
7. Beri keputusan `approved` hanya setelah bukti dan batas penggunaan ditinjau. Jika ragu, gunakan `unclear` dan jangan masukkan ke tokenizer atau training.

## Hubungan dengan korpus JSONL

Setiap contoh korpus harus menunjuk ke `source_id` yang tercatat di register dan membawa metadata lisensi serta tanggal akses sesuai skema di `docs/OWN_NEURAL_NETWORK_ROADMAP.md`. Jangan mengubah status `quality_review` menjadi `passed` hanya karena lisensi sumber disetujui: pemeriksaan kualitas isi adalah keputusan terpisah.

Data percakapan pengguna, riwayat chat, dan Learning Lab tetap dikecualikan secara default. Persetujuan menyimpan pengetahuan tidak sama dengan persetujuan memakai data untuk pelatihan.

## Status saat ini

Repositori saat ini hanya menyediakan fixture sintetis delapan contoh untuk pengujian perangkat lunak. Semua contoh fixture masih berstatus tinjauan kualitas `pending`, belum terbagi ke train/validation/test, dan tidak boleh digunakan untuk melatih model. Jangan mengisi kekosongan dataset dengan mengubah fixture tersebut menjadi data produksi.


## Register CSV dan pemeriksaan otomatis

Isi file `docs/dataset_source_register.csv` (satu baris per sumber), lalu jalankan:

```bash
python tools/corpus/validate_source_register.py docs/dataset_source_register.csv
```

Validator memeriksa kolom wajib, ID sumber unik, URL bukti, tanggal akses, nilai izin, dan menolak keputusan `approved` jika `training_permission` bukan `yes` atau bukti/alasan persetujuan kosong. Register kosong sengaja dinyatakan belum siap.

**Pemeriksaan silang otomatis:** tokenizer dan pemuat data training memeriksa setiap `source_id` terhadap register yang disetujui, termasuk izin `training_permission=yes`, kecocokan judul/URI, dan kecocokan lisensi sumber. Hash register dicatat dalam metadata training dan diperiksa kembali saat melanjutkan checkpoint. Pemeriksaan otomatis ini tidak menggantikan peninjauan manusia atas keabsahan izin dan kualitas isi sumber.
