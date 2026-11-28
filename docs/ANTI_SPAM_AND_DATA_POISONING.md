# Tata AI — Kebijakan Anti-Spam dan Anti-Data-Poisoning

Dokumen ini menetapkan batas keamanan untuk penerimaan percakapan, data pengetahuan, dan bahan training. Ini bukan klaim bahwa server multi-perangkat sudah tersedia.

## Status arsitektur saat ini

- Aplikasi Android Tata AI saat ini terutama berjalan lokal/offline.
- Repository belum memiliki endpoint server/client multi-perangkat yang dapat menerapkan pembatasan request jaringan.
- Data korpus training melewati validator, pemeriksaan privasi/kualitas, register sumber, dan persetujuan sebelum dapat dimuat trainer.
- Percakapan pengguna tidak otomatis menjadi pengetahuan atau bahan training.

## Perlindungan yang sudah diterapkan pada jalur korpus

Validator korpus menolak atau menandai:

- ID contoh dan percakapan yang duplikat.
- Alamat email, nomor telepon, kredensial yang terdeteksi, dan private key.
- File korpus di atas 100 MiB.
- Lebih dari 100.000 record.
- Baris JSONL di atas 1 MiB.
- Lebih dari 40 pesan per contoh.
- Pesan di atas 20.000 karakter atau total percakapan di atas 100.000 karakter.
- Format percakapan, lisensi, sumber, atau status review yang tidak valid.

Batas ini adalah pagar keselamatan untuk proses impor korpus, bukan kuota chat real-time. Validator tetap memerlukan review manusia; deteksi pola tidak menjamin semua data pribadi atau data berbahaya terdeteksi.

## Alur wajib untuk penerimaan chat dan sinkronisasi server di masa depan

Sebelum fitur server/client diaktifkan, implementasikan semua lapisan berikut:

1. **Identitas dan autentikasi**
   - Setiap perangkat memiliki identitas terdaftar dan kredensial unik.
   - Jangan mempercayai `device_id`, nama pengguna, atau role yang dikirim bebas oleh klien.
   - Admin dapat mencabut perangkat atau sesi yang disalahgunakan.

2. **Batas request**
   - Batas ukuran body, panjang pesan, jumlah lampiran, dan jenis berkas.
   - Rate limit terpisah per akun dan perangkat; tambahkan pembatasan jaringan sebagai sinyal tambahan, bukan satu-satunya identitas.
   - Gunakan token bucket/sliding window dengan respons penolakan dan waktu tunggu yang jelas.
   - Terapkan batas konkurensi dan timeout agar request lambat tidak menghabiskan sumber daya.

3. **Duplikasi dan integritas**
   - Setiap operasi tulis membawa idempotency key unik.
   - Tolak pengiriman ulang operasi yang sama; jangan membuat record baru hanya karena klien mengulang request setelah timeout.
   - Validasi skema dan ukuran sebelum membuka transaksi database.
   - Gunakan transaksi atomik, indeks unik, dan backup.

4. **Karantina dan moderasi data**
   - Pesan chat biasa tidak otomatis menjadi pengetahuan.
   - Kiriman yang diusulkan sebagai pengetahuan masuk ke status `perlu_ditinjau`.
   - Deteksi duplikat, pola spam, data pribadi, prompt injection, dan konflik fakta sebagai sinyal review.
   - Jangan otomatis memblokir pengguna hanya berdasarkan satu sinyal; sediakan pembatasan bertahap dan jalur banding Admin.

5. **Perlindungan training**
   - Hanya data yang lulus pemeriksaan privasi dan kualitas serta terikat pada sumber yang disetujui yang dapat dipakai.
   - Izin training dan lisensi harus cocok persis dengan register sumber.
   - Jangan pernah melatih model langsung dari chat live, hasil pencarian mentah, atau data karantina.
   - Catat versi dataset, hash, reviewer, dan alasan persetujuan/penolakan.

6. **Audit dan pemulihan**
   - Catat waktu, perangkat, jenis tindakan, hasil pembatasan, dan alasan penolakan; minimalkan isi pesan sensitif dalam log.
   - Sediakan penguncian sementara, pencabutan perangkat, backup, dan pemulihan.
   - Jangan menghapus data sah secara otomatis sebagai respons terhadap lonjakan trafik.

## Kriteria penerimaan sebelum server/client dirilis

- Request tanpa autentikasi ditolak.
- Rate limit akun/perangkat diuji untuk burst, pengulangan, dan banyak koneksi bersamaan.
- Retry dengan idempotency key tidak menggandakan data.
- Payload terlalu besar, malformed, dan duplikat ditolak tanpa membuat database rusak.
- Data karantina tidak muncul sebagai pengetahuan aktif dan tidak dapat dimuat trainer.
- Revoke perangkat menghentikan akses perangkat tersebut.
- Pengujian beban dan pemulihan backup lulus.

**Catatan:** sampai endpoint server/client benar-benar dibuat dan diuji, perlindungan rate limit jaringan belum dapat dinyatakan aktif. Jangan mengiklankan fitur tersebut sebagai sudah tersedia.
