package com.harmanibike.voiceo3

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/**
 * Eksperimen tokenizer Character-BPE Tata.
 * Urutan ID harus identik dengan tools.model.training_data.tokenizer_vocabulary.
 * Belum dipakai oleh engine percakapan produksi.
 */
class TataCharacterBpeTokenizer(
    alphabet: List<String>,
    merges: List<Pair<String, String>>
) {
    companion object {
        val SPECIAL_TOKENS = listOf("<pad>", "<unk>", "<bos>", "<eos>")
        private const val MAX_JSON_CHARS = 16 * 1024 * 1024
        private const val MAX_TOKEN_ENTRIES = 100_000

        /**
         * Membaca format JSON yang ditulis oleh CharacterBPE.to_dict() Python.
         * Ukuran dan seluruh field struktural diperiksa sebelum tokenizer dibuat.
         */
        fun fromJson(json: String): TataCharacterBpeTokenizer {
            require(json.length <= MAX_JSON_CHARS) { "File tokenizer terlalu besar" }
            try {
                val payload = JSONObject(json)
                require(payload.get("format") == "tata-character-bpe") {
                    "Format tokenizer tidak dikenal"
                }
                require(payload.getInteger("format_version") == 1) {
                    "Versi format tokenizer tidak didukung"
                }

                val specialTokens = payload.getJSONArray("special_tokens").toStringList()
                require(specialTokens == SPECIAL_TOKENS) {
                    "Daftar token khusus tidak cocok"
                }

                val alphabetArray = payload.getJSONArray("alphabet")
                require(alphabetArray.length() <= MAX_TOKEN_ENTRIES) {
                    "Alphabet tokenizer terlalu besar"
                }
                val alphabet = alphabetArray.toStringList()

                val mergesArray = payload.getJSONArray("merges")
                require(mergesArray.length() <= MAX_TOKEN_ENTRIES) {
                    "Daftar merge tokenizer terlalu besar"
                }
                val merges = ArrayList<Pair<String, String>>(mergesArray.length())
                for (index in 0 until mergesArray.length()) {
                    val pair = mergesArray.getJSONArray(index)
                    require(pair.length() == 2) { "Setiap merge harus berisi dua token" }
                    val leftValue = pair.get(0)
                    val rightValue = pair.get(1)
                    require(leftValue is String && rightValue is String) {
                        "Setiap bagian merge harus berupa teks"
                    }
                    merges.add(leftValue to rightValue)
                }

                val tokenizer = TataCharacterBpeTokenizer(alphabet, merges)
                require(payload.getInteger("vocab_size") == tokenizer.idToToken.size) {
                    "Ukuran vocabulary tokenizer tidak cocok"
                }
                return tokenizer
            } catch (error: JSONException) {
                throw IllegalArgumentException("JSON tokenizer tidak valid", error)
            }
        }

        private fun JSONObject.getInteger(name: String): Int {
            val value = get(name)
            require(value is Number) { "Field $name harus berupa angka" }
            val number = value.toDouble()
            require(number.isFinite() && number % 1.0 == 0.0 &&
                number >= Int.MIN_VALUE && number <= Int.MAX_VALUE
            ) { "Field $name harus berupa bilangan bulat" }
            return number.toInt()
        }

        private fun JSONArray.toStringList(): List<String> =
            (0 until length()).map { index ->
                val value = get(index)
                require(value is String) { "Elemen array harus berupa teks" }
                value
            }
    }

    val alphabet: List<String> = alphabet.toList()
    val merges: List<Pair<String, String>> = merges.toList()
    val idToToken: List<String>
    private val tokenToId: Map<String, Int>

    init {
        require(this.alphabet.all { it.codePointCount(0, it.length) == 1 }) {
            "Setiap entri alphabet harus satu Unicode code point"
        }
        require(this.alphabet.distinct().size == this.alphabet.size) {
            "Alphabet mengandung token duplikat"
        }
        val known = (SPECIAL_TOKENS + this.alphabet).toMutableSet()
        val ordered = (SPECIAL_TOKENS + this.alphabet).toMutableList()
        for ((left, right) in this.merges) {
            require(left.isNotEmpty() && right.isNotEmpty()) { "Merge kosong" }
            require(left in known && right in known) { "Merge memakai token yang belum tersedia" }
            val merged = left + right
            require(known.add(merged)) { "Merge menghasilkan token duplikat" }
            ordered.add(merged)
        }
        idToToken = ordered
        tokenToId = ordered.withIndex().associate { it.value to it.index }
    }

    fun encode(text: String): List<Int> {
        var tokens = text.codePoints().toArray().map { cp ->
            String(Character.toChars(cp)).let { if (it in alphabet) it else "<unk>" }
        }
        for ((left, right) in merges) {
            val merged = left + right
            val output = ArrayList<String>(tokens.size)
            var index = 0
            while (index < tokens.size) {
                if (index + 1 < tokens.size &&
                    tokens[index] == left && tokens[index + 1] == right
                ) {
                    output.add(merged)
                    index += 2
                } else {
                    output.add(tokens[index])
                    index += 1
                }
            }
            tokens = output
        }
        return tokens.map { tokenToId.getValue(it) }
    }

    fun decode(ids: List<Int>): String = buildString {
        for (id in ids) {
            val token = idToToken.getOrNull(id) ?: error("ID token di luar vocabulary: $id")
            if (token !in SPECIAL_TOKENS) append(token)
        }
    }
}
