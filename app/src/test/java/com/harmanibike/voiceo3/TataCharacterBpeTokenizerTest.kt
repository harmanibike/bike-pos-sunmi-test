package com.harmanibike.voiceo3

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.json.JSONObject
import org.junit.Test

class TataCharacterBpeTokenizerTest {
    private fun validTokenizerJson(
        format: String = "tata-character-bpe",
        version: String = "1",
        specialTokens: String = """["<pad>","<unk>","<bos>","<eos>"]""",
        alphabet: String = """[" ","a","b","c","i"]""",
        merges: String = """[["a","b"],["ab","c"]]""",
        vocabSize: String = "11"
    ) = """{"format":"$format","format_version":$version,"special_tokens":$specialTokens,"alphabet":$alphabet,"merges":$merges,"vocab_size":$vocabSize}"""

    @Test
    fun matchesPythonUsingSharedParityFixture() {
        val fixtureText = javaClass.getResourceAsStream("/character_bpe_parity.json")
            ?.bufferedReader(Charsets.UTF_8)
            ?.use { it.readText() }
            ?: error("Shared tokenizer parity fixture is missing")
        val fixture = JSONObject(fixtureText)
        fun assertCase(case: JSONObject) {
            val caseTokenizer = TataCharacterBpeTokenizer.fromJson(
                case.getJSONObject("tokenizer").toString()
            )
            val samples = case.getJSONArray("samples")
            for (index in 0 until samples.length()) {
                val sample = samples.getJSONObject(index)
                val sampleText = sample.getString("text")
                val expectedIds = sample.getJSONArray("token_ids")
                assertEquals(
                    (0 until expectedIds.length()).map { expectedIds.getInt(it) },
                    caseTokenizer.encode(sampleText)
                )
                assertEquals(sample.getString("decoded"), caseTokenizer.decode(caseTokenizer.encode(sampleText)))
            }
        }
        assertCase(fixture)
        val additional = fixture.getJSONArray("additional_cases")
        for (index in 0 until additional.length()) {
            assertCase(additional.getJSONObject(index))
        }
    }

    @Test
    fun matchesPythonVocabularyOrderAndMergeEncoding() {
        val tokenizer = TataCharacterBpeTokenizer(
            alphabet = listOf(" ", "a", "b", "c", "i"),
            merges = listOf("a" to "b", "ab" to "c")
        )
        assertEquals(
            listOf("<pad>", "<unk>", "<bos>", "<eos>", " ", "a", "b", "c", "i", "ab", "abc"),
            tokenizer.idToToken
        )
        assertEquals(listOf(10, 4, 8), tokenizer.encode("abc i"))
        assertEquals("abc i", tokenizer.decode(tokenizer.encode("abc i")))
    }

    @Test
    fun loadsPythonTokenizerJsonAndPreservesTokenIds() {
        val tokenizer = TataCharacterBpeTokenizer.fromJson(validTokenizerJson())
        assertEquals(11, tokenizer.idToToken.size)
        assertEquals(listOf(10, 4, 8), tokenizer.encode("abc i"))
        assertEquals("abc i", tokenizer.decode(listOf(10, 4, 8)))
    }

    @Test
    fun rejectsUnknownFormatAndVersion() {
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(validTokenizerJson(format = "other"))
        }
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(validTokenizerJson(version = "2"))
        }
    }

    @Test
    fun rejectsMismatchedSpecialTokensAndVocabularySize() {
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(
                validTokenizerJson(specialTokens = """["<unk>"]""")
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(validTokenizerJson(vocabSize = "10"))
        }
    }

    @Test
    fun rejectsMalformedMergeAndNonStringAlphabetEntries() {
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(
                validTokenizerJson(merges = """[["a"]]""")
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(
                validTokenizerJson(alphabet = """["a",7]""")
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer.fromJson(
                validTokenizerJson(merges = """[["a",7]]""")
            )
        }
    }

    @Test
    fun handlesSupplementaryUnicodeCodePointsLikePython() {
        val tokenizer = TataCharacterBpeTokenizer(
            alphabet = listOf("a", "🧩"),
            merges = emptyList()
        )
        assertEquals(listOf(4, 5), tokenizer.encode("a🧩"))
        assertEquals("a🧩", tokenizer.decode(listOf(4, 5)))
    }

    @Test
    fun unknownCharactersUseUnknownToken() {
        val tokenizer = TataCharacterBpeTokenizer(listOf("a"), emptyList())
        assertEquals(listOf(4, 1), tokenizer.encode("a🧩"))
    }

    @Test
    fun rejectsInvalidMergeOrder() {
        assertThrows(IllegalArgumentException::class.java) {
            TataCharacterBpeTokenizer(listOf("a", "b"), listOf("ab" to "c"))
        }
    }
}
