package com.harmanibike.tata.onnxpoc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TataOnnxRuntimeDeviceTest {

    @Test
    fun appDoesNotRequestInternetPermission() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val packageInfo = context.packageManager.getPackageInfo(
            context.packageName,
            android.content.pm.PackageManager.GET_PERMISSIONS
        )
        val requested = packageInfo.requestedPermissions.orEmpty().toSet()
        assertFalse("PoC must not request INTERNET permission", requested.contains("android.permission.INTERNET"))
    }

    @Test
    fun androidOnnxLogitsMatchPyTorchReferenceForDynamicShapes() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val modelFile = File(context.cacheDir, "tata_transformer_synthetic.onnx")
        instrumentation.context.assets.open("tata_transformer_synthetic.onnx").use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }
        val reference = instrumentation.context.assets.open("tata_transformer_pytorch_reference.json")
            .bufferedReader().use { it.readText() }
        val referenceJson = JSONObject(reference)
        val environment = OrtEnvironment.getEnvironment()

        TataOnnxRuntimeProbe.open(modelFile).use { probe ->
            assertEquals(setOf("input_ids"), probe.inputNames)
            assertEquals(setOf("logits"), probe.outputNames)

            for (caseName in listOf("1x4", "2x5")) {
                val sample = referenceJson.getJSONObject(caseName)
                val inputRows = sample.getJSONArray("input_ids")
                val batch = inputRows.length()
                val sequence = inputRows.getJSONArray(0).length()
                val tokenIds = LongArray(batch * sequence)
                for (b in 0 until batch) {
                    val row = inputRows.getJSONArray(b)
                    assertEquals(sequence, row.length())
                    for (s in 0 until sequence) tokenIds[b * sequence + s] = row.getLong(s)
                }
                val expected = sample.getJSONArray("logits")
                OnnxTensor.createTensor(
                    environment,
                    LongBuffer.wrap(tokenIds),
                    longArrayOf(batch.toLong(), sequence.toLong())
                ).use { tensor ->
                    probe.run(mapOf("input_ids" to tensor)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val actual = result[0].value as Array<Array<FloatArray>>
                        assertEquals(batch, actual.size)
                        for (b in 0 until batch) {
                            val expectedSequence = expected.getJSONArray(b)
                            assertEquals(sequence, actual[b].size)
                            for (s in 0 until sequence) {
                                val expectedToken = expectedSequence.getJSONArray(s)
                                assertEquals(8, actual[b][s].size)
                                for (v in 0 until 8) {
                                    val expectedValue = expectedToken.getDouble(v).toFloat()
                                    val actualValue = actual[b][s][v]
                                    assertFalse(actualValue.isNaN() || actualValue.isInfinite())
                                    val tolerance = 1e-4f + 1e-3f * kotlin.math.abs(expectedValue)
                                    org.junit.Assert.assertEquals(
                                        "case=$caseName batch=$b sequence=$s vocab=$v",
                                        expectedValue, actualValue, tolerance
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        modelFile.delete()
    }

    @Test
    fun opensLocalSyntheticModelAndRunsIdentityInferenceOffline() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelFile = File(context.cacheDir, "tiny_identity.onnx")
        InstrumentationRegistry.getInstrumentation().context.assets.open("tiny_identity.onnx").use { input ->
            modelFile.outputStream().use { output -> input.copyTo(output) }
        }

        val environment = OrtEnvironment.getEnvironment()
        TataOnnxRuntimeProbe.open(modelFile).use { probe ->
            assertEquals(setOf("input"), probe.inputNames)
            assertEquals(setOf("output"), probe.outputNames)

            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(floatArrayOf(1.25f, -3.5f)),
                longArrayOf(1, 2)
            ).use { tensor ->
                probe.run(mapOf("input" to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value as Array<FloatArray>
                    assertEquals(1, output.size)
                    assertArrayEquals(floatArrayOf(1.25f, -3.5f), output[0], 0.00001f)
                }
            }
        }
        modelFile.delete()
    }
}
