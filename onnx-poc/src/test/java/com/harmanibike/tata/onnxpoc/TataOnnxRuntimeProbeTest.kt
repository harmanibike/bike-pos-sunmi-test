package com.harmanibike.tata.onnxpoc

import java.io.File
import org.junit.Assert.assertThrows
import org.junit.Test

class TataOnnxRuntimeProbeTest {

    @Test
    fun rejectsMissingModelFileBeforeRuntimeInitialization() {
        val missing = File(System.getProperty("java.io.tmpdir"), "tata-missing-model.onnx")
        missing.delete()

        assertThrows(IllegalArgumentException::class.java) {
            TataOnnxRuntimeProbe.open(missing)
        }
    }

    @Test
    fun rejectsEmptyModelFileBeforeRuntimeInitialization() {
        val empty = File.createTempFile("tata-empty-model", ".onnx")
        try {
            assertThrows(IllegalArgumentException::class.java) {
                TataOnnxRuntimeProbe.open(empty)
            }
        } finally {
            empty.delete()
        }
    }
}
