package com.harmanibike.tata.onnxpoc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File

/**
 * Experimental local-file ONNX session probe.
 *
 * This class deliberately has no network, asset-download, chat-engine, or app integration.
 * Callers must provide a model file already present on the device.
 */
class TataOnnxRuntimeProbe private constructor(
    private val session: OrtSession
) : AutoCloseable {

    val inputNames: Set<String>
        get() = session.inputNames

    val outputNames: Set<String>
        get() = session.outputNames

    /** Runs caller-supplied tensors against the already-open local model. */
    fun run(inputs: Map<String, OnnxTensor>): OrtSession.Result = session.run(inputs)

    override fun close() {
        session.close()
    }

    companion object {
        /**
         * Opens an ONNX model from a local file only. The caller owns this probe and must close it.
         */
        fun open(modelFile: File): TataOnnxRuntimeProbe {
            require(modelFile.isFile) { "ONNX model file does not exist" }
            require(modelFile.canRead()) { "ONNX model file is not readable" }
            require(modelFile.length() > 0L) { "ONNX model file is empty" }

            val environment = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            return try {
                TataOnnxRuntimeProbe(environment.createSession(modelFile.absolutePath, options))
            } finally {
                options.close()
            }
        }
    }
}
