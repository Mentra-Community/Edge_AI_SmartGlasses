package com.mentra.gemma

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.ByteArrayOutputStream
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import expo.modules.kotlin.exception.Exceptions
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "MentraGemma"

class MentraGemmaModule : Module() {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private var engine: Engine? = null
  private var conversation: Conversation? = null
  private var loadedPath: String? = null
  private val generating = AtomicBoolean(false)

  override fun definition() = ModuleDefinition {
    Name("MentraGemma")
    Events("token", "done", "error")

    AsyncFunction("isSupported") {
      true
    }

    AsyncFunction("isLoaded") {
      engine != null && conversation != null
    }

    AsyncFunction("load") { options: Map<String, Any?> ->
      val modelPath = (options["modelPath"] as? String)?.takeIf { it.isNotBlank() }
        ?: throw IllegalArgumentException("modelPath is required")
      val file = File(modelPath)
      if (!file.exists()) {
        throw IllegalArgumentException("Model file not found at $modelPath")
      }
      if (!file.canRead()) {
        throw IllegalArgumentException(
          "Model file at $modelPath is not readable. Push it into the app's files dir " +
            "(e.g. /data/data/<your-app-id>/files/), or chmod a+r the file."
        )
      }

      val maxNumTokens = (options["maxNumTokens"] as? Number)?.toInt() ?: 4096
      val topK = (options["topK"] as? Number)?.toInt() ?: 64
      val topP = (options["topP"] as? Number)?.toDouble() ?: 0.95
      val temperature = (options["temperature"] as? Number)?.toDouble() ?: 1.0
      val systemPrompt = options["systemPrompt"] as? String
      val enableVision = (options["enableVision"] as? Boolean) ?: false
      val enableAudio = (options["enableAudio"] as? Boolean) ?: false
      val backendOption = (options["backend"] as? String)?.lowercase() ?: "auto"
      val visionBackendOption = (options["visionBackend"] as? String)?.lowercase() ?: "gpu"

      tearDown()

      // Decide CPU vs GPU. GPU allocates ~3GB VRAM for Gemma 4 E2B; on phones with <12GB RAM
      // this competes with Android's RenderThread and SIGSEGV-crashes the app (uncatchable).
      // "auto" falls back to CPU below 12GB.
      val ctx = reactContext()
      val totalRamGb = try {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        mi.totalMem / (1024L * 1024L * 1024L)
      } catch (t: Throwable) {
        Log.w(TAG, "Could not read totalMem, assuming low-RAM", t)
        0L
      }
      val backend: Backend = when (backendOption) {
        "gpu" -> {
          Log.w(TAG, "Forced GPU backend (totalRam=${totalRamGb}GB). <12GB devices may SIGSEGV.")
          Backend.GPU()
        }
        "cpu" -> Backend.CPU()
        else -> {
          if (totalRamGb >= 12) {
            Log.d(TAG, "auto-selecting GPU backend (totalRam=${totalRamGb}GB >= 12GB)")
            Backend.GPU()
          } else {
            Log.d(TAG, "auto-selecting CPU backend (totalRam=${totalRamGb}GB < 12GB)")
            Backend.CPU()
          }
        }
      }

      Log.d(TAG, "Loading model from $modelPath (maxTokens=$maxNumTokens, backend=$backendOption→${if (backend is Backend.GPU) "GPU" else "CPU"}, vision=$enableVision, audio=$enableAudio)")

      val visionBackend: Backend? = if (enableVision) {
        if (visionBackendOption == "cpu") Backend.CPU() else Backend.GPU()
      } else null

      val engineConfig = EngineConfig(
        modelPath = modelPath,
        backend = backend,
        visionBackend = visionBackend,
        audioBackend = if (enableAudio) Backend.CPU() else null,
        maxNumTokens = maxNumTokens,
      )

      val newEngine = try {
        Engine(engineConfig).also { it.initialize() }
      } catch (t: Throwable) {
        Log.e(TAG, "Engine initialization failed", t)
        throw IllegalStateException("Engine initialize failed: ${t.message}", t)
      }

      val samplerConfig = SamplerConfig(
        topK = topK,
        topP = topP,
        temperature = temperature,
      )

      val conversationConfig = if (systemPrompt.isNullOrBlank()) {
        ConversationConfig(samplerConfig = samplerConfig)
      } else {
        ConversationConfig(
          samplerConfig = samplerConfig,
          systemInstruction = Contents.of(listOf(Content.Text(systemPrompt))),
        )
      }

      val newConversation = try {
        newEngine.createConversation(conversationConfig)
      } catch (t: Throwable) {
        Log.e(TAG, "createConversation failed", t)
        try { newEngine.close() } catch (_: Throwable) {}
        throw IllegalStateException("createConversation failed: ${t.message}", t)
      }

      engine = newEngine
      conversation = newConversation
      loadedPath = modelPath
      Log.d(TAG, "Model ready.")
    }

    AsyncFunction("resetConversation") { systemPrompt: String? ->
      val currentEngine = engine ?: throw IllegalStateException("Model not loaded.")
      val currentConversation = conversation
      try { currentConversation?.close() } catch (_: Throwable) {}

      val samplerConfig = SamplerConfig(topK = 64, topP = 0.95, temperature = 1.0)
      val config = if (systemPrompt.isNullOrBlank()) {
        ConversationConfig(samplerConfig = samplerConfig)
      } else {
        ConversationConfig(
          samplerConfig = samplerConfig,
          systemInstruction = Contents.of(listOf(Content.Text(systemPrompt))),
        )
      }
      conversation = currentEngine.createConversation(config)
    }

    AsyncFunction("unload") {
      tearDown()
    }

    AsyncFunction("cancelGeneration") {
      try { conversation?.cancelProcess() } catch (t: Throwable) {
        Log.w(TAG, "cancelProcess failed", t)
      }
      generating.set(false)
    }

    AsyncFunction("generateStream") { prompt: String, enableThinking: Boolean, image: ByteArray? ->
      val conv = conversation ?: throw IllegalStateException("Model not loaded. Call load() first.")
      if (!generating.compareAndSet(false, true)) {
        throw IllegalStateException("A generation is already in progress.")
      }

      val contents = if (image != null && image.isNotEmpty()) {
        // LiteRT expects PNG bytes for Content.ImageBytes. The glasses send JPEG, so we
        // decode → re-encode as PNG here.
        val png = jpegToPng(image)
        if (png != null) {
          Log.d(TAG, "generateStream with image · jpeg=${image.size}B → png=${png.size}B")
          Contents.of(listOf(Content.ImageBytes(png), Content.Text(prompt)))
        } else {
          Log.w(TAG, "generateStream · failed to transcode image (${image.size} bytes), sending text-only")
          Contents.of(listOf(Content.Text(prompt)))
        }
      } else {
        Contents.of(listOf(Content.Text(prompt)))
      }
      val extra = if (enableThinking) mapOf("enable_thinking" to "true") else emptyMap()

      try {
        conv.sendMessageAsync(
          contents,
          object : MessageCallback {
            override fun onMessage(message: Message) {
              val text = try { message.toString() } catch (_: Throwable) { "" }
              val thinking = try { message.channels["thought"]?.toString() } catch (_: Throwable) { null }
              try {
                sendEvent(
                  "token",
                  mapOf(
                    "partial" to text,
                    "thinking" to (thinking ?: ""),
                  ),
                )
              } catch (_: Throwable) {
                // Swallow event-emit errors so the stream keeps running.
              }
            }

            override fun onDone() {
              generating.set(false)
              try { sendEvent("done", mapOf("ok" to true)) } catch (_: Throwable) {}
            }

            override fun onError(throwable: Throwable) {
              generating.set(false)
              Log.e(TAG, "stream error", throwable)
              try {
                sendEvent("error", mapOf("message" to (throwable.message ?: throwable.javaClass.simpleName)))
              } catch (_: Throwable) {}
            }
          },
          extra,
        )
      } catch (t: Throwable) {
        generating.set(false)
        Log.e(TAG, "sendMessageAsync failed to start", t)
        throw IllegalStateException("Generation failed to start: ${t.message}", t)
      }
    }

    OnDestroy {
      tearDown()
      scope.cancel()
    }
  }

  private fun tearDown() {
    generating.set(false)
    try { conversation?.close() } catch (_: Throwable) {}
    try { engine?.close() } catch (_: Throwable) {}
    conversation = null
    engine = null
    loadedPath = null
  }

  @Suppress("unused")
  private fun reactContext() = appContext.reactContext ?: throw Exceptions.ReactContextLost()

  private fun jpegToPng(jpeg: ByteArray): ByteArray? {
    return try {
      val bitmap: Bitmap? = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
      if (bitmap == null) {
        Log.w(TAG, "BitmapFactory could not decode image bytes")
        return null
      }
      val out = ByteArrayOutputStream(jpeg.size)
      val ok = bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
      bitmap.recycle()
      if (!ok) null else out.toByteArray()
    } catch (t: Throwable) {
      Log.w(TAG, "jpegToPng failed", t)
      null
    }
  }
}
