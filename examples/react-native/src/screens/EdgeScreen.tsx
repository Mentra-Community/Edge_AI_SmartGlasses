import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TextInput, Pressable, ScrollView, StyleSheet, Animated, Easing, Image } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Path } from 'react-native-svg';
import { createAudioPlayer, setAudioModeAsync, type AudioPlayer } from 'expo-audio';
import { File, Paths } from 'expo-file-system';
import { fileModelPath } from 'react-native-sherpa-onnx';
import { createTTS, saveAudioToFile, type TtsEngine } from 'react-native-sherpa-onnx/tts';
import {
  createStreamingSTT,
  type StreamingSttEngine,
  type SttStream,
} from 'react-native-sherpa-onnx/stt';
import {
  ModelCategory,
  ensureModelByCategory,
  refreshModelsByCategory,
} from 'react-native-sherpa-onnx/download';
import BluetoothSdk, { type MicPcmEvent } from '@mentra/bluetooth-sdk';
import MentraDirectReceiver, {
  type DirectPhotoUploadEvent,
} from '../../modules/mentra-direct-receiver';
import MentraGemma, {
  type GemmaTokenEvent,
  type GemmaDoneEvent,
  type GemmaErrorEvent,
} from '../../modules/mentra-gemma';
import { useScrollBottomPadding } from '../components/keyboardLayout';
import { isGlassesConnected } from '../sdkFormat';
import type { MentraSdkModel } from '../useMentraSdk';

const TTS_MODEL_ID = 'vits-piper-en_US-lessac-low';
const ASR_MODEL_ID = 'sherpa-onnx-streaming-zipformer-en-2023-06-21-mobile';
const ASR_MODEL_TYPE = 'transducer' as const;
const MIC_SAMPLE_RATE = 16000;
const GEMMA_MODEL_FILENAME = 'gemma-3n-E2B-it-int4.litertlm';
const PHOTO_APP_ID = 'com.mentra.examples.reactnative';
const PHOTO_TIMEOUT_MS = 8000;
const GEMMA_SYSTEM_PROMPT =
  'You are a voice assistant on smart glasses. Keep answers short — usually 1-10 words. Single-word answers like "Yes" or "No" are fine. Avoid extra explanation or detail unless asked.';
// Order longest-first so "hey ment" doesn't shadow "hey mentra".
const WAKE_WORDS = [
  'hey mentra',
  'hey mantra',
  'hey mentor',
  'hay mentra',
  'hay mantra',
  'hay mentor',
  'aye mentra',
  'aye mantra',
  'aye mentor',
  'ay mentra',
  'ay mantra',
  'ay mentor',
  'a mentra',
  'a mantra',
  'a mentor',
  'eh mentra',
  'eh mantra',
  'eh mentor',
  'hey ment',
  'hay ment',
  'aye ment',
  'ay ment',
  'a ment',
  'eh ment',
] as const;
const SILENCE_MS = 800;

function findWakeWordEnd(text: string): number | null {
  const lower = text.toLowerCase();
  for (const word of WAKE_WORDS) {
    const idx = lower.indexOf(word);
    if (idx >= 0) return idx + word.length;
  }
  return null;
}

type TtsStatus =
  | { kind: 'idle' }
  | { kind: 'preparing'; phase: string; percent: number | null }
  | { kind: 'ready' }
  | { kind: 'generating' }
  | { kind: 'playing' }
  | { kind: 'error'; message: string };

type AsrStatus =
  | { kind: 'idle' }
  | { kind: 'preparing'; phase: string; percent: number | null }
  | { kind: 'ready' }
  | { kind: 'listening' }
  | { kind: 'error'; message: string };

type LlmStatus =
  | { kind: 'unloaded' }
  | { kind: 'loading' }
  | { kind: 'ready' }
  | { kind: 'generating' }
  | { kind: 'error'; message: string };

type LoopState =
  | { kind: 'off' }
  | { kind: 'preparing'; phase: string }
  | { kind: 'listening' }
  | { kind: 'capturing'; capturedText: string }
  | { kind: 'thinking'; query: string }
  | { kind: 'speaking'; response: string }
  | { kind: 'error'; message: string };

type BootStatus =
  | { kind: 'loading'; step: string }
  | { kind: 'done'; warning?: string };

type CycleTimings = {
  captureMs: number;
  gemmaMs: number;
  ttsGenMs: number;
  ttsPlayMs: number;
  totalMs: number;
};

export function EdgeScreen({ sdk }: { sdk: MentraSdkModel }) {
  const scrollBottomPadding = useScrollBottomPadding();
  const connected = isGlassesConnected(sdk.glassesStatus);

  // TTS state
  const [prompt, setPrompt] = useState('Hello world from sherpa onnx.');
  const [ttsStatus, setTtsStatus] = useState<TtsStatus>({ kind: 'idle' });
  const ttsRef = useRef<TtsEngine | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const playerSubRef = useRef<{ remove: () => void } | null>(null);

  // ASR state
  const [asrStatus, setAsrStatus] = useState<AsrStatus>({ kind: 'idle' });
  const [finalText, setFinalText] = useState('');
  const [partialText, setPartialText] = useState('');
  const sttRef = useRef<StreamingSttEngine | null>(null);
  const streamRef = useRef<SttStream | null>(null);
  const micSubRef = useRef<{ remove: () => void } | null>(null);
  const processingRef = useRef<Promise<void>>(Promise.resolve());
  const finalTextRef = useRef('');
  const partialTextRef = useRef('');

  // Loop (orchestrated ASR → Gemma → TTS) state
  const [loopState, setLoopStateRaw] = useState<LoopState>({ kind: 'off' });
  const loopStateRef = useRef<LoopState>({ kind: 'off' });
  const loopActiveRef = useRef(false);
  const wakeWordEndOffsetRef = useRef<number | null>(null);
  const loopLastTextRef = useRef('');
  const loopLastChangeAtRef = useRef(0);
  const loopSilenceTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const loopResponseRef = useRef('');
  const loopLastPartialRef = useRef('');

  // Photo capture for vision queries
  const photoUploadUrlRef = useRef<string | null>(null);
  const photoReceiverRunningRef = useRef(false);
  const activePhotoIdRef = useRef<string | null>(null);
  const latestPhotoBytesRef = useRef<Uint8Array | null>(null);
  const latestPhotoUriRef = useRef<string | null>(null);
  const photoSubRef = useRef<{ remove: () => void } | null>(null);
  const photoReadyResolveRef = useRef<((bytes: Uint8Array | null) => void) | null>(null);
  const photoReadyPromiseRef = useRef<Promise<Uint8Array | null> | null>(null);
  const [photoStatus, setPhotoStatus] = useState<'idle' | 'capturing' | 'ready' | 'failed'>('idle');
  const visionEnabledRef = useRef(false);
  const setLoopState = (s: LoopState) => {
    loopStateRef.current = s;
    setLoopStateRaw(s);
  };

  // Boot status & cycle timings
  const [bootStatus, setBootStatus] = useState<BootStatus>({
    kind: 'loading',
    step: 'starting',
  });
  const [lastTimings, setLastTimings] = useState<CycleTimings | null>(null);
  const [turnHistory, setTurnHistory] = useState<{ query: string; response: string; photoUri?: string }[]>([]);
  const currentQueryRef = useRef<string>('');

  const timingsRef = useRef<{
    wakeAt?: number;
    silenceAt?: number;
    gemmaDoneAt?: number;
    ttsReadyAt?: number;
    playStartedAt?: number;
  }>({});

  // LLM (Gemma 4) state
  const [llmStatus, setLlmStatus] = useState<LlmStatus>({ kind: 'unloaded' });
  const [llmPrompt, setLlmPrompt] = useState('Explain on-device AI in one sentence.');
  const [llmAnswer, setLlmAnswer] = useState('');
  const [llmThinking, setLlmThinking] = useState('');
  const [enableThinking, setEnableThinking] = useState(false);
  const gemmaModelFile = new File(Paths.document, GEMMA_MODEL_FILENAME);
  const gemmaModelPath = gemmaModelFile.uri.replace(/^file:\/\//, '');
  const llmTokenSubRef = useRef<{ remove: () => void } | null>(null);
  const llmDoneSubRef = useRef<{ remove: () => void } | null>(null);
  const llmErrorSubRef = useRef<{ remove: () => void } | null>(null);

  useEffect(() => {
    // Wire up Gemma streaming event listeners once.
    llmTokenSubRef.current = MentraGemma.addListener('token', (event: GemmaTokenEvent) => {
      if (event.partial) {
        setLlmAnswer((current) => current + event.partial);
        loopResponseRef.current += event.partial;
      }
      if (event.thinking) setLlmThinking((current) => current + event.thinking);
    });
    llmDoneSubRef.current = MentraGemma.addListener('done', (_event: GemmaDoneEvent) => {
      setLlmStatus({ kind: 'ready' });
      const now = Date.now();
      if (loopActiveRef.current && loopStateRef.current.kind === 'thinking') {
        timingsRef.current.gemmaDoneAt = now;
        const silenceAt = timingsRef.current.silenceAt ?? now;
        console.log(`[Edge Loop] gemma done in ${now - silenceAt}ms · response: ${JSON.stringify(loopResponseRef.current.trim())}`);
        void fireLoopSpeak();
      } else {
        console.log('[Edge LLM] generation done');
      }
    });
    llmErrorSubRef.current = MentraGemma.addListener('error', (event: GemmaErrorEvent) => {
      console.error('[Edge LLM] stream error', event.message);
      setLlmStatus({ kind: 'error', message: event.message });
      if (loopActiveRef.current) {
        loopActiveRef.current = false;
        setLoopState({ kind: 'error', message: `gemma: ${event.message}` });
      }
    });

    // Auto-load all models on first open.
    let cancelled = false;
    void (async () => {
      const bootStart = Date.now();
      try {
        // (1) Pre-warm the audio session once so per-cycle TTS playback doesn't pay this cost.
        setBootStatus({ kind: 'loading', step: 'audio session' });
        await setAudioModeAsync({ interruptionMode: 'duckOthers', playsInSilentMode: true });

        setBootStatus({ kind: 'loading', step: 'TTS engine (vits-piper-en_US-lessac-low)' });
        const t0 = Date.now();
        await ensureTtsEngine();
        console.log(`[Edge Boot] TTS ready in ${Date.now() - t0}ms`);
        if (cancelled) return;

        setBootStatus({ kind: 'loading', step: 'STT engine (streaming-zipformer-en)' });
        const t1 = Date.now();
        await ensureSttEngine();
        console.log(`[Edge Boot] STT ready in ${Date.now() - t1}ms`);
        if (cancelled) return;

        if (gemmaModelFile.exists) {
          setBootStatus({ kind: 'loading', step: 'Gemma 3n E2B (LiteRT LM)' });
          setLlmStatus({ kind: 'loading' });
          const t2 = Date.now();
          // Try vision-enabled first; fall back to text-only if the runtime can't compile the
          // vision executor on this device (Mali GPUs without OpenCL, CPU vision unsupported, etc.).
          try {
            await MentraGemma.load({
              modelPath: gemmaModelPath,
              maxNumTokens: 4096,
              systemPrompt: GEMMA_SYSTEM_PROMPT,
              backend: 'cpu',
              enableVision: true,
              visionBackend: 'gpu',
            });
            visionEnabledRef.current = true;
            console.log('[Edge Boot] Gemma loaded WITH vision');
          } catch (visErr) {
            console.warn('[Edge Boot] vision init failed, falling back to text-only', visErr);
            await MentraGemma.load({
              modelPath: gemmaModelPath,
              maxNumTokens: 4096,
              systemPrompt: GEMMA_SYSTEM_PROMPT,
              backend: 'cpu',
              enableVision: false,
            });
            visionEnabledRef.current = false;
            console.log('[Edge Boot] Gemma loaded TEXT-ONLY (vision unavailable on this device)');
          }
          console.log(`[Edge Boot] Gemma ready in ${Date.now() - t2}ms`);
          if (cancelled) return;
          setLlmStatus({ kind: 'ready' });
          setBootStatus({ kind: 'done' });
        } else {
          setBootStatus({
            kind: 'done',
            warning: `Gemma model not found at ${gemmaModelPath}. Loop will not work until you push the .litertlm file.`,
          });
        }
        console.log(`[Edge Boot] total ${Date.now() - bootStart}ms`);
      } catch (error) {
        console.error('[Edge Boot] failed', error);
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        setBootStatus({ kind: 'done', warning: `Boot error: ${message}` });
      }
    })();

    return () => {
      cancelled = true;
      loopActiveRef.current = false;
      stopSilenceTimer();

      tearDownPlayer();
      void ttsRef.current?.destroy().catch(() => undefined);
      ttsRef.current = null;

      micSubRef.current?.remove();
      micSubRef.current = null;
      void streamRef.current?.release().catch(() => undefined);
      streamRef.current = null;
      void sttRef.current?.destroy().catch(() => undefined);
      sttRef.current = null;
      void BluetoothSdk.setMicState(false).catch(() => undefined);

      llmTokenSubRef.current?.remove();
      llmDoneSubRef.current?.remove();
      llmErrorSubRef.current?.remove();
      photoSubRef.current?.remove();
      photoSubRef.current = null;
      void MentraDirectReceiver.stopPhotoReceiver().catch(() => undefined);
      void MentraGemma.unload().catch(() => undefined);
    };
  }, []);

  // ---------- TTS ----------
  const ensureTtsEngine = async () => {
    if (ttsRef.current) return ttsRef.current;

    setTtsStatus({ kind: 'preparing', phase: 'fetching model registry', percent: null });
    const models = await refreshModelsByCategory(ModelCategory.Tts);
    console.log('[Edge TTS] registry has', models.length, 'models');
    if (!models.some((m) => m.id === TTS_MODEL_ID)) {
      const sample = models.slice(0, 5).map((m) => m.id).join(', ');
      throw new Error(`Model id ${TTS_MODEL_ID} not in registry (sample: ${sample})`);
    }

    setTtsStatus({ kind: 'preparing', phase: 'downloading model', percent: 0 });
    const ready = await ensureModelByCategory(ModelCategory.Tts, TTS_MODEL_ID, {
      onProgress: (p) => {
        console.log('[Edge TTS] progress', p);
        setTtsStatus({
          kind: 'preparing',
          phase: p.phase ?? 'preparing',
          percent: typeof p.percent === 'number' ? p.percent : null,
        });
      },
    });
    console.log('[Edge TTS] model ready at', ready.localPath);

    setTtsStatus({ kind: 'preparing', phase: 'loading engine', percent: null });
    const tts = await createTTS({
      modelPath: fileModelPath(ready.localPath),
      modelType: 'vits',
      numThreads: 2,
    });
    ttsRef.current = tts;
    return tts;
  };

  // Stop current playback but keep the player object alive for reuse across cycles.
  const stopPlayback = () => {
    if (playerRef.current) {
      try { playerRef.current.pause(); } catch { /* already torn down */ }
    }
  };

  const tearDownPlayer = () => {
    playerSubRef.current?.remove();
    playerSubRef.current = null;
    if (playerRef.current) {
      try { playerRef.current.pause(); playerRef.current.remove(); } catch { /* ignore */ }
      playerRef.current = null;
    }
  };

  const handlePlaybackFinish = () => {
    stopPlayback();
    if (loopActiveRef.current && loopStateRef.current.kind === 'speaking') {
      const playEnd = Date.now();
      const t = timingsRef.current;
      if (t.wakeAt && t.silenceAt && t.gemmaDoneAt && t.ttsReadyAt && t.playStartedAt) {
        const cycle: CycleTimings = {
          captureMs: t.silenceAt - t.wakeAt,
          gemmaMs: t.gemmaDoneAt - t.silenceAt,
          ttsGenMs: t.ttsReadyAt - t.gemmaDoneAt,
          ttsPlayMs: playEnd - t.playStartedAt,
          totalMs: playEnd - t.silenceAt,
        };
        console.log(
          `[Edge Loop] cycle complete · capture=${cycle.captureMs}ms · gemma=${cycle.gemmaMs}ms · ttsGen=${cycle.ttsGenMs}ms · ttsPlay=${cycle.ttsPlayMs}ms · total(silence→playEnd)=${cycle.totalMs}ms`,
        );
        setLastTimings(cycle);
      }
      const query = currentQueryRef.current;
      const response = loopResponseRef.current.trim();
      const photoUri = latestPhotoUriRef.current ?? undefined;
      if (query || response) {
        setTurnHistory((prev) => [{ query, response, photoUri }, ...prev].slice(0, 5));
      }
      void restartLoopListening();
    } else {
      setTtsStatus({ kind: 'ready' });
    }
  };

  const playWavUri = (uri: string) => {
    // Per-cycle fresh player. Reusing via player.replace() raced with didJustFinish for the
    // previous source, which truncated playback. Audio session is pre-warmed at boot, so the
    // remaining createAudioPlayer cost is small.
    tearDownPlayer();
    const player = createAudioPlayer({ uri }, { updateInterval: 250 });
    playerRef.current = player;
    playerSubRef.current = player.addListener('playbackStatusUpdate', (s) => {
      if (s.didJustFinish) handlePlaybackFinish();
    });
    player.play();
  };

  const onGenerate = async () => {
    const text = prompt.trim();
    if (!text) return;

    try {
      const tts = await ensureTtsEngine();
      setTtsStatus({ kind: 'generating' });
      const audio = await tts.generateSpeech(text);

      const file = new File(Paths.cache, 'edge-tts.wav');
      file.create({ intermediates: true, overwrite: true });
      await saveAudioToFile(audio, file.uri.replace(/^file:\/\//, ''));

      setTtsStatus({ kind: 'playing' });
      playWavUri(file.uri);
    } catch (error) {
      console.error('[Edge TTS] generate failed', error);
      const message = error instanceof Error ? error.message : String(error);
      setTtsStatus({ kind: 'error', message });
    }
  };

  // ---------- ASR ----------
  const ensureSttEngine = async () => {
    if (sttRef.current) return sttRef.current;

    setAsrStatus({ kind: 'preparing', phase: 'fetching model registry', percent: null });
    const models = await refreshModelsByCategory(ModelCategory.Stt);
    console.log('[Edge ASR] registry has', models.length, 'models');
    if (!models.some((m) => m.id === ASR_MODEL_ID)) {
      const sample = models.slice(0, 5).map((m) => m.id).join(', ');
      throw new Error(`Model id ${ASR_MODEL_ID} not in registry (sample: ${sample})`);
    }

    setAsrStatus({ kind: 'preparing', phase: 'downloading 349MB model', percent: 0 });
    const ready = await ensureModelByCategory(ModelCategory.Stt, ASR_MODEL_ID, {
      onProgress: (p) => {
        console.log('[Edge ASR] progress', p);
        setAsrStatus({
          kind: 'preparing',
          phase: p.phase ?? 'preparing',
          percent: typeof p.percent === 'number' ? p.percent : null,
        });
      },
    });
    console.log('[Edge ASR] model ready at', ready.localPath);

    setAsrStatus({ kind: 'preparing', phase: 'loading recognizer', percent: null });
    const stt = await createStreamingSTT({
      modelPath: fileModelPath(ready.localPath),
      modelType: ASR_MODEL_TYPE,
      numThreads: 4,
    });
    sttRef.current = stt;
    return stt;
  };

  const handleMicChunk = async (payload: MicPcmEvent) => {
    const stream = streamRef.current;
    if (!stream) return;
    // While Gemma is generating / TTS is speaking, skip ASR work. The mic stays on so the
    // glasses keep streaming audio (so it's available the moment we restart), but we drop
    // chunks to save CPU and avoid garbage-text picking up from TTS bleed. We reset() the
    // stream when transitioning back to listening anyway.
    if (loopActiveRef.current) {
      const kind = loopStateRef.current.kind;
      if (kind === 'thinking' || kind === 'speaking') return;
    }

    // Copy bytes to a fresh aligned buffer, then view as little-endian int16, then normalize to [-1, 1].
    const src = new Uint8Array(payload.pcm);
    const aligned = new Uint8Array(src.byteLength);
    aligned.set(src);
    const i16 = new Int16Array(aligned.buffer);
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;

    try {
      const { result, isEndpoint } = await stream.processAudioChunk(f32, MIC_SAMPLE_RATE);
      if (isEndpoint) {
        if (result.text) {
          const newFinal = finalTextRef.current
            ? `${finalTextRef.current} ${result.text}`
            : result.text;
          finalTextRef.current = newFinal;
          setFinalText(newFinal);
          console.log('[Edge ASR] endpoint:', result.text);
        }
        partialTextRef.current = '';
        setPartialText('');
        await stream.reset();
      } else {
        partialTextRef.current = result.text;
        setPartialText(result.text);
      }
      runLoopOnTranscriptUpdate();
    } catch (error) {
      console.error('[Edge ASR] chunk error', error);
    }
  };

  const runLoopOnTranscriptUpdate = () => {
    if (!loopActiveRef.current) return;
    const state = loopStateRef.current;
    if (state.kind !== 'listening' && state.kind !== 'capturing') return;

    const combined = `${finalTextRef.current} ${partialTextRef.current}`.replace(/\s+/g, ' ').trim();
    const partial = partialTextRef.current;
    // Anchor silence on partial (intermediate) growth only. Endpoint commits drop partial → '',
    // which happens AFTER silence has already begun; treating that as a "change" would falsely
    // reset our silence timer.
    const partialGrew = partial.length > 0 && partial !== loopLastPartialRef.current;
    loopLastPartialRef.current = partial;

    if (state.kind === 'listening') {
      const wakeEnd = findWakeWordEnd(combined);
      if (wakeEnd !== null) {
        wakeWordEndOffsetRef.current = wakeEnd;
        const captured = combined.substring(wakeEnd).trim();
        const now = Date.now();
        timingsRef.current = { wakeAt: now };
        console.log('[Edge Loop] wake word detected, capture starts:', JSON.stringify(captured));
        setLoopState({ kind: 'capturing', capturedText: captured });
        loopLastChangeAtRef.current = now;
        // Fire camera capture in parallel — by the time silence triggers, the JPEG
        // should be on the phone and ready to attach to Gemma.
        void requestPhotoOnWakeWord();
      }
      return;
    }

    // capturing
    if (wakeWordEndOffsetRef.current === null) return;
    const captured = combined.substring(wakeWordEndOffsetRef.current).trim();
    if (captured !== state.capturedText) {
      setLoopState({ kind: 'capturing', capturedText: captured });
    }
    if (partialGrew) {
      loopLastChangeAtRef.current = Date.now();
    }
  };

  const subscribeMic = () => {
    if (micSubRef.current) return; // idempotent — mic stays subscribed across loop cycles
    micSubRef.current = BluetoothSdk.addListener('mic_pcm', (payload: MicPcmEvent) => {
      // Serialize processing — concurrent processAudioChunk calls would interleave on the native stream.
      processingRef.current = processingRef.current
        .then(() => handleMicChunk(payload))
        .catch((error) => console.error('[Edge ASR] queue error', error));
    });
  };

  const resetTranscriptBuffers = () => {
    finalTextRef.current = '';
    partialTextRef.current = '';
    setFinalText('');
    setPartialText('');
    loopLastTextRef.current = '';
  };

  const startListening = async () => {
    try {
      if (!connected) throw new Error('Connect glasses first to stream the microphone.');
      const engine = await ensureSttEngine();
      const stream = await engine.createStream();
      streamRef.current = stream;
      resetTranscriptBuffers();
      subscribeMic();
      await BluetoothSdk.setMicState(true, true, false);
      setAsrStatus({ kind: 'listening' });
      console.log('[Edge ASR] listening');
    } catch (error) {
      console.error('[Edge ASR] start failed', error);
      const message = error instanceof Error ? error.message : String(error);
      setAsrStatus({ kind: 'error', message });
      micSubRef.current?.remove();
      micSubRef.current = null;
      void streamRef.current?.release().catch(() => undefined);
      streamRef.current = null;
    }
  };

  // Stop receiving audio but keep the STT stream alive for fast reuse next cycle.
  const pauseMic = async () => {
    await BluetoothSdk.setMicState(false).catch(() => undefined);
    micSubRef.current?.remove();
    micSubRef.current = null;
    await processingRef.current.catch(() => undefined);
  };

  const releaseMic = async () => {
    await pauseMic();
    if (streamRef.current) {
      await streamRef.current.release().catch(() => undefined);
      streamRef.current = null;
    }
  };

  const stopListening = async () => {
    try {
      await releaseMic();
      setAsrStatus({ kind: 'ready' });
      console.log('[Edge ASR] stopped');
    } catch (error) {
      console.error('[Edge ASR] stop failed', error);
      const message = error instanceof Error ? error.message : String(error);
      setAsrStatus({ kind: 'error', message });
    }
  };

  // ---------- PHOTO CAPTURE ----------
  const handlePhotoUpload = async (payload: DirectPhotoUploadEvent) => {
    if (activePhotoIdRef.current && payload.requestId && payload.requestId !== activePhotoIdRef.current) {
      console.log('[Edge Photo] ignoring stale upload', payload.requestId);
      return;
    }
    try {
      const res = await fetch(payload.fileUri);
      const buf = await res.arrayBuffer();
      const bytes = new Uint8Array(buf);
      latestPhotoBytesRef.current = bytes;
      latestPhotoUriRef.current = payload.fileUri;
      setPhotoStatus('ready');
      console.log(`[Edge Photo] received ${bytes.byteLength} bytes at ${payload.fileUri}`);
      photoReadyResolveRef.current?.(bytes);
      photoReadyResolveRef.current = null;
    } catch (e) {
      console.error('[Edge Photo] failed to read file', e);
      setPhotoStatus('failed');
      photoReadyResolveRef.current?.(null);
      photoReadyResolveRef.current = null;
    }
  };

  const ensurePhotoReceiver = async () => {
    if (photoReceiverRunningRef.current && photoUploadUrlRef.current) {
      return photoUploadUrlRef.current;
    }
    const r = await MentraDirectReceiver.startPhotoReceiver();
    photoReceiverRunningRef.current = true;
    photoUploadUrlRef.current = r.uploadUrl;
    console.log('[Edge Photo] receiver ready at', r.uploadUrl);
    return r.uploadUrl;
  };

  const stopPhotoReceiver = async () => {
    await MentraDirectReceiver.stopPhotoReceiver().catch(() => undefined);
    photoReceiverRunningRef.current = false;
    photoUploadUrlRef.current = null;
  };

  const requestPhotoOnWakeWord = async () => {
    const url = photoUploadUrlRef.current;
    if (!url) {
      console.warn('[Edge Photo] no upload URL — receiver not started');
      return;
    }
    const requestId = `edge-${Date.now()}`;
    activePhotoIdRef.current = requestId;
    latestPhotoBytesRef.current = null;
    latestPhotoUriRef.current = null;
    setPhotoStatus('capturing');
    photoReadyPromiseRef.current = new Promise<Uint8Array | null>((resolve) => {
      photoReadyResolveRef.current = resolve;
      setTimeout(() => {
        if (photoReadyResolveRef.current === resolve) {
          console.warn('[Edge Photo] timed out waiting for photo');
          photoReadyResolveRef.current = null;
          setPhotoStatus('failed');
          resolve(null);
        }
      }, PHOTO_TIMEOUT_MS);
    });
    try {
      console.log('[Edge Photo] requesting photo', requestId);
      await BluetoothSdk.photoRequest(
        requestId,
        PHOTO_APP_ID,
        'medium',
        url,
        null,
        'medium',
        false,
        true,
      );
    } catch (e) {
      console.error('[Edge Photo] photoRequest failed', e);
      photoReadyResolveRef.current?.(null);
      photoReadyResolveRef.current = null;
      setPhotoStatus('failed');
    }
  };

  // ---------- LOOP ----------
  const stopSilenceTimer = () => {
    if (loopSilenceTimerRef.current) {
      clearInterval(loopSilenceTimerRef.current);
      loopSilenceTimerRef.current = null;
    }
  };

  const startSilenceTimer = () => {
    stopSilenceTimer();
    loopSilenceTimerRef.current = setInterval(() => {
      if (!loopActiveRef.current) return;
      const state = loopStateRef.current;
      if (state.kind !== 'capturing') return;
      const captured = state.capturedText.trim();
      if (!captured) return;
      if (Date.now() - loopLastChangeAtRef.current < SILENCE_MS) return;
      void fireLoopGemma(captured);
    }, 400);
  };

  // Full mic startup — called once at startLoop. Mic and listener stay alive across cycles.
  const startMicForLoop = async () => {
    if (!sttRef.current) await ensureSttEngine();
    const engine = sttRef.current;
    if (!engine) throw new Error('STT engine missing.');
    if (streamRef.current) {
      await streamRef.current.reset();
    } else {
      streamRef.current = await engine.createStream();
    }
    resetTranscriptBuffers();
    wakeWordEndOffsetRef.current = null;
    loopLastChangeAtRef.current = Date.now();
    loopLastPartialRef.current = '';
    subscribeMic();
    await BluetoothSdk.setMicState(true, true, false);
    setAsrStatus({ kind: 'listening' });
    setLoopState({ kind: 'listening' });
    startSilenceTimer();
  };

  // Between cycles — mic is already running, just reset stream state and transcript buffers.
  const prepareNextListenCycle = async () => {
    // Drain any in-flight chunk callbacks so they don't write into our refs after reset.
    await processingRef.current.catch(() => undefined);
    if (streamRef.current) {
      await streamRef.current.reset();
    } else if (sttRef.current) {
      streamRef.current = await sttRef.current.createStream();
    }
    resetTranscriptBuffers();
    wakeWordEndOffsetRef.current = null;
    loopLastChangeAtRef.current = Date.now();
    loopLastPartialRef.current = '';
    latestPhotoBytesRef.current = null;
    latestPhotoUriRef.current = null;
    activePhotoIdRef.current = null;
    photoReadyPromiseRef.current = null;
    photoReadyResolveRef.current = null;
    setPhotoStatus('idle');
    setAsrStatus({ kind: 'listening' });
    setLoopState({ kind: 'listening' });
    startSilenceTimer();
  };

  const startLoop = async () => {
    try {
      if (!connected) throw new Error('Connect glasses first.');
      const gemmaLoaded = await MentraGemma.isLoaded();
      if (!gemmaLoaded) {
        throw new Error('Load Gemma first using the LLM card below.');
      }
      loopActiveRef.current = true;
      setLoopState({ kind: 'preparing', phase: 'loading STT recognizer' });
      await ensureSttEngine();
      setLoopState({ kind: 'preparing', phase: 'loading TTS engine' });
      await ensureTtsEngine();
      setLoopState({ kind: 'preparing', phase: 'starting photo receiver' });
      try {
        await ensurePhotoReceiver();
        if (!photoSubRef.current) {
          photoSubRef.current = MentraDirectReceiver.addListener('photoUpload', handlePhotoUpload);
        }
      } catch (e) {
        console.warn('[Edge Photo] failed to start receiver — vision queries disabled', e);
      }
      await startMicForLoop();
      console.log('[Edge Loop] started');
    } catch (error) {
      console.error('[Edge Loop] start failed', error);
      const message = error instanceof Error ? error.message : String(error);
      loopActiveRef.current = false;
      setLoopState({ kind: 'error', message });
    }
  };

  const stopLoop = async () => {
    console.log('[Edge Loop] stopping');
    loopActiveRef.current = false;
    stopSilenceTimer();
    await MentraGemma.cancelGeneration().catch(() => undefined);
    stopPlayback();
    await releaseMic().catch(() => undefined);
    photoSubRef.current?.remove();
    photoSubRef.current = null;
    await stopPhotoReceiver().catch(() => undefined);
    latestPhotoBytesRef.current = null;
    activePhotoIdRef.current = null;
    photoReadyPromiseRef.current = null;
    photoReadyResolveRef.current = null;
    setPhotoStatus('idle');
    setAsrStatus({ kind: 'ready' });
    setLoopState({ kind: 'off' });
  };

  const fireLoopGemma = async (captured: string) => {
    const now = Date.now();
    const wakeAt = timingsRef.current.wakeAt ?? now;
    timingsRef.current.silenceAt = now;
    console.log(`[Edge Loop] silence triggered after ${now - wakeAt}ms capture window, firing gemma:`, JSON.stringify(captured));
    currentQueryRef.current = captured;
    stopSilenceTimer();
    // Mic stays on across the cycle. handleMicChunk short-circuits while state is thinking/speaking,
    // and prepareNextListenCycle calls stream.reset() before transitioning back to listening.
    setAsrStatus({ kind: 'ready' });
    setLoopState({ kind: 'thinking', query: captured });
    loopResponseRef.current = '';
    setLlmAnswer('');
    setLlmThinking('');
    try {
      const photoBytes = visionEnabledRef.current && photoReadyPromiseRef.current
        ? await photoReadyPromiseRef.current
        : null;
      if (photoBytes) {
        console.log(`[Edge Loop] attaching photo to gemma (${photoBytes.byteLength} bytes)`);
      } else if (!visionEnabledRef.current) {
        console.log('[Edge Loop] vision disabled — text-only');
      } else {
        console.log('[Edge Loop] no photo ready, sending text-only');
      }
      await MentraGemma.generateStream(captured, false, photoBytes);
    } catch (error) {
      console.error('[Edge Loop] generateStream failed', error);
      const message = error instanceof Error ? error.message : String(error);
      loopActiveRef.current = false;
      setLoopState({ kind: 'error', message });
    }
  };

  const fireLoopSpeak = async () => {
    const text = loopResponseRef.current.trim();
    if (!text) {
      console.warn('[Edge Loop] empty gemma response, restarting cycle');
      if (loopActiveRef.current) void restartLoopListening();
      return;
    }
    setLoopState({ kind: 'speaking', response: text });
    try {
      const ttsStart = Date.now();
      const tts = await ensureTtsEngine();
      const audio = await tts.generateSpeech(text);
      // Prepend ~350ms of silence so the audio HAL spin-up doesn't eat the first phoneme.
      const padSamples = Math.floor((audio.sampleRate * 350) / 1000);
      const paddedSamples: number[] = new Array(padSamples + audio.samples.length);
      for (let i = 0; i < padSamples; i++) paddedSamples[i] = 0;
      for (let i = 0; i < audio.samples.length; i++) paddedSamples[padSamples + i] = audio.samples[i]!;
      const paddedAudio = { samples: paddedSamples, sampleRate: audio.sampleRate };
      const file = new File(Paths.cache, 'edge-loop-tts.wav');
      file.create({ intermediates: true, overwrite: true });
      await saveAudioToFile(paddedAudio, file.uri.replace(/^file:\/\//, ''));
      timingsRef.current.ttsReadyAt = Date.now();
      console.log(`[Edge Loop] TTS generated in ${timingsRef.current.ttsReadyAt - ttsStart}ms (${audio.samples.length} samples @ ${audio.sampleRate}Hz)`);

      timingsRef.current.playStartedAt = Date.now();
      playWavUri(file.uri);
    } catch (error) {
      console.error('[Edge Loop] speak failed', error);
      const message = error instanceof Error ? error.message : String(error);
      loopActiveRef.current = false;
      setLoopState({ kind: 'error', message });
    }
  };

  const restartLoopListening = async () => {
    if (!loopActiveRef.current) return;
    try {
      console.log('[Edge Loop] restarting listening (mic stays on)');
      await prepareNextListenCycle();
    } catch (error) {
      console.error('[Edge Loop] restart failed', error);
      const message = error instanceof Error ? error.message : String(error);
      loopActiveRef.current = false;
      setLoopState({ kind: 'error', message });
    }
  };

  // ---------- LLM (Gemma 4) ----------
  const loadGemma = async () => {
    setLlmStatus({ kind: 'loading' });
    setLlmAnswer('');
    setLlmThinking('');
    try {
      if (!gemmaModelFile.exists) {
        throw new Error(
          `Model file not found at ${gemmaModelPath}. ` +
            `Push the ${GEMMA_MODEL_FILENAME} file there with: ` +
            `adb push ${GEMMA_MODEL_FILENAME} /data/local/tmp/ && ` +
            `adb shell run-as com.mentra.bluetoothsdk.example cp /data/local/tmp/${GEMMA_MODEL_FILENAME} files/`,
        );
      }
      console.log('[Edge LLM] loading model from', gemmaModelPath);
      await MentraGemma.load({
        modelPath: gemmaModelPath,
        maxNumTokens: 4096,
        systemPrompt: GEMMA_SYSTEM_PROMPT,
      });
      console.log('[Edge LLM] model loaded');
      setLlmStatus({ kind: 'ready' });
    } catch (error) {
      console.error('[Edge LLM] load failed', error);
      const message = error instanceof Error ? error.message : String(error);
      setLlmStatus({ kind: 'error', message });
    }
  };

  const askGemma = async () => {
    const text = llmPrompt.trim();
    if (!text) return;
    setLlmAnswer('');
    setLlmThinking('');
    setLlmStatus({ kind: 'generating' });
    try {
      await MentraGemma.generateStream(text, enableThinking);
      console.log('[Edge LLM] stream started');
    } catch (error) {
      console.error('[Edge LLM] generateStream failed', error);
      const message = error instanceof Error ? error.message : String(error);
      setLlmStatus({ kind: 'error', message });
    }
  };

  const resetGemmaConversation = async () => {
    try {
      await MentraGemma.resetConversation(GEMMA_SYSTEM_PROMPT);
      setLlmAnswer('');
      setLlmThinking('');
      console.log('[Edge LLM] conversation reset');
    } catch (error) {
      console.error('[Edge LLM] reset failed', error);
    }
  };

  const cancelGemma = async () => {
    try {
      await MentraGemma.cancelGeneration();
      console.log('[Edge LLM] generation cancelled');
    } catch (error) {
      console.error('[Edge LLM] cancel failed', error);
    }
  };

  // ---------- UI helpers ----------
  const ttsBusy =
    ttsStatus.kind === 'preparing' ||
    ttsStatus.kind === 'generating' ||
    ttsStatus.kind === 'playing';
  const canGenerate = !ttsBusy && prompt.trim().length > 0;
  const ttsButtonLabel = (() => {
    switch (ttsStatus.kind) {
      case 'preparing':
        return ttsStatus.percent !== null
          ? `${ttsStatus.phase} ${Math.round(ttsStatus.percent)}%`
          : ttsStatus.phase;
      case 'generating':
        return 'Generating...';
      case 'playing':
        return 'Playing...';
      default:
        return 'Generate';
    }
  })();

  const asrBusy = asrStatus.kind === 'preparing';
  const asrListening = asrStatus.kind === 'listening';
  const asrButtonLabel = (() => {
    switch (asrStatus.kind) {
      case 'preparing':
        return asrStatus.percent !== null
          ? `${asrStatus.phase} ${Math.round(asrStatus.percent)}%`
          : asrStatus.phase;
      case 'listening':
        return 'Stop listening';
      default:
        return connected ? 'Start listening' : 'Connect glasses to listen';
    }
  })();
  const canToggleAsr = (asrListening || (!asrBusy && connected));

  const llmIdle = llmStatus.kind === 'unloaded';
  const llmReady = llmStatus.kind === 'ready';
  const llmBusy = llmStatus.kind === 'loading' || llmStatus.kind === 'generating';
  const canAskGemma = llmReady && llmPrompt.trim().length > 0;
  const llmButtonLabel = (() => {
    switch (llmStatus.kind) {
      case 'unloaded':
      case 'error':
        return 'Load Gemma';
      case 'loading':
        return 'Loading model (5-30s)...';
      case 'generating':
        return 'Generating...';
      case 'ready':
        return canAskGemma ? 'Ask Gemma' : 'Type a prompt';
    }
  })();

  const loopActive = loopState.kind !== 'off' && loopState.kind !== 'error';
  const booting = bootStatus.kind === 'loading';
  const dimLowerCards = loopActive || booting;
  const startDisabled = booting || (!connected && !loopActive);

  // Status orb color reflects current loop phase.
  const orbColor = (() => {
    switch (loopState.kind) {
      case 'off': return '#3A413E';
      case 'preparing': return '#FFC857';
      case 'listening': return ACCENT;
      case 'capturing': return '#FFC857';
      case 'thinking': return '#5BB8FF';
      case 'speaking': return ACCENT_BRIGHT;
      case 'error': return ALARM;
    }
  })();

  const heroHeadline = booting
    ? bootStatus.step
    : loopHeadline(loopState);

  return (
    <ScrollView
      keyboardDismissMode="interactive"
      keyboardShouldPersistTaps="handled"
      style={s.screen}
      contentContainerStyle={{ paddingBottom: scrollBottomPadding + 32, paddingTop: 8 }}>

      {/* Header */}
      <View style={s.header}>
        <View style={s.headerLeft}>
          <View style={s.logoChip}>
            <Svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
              <Path d="M12 2.5 13.6 8 19 9.5 13.6 11 12 16.5 10.4 11 5 9.5 10.4 8 12 2.5z" />
              <Path d="M18.5 14.5 19.2 16.8 21.5 17.5 19.2 18.2 18.5 20.5 17.8 18.2 15.5 17.5 17.8 16.8 18.5 14.5z" />
            </Svg>
          </View>
          <View>
            <Text style={s.brandEyebrow}>MENTRA · ON-DEVICE</Text>
            <Text style={s.brandTitle}>Edge AI Glasses</Text>
          </View>
        </View>
        <View style={s.connPill}>
          <View style={[s.connPillDot, connected && s.connPillDotOn]} />
          <Text style={s.connPillText}>{connected ? 'LIVE' : 'OFFLINE'}</Text>
        </View>
      </View>

      {/* HERO */}
      <View style={s.hero}>
        <View style={s.heroOrnament} />
        <View style={s.heroHeader}>
          <Text style={s.heroEyebrow}>{booting ? 'BOOTING' : 'VOICE LOOP'}</Text>
          <View style={s.statusOrbWrap}>
            <View style={[s.statusOrb, { backgroundColor: orbColor }]} />
          </View>
        </View>
        <Text style={s.heroHeadline} numberOfLines={2}>
          {heroHeadline}
        </Text>

        {loopState.kind === 'capturing' && loopState.capturedText ? (
          <View style={s.captureBox}>
            <View style={s.captureHeader}>
              <Text style={s.captureLabel}>HEARING</Text>
              {photoStatus !== 'idle' ? (
                <Text style={[
                  s.photoChip,
                  photoStatus === 'ready' && s.photoChipReady,
                  photoStatus === 'failed' && s.photoChipFailed,
                ]}>
                  {photoStatus === 'capturing' ? '◌ Photo' : photoStatus === 'ready' ? '◉ Photo ready' : '✕ Photo failed'}
                </Text>
              ) : null}
            </View>
            <Text style={s.captureText}>{loopState.capturedText}</Text>
          </View>
        ) : null}

        {lastTimings ? (
          <View style={s.timingStrip}>
            <Stat label="cap" value={lastTimings.captureMs} />
            <View style={s.statDivider} />
            <Stat label="gem" value={lastTimings.gemmaMs} />
            <View style={s.statDivider} />
            <Stat label="tts" value={lastTimings.ttsGenMs} />
            <View style={s.statDivider} />
            <Stat label="play" value={lastTimings.ttsPlayMs} />
            <View style={s.statDivider} />
            <Stat label="total" value={lastTimings.totalMs} highlight />
          </View>
        ) : null}

        <Pressable disabled={startDisabled} onPress={loopActive ? stopLoop : startLoop}>
          <LinearGradient
            colors={
              loopActive
                ? [ALARM, '#C13030']
                : startDisabled
                  ? ['#262E2C', '#1E2624']
                  : [ACCENT, '#2FA565']
            }
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
            style={s.heroBtn}>
            <View style={[s.heroBtnDot, { backgroundColor: '#0A1410' }]} />
            <Text style={[s.heroBtnText, startDisabled && !loopActive && s.heroBtnTextDisabled]}>
              {booting
                ? 'Loading Models'
                : !connected && !loopActive
                  ? 'Connect Glasses'
                  : loopActive
                    ? 'Stop Loop'
                    : 'Start Loop'}
            </Text>
          </LinearGradient>
        </Pressable>

        {bootStatus.kind === 'done' && bootStatus.warning ? (
          <Text style={s.heroFootnote}>{bootStatus.warning}</Text>
        ) : null}
      </View>

      {/* HISTORY */}
      {turnHistory.length > 0 ? (
        <View style={s.section}>
          <View style={s.sectionHeader}>
            <Text style={s.sectionLabel}>EXCHANGES</Text>
            <Text style={s.sectionCount}>{turnHistory.length.toString().padStart(2, '0')}</Text>
          </View>
          <View style={{ gap: 8 }}>
            {turnHistory.map((turn, idx) => (
              <View key={idx} style={[s.exchange, idx === 0 && s.exchangeLatest]}>
                <View style={s.exchangeBody}>
                  {turn.photoUri ? (
                    <Image source={{ uri: turn.photoUri }} style={s.exchangeThumb} resizeMode="cover" />
                  ) : null}
                  <View style={s.exchangeTextStack}>
                    <View style={s.exchangeLine}>
                      <Text style={s.speakerYou}>YOU</Text>
                      <Text style={s.exchangeText}>{turn.query || '—'}</Text>
                    </View>
                    <View style={s.exchangeDivider} />
                    <View style={s.exchangeLine}>
                      <Text style={s.speakerEdge}>EDGE</Text>
                      <Text style={[s.exchangeText, s.exchangeTextAi]}>{turn.response || '—'}</Text>
                    </View>
                  </View>
                </View>
              </View>
            ))}
          </View>
        </View>
      ) : null}

      {/* DIVIDER */}
      <View style={s.divider}>
        <View style={s.dividerLine} />
        <Text style={s.dividerText}>MANUAL CONTROLS</Text>
        <View style={s.dividerLine} />
      </View>

      <View style={dimLowerCards ? s.dimmed : null} pointerEvents={dimLowerCards ? 'none' : 'auto'}>

        {/* TTS */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardEyebrow}>TEXT → SPEECH</Text>
            <Text style={s.cardModel}>piper · lessac low</Text>
          </View>
          <View style={s.scrollInputWrap}>
            <ScrollView
              style={s.scrollInputScroll}
              contentContainerStyle={s.scrollInputContent}
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}>
              <TextInput
                value={prompt}
                onChangeText={setPrompt}
                placeholder="Type something to speak..."
                placeholderTextColor={MUTED_DIM}
                multiline
                style={s.scrollInputText}
                editable={!ttsBusy}
              />
            </ScrollView>
          </View>
          <Pressable disabled={!canGenerate} onPress={onGenerate}>
            <View style={[s.cardBtn, !canGenerate && s.cardBtnDisabled]}>
              <Text style={s.cardBtnText}>{ttsButtonLabel}</Text>
            </View>
          </Pressable>
          <Text style={s.cardStatus}>{ttsStatusText(ttsStatus)}</Text>
        </View>

        {/* ASR */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardEyebrow}>SPEECH → TEXT</Text>
            <Text style={s.cardModel}>zipformer · glasses mic</Text>
          </View>
          <View style={s.transcriptWrap}>
            <ScrollView
              style={s.transcriptScroll}
              contentContainerStyle={s.transcriptContent}
              showsVerticalScrollIndicator={false}>
              {finalText ? <Text style={s.transcriptFinal}>{finalText}</Text> : null}
              {partialText ? <Text style={s.transcriptPartial}>{partialText}</Text> : null}
              {!finalText && !partialText ? (
                <Text style={s.transcriptPlaceholder}>
                  {asrListening ? '◉ Listening to glasses mic' : 'Press Start to transcribe'}
                </Text>
              ) : null}
            </ScrollView>
          </View>
          <Pressable disabled={!canToggleAsr} onPress={asrListening ? stopListening : startListening}>
            <View style={[
              s.cardBtn,
              asrListening && s.cardBtnAlarm,
              !canToggleAsr && !asrListening && s.cardBtnDisabled,
            ]}>
              <Text style={s.cardBtnText}>{asrButtonLabel}</Text>
            </View>
          </Pressable>
          <Text style={s.cardStatus}>{asrStatusText(asrStatus)}</Text>
        </View>

        {/* LLM */}
        <View style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardEyebrow}>ON-DEVICE LLM</Text>
            <Text style={s.cardModel}>gemma 3n · e2b · litert</Text>
          </View>
          {llmReady || llmStatus.kind === 'generating' ? (
            <>
              <View style={s.scrollInputWrap}>
                <ScrollView
                  style={s.scrollInputScroll}
                  contentContainerStyle={s.scrollInputContent}
                  keyboardShouldPersistTaps="handled"
                  showsVerticalScrollIndicator={false}>
                  <TextInput
                    value={llmPrompt}
                    onChangeText={setLlmPrompt}
                    placeholder="Ask Gemma anything..."
                    placeholderTextColor={MUTED_DIM}
                    multiline
                    style={s.scrollInputText}
                    editable={llmStatus.kind === 'ready'}
                  />
                </ScrollView>
              </View>
              <View style={s.chipsRow}>
                <Pressable
                  onPress={() => setEnableThinking((v) => !v)}
                  style={[s.toggleChip, enableThinking && s.toggleChipOn]}>
                  <Text style={[s.toggleChipText, enableThinking && s.toggleChipTextOn]}>
                    Thinking · {enableThinking ? 'On' : 'Off'}
                  </Text>
                </Pressable>
                <Pressable onPress={resetGemmaConversation} style={s.toggleChip}>
                  <Text style={s.toggleChipText}>Reset Chat</Text>
                </Pressable>
              </View>
              {llmThinking ? (
                <View style={s.thinkingBox}>
                  <Text style={s.thinkingLabel}>REASONING</Text>
                  <Text style={s.thinkingText}>{llmThinking}</Text>
                </View>
              ) : null}
              <View style={s.transcriptWrap}>
                <ScrollView
                  style={s.transcriptScroll}
                  contentContainerStyle={s.transcriptContent}
                  showsVerticalScrollIndicator={false}>
                  {llmAnswer ? (
                    <Text style={s.transcriptFinal}>{llmAnswer}</Text>
                  ) : (
                    <Text style={s.transcriptPlaceholder}>
                      {llmStatus.kind === 'generating' ? '◌ Generating...' : 'Tap Ask Gemma'}
                    </Text>
                  )}
                </ScrollView>
              </View>
            </>
          ) : (
            <View style={s.modelPathBox}>
              <Text style={s.modelPathLabel}>EXPECTED PATH</Text>
              <Text style={s.modelPathMono}>{gemmaModelPath}</Text>
            </View>
          )}
          <Pressable
            disabled={llmBusy || (llmReady && !canAskGemma)}
            onPress={llmIdle || llmStatus.kind === 'error' ? loadGemma : askGemma}>
            <View style={[s.cardBtn, (llmBusy || (llmReady && !canAskGemma)) && s.cardBtnDisabled]}>
              <Text style={s.cardBtnText}>{llmButtonLabel}</Text>
            </View>
          </Pressable>
          {llmStatus.kind === 'generating' ? (
            <Pressable onPress={cancelGemma}>
              <View style={[s.cardBtn, s.cardBtnAlarm]}>
                <Text style={s.cardBtnText}>Cancel</Text>
              </View>
            </Pressable>
          ) : null}
          <Text style={s.cardStatus}>{llmStatusText(llmStatus)}</Text>
        </View>
      </View>
    </ScrollView>
  );
}

function Stat({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <View style={s.stat}>
      <Text style={s.statLabel}>{label}</Text>
      <View style={s.statRow}>
        <Text style={[s.statValue, highlight && s.statValueHi]}>{value}</Text>
        <Text style={s.statUnit}>ms</Text>
      </View>
    </View>
  );
}

function loopHeadline(state: LoopState): string {
  switch (state.kind) {
    case 'off': return 'Idle. Tap Start.';
    case 'preparing': return state.phase;
    case 'listening': return 'Listening for wake word';
    case 'capturing': return state.capturedText ? 'Hearing you...' : 'Wake detected — go';
    case 'thinking': return 'Thinking';
    case 'speaking': return 'Speaking';
    case 'error': return state.message;
  }
}

function ttsStatusText(status: TtsStatus): string {
  switch (status.kind) {
    case 'idle': return 'Engine cached on-device';
    case 'preparing':
      return status.percent !== null
        ? `${status.phase} · ${Math.round(status.percent)}%`
        : status.phase;
    case 'ready': return 'Ready';
    case 'generating': return 'Synthesizing on device...';
    case 'playing': return 'Playing audio';
    case 'error': return `Error · ${status.message}`;
  }
}

function llmStatusText(status: LlmStatus): string {
  switch (status.kind) {
    case 'unloaded': return 'Push .litertlm to expected path, then load.';
    case 'loading': return 'Loading weights into memory (5-30s)...';
    case 'ready': return 'Multi-turn memory active. Tap Reset to clear.';
    case 'generating': return 'Streaming tokens from on-device runtime';
    case 'error': return `Error · ${status.message}`;
  }
}

function asrStatusText(status: AsrStatus): string {
  switch (status.kind) {
    case 'idle': return 'Streaming Zipformer · runs on device';
    case 'preparing':
      return status.percent !== null
        ? `${status.phase} · ${Math.round(status.percent)}%`
        : status.phase;
    case 'ready': return 'Ready';
    case 'listening': return 'Streaming glasses mic @ 16kHz PCM';
    case 'error': return `Error · ${status.message}`;
  }
}

// === EDGE PAGE THEME ===
const ACCENT = '#7DD89E';
const ACCENT_BRIGHT = '#3DFFB0';
const ALARM = '#FF5C5C';
const BG = '#0A0E0C';
const SURFACE = '#13191A';
const SURFACE_LIFT = '#0F1414';
const HAIRLINE = 'rgba(255,255,255,0.07)';
const HAIRLINE_STRONG = 'rgba(255,255,255,0.14)';
const INK = '#F0F4F1';
const MUTED = '#8B9491';
const MUTED_DIM = '#5A6360';
const MONO = 'Courier';

const s = StyleSheet.create({
  screen: { flex: 1, backgroundColor: BG },

  /* HEADER */
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, paddingTop: 8, paddingBottom: 18,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  logoChip: {
    width: 38, height: 38, borderRadius: 12,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(125,216,158,0.08)',
    borderWidth: 1, borderColor: 'rgba(125,216,158,0.22)',
  },
  brandEyebrow: { fontSize: 9, fontWeight: '700', letterSpacing: 1.4, color: MUTED, fontFamily: MONO },
  brandTitle: { fontSize: 22, fontWeight: '800', color: INK, letterSpacing: -0.6, marginTop: -1 },
  connPill: {
    flexDirection: 'row', alignItems: 'center', gap: 7,
    paddingVertical: 6, paddingHorizontal: 11,
    borderRadius: 999, borderWidth: 1, borderColor: HAIRLINE,
  },
  connPillDot: { width: 6, height: 6, borderRadius: 999, backgroundColor: MUTED_DIM },
  connPillDotOn: { backgroundColor: ACCENT_BRIGHT },
  connPillText: { fontSize: 9, fontWeight: '700', color: INK, letterSpacing: 1.4, fontFamily: MONO },

  /* HERO */
  hero: {
    marginHorizontal: 16, padding: 22,
    borderRadius: 24, backgroundColor: SURFACE,
    borderWidth: 1, borderColor: HAIRLINE,
    gap: 16, overflow: 'hidden',
  },
  heroOrnament: {
    position: 'absolute', top: -80, right: -80, width: 220, height: 220,
    borderRadius: 999, borderWidth: 1, borderColor: 'rgba(125,216,158,0.06)',
  },
  heroHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  heroEyebrow: { fontSize: 10, fontWeight: '700', letterSpacing: 1.8, color: MUTED, fontFamily: MONO },
  statusOrbWrap: { width: 18, height: 18, alignItems: 'center', justifyContent: 'center' },
  statusOrb: { width: 10, height: 10, borderRadius: 999 },
  statusOrbHalo: { position: 'absolute', width: 14, height: 14, borderRadius: 999 },
  heroHeadline: { fontSize: 22, fontWeight: '700', color: INK, letterSpacing: -0.4, lineHeight: 28 },

  captureBox: {
    padding: 12, borderRadius: 14,
    backgroundColor: 'rgba(255,200,87,0.06)',
    borderWidth: 1, borderColor: 'rgba(255,200,87,0.18)',
    gap: 6,
  },
  captureHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  captureLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.6, color: '#FFC857', fontFamily: MONO },
  photoChip: { fontSize: 9, fontWeight: '700', letterSpacing: 1.2, color: MUTED, fontFamily: MONO },
  photoChipReady: { color: ACCENT_BRIGHT },
  photoChipFailed: { color: ALARM },
  captureText: { fontSize: 15, color: INK, lineHeight: 21, fontStyle: 'italic' },

  timingStrip: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingVertical: 12, paddingHorizontal: 14,
    backgroundColor: SURFACE_LIFT, borderRadius: 14,
    borderWidth: 1, borderColor: HAIRLINE,
  },
  stat: { gap: 2 },
  statLabel: { fontSize: 8, fontWeight: '700', letterSpacing: 1.4, color: MUTED, fontFamily: MONO },
  statRow: { flexDirection: 'row', alignItems: 'baseline', gap: 2 },
  statValue: { fontSize: 16, fontWeight: '700', color: INK, fontFamily: MONO },
  statValueHi: { color: ACCENT_BRIGHT },
  statUnit: { fontSize: 9, fontWeight: '700', color: MUTED, fontFamily: MONO },
  statDivider: { width: 1, height: 26, backgroundColor: HAIRLINE_STRONG },

  heroBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 10,
    paddingVertical: 18, borderRadius: 14,
  },
  heroBtnDot: { width: 6, height: 6, borderRadius: 999 },
  heroBtnText: { fontSize: 12, fontWeight: '800', color: '#0A1410', letterSpacing: 2, fontFamily: MONO },
  heroBtnTextDisabled: { color: MUTED },
  heroFootnote: { fontSize: 10, color: MUTED, lineHeight: 16, fontFamily: MONO, letterSpacing: 0.4 },

  /* SECTION */
  section: { marginHorizontal: 16, marginTop: 24, gap: 10 },
  sectionHeader: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between', paddingHorizontal: 4 },
  sectionLabel: { fontSize: 10, fontWeight: '700', letterSpacing: 1.8, color: MUTED, fontFamily: MONO },
  sectionCount: { fontSize: 10, fontWeight: '700', color: ACCENT, fontFamily: MONO },

  exchange: {
    padding: 14, borderRadius: 16,
    backgroundColor: SURFACE,
    borderWidth: 1, borderColor: HAIRLINE,
  },
  exchangeLatest: { borderColor: 'rgba(125,216,158,0.4)' },
  exchangeBody: { flexDirection: 'row', alignItems: 'stretch', gap: 12 },
  exchangeThumb: {
    width: 72, height: 72, borderRadius: 12,
    backgroundColor: SURFACE_LIFT,
    borderWidth: 1, borderColor: HAIRLINE_STRONG,
  },
  exchangeTextStack: { flex: 1, gap: 8, justifyContent: 'center' },
  exchangeLine: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  speakerYou: { width: 40, fontSize: 9, fontWeight: '700', letterSpacing: 1.4, color: MUTED, fontFamily: MONO, paddingTop: 3 },
  speakerEdge: { width: 40, fontSize: 9, fontWeight: '700', letterSpacing: 1.4, color: ACCENT, fontFamily: MONO, paddingTop: 3 },
  exchangeText: { flex: 1, fontSize: 14, color: INK, lineHeight: 20 },
  exchangeTextAi: { color: INK, fontWeight: '500' },
  exchangeDivider: { height: 1, backgroundColor: HAIRLINE },

  /* DIVIDER */
  divider: { flexDirection: 'row', alignItems: 'center', gap: 12, marginHorizontal: 16, marginTop: 28, marginBottom: 4 },
  dividerLine: { flex: 1, height: 1, backgroundColor: HAIRLINE },
  dividerText: { fontSize: 9, fontWeight: '700', letterSpacing: 2.4, color: MUTED, fontFamily: MONO },

  /* CARD */
  dimmed: { opacity: 0.35 },
  card: {
    marginHorizontal: 16, marginTop: 10,
    padding: 16, borderRadius: 18,
    backgroundColor: SURFACE, borderWidth: 1, borderColor: HAIRLINE,
    gap: 12,
  },
  cardHeader: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between' },
  cardEyebrow: { fontSize: 10, fontWeight: '700', letterSpacing: 1.6, color: ACCENT, fontFamily: MONO },
  cardModel: { fontSize: 9, fontWeight: '600', color: MUTED_DIM, fontFamily: MONO, letterSpacing: 0.4 },

  scrollInputWrap: {
    height: 100, borderRadius: 12,
    borderWidth: 1, borderColor: HAIRLINE,
    backgroundColor: SURFACE_LIFT,
  },
  scrollInputScroll: { flex: 1 },
  scrollInputContent: { padding: 12 },
  scrollInputText: { fontSize: 14, color: INK, lineHeight: 19, textAlignVertical: 'top', padding: 0, minHeight: 76 },

  transcriptWrap: {
    height: 120, borderRadius: 12,
    borderWidth: 1, borderColor: HAIRLINE,
    backgroundColor: SURFACE_LIFT,
  },
  transcriptScroll: { flex: 1 },
  transcriptContent: { padding: 12, gap: 4 },
  transcriptFinal: { fontSize: 14, color: INK, lineHeight: 19 },
  transcriptPartial: { fontSize: 14, color: MUTED, lineHeight: 19, fontStyle: 'italic' },
  transcriptPlaceholder: { fontSize: 11, color: MUTED_DIM, fontFamily: MONO, letterSpacing: 0.4 },

  cardBtn: {
    paddingVertical: 13, alignItems: 'center', justifyContent: 'center',
    borderRadius: 12,
    backgroundColor: SURFACE_LIFT,
    borderWidth: 1, borderColor: HAIRLINE_STRONG,
  },
  cardBtnAlarm: { backgroundColor: '#2A1414', borderColor: 'rgba(255,92,92,0.4)' },
  cardBtnDisabled: { opacity: 0.4 },
  cardBtnText: { fontSize: 11, fontWeight: '800', color: INK, letterSpacing: 1.8, fontFamily: MONO },
  cardStatus: { fontSize: 10, color: MUTED, lineHeight: 16, fontFamily: MONO, letterSpacing: 0.4 },

  chipsRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
  toggleChip: {
    paddingVertical: 7, paddingHorizontal: 11,
    borderRadius: 999, borderWidth: 1, borderColor: HAIRLINE_STRONG,
    backgroundColor: SURFACE_LIFT,
  },
  toggleChipOn: { backgroundColor: 'rgba(125,216,158,0.12)', borderColor: 'rgba(125,216,158,0.45)' },
  toggleChipText: { fontSize: 10, fontWeight: '700', color: INK, letterSpacing: 1.2, fontFamily: MONO },
  toggleChipTextOn: { color: ACCENT_BRIGHT },

  thinkingBox: {
    padding: 12, borderRadius: 12,
    backgroundColor: 'rgba(91,184,255,0.05)',
    borderWidth: 1, borderColor: 'rgba(91,184,255,0.22)',
    gap: 6,
  },
  thinkingLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.6, color: '#5BB8FF', fontFamily: MONO },
  thinkingText: { fontSize: 12, color: MUTED, lineHeight: 17, fontStyle: 'italic' },

  modelPathBox: {
    padding: 12, borderRadius: 12,
    backgroundColor: SURFACE_LIFT,
    borderWidth: 1, borderColor: HAIRLINE,
    gap: 6,
  },
  modelPathLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.6, color: MUTED, fontFamily: MONO },
  modelPathMono: { fontSize: 10, color: INK, fontFamily: MONO, lineHeight: 15 },
});
