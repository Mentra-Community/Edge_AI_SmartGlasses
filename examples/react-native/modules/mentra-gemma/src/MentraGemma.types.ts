export type GemmaLoadOptions = {
  /** Absolute path to the .litertlm model file on the device filesystem. */
  modelPath: string;
  /** Maximum total tokens (prompt + response). Default 4096, max 32768. */
  maxNumTokens?: number;
  /** Top-K sampling cutoff. Default 64. */
  topK?: number;
  /** Top-P nucleus sampling. Default 0.95. */
  topP?: number;
  /** Sampling temperature. Default 1.0. */
  temperature?: number;
  /** Optional system prompt set on conversation creation. */
  systemPrompt?: string;
  /** Enable vision (image input). Default false. */
  enableVision?: boolean;
  /** Backend for the vision encoder when enableVision=true. 'gpu' is faster but requires
   *  OpenCL drivers; 'cpu' works on devices without OpenCL. Default 'gpu'. */
  visionBackend?: 'gpu' | 'cpu';
  /** Enable audio input. Forces audioBackend = CPU. Default false. */
  enableAudio?: boolean;
  /**
   * Inference backend.
   * - 'cpu' (safe, default for low-RAM phones)
   * - 'gpu' (2-5x faster but allocates ~3GB VRAM; SIGSEGV on phones with <12GB RAM)
   * - 'auto' (default): GPU if device has >=12GB RAM, else CPU.
   */
  backend?: 'cpu' | 'gpu' | 'auto';
};

export type GemmaTokenEvent = {
  /** Most recent chunk emitted by the LiteRT runtime (not the accumulated response). */
  partial: string;
  /** Thinking/reasoning chunk for Gemma 4 with enable_thinking. Empty string if not present. */
  thinking: string;
};

export type GemmaDoneEvent = {
  ok: boolean;
};

export type GemmaErrorEvent = {
  message: string;
};

export type MentraGemmaModuleEvents = {
  token: (event: GemmaTokenEvent) => void;
  done: (event: GemmaDoneEvent) => void;
  error: (event: GemmaErrorEvent) => void;
};
