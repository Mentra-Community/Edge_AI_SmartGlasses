import { NativeModule, requireNativeModule } from 'expo';

import type { GemmaLoadOptions, MentraGemmaModuleEvents } from './MentraGemma.types';

declare class MentraGemmaModule extends NativeModule<MentraGemmaModuleEvents> {
  isSupported(): Promise<boolean>;
  isLoaded(): Promise<boolean>;
  load(options: GemmaLoadOptions): Promise<void>;
  resetConversation(systemPrompt?: string | null): Promise<void>;
  unload(): Promise<void>;
  cancelGeneration(): Promise<void>;
  /** Streaming generation. Emits 'token' events with {partial, thinking}, 'done' at end, 'error' on failure.
   *  If `image` is a non-empty Uint8Array (raw PNG/JPEG bytes), it's prepended to the prompt for
   *  vision-enabled models. Model must have been loaded with `enableVision: true`. */
  generateStream(prompt: string, enableThinking: boolean, image?: Uint8Array | null): Promise<void>;
}

export default requireNativeModule<MentraGemmaModule>('MentraGemma');
