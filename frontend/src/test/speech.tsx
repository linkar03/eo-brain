// Test helpers: a scriptable stand-in for the browser's SpeechRecognition,
// and a minimal renderHook on react-dom (no testing library needed).

import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { vi } from 'vitest';
import type {
  SpeechRecognitionErrorCode,
  SpeechRecognitionLike,
  SpeechRecognitionResultLike,
} from '../types/speech-recognition';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** One recogniser per session, as the hook creates them; the test fires its events. */
export class MockRecognition extends EventTarget implements SpeechRecognitionLike {
  static instances: MockRecognition[] = [];

  static get last(): MockRecognition {
    const rec = MockRecognition.instances[MockRecognition.instances.length - 1];
    if (!rec) throw new Error('no recogniser was created');
    return rec;
  }

  lang = '';
  continuous = true;
  interimResults = false;
  maxAlternatives = 0;
  onstart: SpeechRecognitionLike['onstart'] = null;
  onresult: SpeechRecognitionLike['onresult'] = null;
  onerror: SpeechRecognitionLike['onerror'] = null;
  onend: SpeechRecognitionLike['onend'] = null;
  start = vi.fn();
  stop = vi.fn();
  abort = vi.fn();

  constructor() {
    super();
    MockRecognition.instances.push(this);
  }

  get detached(): boolean {
    return !this.onstart && !this.onresult && !this.onerror && !this.onend;
  }

  /** A result event; `results` is the whole session so far, each [transcript, isFinal]. */
  emitResult(...results: Array<[string, boolean]>) {
    const list: SpeechRecognitionResultLike[] = results.map(([transcript, isFinal]) =>
      Object.assign([{ transcript, confidence: 0.9 }], { isFinal }),
    );
    const event = Object.assign(new Event('result'), { resultIndex: 0, results: list });
    act(() => { this.onresult?.(event); });
  }

  /** The service started listening (after the permission prompt, on first use). */
  emitStart() {
    act(() => { this.onstart?.(new Event('start')); });
  }

  emitError(error: SpeechRecognitionErrorCode | string) {
    const event = Object.assign(new Event('error'), {
      error: error as SpeechRecognitionErrorCode,
      message: '',
    });
    act(() => { this.onerror?.(event); });
  }

  emitEnd() {
    act(() => { this.onend?.(new Event('end')); });
  }
}

/** Expose MockRecognition under the given global name (as Chrome: webkit-prefixed). */
export function installSpeechRecognition(
  name: 'SpeechRecognition' | 'webkitSpeechRecognition' = 'webkitSpeechRecognition',
) {
  MockRecognition.instances = [];
  vi.stubGlobal(name, MockRecognition);
}

/** Render a hook in a throwaway root; `rerender` passes new props. */
export function renderHook<P, R>(hook: (props: P) => R, initialProps: P) {
  let props = initialProps;
  const result = { current: undefined as R };
  function Probe() {
    result.current = hook(props);
    return null;
  }
  const root = createRoot(document.createElement('div'));
  act(() => root.render(<Probe />));
  return {
    result,
    rerender(next: P) {
      props = next;
      act(() => root.render(<Probe />));
    },
    unmount() {
      act(() => root.unmount());
    },
  };
}
