// The part of the Web Speech API's recognition interface that the app uses.
//
// TypeScript's lib.dom does not reliably declare SpeechRecognition (it was
// removed in TypeScript 4.4, and Chrome and Safari still expose it as
// webkitSpeechRecognition), so the app declares the subset it needs here,
// under its own names, instead of using `any`. The browser objects match
// these shapes structurally.
//
// https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition

/** SpeechRecognitionErrorEvent.error values in the specification (MDN). */
export type SpeechRecognitionErrorCode =
  | 'aborted'
  | 'audio-capture'
  | 'bad-grammar'
  | 'language-not-supported'
  | 'network'
  | 'no-speech'
  | 'not-allowed'
  | 'phrases-not-supported'
  | 'service-not-allowed';

export interface SpeechRecognitionAlternativeLike {
  readonly transcript: string;
  readonly confidence: number;
}

export interface SpeechRecognitionResultLike {
  readonly isFinal: boolean;
  readonly length: number;
  readonly [index: number]: SpeechRecognitionAlternativeLike;
}

export interface SpeechRecognitionResultListLike {
  readonly length: number;
  readonly [index: number]: SpeechRecognitionResultLike;
}

export interface SpeechRecognitionEventLike extends Event {
  readonly resultIndex: number;
  /** Every result of the session so far: final ones, then the current interim ones. */
  readonly results: SpeechRecognitionResultListLike;
}

export interface SpeechRecognitionErrorEventLike extends Event {
  readonly error: SpeechRecognitionErrorCode;
  readonly message: string;
}

export interface SpeechRecognitionLike extends EventTarget {
  /** BCP 47 language tag, e.g. "it-IT". */
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: ((event: Event) => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: ((event: Event) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

export interface SpeechRecognitionConstructor {
  new (): SpeechRecognitionLike;
}

/** Where browsers expose the constructor: unprefixed, or with the webkit prefix. */
export interface SpeechRecognitionGlobals {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
}
