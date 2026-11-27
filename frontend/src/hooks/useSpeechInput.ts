import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  SpeechRecognitionConstructor,
  SpeechRecognitionErrorCode,
  SpeechRecognitionGlobals,
  SpeechRecognitionLike,
} from '../types/speech-recognition';
import {
  HOLD_MS,
  IDLE,
  MAX_HANDS_FREE_MS,
  SILENCE_STOP_MS,
  gestureStep,
  handsFree,
} from './micGesture';
import type { GestureEvent, GestureMode, GestureState } from './micGesture';

/**
 * Dictation into the chat input with the browser's speech recognition
 * (Web Speech API).
 *
 * The hook only ever writes into the text box through `onText`; it never
 * sends anything. The user reviews the text and sends it like typed text.
 *
 * - The words are appended to the text that was in the box when listening
 *   started, with interim (not yet final) words shown as they come.
 * - The microphone button is driven by `press` / `release` (pointer or key)
 *   and `click` (an activation with no separate release). Hold it to talk,
 *   or tap it to start and tap again to stop: see ./micGesture.ts for the
 *   rules, the threshold and the edge cases.
 * - Recognition is continuous (`continuous = true`): a pause does not end
 *   it. Browsers still end a session on their own (Chrome after a long
 *   silence and at a maximum length); while the user still wants to listen,
 *   the hook starts a new session at once and keeps accumulating. Each
 *   session's words are committed when it ends and the next one starts from
 *   an empty result list, so nothing is written twice.
 * - Hands-free listening (a tap) stops by itself after SILENCE_STOP_MS with
 *   no recognised word, and after MAX_HANDS_FREE_MS in any case.
 * - `stop()` lets the browser finish the words already spoken; `cancel()`
 *   (the user edits the box, the message is sent, the chat is closed) stops
 *   at once and drops anything not yet transcribed.
 * - Each session uses a new recogniser and its handlers are detached when it
 *   is replaced or cancelled, so a late event from an old one changes nothing.
 */

export type SpeechLang = 'it-IT' | 'en-US';

export const SPEECH_LANGS: ReadonlyArray<{ tag: SpeechLang; short: string; name: string }> = [
  { tag: 'it-IT', short: 'IT', name: 'Italian' },
  { tag: 'en-US', short: 'EN', name: 'English' },
];

export function speechLangName(lang: SpeechLang): string {
  return SPEECH_LANGS.find(l => l.tag === lang)?.name ?? lang;
}

/** English when the browser's language is English, Italian otherwise (the orchestrator's default). */
export function defaultSpeechLang(browserLang: string | undefined): SpeechLang {
  const primary = (browserLang ?? '').trim().toLowerCase().split(/[-_]/)[0];
  return primary === 'en' ? 'en-US' : 'it-IT';
}

/** The recognition constructor, unprefixed or webkit-prefixed; null where the API is missing. */
export function getSpeechRecognition(
  scope: SpeechRecognitionGlobals = globalThis as SpeechRecognitionGlobals,
): SpeechRecognitionConstructor | null {
  return scope.SpeechRecognition ?? scope.webkitSpeechRecognition ?? null;
}

/** Text that was in the box, then the dictated words, separated by one space. */
export function joinDictation(base: string, dictated: string): string {
  const spoken = dictated.replace(/\s+/g, ' ').trim();
  if (!spoken) return base;
  const head = base.replace(/\s+$/, '');
  return head ? `${head} ${spoken}` : spoken;
}

/**
 * Error codes the API reports, plus 'start-failed' when start() itself throws
 * and 'restart-failed' when the browser keeps ending sessions as soon as they start.
 */
export type SpeechFailure = SpeechRecognitionErrorCode | 'start-failed' | 'restart-failed';

export function speechErrorMessage(code: SpeechFailure, lang: SpeechLang): string {
  switch (code) {
    case 'not-allowed':
      return 'Microphone access was denied. Allow it for this site in the browser settings, then try again.';
    case 'service-not-allowed':
      return 'The browser does not allow speech recognition on this page.';
    case 'audio-capture':
      return 'No microphone was found, or it could not be used.';
    case 'no-speech':
      return 'No speech was heard. Press the microphone and try again.';
    case 'network':
      return 'Speech recognition failed: network error. Check the connection and try again.';
    case 'language-not-supported':
      return `This browser cannot recognise ${speechLangName(lang)}.`;
    case 'aborted':
      return 'Listening was interrupted. Press the microphone to try again.';
    case 'restart-failed':
      return 'Speech recognition kept stopping by itself. Try again, or type the request.';
    default:
      return 'Speech recognition failed. Try again, or type the request.';
  }
}

export type SpeechStatus = 'idle' | 'listening' | 'error';

/** A session ending this soon after it started, with no word, counts as a failed restart. */
const QUICK_END_MS = 1000;
/** This many failed restarts in a row give up with 'restart-failed'. */
const MAX_QUICK_ENDS = 3;

interface Session {
  /** The text in the box when listening started. */
  base: string;
  /** Words of the sessions that already ended. */
  committed: string;
  /** Words of the running session (its whole result list). */
  current: string;
  rec: SpeechRecognitionLike | null;
  segmentStartedAt: number;
  segmentHeard: boolean;
  quickEnds: number;
  /** The recogniser reported 'start' at least once. */
  started: boolean;
  /** stop() was called: let the browser finish, then end; no restart. */
  stopping: boolean;
}

function detach(rec: SpeechRecognitionLike) {
  rec.onstart = null;
  rec.onresult = null;
  rec.onerror = null;
  rec.onend = null;
}

function sessionText(s: Session): string {
  return joinDictation(s.base, [s.committed, s.current].filter(Boolean).join(' '));
}

/** Why listening stopped by itself; null when it did not. */
export type SpeechNotice = 'silence' | 'max-length' | null;

export function speechNoticeMessage(notice: SpeechNotice): string | null {
  if (notice === 'silence') {
    return `Stopped after ${SILENCE_STOP_MS / 1000} seconds of silence. The words so far are in the box.`;
  }
  if (notice === 'max-length') {
    return `Stopped after ${MAX_HANDS_FREE_MS / 60_000} minutes. The words so far are in the box.`;
  }
  return null;
}

export interface SpeechInput {
  /** False where the browser has no speech recognition: render no control at all. */
  supported: boolean;
  status: SpeechStatus;
  listening: boolean;
  /** The gesture: 'pressed' (just pressed, not yet a hold), 'hold' (push-to-talk), 'toggle' (hands-free). */
  mode: GestureMode;
  /** The hands-free safety stops apply (a tap, or a hold whose release may have been lost to a permission prompt). */
  handsFree: boolean;
  /** Error code of the last failure, or null. */
  error: SpeechFailure | null;
  /** Readable message for `error`, or null. */
  message: string | null;
  /** Why listening last stopped by itself (hands-free safety stop), or null. */
  notice: SpeechNotice;
  /** The button went down (pointer or key); `currentText` is the box's content. */
  press: (currentText: string) => void;
  /** The button came up. */
  release: () => void;
  /** An activation with no separate press and release (assistive technology): a tap. */
  click: (currentText: string) => void;
  /**
   * The press can no longer be followed (touch cancelled, the button lost focus
   * during a key hold): stop if pressed or held, or go hands-free if the
   * recogniser has not started yet (see micGesture.ts).
   */
  endPress: () => void;
  /** Stop listening, keeping the words already spoken. */
  stop: () => void;
  /** Stop listening now, dropping anything not yet transcribed, and clear any error. */
  cancel: () => void;
}

export function useSpeechInput({
  lang,
  onText,
}: {
  lang: SpeechLang;
  /** Receives the whole new content of the box. */
  onText: (text: string) => void;
}): SpeechInput {
  const [Recognition] = useState(() => getSpeechRecognition());
  const [status, setStatus] = useState<SpeechStatus>('idle');
  const [error, setError] = useState<SpeechFailure | null>(null);
  const [notice, setNotice] = useState<SpeechNotice>(null);
  const [gestureView, setGestureView] = useState<GestureState>(IDLE);
  const session = useRef<Session | null>(null);
  const gesture = useRef<GestureState>(IDLE);
  const timers = useRef<{ hold?: number; silence?: number; max?: number }>({});

  const onTextRef = useRef(onText);
  useEffect(() => { onTextRef.current = onText; }, [onText]);
  const langRef = useRef(lang);
  langRef.current = lang;

  // ------------------------------------------------------------ timers

  const clearTimer = (name: 'hold' | 'silence' | 'max') => {
    const id = timers.current[name];
    if (id !== undefined) window.clearTimeout(id);
    timers.current[name] = undefined;
  };
  const clearTimers = () => { clearTimer('hold'); clearTimer('silence'); clearTimer('max'); };

  // Assigned below; timers and recogniser handlers call through these refs so
  // they always reach the current closures.
  const dispatchRef = useRef<(event: GestureEvent) => void>(() => {});
  const safetyStopRef = useRef<(why: Exclude<SpeechNotice, null>) => void>(() => {});

  const armSilence = () => {
    clearTimer('silence');
    timers.current.silence = window.setTimeout(() => safetyStopRef.current('silence'), SILENCE_STOP_MS);
  };

  /**
   * Hands-free safety timers run exactly while hands-free listening is active.
   * The silence count starts when the recogniser does, so time spent on the
   * permission prompt is not taken as silence; the maximum length counts from
   * the start of hands-free, so a recogniser that never starts still ends.
   */
  const syncSafetyTimers = () => {
    const s = session.current;
    if (s && !s.stopping && handsFree(gesture.current)) {
      if (s.started && timers.current.silence === undefined) armSilence();
      if (timers.current.max === undefined) {
        timers.current.max = window.setTimeout(() => safetyStopRef.current('max-length'), MAX_HANDS_FREE_MS);
      }
    } else {
      clearTimer('silence');
      clearTimer('max');
    }
  };

  const setGesture = (next: GestureState) => {
    gesture.current = next;
    setGestureView(next);
  };

  // ------------------------------------------------------------ sessions

  const finish = (s: Session) => {
    if (session.current === s) session.current = null;
    if (s.rec) detach(s.rec);
    clearTimers();
    if (gesture.current.mode !== 'idle') setGesture(IDLE);
  };

  const fail = (s: Session, code: SpeechFailure) => {
    if (s.rec) {
      detach(s.rec);
      try { s.rec.abort(); } catch { /* already ended */ }
    }
    finish(s);
    setError(code);
    setStatus('error');
  };

  const beginSegment = (s: Session) => {
    if (!Recognition) return;
    let rec: SpeechRecognitionLike;
    try {
      rec = new Recognition();
    } catch {
      fail(s, 'start-failed');
      return;
    }
    rec.lang = langRef.current;
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    s.rec = rec;
    s.current = '';
    s.segmentStartedAt = Date.now();
    s.segmentHeard = false;

    rec.onstart = () => {
      if (session.current !== s || s.rec !== rec) return;
      if (!s.started) {
        s.started = true;
        dispatchRef.current({ type: 'started', at: Date.now() });
      }
    };
    rec.onresult = event => {
      if (session.current !== s || s.rec !== rec) return;
      // `results` holds this session's whole list (final results, then interim
      // ones), so rebuilding its text from all of them is idempotent.
      let spoken = '';
      for (let i = 0; i < event.results.length; i++) {
        spoken += event.results[i][0]?.transcript ?? '';
      }
      s.current = spoken;
      s.segmentHeard = s.segmentHeard || spoken.trim() !== '';
      if (timers.current.silence !== undefined) armSilence();
      onTextRef.current(sessionText(s));
    };
    rec.onerror = event => {
      if (session.current !== s || s.rec !== rec) return;
      // Silence ends the browser's session ('no-speech', then 'end'): while the
      // user still wants to listen, 'end' starts the next one.
      if (event.error === 'no-speech') return;
      if (s.stopping && event.error === 'aborted') return;
      fail(s, event.error);
    };
    rec.onend = () => {
      if (session.current !== s || s.rec !== rec) return;
      s.committed = [s.committed, s.current].filter(Boolean).join(' ');
      s.current = '';
      detach(rec);
      if (s.stopping) {
        finish(s);
        return;
      }
      // Ended on its own while the user still wants to listen: start again.
      const quick = Date.now() - s.segmentStartedAt < QUICK_END_MS && !s.segmentHeard;
      s.quickEnds = quick ? s.quickEnds + 1 : 0;
      if (s.quickEnds >= MAX_QUICK_ENDS) {
        fail(s, 'restart-failed');
        return;
      }
      beginSegment(s);
    };

    try {
      rec.start();
    } catch {
      fail(s, 'start-failed');
    }
  };

  const cancel = useCallback(() => {
    const s = session.current;
    session.current = null;
    if (s?.rec) {
      detach(s.rec);
      try { s.rec.abort(); } catch { /* already ended */ }
    }
    clearTimers();
    setGesture(IDLE);
    setStatus('idle');
    setError(null);
    setNotice(null);
  }, []);

  const startListening = (currentText: string) => {
    if (!Recognition) return;
    const old = session.current;
    session.current = null;
    if (old?.rec) {
      detach(old.rec);
      try { old.rec.abort(); } catch { /* already ended */ }
    }
    const s: Session = {
      base: currentText, committed: '', current: '', rec: null, segmentStartedAt: 0,
      segmentHeard: false, quickEnds: 0, started: false, stopping: false,
    };
    session.current = s;
    setError(null);
    setNotice(null);
    setStatus('listening');
    beginSegment(s);
  };

  /** Stop, keeping the words: the current recogniser finishes, then the session ends. */
  const stopListening = () => {
    const s = session.current;
    clearTimers();
    if (!s) return;
    s.stopping = true;
    const rec = s.rec;
    if (!rec) {
      finish(s);
    } else {
      try {
        rec.stop();
      } catch {
        detach(rec);
        finish(s);
      }
    }
    setStatus(prev => (prev === 'listening' ? 'idle' : prev));
  };

  // ------------------------------------------------------------ gesture

  const pendingText = useRef('');

  const dispatch = (event: GestureEvent) => {
    const { state, effect } = gestureStep(gesture.current, event);
    setGesture(state);
    if (event.type === 'press' && effect === 'start') {
      clearTimer('hold');
      timers.current.hold = window.setTimeout(() => {
        timers.current.hold = undefined;
        dispatchRef.current({ type: 'hold-timer' });
      }, HOLD_MS);
    }
    if (state.mode !== 'pressed') clearTimer('hold');
    if (effect === 'start') startListening(pendingText.current);
    if (effect === 'stop') stopListening();
    syncSafetyTimers();
  };
  dispatchRef.current = dispatch;

  safetyStopRef.current = why => {
    if (!session.current) return;
    stopListening();
    setGesture(IDLE);
    setNotice(why);
  };

  const press = useCallback((currentText: string) => {
    pendingText.current = currentText;
    dispatchRef.current({ type: 'press', at: Date.now() });
  }, []);

  const release = useCallback(() => {
    dispatchRef.current({ type: 'release', at: Date.now() });
  }, []);

  const click = useCallback((currentText: string) => {
    pendingText.current = currentText;
    dispatchRef.current({ type: 'click' });
  }, []);

  const endPress = useCallback(() => {
    const mode = gesture.current.mode;
    if (mode === 'pressed' || mode === 'hold') dispatchRef.current({ type: 'interrupt' });
  }, []);

  const stop = useCallback(() => {
    if (!session.current) return;
    stopListening();
    setGesture(IDLE);
  }, []);

  // Leaving the page stops listening: a hold's release would be lost, and the
  // microphone must not stay open unattended. A hidden tab always stops; a
  // window blur stops once the recogniser has started (before that, it is
  // usually the permission prompt taking the focus: see micGesture.ts).
  const active = gestureView.mode !== 'idle';
  useEffect(() => {
    if (!active) return;
    const interrupt = () => dispatchRef.current({ type: 'interrupt' });
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') dispatchRef.current({ type: 'leave' });
    };
    window.addEventListener('blur', interrupt);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('blur', interrupt);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [active]);

  // A new language applies to the next press: finish the current words and
  // clear a message about the old language.
  useEffect(() => {
    if (session.current) {
      stopListening();
      setGesture(IDLE);
    }
    setError(null);
    setNotice(null);
    setStatus(prev => (prev === 'error' ? 'idle' : prev));
  }, [lang]);

  // Unmount: release the microphone.
  useEffect(() => () => {
    const s = session.current;
    session.current = null;
    clearTimers();
    if (s?.rec) {
      detach(s.rec);
      try { s.rec.abort(); } catch { /* already ended */ }
    }
  }, []);

  return {
    supported: Recognition !== null,
    status,
    listening: status === 'listening',
    mode: gestureView.mode,
    handsFree: handsFree(gestureView),
    error,
    message: error ? speechErrorMessage(error, lang) : null,
    notice,
    press,
    release,
    click,
    endPress,
    stop,
    cancel,
  };
}
