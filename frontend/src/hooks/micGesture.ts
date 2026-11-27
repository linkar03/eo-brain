/**
 * The microphone button's gesture: hold to talk, or tap to toggle.
 *
 * A pure state machine, so the timing rules can be tested without a browser.
 * The hook (useSpeechInput) feeds it events and carries out its effects:
 * 'start' opens a listening session, 'stop' ends it keeping the words spoken.
 *
 * - Listening starts on the press itself, so no word is lost while the
 *   gesture is still undecided.
 * - Released before HOLD_MS: a tap. Listening goes on hands-free ('toggle')
 *   until the next press, which stops it.
 * - Held for HOLD_MS or longer: push-to-talk ('hold'). Listening goes on for
 *   as long as the button is held, through pauses, and stops on release.
 * - An interruption after the recogniser has started (the press is cancelled
 *   by the browser, the window loses focus, the button loses focus during a
 *   key hold) stops, keeping the words: the release may never arrive, and the
 *   microphone must not stay open on a page the user left. A hidden page
 *   ('leave') always stops.
 * - The permission prompt: on first use, starting recognition makes the
 *   browser ask for the microphone. The prompt can take the pointer (a touch
 *   is cancelled), the focus (the window blurs) or the release, and the
 *   recogniser's 'start' event comes only once the user has answered.
 *   - An interruption before 'start' does not stop: no audio is captured yet,
 *     and the prompt is the usual cause. The press becomes hands-free, so the
 *     next press stops it and the hands-free safety stops apply once it starts.
 *   - A press whose recognition started more than START_GRACE_MS after it is
 *     'unconfirmed': a release still stops it as usual, but if none ever
 *     arrives the hands-free safety stops (silence, maximum length) apply to
 *     it as to a tap.
 */

/** Shorter presses are taps (toggle); longer ones are holds (push-to-talk). */
export const HOLD_MS = 300;
/** Recognition starting later than this after the press means a prompt may have taken the release. */
export const START_GRACE_MS = 800;
/** Hands-free listening stops after this long without any recognised word. */
export const SILENCE_STOP_MS = 10_000;
/** Hands-free listening never lasts longer than this. */
export const MAX_HANDS_FREE_MS = 120_000;

export type GestureMode = 'idle' | 'pressed' | 'hold' | 'toggle';
export type PressSource = 'pointer' | 'key' | 'click';

export interface GestureState {
  mode: GestureMode;
  pressedAt: number | null;
  /** Recognition started late: the release may have gone to a permission prompt. */
  unconfirmed: boolean;
  /** The recogniser reported 'start': audio is being captured. */
  started: boolean;
}

export type GestureEvent =
  | { type: 'press'; at: number }
  /** A complete activation with no separate release (assistive technology, a synthetic click). */
  | { type: 'click' }
  | { type: 'release'; at: number }
  /** HOLD_MS elapsed since the press, still pressed. */
  | { type: 'hold-timer' }
  /** The press or the session was interrupted: pointer cancel, window blur, the button losing focus. */
  | { type: 'interrupt' }
  /** The page was hidden: stop in any case. */
  | { type: 'leave' }
  /** The recogniser reported 'start' (permission granted, audio flowing). */
  | { type: 'started'; at: number }
  /** Listening ended for any other reason (error, safety stop, cancel, the text was edited). */
  | { type: 'ended' };

export type GestureEffect = 'start' | 'stop' | null;

export const IDLE: GestureState = { mode: 'idle', pressedAt: null, unconfirmed: false, started: false };

export function gestureStep(state: GestureState, event: GestureEvent): { state: GestureState; effect: GestureEffect } {
  const stay = { state, effect: null as GestureEffect };
  switch (event.type) {
    case 'press':
      if (state.mode === 'idle') {
        return { state: { ...IDLE, mode: 'pressed', pressedAt: event.at }, effect: 'start' };
      }
      if (state.mode === 'toggle') return { state: IDLE, effect: 'stop' };   // the second tap
      return stay;                                                           // already pressed
    case 'click':
      if (state.mode === 'idle') return { state: { ...IDLE, mode: 'toggle' }, effect: 'start' };
      if (state.mode === 'toggle') return { state: IDLE, effect: 'stop' };
      return stay;
    case 'hold-timer':
      return state.mode === 'pressed' ? { state: { ...state, mode: 'hold' }, effect: null } : stay;
    case 'release':
      if (state.mode === 'pressed') {
        const held = state.pressedAt === null ? 0 : event.at - state.pressedAt;
        if (held < HOLD_MS) return { state: { ...state, mode: 'toggle' }, effect: null };
        return { state: IDLE, effect: 'stop' };
      }
      if (state.mode === 'hold') return { state: IDLE, effect: 'stop' };
      return stay;   // idle (the release of the tap that stopped) or toggle
    case 'interrupt':
      if (state.mode === 'idle') return stay;
      // Before 'start' no audio is captured, and a permission prompt is the
      // usual cause: carry on hands-free instead of dropping the first press.
      if (!state.started) return { state: { ...state, mode: 'toggle' }, effect: null };
      return { state: IDLE, effect: 'stop' };
    case 'leave':
      return state.mode === 'idle' ? stay : { state: IDLE, effect: 'stop' };
    case 'started': {
      if (state.mode === 'idle' || state.started) return stay;
      const late = (state.mode === 'pressed' || state.mode === 'hold') && state.pressedAt !== null
        && event.at - state.pressedAt > START_GRACE_MS;
      return { state: { ...state, started: true, unconfirmed: state.unconfirmed || late }, effect: null };
    }
    case 'ended':
      return state.mode === 'idle' ? stay : { state: IDLE, effect: null };
    default:
      return stay;
  }
}

/** Whether the hands-free safety stops (silence, maximum length) apply. */
export function handsFree(state: GestureState): boolean {
  return state.mode === 'toggle' || (state.unconfirmed && state.mode !== 'idle');
}
