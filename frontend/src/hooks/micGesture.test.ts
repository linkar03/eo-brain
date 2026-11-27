import { describe, expect, it } from 'vitest';
import { HOLD_MS, IDLE, START_GRACE_MS, gestureStep, handsFree } from './micGesture';
import type { GestureEvent, GestureState } from './micGesture';

/** Run events from idle; returns the final state and every effect, in order. */
function run(...events: GestureEvent[]) {
  let state: GestureState = IDLE;
  const effects: string[] = [];
  for (const e of events) {
    const out = gestureStep(state, e);
    state = out.state;
    if (out.effect) effects.push(out.effect);
  }
  return { state, effects };
}

describe('mic gesture', () => {
  it('a press starts listening at once, before the gesture is decided', () => {
    const { state, effects } = run({ type: 'press', at: 0 });
    expect(state.mode).toBe('pressed');
    expect(effects).toEqual(['start']);
  });

  it(`a release before ${HOLD_MS} ms is a tap: hands-free, no stop`, () => {
    const { state, effects } = run({ type: 'press', at: 0 }, { type: 'release', at: HOLD_MS - 1 });
    expect(state.mode).toBe('toggle');
    expect(handsFree(state)).toBe(true);
    expect(effects).toEqual(['start']);
  });

  it('the next press stops hands-free listening, and its release does nothing', () => {
    const { state, effects } = run(
      { type: 'press', at: 0 }, { type: 'release', at: 100 },
      { type: 'press', at: 5000 }, { type: 'release', at: 5100 },
    );
    expect(state).toEqual(IDLE);
    expect(effects).toEqual(['start', 'stop']);
  });

  it('a hold is push-to-talk: no safety stops, the release stops', () => {
    const held = run({ type: 'press', at: 0 }, { type: 'hold-timer' });
    expect(held.state.mode).toBe('hold');
    expect(handsFree(held.state)).toBe(false);
    const released = run({ type: 'press', at: 0 }, { type: 'hold-timer' }, { type: 'release', at: 9000 });
    expect(released.state).toEqual(IDLE);
    expect(released.effects).toEqual(['start', 'stop']);
  });

  it('a release at the threshold counts as a hold even if the timer has not fired yet', () => {
    const { state, effects } = run({ type: 'press', at: 0 }, { type: 'release', at: HOLD_MS });
    expect(state).toEqual(IDLE);
    expect(effects).toEqual(['start', 'stop']);
  });

  const STARTED: GestureEvent = { type: 'started', at: 100 };
  const prefixes: Record<string, GestureEvent[]> = {
    pressed: [{ type: 'press', at: 0 }, STARTED],
    hold: [{ type: 'press', at: 0 }, STARTED, { type: 'hold-timer' }],
    toggle: [{ type: 'press', at: 0 }, STARTED, { type: 'release', at: 150 }],
  };

  it.each(['pressed', 'hold', 'toggle'])('an interruption in %s after the start stops', mode => {
    const { state, effects } = run(...prefixes[mode], { type: 'interrupt' });
    expect(state).toEqual(IDLE);
    expect(effects).toEqual(['start', 'stop']);
  });

  it.each(['pressed', 'hold'])('an interruption in %s before the start (a permission prompt) goes hands-free', mode => {
    const prefix = prefixes[mode].filter(e => e.type !== 'started');
    const { state, effects } = run(...prefix, { type: 'interrupt' });
    expect(state.mode).toBe('toggle');
    expect(handsFree(state)).toBe(true);
    expect(effects).toEqual(['start']);
    // the release that may follow changes nothing; the start arrives; the next press stops
    const after = run(...prefix, { type: 'interrupt' }, { type: 'release', at: 2000 }, { type: 'started', at: 5000 },
      { type: 'press', at: 9000 });
    expect(after).toEqual({ state: IDLE, effects: ['start', 'stop'] });
  });

  it.each(['pressed', 'hold', 'toggle'])('leaving the page in %s stops, started or not', mode => {
    expect(run(...prefixes[mode], { type: 'leave' })).toEqual({ state: IDLE, effects: ['start', 'stop'] });
    const notStarted = prefixes[mode].filter(e => e.type !== 'started');
    expect(run(...notStarted, { type: 'leave' })).toEqual({ state: IDLE, effects: ['start', 'stop'] });
  });

  it('an interruption, a departure or a release while idle does nothing', () => {
    expect(run({ type: 'interrupt' }, { type: 'leave' }, { type: 'release', at: 0 }, { type: 'hold-timer' }))
      .toEqual({ state: IDLE, effects: [] });
  });

  it('a second press while still pressed is ignored (key repeat, a second finger)', () => {
    const { state, effects } = run({ type: 'press', at: 0 }, { type: 'press', at: 50 });
    expect(state.pressedAt).toBe(0);
    expect(effects).toEqual(['start']);
  });

  it('recognition starting late (a permission prompt) makes the press unconfirmed: hands-free safety applies', () => {
    const { state } = run({ type: 'press', at: 0 }, { type: 'hold-timer' }, { type: 'started', at: START_GRACE_MS + 1 });
    expect(state.mode).toBe('hold');
    expect(state.unconfirmed).toBe(true);
    expect(handsFree(state)).toBe(true);
    // ...and its release, if it does arrive, still stops
    const out = gestureStep(state, { type: 'release', at: 20_000 });
    expect(out.state).toEqual(IDLE);
    expect(out.effect).toBe('stop');
  });

  it('recognition starting promptly keeps a hold a hold', () => {
    const { state } = run({ type: 'press', at: 0 }, { type: 'started', at: START_GRACE_MS }, { type: 'hold-timer' });
    expect(state.unconfirmed).toBe(false);
    expect(state.started).toBe(true);
    expect(handsFree(state)).toBe(false);
  });

  it('only the first start counts (restarts after the browser ends a session)', () => {
    const { state } = run({ type: 'press', at: 0 }, { type: 'started', at: 100 }, { type: 'hold-timer' },
      { type: 'started', at: START_GRACE_MS + 5000 });
    expect(state.unconfirmed).toBe(false);
  });

  it('a click (no press or release of its own) toggles', () => {
    expect(run({ type: 'click' }).state.mode).toBe('toggle');
    expect(run({ type: 'click' }, { type: 'click' })).toEqual({ state: IDLE, effects: ['start', 'stop'] });
  });

  it('when listening ends by itself the gesture ends without another stop', () => {
    const { state, effects } = run({ type: 'press', at: 0 }, { type: 'hold-timer' }, { type: 'ended' });
    expect(state).toEqual(IDLE);
    expect(effects).toEqual(['start']);
  });
});
