// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import {
  defaultSpeechLang,
  getSpeechRecognition,
  joinDictation,
  speechErrorMessage,
  speechNoticeMessage,
  useSpeechInput,
} from './useSpeechInput';
import type { SpeechLang } from './useSpeechInput';
import { HOLD_MS, MAX_HANDS_FREE_MS, SILENCE_STOP_MS, START_GRACE_MS } from './micGesture';
import type { SpeechRecognitionConstructor } from '../types/speech-recognition';
import { MockRecognition, installSpeechRecognition, renderHook } from '../test/speech';

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-11-27T09:00:00Z'));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/** The hook wired to a fake text box; `text()` is the box's content. */
function setup(lang: SpeechLang = 'it-IT') {
  let text = '';
  const onText = vi.fn((next: string) => { text = next; });
  const hook = renderHook(
    (props: { lang: SpeechLang }) => useSpeechInput({ lang: props.lang, onText }),
    { lang },
  );
  const h = {
    ...hook,
    onText,
    text: () => text,
    get s() { return hook.result.current; },
    act(fn: () => void) { act(fn); },
    wait(ms: number) { act(() => { vi.advanceTimersByTime(ms); }); },
    /** A press, with the microphone already allowed: the recogniser starts at once. */
    press(current = text) {
      const before = MockRecognition.instances.length;
      const rec = h.pressPrompted(current);
      if (MockRecognition.instances.length > before) rec.emitStart();
      return rec;
    },
    /** A first press: the recogniser waits on the permission prompt (call `emitStart()` for "Allow"). */
    pressPrompted(current = text) {
      text = current;
      act(() => hook.result.current.press(current));
      return MockRecognition.last;
    },
    release() { act(() => hook.result.current.release()); },
    /** A short press: hands-free listening. */
    tap(current = text) {
      const rec = h.press(current);
      h.wait(HOLD_MS / 3);
      h.release();
      return rec;
    },
    /** Press and keep holding past the threshold: push-to-talk. */
    hold(current = text) {
      const rec = h.press(current);
      h.wait(HOLD_MS + 50);
      return rec;
    },
  };
  return h;
}

describe('helpers', () => {
  it.each([
    ['it-IT', 'it-IT'],
    ['it', 'it-IT'],
    ['it-CH', 'it-IT'],
    ['en-US', 'en-US'],
    ['en-GB', 'en-US'],
    ['EN', 'en-US'],
    ['de-DE', 'it-IT'],
    ['fr', 'it-IT'],
    ['', 'it-IT'],
    [undefined, 'it-IT'],
  ])('defaultSpeechLang(%s) = %s', (browser, expected) => {
    expect(defaultSpeechLang(browser)).toBe(expected);
  });

  it.each([
    ['', 'Firenze', 'Firenze'],
    ['NDVI near', 'Firenze', 'NDVI near Firenze'],
    ['NDVI near ', ' Firenze ', 'NDVI near Firenze'],
    ['NDVI near', '  vegetation   in  Milan ', 'NDVI near vegetation in Milan'],
    ['typed text ', '', 'typed text '],
    ['typed text', '   ', 'typed text'],
  ])('joinDictation(%j, %j) = %j', (base, dictated, expected) => {
    expect(joinDictation(base, dictated)).toBe(expected);
  });

  it('finds the unprefixed constructor first, then the webkit one', () => {
    const a = MockRecognition as unknown as SpeechRecognitionConstructor;
    const b = class extends MockRecognition {} as unknown as SpeechRecognitionConstructor;
    expect(getSpeechRecognition({ SpeechRecognition: a, webkitSpeechRecognition: b })).toBe(a);
    expect(getSpeechRecognition({ webkitSpeechRecognition: b })).toBe(b);
    expect(getSpeechRecognition({})).toBeNull();
  });

  it('explains a safety stop', () => {
    expect(speechNoticeMessage('silence')).toBe('Stopped after 10 seconds of silence. The words so far are in the box.');
    expect(speechNoticeMessage('max-length')).toBe('Stopped after 2 minutes. The words so far are in the box.');
    expect(speechNoticeMessage(null)).toBeNull();
  });
});

describe('unsupported browser', () => {
  it('reports unsupported and does nothing', () => {
    const h = setup();
    expect(h.s.supported).toBe(false);
    h.act(() => h.s.press('text'));
    h.release();
    h.act(() => h.s.click('text'));
    h.act(() => h.s.stop());
    h.act(() => h.s.cancel());
    expect(h.s.status).toBe('idle');
    expect(h.s.message).toBeNull();
    expect(h.onText).not.toHaveBeenCalled();
    h.unmount();
  });

  it('works with the unprefixed constructor too', () => {
    installSpeechRecognition('SpeechRecognition');
    const h = setup();
    expect(h.s.supported).toBe(true);
    h.press();
    expect(MockRecognition.last.start).toHaveBeenCalledOnce();
    h.unmount();
  });
});

describe('push-to-talk (hold)', () => {
  it('starts on the press, continuous, with interim results in the chosen language', () => {
    installSpeechRecognition();
    const h = setup('en-US');
    const rec = h.press();
    expect(rec.lang).toBe('en-US');
    expect(rec.continuous).toBe(true);
    expect(rec.interimResults).toBe(true);
    expect(rec.maxAlternatives).toBe(1);
    expect(rec.start).toHaveBeenCalledOnce();
    expect(h.s.listening).toBe(true);
    expect(h.s.mode).toBe('pressed');
    h.wait(HOLD_MS);
    expect(h.s.mode).toBe('hold');
    expect(h.s.handsFree).toBe(false);
    h.unmount();
  });

  it('keeps listening through a long pause while held, and stops on release keeping the words', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    rec.emitResult(['analizza la', false]);
    h.wait(SILENCE_STOP_MS * 3);                    // silence while held: no safety stop
    expect(h.s.listening).toBe(true);
    expect(rec.stop).not.toHaveBeenCalled();

    h.release();
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(rec.abort).not.toHaveBeenCalled();
    expect(h.s.status).toBe('idle');
    expect(h.s.mode).toBe('idle');
    rec.emitResult(['analizza la vegetazione', true]);   // the last words still arrive
    expect(h.text()).toBe('analizza la vegetazione');
    rec.emitEnd();
    expect(rec.detached).toBe(true);
    expect(MockRecognition.instances).toHaveLength(1);    // no restart after a release
    h.unmount();
  });

  it('restarts transparently when the browser ends the session while held, without repeating words', () => {
    installSpeechRecognition();
    const h = setup();
    const first = h.hold('NDVI');
    first.emitResult(['vegetazione', true], [' a Fi', false]);
    expect(h.text()).toBe('NDVI vegetazione a Fi');
    h.wait(5000);
    first.emitEnd();                                  // Chrome: maximum length or long silence

    expect(MockRecognition.instances).toHaveLength(2);
    const second = MockRecognition.last;
    expect(second.start).toHaveBeenCalledOnce();
    expect(second.continuous).toBe(true);
    expect(first.detached).toBe(true);
    expect(h.s.listening).toBe(true);
    expect(h.s.mode).toBe('hold');
    expect(h.text()).toBe('NDVI vegetazione a Fi');

    second.emitResult(['Firenze', false]);           // a fresh result list
    expect(h.text()).toBe('NDVI vegetazione a Fi Firenze');
    second.emitResult(['Firenze nel 2024', true]);
    expect(h.text()).toBe('NDVI vegetazione a Fi Firenze nel 2024');

    first.emitResult(['stale', true]);               // the old recogniser is ignored
    expect(h.text()).toBe('NDVI vegetazione a Fi Firenze nel 2024');

    h.release();
    expect(second.stop).toHaveBeenCalledOnce();
    second.emitEnd();
    expect(MockRecognition.instances).toHaveLength(2);
    expect(h.s.status).toBe('idle');
    h.unmount();
  });

  it('"no-speech" while held is not an error: the session restarts', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    h.wait(8000);
    rec.emitError('no-speech');
    rec.emitEnd();
    expect(h.s.status).toBe('listening');
    expect(h.s.message).toBeNull();
    expect(MockRecognition.instances).toHaveLength(2);
    h.unmount();
  });

  it('gives up when every restart ends at once with no words', () => {
    installSpeechRecognition();
    const h = setup();
    h.hold();
    for (let i = 0; i < 3; i++) MockRecognition.last.emitEnd();
    expect(h.s.status).toBe('error');
    expect(h.s.error).toBe('restart-failed');
    expect(h.s.message).toMatch(/kept stopping by itself/);
    expect(h.s.mode).toBe('idle');
    h.unmount();
  });

  it('a pause long enough between restarts is not counted as a failed restart', () => {
    installSpeechRecognition();
    const h = setup();
    h.hold();
    for (let i = 0; i < 5; i++) {
      h.wait(2000);
      MockRecognition.last.emitEnd();
    }
    expect(h.s.status).toBe('listening');
    expect(MockRecognition.instances).toHaveLength(6);
    h.unmount();
  });

  it('losing the window stops a hold (the release would never arrive), keeping the words', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    rec.emitResult(['acqua a Pisa', false]);
    h.act(() => { window.dispatchEvent(new Event('blur')); });
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.mode).toBe('idle');
    expect(h.text()).toBe('acqua a Pisa');
    h.unmount();
  });

  it('a hidden page stops listening', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
    h.act(() => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(rec.stop).toHaveBeenCalledOnce();
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    h.unmount();
  });

  it('endPress (pointer cancel, the button losing focus) stops a press or a hold, not hands-free', () => {
    installSpeechRecognition();
    const h = setup();
    let rec = h.hold();
    h.act(() => h.s.endPress());
    expect(rec.stop).toHaveBeenCalledOnce();
    rec.emitEnd();

    rec = h.tap();
    h.act(() => h.s.endPress());
    expect(rec.stop).not.toHaveBeenCalled();
    expect(h.s.mode).toBe('toggle');
    h.unmount();
  });
});

describe('the permission prompt on first use', () => {
  it('a late start makes the hold unconfirmed: hands-free safety stops apply if no release arrives', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.wait(START_GRACE_MS + 2000);                   // the prompt is open; the release goes to it
    expect(h.s.mode).toBe('hold');
    rec.emitStart();                                 // the user allowed the microphone
    expect(h.s.handsFree).toBe(true);
    rec.emitResult(['vegetazione', false]);
    h.wait(SILENCE_STOP_MS - 1);
    expect(rec.stop).not.toHaveBeenCalled();
    h.wait(1);
    expect(rec.stop).toHaveBeenCalledOnce();         // the microphone does not stay open
    expect(h.s.mode).toBe('idle');
    expect(h.s.notice).toBe('silence');
    expect(h.text()).toBe('vegetazione');
    h.unmount();
  });

  it('an unconfirmed hold still stops on its release if it does arrive', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.wait(START_GRACE_MS + 500);
    rec.emitStart();
    h.release();
    expect(rec.stop).toHaveBeenCalledOnce();
    h.unmount();
  });

  it('a prompt-free start is not unconfirmed', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.wait(100);
    rec.emitStart();
    h.wait(HOLD_MS);
    expect(h.s.mode).toBe('hold');
    expect(h.s.handsFree).toBe(false);
    h.unmount();
  });

  it.each([
    ['the prompt cancels the touch', (h: ReturnType<typeof setup>) => h.act(() => h.s.endPress())],
    ['the prompt takes the focus', (h: ReturnType<typeof setup>) => h.act(() => { window.dispatchEvent(new Event('blur')); })],
  ])('%s before the recogniser starts: the press goes on hands-free, not stopped', (_why, interrupt) => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.wait(HOLD_MS + 50);
    interrupt(h);
    expect(rec.stop).not.toHaveBeenCalled();
    expect(h.s.mode).toBe('toggle');
    expect(h.s.handsFree).toBe(true);

    h.wait(SILENCE_STOP_MS * 2);                     // answering the prompt is not silence
    expect(rec.stop).not.toHaveBeenCalled();
    rec.emitStart();                                 // allowed
    rec.emitResult(['acqua', false]);
    h.wait(SILENCE_STOP_MS);                         // ...but silence after the start is
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.notice).toBe('silence');
    expect(h.text()).toBe('acqua');
    h.unmount();
  });

  it('a press interrupted before the start is stopped by the next press', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.act(() => h.s.endPress());
    rec.emitStart();
    h.press();
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.mode).toBe('idle');
    h.unmount();
  });

  it('a recogniser that never starts is still ended by the maximum length', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.act(() => h.s.endPress());
    h.wait(MAX_HANDS_FREE_MS);
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.notice).toBe('max-length');
    h.unmount();
  });

  it('a hidden page stops even before the recogniser starts', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
    h.act(() => { document.dispatchEvent(new Event('visibilitychange')); });
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.mode).toBe('idle');
    h.unmount();
  });

  it('a denied permission ends the press with its message', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.pressPrompted();
    h.act(() => h.s.endPress());                     // the prompt took the touch
    rec.emitError('not-allowed');
    expect(h.s.mode).toBe('idle');
    expect(h.s.listening).toBe(false);
    expect(h.s.error).toBe('not-allowed');
    h.unmount();
  });
});

describe('hands-free (tap)', () => {
  it('a short press keeps listening after the release; the next press stops it', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.tap();
    expect(h.s.mode).toBe('toggle');
    expect(h.s.handsFree).toBe(true);
    expect(h.s.listening).toBe(true);
    expect(rec.stop).not.toHaveBeenCalled();
    rec.emitResult(['ndwi', false]);

    h.press();                                         // second tap: stop on the press
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(MockRecognition.instances).toHaveLength(1); // it does not start a new session
    h.release();                                       // its release changes nothing
    expect(h.s.mode).toBe('idle');
    expect(MockRecognition.instances).toHaveLength(1);
    h.unmount();
  });

  it('does not stop on a pause, but stops after the silence limit keeping the words', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.tap('NDVI');
    h.wait(SILENCE_STOP_MS - 1000);
    rec.emitResult(['Firenze', false]);              // speech resets the silence clock
    h.wait(SILENCE_STOP_MS - 1000);
    expect(h.s.listening).toBe(true);
    h.wait(1000);
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.status).toBe('idle');
    expect(h.s.notice).toBe('silence');
    expect(h.text()).toBe('NDVI Firenze');
    h.unmount();
  });

  it('restarts across browser-ended sessions and still has the safety stops', () => {
    installSpeechRecognition();
    const h = setup();
    const first = h.tap();
    first.emitResult(['allagamenti', true]);
    h.wait(3000);
    first.emitEnd();
    const second = MockRecognition.last;
    expect(second).not.toBe(first);
    second.emitResult(['a Venezia', false]);
    expect(h.text()).toBe('allagamenti a Venezia');
    h.wait(SILENCE_STOP_MS);
    expect(second.stop).toHaveBeenCalledOnce();
    h.unmount();
  });

  it('never lasts longer than the maximum, even with constant speech (a TV in the room)', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.tap();
    for (let t = 0; t < MAX_HANDS_FREE_MS; t += 5000) {
      rec.emitResult([`parola ${t}`, false]);
      h.wait(5000);
    }
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.notice).toBe('max-length');
    h.unmount();
  });

  it('a click (assistive technology) is a tap', () => {
    installSpeechRecognition();
    const h = setup();
    h.act(() => h.s.click(''));
    expect(h.s.mode).toBe('toggle');
    const rec = MockRecognition.last;
    h.act(() => h.s.click(h.text()));
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.mode).toBe('idle');
    h.unmount();
  });

  it('a new press clears the notice of the last safety stop', () => {
    installSpeechRecognition();
    const h = setup();
    h.tap();
    h.wait(SILENCE_STOP_MS);
    expect(h.s.notice).toBe('silence');
    h.press();
    expect(h.s.notice).toBeNull();
    h.unmount();
  });
});

describe('transcript', () => {
  it('shows interim words, then the final text, and never does anything else', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    rec.emitResult(['vegetazione a', false]);
    expect(h.text()).toBe('vegetazione a');
    rec.emitResult(['vegetazione a Firenze', false]);
    expect(h.text()).toBe('vegetazione a Firenze');
    rec.emitResult(['vegetazione a Firenze nel 2024', true]);
    expect(h.text()).toBe('vegetazione a Firenze nel 2024');
    h.release();
    rec.emitEnd();
    expect(h.text()).toBe('vegetazione a Firenze nel 2024');
    expect(h.s.status).toBe('idle');
    // The hook's only output is the text box.
    expect(h.onText).toHaveBeenCalledTimes(3);
    h.unmount();
  });

  it('joins final and interim results of the session', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    rec.emitResult(['analyze vegetation', true], [' in Milan', false]);
    expect(h.text()).toBe('analyze vegetation in Milan');
    h.unmount();
  });

  it('appends to the text that was in the box, and a second dictation to the first', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold('NDVI near ');
    rec.emitResult(['Firenze centro', true]);
    expect(h.text()).toBe('NDVI near Firenze centro');
    h.release();
    rec.emitEnd();

    const again = h.hold();
    again.emitResult([' last year', true]);
    expect(h.text()).toBe('NDVI near Firenze centro last year');
    h.unmount();
  });

  it('an empty result leaves the box as it was', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold('typed ');
    rec.emitResult(['', false]);
    expect(h.text()).toBe('typed ');
    h.unmount();
  });

  it('a new language is used from the next press; the current words are kept', () => {
    installSpeechRecognition();
    const h = setup('it-IT');
    const rec = h.tap();
    h.rerender({ lang: 'en-US' });
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(h.s.status).toBe('idle');
    expect(h.s.mode).toBe('idle');
    rec.emitEnd();
    expect(h.press().lang).toBe('en-US');
    h.unmount();
  });
});

describe('errors', () => {
  it.each([
    ['not-allowed', /Microphone access was denied/],
    ['service-not-allowed', /does not allow speech recognition/],
    ['audio-capture', /No microphone was found/],
    ['network', /network error/],
    ['language-not-supported', /cannot recognise Italian/],
    ['aborted', /Listening was interrupted/],
    ['bad-grammar', /Speech recognition failed\. Try again/],
    ['phrases-not-supported', /Speech recognition failed\. Try again/],
    ['something-new', /Speech recognition failed\. Try again/],
  ])('%s gives a readable message and ends the session', (code, message) => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold('typed');
    rec.emitError(code);

    expect(h.s.status).toBe('error');
    expect(h.s.listening).toBe(false);
    expect(h.s.mode).toBe('idle');
    expect(h.s.error).toBe(code);
    expect(h.s.message).toMatch(message);
    expect(rec.detached).toBe(true);
    expect(MockRecognition.instances).toHaveLength(1);   // an error is not restarted

    rec.emitResult(['late words', true]);
    expect(h.text()).toBe('typed');
    h.release();                                         // the release after an error does nothing
    expect(MockRecognition.instances).toHaveLength(1);
    h.unmount();
  });

  it('names the selected language when it is not supported', () => {
    expect(speechErrorMessage('language-not-supported', 'en-US')).toBe('This browser cannot recognise English.');
    expect(speechErrorMessage('no-speech', 'it-IT')).toMatch(/No speech was heard/);
  });

  it('reports start() throwing as an error', () => {
    installSpeechRecognition();
    vi.stubGlobal('webkitSpeechRecognition', class extends MockRecognition {
      start = vi.fn(() => { throw new DOMException('already started', 'InvalidStateError'); });
    });
    const h = setup();
    const rec = h.press();
    expect(h.s.status).toBe('error');
    expect(h.s.error).toBe('start-failed');
    expect(h.s.message).toMatch(/Speech recognition failed/);
    expect(h.s.mode).toBe('idle');
    expect(rec.detached).toBe(true);
    h.unmount();
  });

  it('after a stop, "aborted" is not an error; others still are', () => {
    installSpeechRecognition();
    const h = setup();
    let rec = h.hold();
    h.release();
    rec.emitError('aborted');
    rec.emitError('no-speech');
    expect(h.s.status).toBe('idle');
    expect(h.s.message).toBeNull();
    rec.emitEnd();

    rec = h.hold();
    h.release();
    rec.emitError('network');
    expect(h.s.status).toBe('error');
    h.unmount();
  });

  it('the message is cleared by the next press, by cancel and by a language change', () => {
    installSpeechRecognition();
    const h = setup();
    h.hold().emitError('network');
    expect(h.s.message).not.toBeNull();
    h.press();
    expect(h.s.message).toBeNull();
    expect(h.s.status).toBe('listening');

    MockRecognition.last.emitError('network');
    h.act(() => h.s.cancel());
    expect(h.s.message).toBeNull();
    expect(h.s.status).toBe('idle');

    h.hold().emitError('language-not-supported');
    h.rerender({ lang: 'en-US' });
    expect(h.s.message).toBeNull();
    expect(h.s.status).toBe('idle');
    h.unmount();
  });
});

describe('cancel and cleanup', () => {
  it('cancel aborts at once, ends the gesture and ignores anything the old session sends', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold('before');
    rec.emitResult(['spoken', false]);
    h.act(() => h.s.cancel());

    expect(rec.abort).toHaveBeenCalledOnce();
    expect(rec.detached).toBe(true);
    expect(h.s.status).toBe('idle');
    expect(h.s.mode).toBe('idle');
    expect(h.text()).toBe('before spoken');   // what was on screen stays
    rec.emitResult(['spoken more', true]);
    rec.emitEnd();
    expect(h.text()).toBe('before spoken');
    expect(MockRecognition.instances).toHaveLength(1);
    h.release();                               // the release of the cancelled hold does nothing
    expect(MockRecognition.instances).toHaveLength(1);
    h.unmount();
  });

  it('cancel stops the hands-free timers', () => {
    installSpeechRecognition();
    const h = setup();
    h.tap();
    h.act(() => h.s.cancel());
    h.wait(MAX_HANDS_FREE_MS);
    expect(h.s.notice).toBeNull();
    h.unmount();
  });

  it('unmount while listening releases the microphone', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.tap();
    h.unmount();
    expect(rec.abort).toHaveBeenCalledOnce();
    expect(rec.detached).toBe(true);
  });

  it('unmount while idle does nothing', () => {
    installSpeechRecognition();
    const h = setup();
    const rec = h.hold();
    h.release();
    rec.emitEnd();
    h.unmount();
    expect(rec.abort).not.toHaveBeenCalled();
  });
});
