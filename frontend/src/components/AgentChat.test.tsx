// @vitest-environment happy-dom
// Voice input in the chat: the microphone only fills the text box; sending is
// still the user's explicit action.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { AgentChat, JobStatusBadge, dictationStatus } from './AgentChat';
import { HOLD_MS, SILENCE_STOP_MS } from '../hooks/micGesture';
import { MockRecognition, installSpeechRecognition } from '../test/speech';

let container: HTMLDivElement;
let root: Root;
let fetchMock: ReturnType<typeof vi.fn>;

function setBrowserLanguage(language: string) {
  Object.defineProperty(window.navigator, 'language', { value: language, configurable: true });
}

function render() {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(<AgentChat />));
  click(container.querySelector<HTMLButtonElement>('button[title="Open EO Brain Agent"]')!);
}

const q = <T extends Element>(selector: string) => container.querySelector<T>(selector);
const input = () => q<HTMLInputElement>('input[aria-label="Message"]')!;
const mic = () => q<HTMLButtonElement>('button[aria-pressed]');
const languageSelect = () => q<HTMLSelectElement>('select[aria-label="Dictation language"]');
const status = () => q<HTMLElement>('[role="status"]')?.textContent ?? '';

/**
 * Run a user action; a recogniser it creates reports 'start' at once, as with
 * the microphone already allowed. `prompted: true` leaves it waiting on the
 * permission prompt.
 */
function withStart(action: () => void, prompted = false) {
  const before = MockRecognition.instances.length;
  act(action);
  if (!prompted && MockRecognition.instances.length > before) MockRecognition.last.emitStart();
}

function click(el: HTMLElement) {
  withStart(() => { el.click(); });
}

/** Type into the box the way a user does (React sees an input event). */
function type(value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
  act(() => {
    setter.call(input(), value);
    input().dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function pointerDown(el: HTMLElement, pointerType = 'mouse', prompted = false) {
  withStart(() => { el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerId: 1, pointerType })); }, prompted);
}

function pointerUp(el: HTMLElement, pointerType = 'mouse') {
  act(() => { el.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0, pointerId: 1, pointerType })); });
}

function key(el: HTMLElement, type: 'keydown' | 'keyup', k: string, repeat = false) {
  withStart(() => { el.dispatchEvent(new KeyboardEvent(type, { key: k, repeat, bubbles: true, cancelable: true })); });
}

function wait(ms: number) {
  act(() => { vi.advanceTimersByTime(ms); });
}

function pressEnter() {
  act(() => {
    input().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  });
}

function selectLanguage(tag: string) {
  act(() => {
    const select = languageSelect()!;
    select.value = tag;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-11-27T09:00:00Z'));
  setBrowserLanguage('it-IT');
  // The request never resolves: these tests only look at what is sent.
  fetchMock = vi.fn(() => new Promise<Response>(() => {}));
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('AgentChat voice input', () => {
  it('renders no microphone and no language choice where the API is missing', () => {
    render();
    expect(mic()).toBeNull();
    expect(languageSelect()).toBeNull();
    expect(q('[role="status"]')).toBeNull();
    expect(input()).not.toBeNull();
    expect(q('button[aria-label="Send"]')).not.toBeNull();
  });

  it.each([
    ['it-IT', 'it-IT', 'Dictate in Italian'],
    ['en-GB', 'en-US', 'Dictate in English'],
    ['de-DE', 'it-IT', 'Dictate in Italian'],
  ])('browser language %s: dictation defaults to %s', (browser, tag, label) => {
    setBrowserLanguage(browser);
    installSpeechRecognition();
    render();
    expect(languageSelect()!.value).toBe(tag);
    expect(mic()!.getAttribute('aria-label')).toBe(label);
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
    expect(mic()!.type).toBe('button');
  });

  it('holding the microphone dictates while held; releasing stops and sends nothing', () => {
    installSpeechRecognition();
    render();
    type('NDVI');
    pointerDown(mic()!);
    const rec = MockRecognition.last;
    expect(rec.lang).toBe('it-IT');
    expect(rec.continuous).toBe(true);
    expect(mic()!.getAttribute('aria-pressed')).toBe('true');
    expect(mic()!.getAttribute('aria-label')).toBe('Dictate in Italian');   // label stays, state is aria-pressed
    wait(HOLD_MS + 10);
    expect(status()).toMatch(/^Push-to-talk \(Italian\): listening while you hold\. Release to stop\. Nothing is sent until you press Send\.$/);

    rec.emitResult(['a Firenze', false]);
    expect(input().value).toBe('NDVI a Firenze');
    pointerUp(mic()!);
    expect(rec.stop).toHaveBeenCalledOnce();
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
    rec.emitResult(['a Firenze nel 2024', true]);
    rec.emitEnd();
    expect(input().value).toBe('NDVI a Firenze nel 2024');
    expect(status()).toBe('');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('the click that follows a pointer gesture does not toggle again', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!);
    wait(HOLD_MS + 10);
    pointerUp(mic()!);
    click(mic()!);                                    // the browser's click after pointerup
    expect(MockRecognition.instances).toHaveLength(1);
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
  });

  it('a tap starts hands-free dictation, says so, and the next tap stops it', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!);
    wait(50);
    pointerUp(mic()!);
    const rec = MockRecognition.last;
    expect(mic()!.getAttribute('aria-pressed')).toBe('true');
    expect(status()).toMatch(/^Hands-free \(Italian\): listening until you press the microphone again; it stops by itself after 10 seconds of silence\./);
    rec.emitResult(['vegetazione', false]);

    pointerDown(mic()!);
    expect(rec.stop).toHaveBeenCalledOnce();
    pointerUp(mic()!);
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
    expect(MockRecognition.instances).toHaveLength(1);
  });

  it('hands-free dictation stops after the silence limit and says why', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!);
    pointerUp(mic()!);
    wait(SILENCE_STOP_MS);
    expect(MockRecognition.last.stop).toHaveBeenCalledOnce();
    expect(status()).toBe('Stopped after 10 seconds of silence. The words so far are in the box.');
  });

  it('a click with no pointer or key gesture (assistive technology) is a tap', () => {
    installSpeechRecognition();
    render();
    click(mic()!);
    expect(mic()!.getAttribute('aria-pressed')).toBe('true');
    expect(status()).toMatch(/^Hands-free/);
    click(mic()!);
    expect(MockRecognition.last.stop).toHaveBeenCalledOnce();
  });

  it('a cancelled touch stops a hold', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!, 'touch');
    wait(HOLD_MS + 10);
    act(() => { mic()!.dispatchEvent(new PointerEvent('pointercancel', { bubbles: true, pointerId: 1, pointerType: 'touch' })); });
    expect(MockRecognition.last.stop).toHaveBeenCalledOnce();
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
  });

  it('first use: the permission prompt cancels the touch, and the press goes on hands-free once allowed', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!, 'touch', true);
    const rec = MockRecognition.last;
    act(() => { mic()!.dispatchEvent(new PointerEvent('pointercancel', { bubbles: true, pointerId: 1, pointerType: 'touch' })); });
    expect(rec.stop).not.toHaveBeenCalled();
    rec.emitStart();
    expect(status()).toMatch(/^Hands-free \(Italian\)/);
    rec.emitResult(['vegetazione a Pisa', false]);
    expect(input().value).toBe('vegetazione a Pisa');
    pointerDown(mic()!, 'touch');                     // the next press stops it
    expect(rec.stop).toHaveBeenCalledOnce();
    pointerUp(mic()!, 'touch');
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('a right click does not dictate', () => {
    installSpeechRecognition();
    render();
    act(() => { mic()!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 2, pointerId: 1, pointerType: 'mouse' })); });
    expect(MockRecognition.instances).toHaveLength(0);
  });

  it('keyboard: holding Space is push-to-talk, a short Enter press toggles', () => {
    installSpeechRecognition();
    render();
    key(mic()!, 'keydown', ' ');
    wait(HOLD_MS + 10);
    key(mic()!, 'keydown', ' ', true);                // auto-repeat while held: ignored
    expect(MockRecognition.instances).toHaveLength(1);
    expect(status()).toMatch(/^Push-to-talk/);
    key(mic()!, 'keyup', ' ');
    expect(MockRecognition.last.stop).toHaveBeenCalledOnce();
    MockRecognition.last.emitEnd();

    key(mic()!, 'keydown', 'Enter');
    key(mic()!, 'keyup', 'Enter');
    expect(MockRecognition.instances).toHaveLength(2);
    expect(status()).toMatch(/^Hands-free/);
    key(mic()!, 'keydown', 'Enter');
    expect(MockRecognition.last.stop).toHaveBeenCalledOnce();
    key(mic()!, 'keyup', 'Enter');
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
  });

  it('the dictated text can be edited, and is sent only with Enter in the box', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!);
    wait(HOLD_MS + 10);
    const rec = MockRecognition.last;
    rec.emitResult(['vegetazione a Firenzi', true]);
    pointerUp(mic()!);
    rec.emitEnd();

    type('vegetazione a Firenze');
    expect(fetchMock).not.toHaveBeenCalled();
    pressEnter();
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.message).toBe('vegetazione a Firenze');
  });

  it('typing while listening stops listening and keeps what is on screen', () => {
    installSpeechRecognition();
    render();
    type('NDVI');
    click(mic()!);
    const rec = MockRecognition.last;
    rec.emitResult(['Milano', false]);
    expect(input().value).toBe('NDVI Milano');

    type('NDVI Milano 2023');
    expect(rec.abort).toHaveBeenCalledOnce();
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
    rec.emitResult(['Milano centro', true]);
    expect(input().value).toBe('NDVI Milano 2023');
  });

  it('sending stops listening, and a late result does not refill the box', () => {
    installSpeechRecognition();
    render();
    click(mic()!);
    const rec = MockRecognition.last;
    rec.emitResult(['analizza Firenze', false]);
    pressEnter();

    expect(rec.abort).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(JSON.parse(String(fetchMock.mock.calls[0][1].body)).message).toBe('analizza Firenze');
    expect(input().value).toBe('');
    rec.emitResult(['analizza Firenze centro', true]);
    rec.emitEnd();
    expect(input().value).toBe('');
    expect(MockRecognition.instances).toHaveLength(1);
  });

  it('closing the chat stops listening', () => {
    installSpeechRecognition();
    render();
    click(mic()!);
    const rec = MockRecognition.last;
    click(q<HTMLButtonElement>('button[aria-label="Close chat"]')!);
    expect(rec.abort).toHaveBeenCalledOnce();
    expect(rec.detached).toBe(true);
  });

  it('unmounting stops listening', () => {
    installSpeechRecognition();
    render();
    click(mic()!);
    const rec = MockRecognition.last;
    act(() => root.unmount());
    expect(rec.abort).toHaveBeenCalledOnce();
    root = createRoot(document.createElement('div'));   // for afterEach
  });

  it('a denied permission shows a readable message', () => {
    installSpeechRecognition();
    render();
    pointerDown(mic()!);
    MockRecognition.last.emitError('not-allowed');
    pointerUp(mic()!);
    expect(mic()!.getAttribute('aria-pressed')).toBe('false');
    expect(status()).toMatch(/Microphone access was denied/);
    expect(MockRecognition.instances).toHaveLength(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('the language choice applies to the next press', () => {
    installSpeechRecognition();
    render();
    selectLanguage('en-US');
    expect(mic()!.getAttribute('aria-label')).toBe('Dictate in English');
    pointerDown(mic()!);
    expect(MockRecognition.last.lang).toBe('en-US');
    expect(status()).toMatch(/^Listening \(English\)\. Hold to talk, or release now for hands-free\./);
  });

  it('the microphone is a keyboard-operable button', () => {
    installSpeechRecognition();
    render();
    const button = mic()!;
    button.focus();
    expect(document.activeElement).toBe(button);
    expect(button.tagName).toBe('BUTTON');
    expect(button.disabled).toBe(false);
  });
});

describe('dictation status line', () => {
  it('names the mode and how to stop it', () => {
    expect(dictationStatus('pressed', false, 'en-US')).toMatch(/^Listening \(English\)\. Hold to talk, or release now for hands-free\./);
    expect(dictationStatus('hold', false, 'it-IT')).toMatch(/^Push-to-talk \(Italian\): listening while you hold\. Release to stop\./);
    expect(dictationStatus('toggle', true, 'it-IT')).toMatch(/^Hands-free \(Italian\)/);
    // a hold whose release may have gone to a permission prompt is described as hands-free
    expect(dictationStatus('hold', true, 'it-IT')).toMatch(/^Hands-free \(Italian\)/);
  });
});

describe('job status badge', () => {
  function badge(props: Parameters<typeof JobStatusBadge>[0]) {
    act(() => root.render(<JobStatusBadge {...props} />));
    return container.querySelector<HTMLElement>('[data-job-status]')!;
  }

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  it('shows the completion message with the coverage', () => {
    const el = badge({ status: 'completed', progress: 100,
      message: 'Completed: cloud-free data covers 99.9% of the requested area (4 acquisition days, 2026-07-03 to 2026-07-28).' });
    expect(el.dataset.jobStatus).toBe('completed');
    expect(el.textContent).toContain('Map updated');
    expect(el.textContent).toContain('covers 99.9% of the requested area');
  });

  it('flags a low-coverage result and shows its reason', () => {
    const el = badge({ status: 'completed', progress: 100, lowCoverage: true,
      message: 'Completed with partial coverage: cloud-free data covers only 5.3% of the requested area (1 acquisition day, 2026-11-13). Reason: ...' });
    expect(el.dataset.jobStatus).toBe('partial');
    expect(el.textContent).toContain('Map updated: partial coverage');
    expect(el.textContent).toContain('covers only 5.3%');
  });

  it('shows why a job failed', () => {
    const el = badge({ status: 'failed', progress: 30, message: 'No cloud-free pixels over the requested area: ...' });
    expect(el.textContent).toContain('Failed');
    expect(el.textContent).toContain('No cloud-free pixels');
  });

  it('shows progress messages only while running', () => {
    const el = badge({ status: 'processing', progress: 40, message: 'Loading 8 scenes...' });
    expect(el.textContent).toContain('40%');
    expect(el.textContent).toContain('Loading 8 scenes...');
  });
});
