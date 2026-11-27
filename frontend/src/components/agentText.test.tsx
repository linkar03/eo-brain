// @vitest-environment happy-dom
// The agent's replies: a small Markdown subset is rendered, everything else is text.
import { afterEach, describe, expect, it } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { AgentText } from './agentText';

let container: HTMLDivElement;
let root: Root;

function render(text: string) {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(<AgentText text={text} />));
  return container;
}

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe('AgentText', () => {
  it('renders **bold** without the asterisks', () => {
    const el = render('There are **27.52 hectares** at the alert level.');
    expect(el.querySelector('strong')?.textContent).toBe('27.52 hectares');
    expect(el.textContent).toBe('There are 27.52 hectares at the alert level.');
  });

  it('renders `code` as code', () => {
    const el = render('Job ID: `7d683fdd-804f`');
    expect(el.querySelector('code')?.textContent).toBe('7d683fdd-804f');
    expect(el.textContent).toBe('Job ID: 7d683fdd-804f');
  });

  it('turns "* " and "- " at the start of a line into bullets, and keeps the lines', () => {
    const el = render('Launched:\n* **Algorithm:** LISA\n- Window: 60 days');
    expect(el.textContent).toBe('Launched:\n• Algorithm: LISA\n• Window: 60 days');
    expect(el.querySelectorAll('strong')).toHaveLength(1);
  });

  it('leaves asterisks and hyphens that are not markup alone', () => {
    const text = 'NDVI = (NIR - red) / (NIR + red); 2 * 3 * 4; an open **marker';
    expect(render(text).textContent).toBe(text);
    expect(container.querySelector('strong')).toBeNull();
  });

  it('never interprets a reply as HTML', () => {
    const text = '<img src=x onerror="alert(1)"> and **<b>bold</b>**';
    const el = render(text);
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('b')).toBeNull();
    expect(el.textContent).toBe('<img src=x onerror="alert(1)"> and <b>bold</b>');
  });
});
