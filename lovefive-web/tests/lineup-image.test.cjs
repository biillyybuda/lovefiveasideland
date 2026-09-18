require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { layoutLineupName, drawLineupImage } = require('../lib/lineup-image.ts');
const segmenter = new Intl.Segmenter('en', { granularity: 'grapheme' });
const measure = (text, size) => [...segmenter.segment(text)].length * size * 0.55;

test('portrait name layout preserves long names, accents and whole emoji without truncation', () => {
  for (const name of ['Bill', 'Sam K', 'Jean-Pierre O’Connor', 'Alexander Montgomery-Sutherland', 'A'.repeat(100), 'José 👨‍👩‍👧‍👦 '.repeat(6)]) {
    const layout = layoutLineupName(name, measure);
    assert.equal(layout.lines.join('').replace(/\s/g, ''), name.replace(/\s/g, ''));
    assert.ok(layout.size >= 40);
    assert.ok(layout.lines.every(line => measure(line, layout.size) <= 418));
    for (const line of layout.lines) assert.ok(!line.startsWith('\u200d') && !line.endsWith('\u200d'));
  }
});

test('export rejects incomplete teams and invalid dates or forecasts before drawing', () => {
  const data = { date: '2026-09-10', blue: ['Bill', 'Jacob', 'Sam K', 'Sug', 'Cam'], red: ['Tom D', 'Gav', 'Kelso', 'Glee', 'Jos'], forecast: { win: 0.745, draw: 0.062, loss: 0.193 } };
  const canvas = { getContext() { throw new Error('Unexpected drawing'); } };
  assert.throws(() => drawLineupImage(canvas, { ...data, blue: data.blue.slice(1) }), /teams of five/);
  assert.throws(() => drawLineupImage(canvas, { ...data, date: '2026-02-30' }), /valid match date/);
  assert.throws(() => drawLineupImage(canvas, { ...data, forecast: { ...data.forecast, win: NaN } }), /forecast/);
});
