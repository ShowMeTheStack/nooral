// Token strips on the stage: a text cut into tokens, each token a cell sized to its text (spaces shown as ·),
// laid out in lines that wrap at a width, with the token's id or a value under it when asked.
//   S.Kit.tokens.strip({prefix, x, y, width, tokens: [text..], cls(i) -> extra classes, sub(i) -> text, size,
//                       h, gap, lineGap, minW, keyOf(i) -> a stable key}) -> {items, h, boxes: [{x, y, w, h}..]}
//   S.Kit.tokens.show(t) -> the token as drawn: a space becomes ·, a line break ⏎
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const TK = (K.tokens = {});
  const r2 = (v) => Math.round(v * 100) / 100;
  TK.show = (t) => String(t).replace(/ /g, '·').replace(/\n/g, '⏎');

  TK.strip = function (o) {
    const P = o.prefix || 'tk', fs = o.size || 15, h = o.h || fs + 14, gap = o.gap === undefined ? 4 : o.gap, lineGap = o.lineGap || (o.sub ? 20 : 8);
    const items = [], boxes = [];
    let x = o.x, y = o.y;
    o.tokens.forEach((t, i) => {
      const w = Math.max(o.minW || 18, S.geom.textWidth(TK.show(t), fs, { mono: true }) + 12);
      if (x + w > o.x + o.width && x > o.x) { x = o.x; y += h + lineGap; }
      const it = { key: P + ':' + (o.keyOf ? o.keyOf(i) : i), kind: 'cell', x: r2(x), y: r2(y), w: r2(w), h, rx: 5, text: TK.show(t), size: fs, cls: 'tok mono ' + (o.cls ? o.cls(i) || '' : '') };
      if (o.sub) { const s = o.sub(i); if (s !== undefined && s !== null && s !== '') it.sub = String(s); }
      if (o.enter) it.enter = o.enter;
      items.push(it);
      boxes.push({ x, y, w, h });
      x += w + gap;
    });
    return { items, h: y + h - o.y + (o.sub ? 16 : 0), boxes };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
