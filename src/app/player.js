// A scene's player: the stage, the narration and the controls (first, back, play, step, a slider). A scene is
// {id, tag, title, blurb, knobs, frames({narrow, knobs}) -> [{items, say, view, narrowView}]}; the frames are built
// again when the stage crosses the phone width, since a scene may lay itself out differently there, and when a
// knob moves. knobs: [{id, label, min, max, step, value, fmt(v) -> text, log}] become sliders under the narration
// (log: the slider moves the exponent, base 10); knobsGo: 'last' shows the last frame after a knob moves.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const PHONE = 600;
  /** narration: `text in backticks` is a value, set in the monospace face */
  const sayNodes = (txt) => String(txt || '').split('`').map((part, i) => (i % 2 ? h('span.v', part) : document.createTextNode(part)));

  class Player {
    constructor(scene) {
      this.scene = scene;
      this.i = 0;
      this.timer = null;
      this.el = h('section.scene', { id: scene.id, dataset: { id: scene.id } });
      this.el.append(h('div.scene-head', h('span.tag', scene.tag), h('h2', scene.title), scene.blurb ? h('p', scene.blurb) : null));
      this.wrap = h('div.stage-wrap');
      this.el.append(this.wrap);
      this.stage = new S.Stage(this.wrap, { label: scene.title });
      this.sayEl = h('div.say', { 'aria-live': 'polite' });
      this.el.append(this.sayEl);
      const btn = (name, label, fn, cls) => { const b = h('button.icon-btn' + (cls ? '.' + cls : ''), { type: 'button', title: label, 'aria-label': label }, icon(name)); b.addEventListener('click', fn); return b; };
      this.bFirst = btn('first', 'Back to the start', () => { this.stop(); this.go(0); });
      this.bPrev = btn('prev', 'Step back', () => { this.stop(); this.go(this.i - 1); });
      this.bPlay = btn('play', 'Play', () => this.toggle(), 'play');
      this.bNext = btn('next', 'Step forward', () => { this.stop(); this.go(this.i + 1); });
      this.slider = h('input', { type: 'range', min: 0, max: 0, value: 0, 'aria-label': 'Step' });
      this.slider.addEventListener('input', () => { this.stop(); this.go(+this.slider.value); });
      this.pos = h('span.pos');
      this.el.append(h('div.controls', this.bFirst, this.bPrev, this.bPlay, this.bNext, this.slider, this.pos));
      this.knobs = {};
      if (scene.knobs && scene.knobs.length) {
        const row = h('div.scene-knobs');
        for (const k of scene.knobs) {
          this.knobs[k.id] = k.value;
          const fmt = k.fmt || ((v) => String(+(+v).toPrecision(3)));
          const out = h('output', fmt(k.value));
          const toPos = (v) => (k.log ? Math.log10(v) : v), toVal = (p) => (k.log ? Math.pow(10, p) : p);
          const inp = h('input', { type: 'range', min: toPos(k.min), max: toPos(k.max), step: k.step || (k.log ? 0.05 : 1), value: toPos(k.value), 'aria-label': k.label });
          inp.addEventListener('input', () => { const v = toVal(+inp.value); this.knobs[k.id] = v; out.textContent = fmt(v); this.stop(); this.rebuild(); });
          row.append(h('label', k.label, inp, out));
        }
        this.el.append(row);
      }
    }
    /** a knob moved: build the frames again, keeping the step */
    rebuild() {
      clearTimeout(this.rebuildT);
      this.rebuildT = setTimeout(() => { this.frames = this.scene.frames({ narrow: this.builtNarrow, knobs: Object.assign({}, this.knobs) }); this.slider.max = this.frames.length - 1; this.size(); this.go(this.scene.knobsGo === 'last' ? this.frames.length - 1 : Math.min(this.i, this.frames.length - 1)); }, 60);
    }
    get narrow() { return (this.wrap.clientWidth || innerWidth) < PHONE; }
    /** build the frames for the stage's width, size the stage to them and show the current one */
    build() {
      const narrow = this.narrow;
      this.frames = this.scene.frames({ narrow, knobs: Object.assign({}, this.knobs) });
      this.builtNarrow = narrow;
      this.slider.max = this.frames.length - 1;
      this.i = Math.min(this.i, this.frames.length - 1);
      this.stage.clear();
      this.size();
      this.go(this.i, true);
    }
    size() {
      const W = this.wrap.clientWidth || 800;
      let ratio = 0;
      for (const f of this.frames) { const v = (this.narrow && f.narrowView) || f.view; if (v) ratio = Math.max(ratio, v.h / v.w); }
      if (!ratio) ratio = 0.5;
      this.wrap.style.height = Math.round(Math.max(220, Math.min(W * ratio, 900))) + 'px';
    }
    resized() { if (this.narrow !== this.builtNarrow) this.build(); else { this.size(); this.stage.show(this.frames[this.i], { instant: true }); } }
    go(i, instant) {
      if (!this.frames) return;
      i = Math.max(0, Math.min(this.frames.length - 1, i));
      this.i = i;
      const f = this.frames[i];
      this.stage.show(f, { instant: instant || (S.app && S.app.settings.animate === false), duration: f.duration || 650 });
      this.sayEl.innerHTML = '';
      this.sayEl.append(h('span.step', 'Step ' + (i + 1) + ' of ' + this.frames.length + (f.title ? ' · ' + f.title : '')), ...sayNodes(f.say));
      this.slider.value = i;
      this.pos.textContent = i + 1 + ' / ' + this.frames.length;
      this.bPrev.disabled = this.bFirst.disabled = i === 0;
      this.bNext.disabled = i === this.frames.length - 1;
    }
    toggle() { if (this.timer) this.stop(); else this.play(); }
    play() {
      if (this.i >= this.frames.length - 1) this.go(0);
      const tick = () => { if (this.i >= this.frames.length - 1) { this.stop(); return; } this.go(this.i + 1); this.timer = setTimeout(tick, this.frames[this.i].hold || 2200); };
      this.timer = setTimeout(tick, 700);
      this.bPlay.replaceChildren(icon('pause')); this.bPlay.title = 'Pause'; this.bPlay.setAttribute('aria-label', 'Pause');
    }
    stop() {
      clearTimeout(this.timer); this.timer = null;
      this.bPlay.replaceChildren(icon('play')); this.bPlay.title = 'Play'; this.bPlay.setAttribute('aria-label', 'Play');
    }
  }
  S.Player = Player;
})(globalThis.Nooral = globalThis.Nooral || {});
