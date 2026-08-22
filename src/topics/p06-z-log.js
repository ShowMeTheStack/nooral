// Part 6: every topic's run keeps the list of machines it fitted (data and parameters) in its result, so the claims
// check can fit the same machines with a reference library and compare. The pictures do not use it.
(function (S) {
  'use strict';
  const P = S.P06;
  for (const t of S.topics.filter((x) => /^p06-/.test(x.family))) {
    if (t.run.logged) continue;
    const run = t.run;
    t.run = function (inp) {
      P.log = [];
      try {
        const out = run.call(this, inp);
        const res = Array.isArray(out) ? { frames: out } : out;
        res.result = Object.assign({}, res.result || {}, { fits: P.log });
        return res;
      } finally { P.log = null; }
    };
    t.run.logged = true;
  }
})(globalThis.Nooral = globalThis.Nooral || {});
