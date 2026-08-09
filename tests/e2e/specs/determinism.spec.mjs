// The same numbers in the page as in node, on every load: a topic computed from stored weights (the part 10 saliency
// test), a network trained by a workflow in the page's worker, and the text lab's engine trained on the CPU. Each is
// computed here in node with the same engines and compared to the last bit.
import { assert } from '../lib.mjs';
import { load } from '../../load.mjs';

const S = load(['core', 'stage', 'topics', 'app', 'content']);
// the numbers the saliency topic narrates, from its own frames' meta
const SAL = `(() => { const t = Nooral.byId('cnn-saliency'); return t.presets.map((p) => { const f = t.run(p.values).frames.find((x) => x.meta && x.meta.phase === 'test').meta; return [f.s0, f.sA, f.sB]; }); })()`;
const salNode = (() => { const t = S.byId('cnn-saliency'); return t.presets.map((p) => { const f = t.run(p.values).frames.find((x) => x.meta && x.meta.phase === 'test').meta; return [f.s0, f.sA, f.sB]; }); })();
// a short run of the text lab's engine (the small size, on the CPU)
const TL_RUN = `(() => { const E = new Nooral.TL.Engine(); E.init({ size: 'small' }); const l = []; for (let i = 0; i < 12; i++) l.push(E.step()); return l; })()`;
const tlNode = (() => { const E = new S.TL.Engine(); E.init({ size: 'small' }); const l = []; for (let i = 0; i < 12; i++) l.push(E.step()); return l; })();

export default function (t) {
  t.test('the saliency topic gives the node numbers on two loads', async (p) => {
    for (let k = 0; k < 2; k++) {
      await p.route('topic=cnn-saliency');
      assert.deepEqual(await p.ev(SAL), salNode, 'load ' + (k + 1));
      await p.reload();
    }
  });

  t.test('a workflow trained in the worker scores as in node', async (p) => {
    const x = S.Content.challenges.get('ne-ch-spiral'), want = S.Challenges.evaluate(x.solution, x).score;
    await p.route('challenges');
    const got = await p.ev("(async () => { const x = Nooral.Content.challenges.get('ne-ch-spiral'); return (await Nooral.Challenges.evaluateAsync(x.solution, x)).score; })()");
    assert.equal(got, want);
  }, { timeout: 90000 });

  t.test('the text lab engine trains to the same losses as in node', async (p) => {
    await p.route('textlab/bpe');
    assert.deepEqual(await p.ev(TL_RUN), tlNode);
  }, { timeout: 90000 });
}
