// Reference fits for the claims check of part 6: every machine the part-6 topics fit with their default input, their
// presets and the extra inputs the check lists is written to tests/ref/p06/jobs.json, fitted again by scikit-learn
// (tests/ref/gen_p06.py, in the local test environment) and stored in tests/ref/p06/sk.json.
// Usage: node tests/ref/gen_p06.mjs   (needs ~/.nooral-venv with scikit-learn)
import { writeFileSync, unlinkSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import { S } from '../claims-lib-p06-08.mjs';
import { svmChecks } from '../claims-p06-svm.mjs';

const here = dirname(fileURLToPath(import.meta.url)), checks = svmChecks(S), jobs = [];
for (const t of S.topics.filter((x) => /^p06-/.test(x.family))) {
  const inputs = [{}].concat((t.presets || []).map((p) => p.values), (checks[t.id] || {}).extra || []);
  for (const v of inputs) {
    const input = S.inputOf(t, Object.assign(S.inputOf(t), v));
    if (S.checkInput(t, input)) continue;
    const r = S.run(t, input);
    (r.result.fits || []).forEach((f, k) => jobs.push(Object.assign({ id: t.id + '|' + JSON.stringify(input) + '|' + k, kind: f.kind, X: f.X, y: f.y, params: f.params }, f.Xe ? { Xe: f.Xe } : {})));
  }
}
writeFileSync(join(here, 'p06', 'jobs.json'), JSON.stringify(jobs));
execFileSync(join(homedir(), '.nooral-venv', 'bin', 'python'), [join(here, 'gen_p06.py')], { stdio: 'inherit' });
unlinkSync(join(here, 'p06', 'jobs.json'));
console.log(jobs.length + ' fits written to tests/ref/p06/sk.json');
