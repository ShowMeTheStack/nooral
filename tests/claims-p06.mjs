// Part 6, support vector machines: the narration of every topic read back against independent working-out.
// Usage: node tests/claims-p06.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { svmChecks } from './claims-p06-svm.mjs';

// formula constants that the narration may write as they are: the margins f = ±1, a street of 2 / ‖w‖, x²
// and the constants 0 and 1 of the margin conditions (m ≥ 1, 0 < α < C, 1 − m)
const ALLOWED = [/±1/g, /[+−]1(?![.,]?\d)/g, /(?<![\d.,−-])[01](?![.,]?\d)/g, /\b2 \/ /g, /[xyz]²/g, /½/g, /\b[23]-D\b/g, /√2/g];

run('claims-p06', /^p06-/, svmChecks(S), ALLOWED);
export { phase };
