// Part 7, unsupervised learning: the narration of every topic read back against independent working-out.
// Usage: node tests/claims-p07.mjs [topic id prefix ...]
import { S, run } from './claims-lib-p06-08.mjs';
import { kmChecks } from './claims-p07-km.mjs';
import { moreChecks } from './claims-p07-more.mjs';

// formula constants the narration may write as they are
const ALLOWED = [/2\^/g, /90%/g, /PC\d\b/g, /n − 1/g, /−2 × /g, /= [01](?![.\d])/g, /[≥≤<>] [01](?![.\d])/g, /\b2 \/ /g, /½/g, /−1 and 1/g, /from −1 to 1/g, /\b1 − /g];

run('claims-p07', /^p07-/, Object.assign({}, kmChecks(S), moreChecks(S)), ALLOWED);
