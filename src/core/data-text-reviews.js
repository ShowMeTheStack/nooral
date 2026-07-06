// A larger set of short reviews, made from templates with varied vocabulary (written for Nooral: places, things and
// wording are invented). Positive and negative reviews talk about the same places and the same aspects with the same
// words, and many need more than counting words: negations ("not bad at all", "could not fault it"), a turn after
// "but" or "although", a mixed review with an overall verdict, an expectation that was or was not met. So a
// bag-of-words model gets most of them but not all, and word order helps.
//   Data 'opinions': {texts, y (0 negative, 1 positive), kinds (the template of each), classes}, 300 of each label
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});

  // aspects: [noun, plural, positive words, negative words]
  const A = {
    food: ['the food', 1, ['fresh', 'tasty', 'delicious', 'warm', 'generous', 'full of flavour'], ['stale', 'bland', 'cold', 'greasy', 'tiny', 'tasteless']],
    soup: ['the soup', 0, ['hot', 'rich', 'delicious', 'thick and tasty', 'fresh'], ['watery', 'cold', 'salty', 'bland', 'lukewarm']],
    cakes: ['the cakes', 1, ['fresh', 'light', 'delicious', 'moist', 'lovely'], ['dry', 'stale', 'heavy', 'too sweet', 'crumbly']],
    coffee: ['the coffee', 0, ['strong', 'smooth', 'rich', 'hot', 'excellent'], ['weak', 'bitter', 'lukewarm', 'burnt', 'awful']],
    service: ['the service', 0, ['quick', 'friendly', 'attentive', 'polite', 'helpful', 'warm'], ['slow', 'rude', 'careless', 'unfriendly', 'muddled', 'cold']],
    staff: ['the staff', 1, ['kind', 'friendly', 'helpful', 'patient', 'cheerful'], ['rude', 'unhelpful', 'grumpy', 'careless', 'impatient']],
    prices: ['the prices', 1, ['fair', 'reasonable', 'low', 'honest', 'good value'], ['too high', 'unfair', 'silly', 'steep', 'a rip off']],
    room: ['the room', 0, ['clean', 'quiet', 'bright', 'spacious', 'comfortable', 'cosy'], ['dirty', 'noisy', 'dark', 'cramped', 'damp', 'shabby']],
    bed: ['the bed', 0, ['comfortable', 'soft', 'big and comfortable', 'warm'], ['hard', 'lumpy', 'creaky', 'too short', 'uncomfortable']],
    breakfast: ['breakfast', 0, ['generous', 'fresh', 'delicious', 'varied'], ['meagre', 'cold', 'stale', 'boring']],
    view: ['the view', 0, ['wonderful', 'stunning', 'lovely', 'beautiful'], ['dull', 'blocked', 'grim', 'ugly']],
    story: ['the story', 0, ['gripping', 'clever', 'moving', 'funny', 'original'], ['dull', 'confusing', 'predictable', 'silly', 'slow']],
    actors: ['the actors', 1, ['brilliant', 'convincing', 'charming', 'superb'], ['wooden', 'flat', 'unconvincing', 'awkward']],
    ending: ['the ending', 0, ['satisfying', 'surprising', 'perfect', 'touching'], ['rushed', 'disappointing', 'pointless', 'confusing']],
    music: ['the music', 0, ['beautiful', 'lively', 'catchy', 'lovely'], ['loud', 'annoying', 'tinny', 'boring']],
    delivery: ['the delivery', 0, ['fast', 'on time', 'early', 'quick'], ['late', 'slow', 'a week late', 'chaotic']],
    packaging: ['the packaging', 0, ['neat', 'sturdy', 'careful', 'tidy'], ['torn', 'soaked', 'crushed', 'flimsy']],
    repair: ['the repair', 0, ['quick', 'careful', 'solid', 'neat', 'perfect'], ['sloppy', 'slow', 'useless', 'half done', 'careless']],
    battery: ['the battery', 0, ['long lasting', 'reliable', 'strong'], ['weak', 'short lived', 'unreliable', 'useless']],
    build: ['the build', 0, ['solid', 'sturdy', 'well made', 'tough'], ['flimsy', 'cheap', 'loose', 'fragile']],
    fit: ['the fit', 0, ['comfortable', 'snug', 'just right', 'perfect'], ['tight', 'stiff', 'awkward', 'too loose']],
    sound: ['the sound', 0, ['clear', 'rich', 'crisp', 'full'], ['tinny', 'muffled', 'harsh', 'thin']],
    seats: ['the seats', 1, ['comfortable', 'roomy', 'soft'], ['cramped', 'hard', 'broken', 'uncomfortable']],
    teacher: ['the teacher', 0, ['patient', 'clear', 'encouraging', 'kind'], ['impatient', 'vague', 'dull', 'unprepared']],
  };
  // places: [name, aspects, present tense?]
  const P = [
    ['the corner cafe', ['food', 'coffee', 'service', 'prices', 'cakes', 'staff', 'music'], 0],
    ['the bakery on the hill', ['cakes', 'coffee', 'staff', 'prices'], 0],
    ['the noodle bar', ['food', 'soup', 'service', 'prices', 'staff'], 0],
    ['the harbour restaurant', ['food', 'service', 'prices', 'view', 'staff'], 0],
    ['the guest house', ['room', 'bed', 'breakfast', 'staff', 'view', 'prices'], 0],
    ['the hotel by the station', ['room', 'bed', 'breakfast', 'service', 'prices'], 0],
    ['the mountain hut', ['room', 'bed', 'food', 'view', 'staff'], 0],
    ['the new film', ['story', 'actors', 'ending', 'music'], 0],
    ['the school play', ['story', 'actors', 'music', 'seats'], 0],
    ['the concert in the park', ['music', 'sound', 'seats', 'prices'], 0],
    ['the online shop', ['delivery', 'packaging', 'service', 'prices'], 0],
    ['the bike shop', ['repair', 'staff', 'prices', 'service'], 0],
    ['the phone repair stall', ['repair', 'service', 'prices', 'staff'], 0],
    ['this kettle', ['build', 'prices'], 1],
    ['these headphones', ['sound', 'battery', 'fit', 'build', 'prices'], 1],
    ['these walking boots', ['fit', 'build', 'prices'], 1],
    ['this torch', ['battery', 'build', 'prices'], 1],
    ['the evening language class', ['teacher', 'prices', 'room'], 0],
    ['the swimming lessons', ['teacher', 'prices', 'staff'], 0],
  ];
  const INT = ['', '', 'really ', 'very ', 'quite ', 'so '];
  const CLOSE = [['I will be back.', 'Highly recommended.', 'Well worth it.', 'We left happy.', 'A real find.', 'Ten out of ten.'],
    ['I will not be back.', 'Avoid it.', 'Not worth the money.', 'We left disappointed.', 'What a shame.', 'Two out of ten.']];
  const VERDICT = [['it was a good visit', 'I liked it', 'I would go again', 'it was worth it', 'I am glad we went', 'it was a success'],
    ['I would not go again', 'it was a let down', 'it was not worth it', 'I was disappointed', 'I wish we had stayed home', 'it was a failure']];

  function generate(seed) {
    const r = D.rng(seed), pick = r.pick, cap = (s) => s[0].toUpperCase() + s.slice(1);
    const parts = () => {
      const [place, asp, present] = pick(P), a = A[pick(asp)], b = A[pick(asp.filter((k) => A[k] !== a).length ? asp.filter((k) => A[k] !== a) : asp)];
      const be = (x) => (present ? (x[1] ? 'are' : 'is') : x[1] ? 'were' : 'was');
      const clause = (x, pos, int = true) => { const w = pick(x[pos ? 2 : 3]); return x[0] + ' ' + be(x) + ' ' + (int && !w.includes(' ') ? pick(INT) : '') + w; };
      const pl = /^these |lessons$/.test(place);
      return { place, a, b, be, clause, present, is: pl ? 'are' : 'is', was: pl ? 'were' : 'was', has: pl ? 'have' : 'has', it: pl ? 'they' : 'it' };
    };
    // each template: (want positive?) -> text with that label
    const T = {
      plain: (pos, q) => cap(q.clause(q.a, pos)) + '. ' + (r() < 0.5 ? pick(CLOSE[pos ? 0 : 1]) : ''),
      two: (pos, q) => cap(q.clause(q.a, pos)) + ' and ' + q.clause(q.b, pos, false) + '.',
      negated: (pos, q) => cap(q.a[0]) + ' ' + q.be(q.a) + (pos ? ' not ' + pick(['', 'at all ', 'in the least ']) + pick(q.a[3]) + ', ' + pick(['quite the opposite.', 'not even a little.', 'whatever others say.', 'which was a relief.']) : ' not ' + pick(['', 'very ', 'exactly ', 'at all ']) + pick(q.a[2]) + '.'),
      but: (pos, q) => cap(q.clause(q.a, !pos)) + ', but ' + q.clause(q.b, pos) + (r() < 0.4 ? ', and that is what matters.' : '.'),
      although: (pos, q) => 'Although ' + q.clause(q.a, !pos, false) + ', ' + q.clause(q.b, pos) + '.',
      overall: (pos, q) => { const x = cap(q.clause(q.a, true)) + '. ' + cap(q.clause(q.b, false)) + '.'; return (r() < 0.5 ? x : cap(q.clause(q.b, false)) + '. ' + cap(q.clause(q.a, true)) + '.') + ' Overall, ' + pick(VERDICT[pos ? 0 : 1]) + '.'; },
      expect: (pos, q) => (pos ? pick(['I expected very little from ' + q.place + ', but ' + q.clause(q.a, true) + '.', 'I was not expecting much from ' + q.place + ', and it turned out to be a pleasant surprise.', 'Everyone said ' + q.place + ' ' + q.was + ' good, and they were right.'])
        : pick(['I expected a lot from ' + q.place + ' after all the praise, but ' + q.clause(q.a, false) + '.', 'Everyone said ' + q.place + ' ' + q.was + ' good. They were wrong.', 'I had high hopes for ' + q.place + ', and they were not met.'])),
      recommend: (pos, q) => (pos ? pick(['I would not hesitate to recommend ' + q.place + '.', 'I cannot recommend ' + q.place + ' enough.', 'Could not fault ' + q.place + ', ' + q.clause(q.a, true, false) + '.', 'Nothing to complain about: ' + q.clause(q.a, true, false) + '.'])
        : pick(['I would think twice before recommending ' + q.place + '.', 'I cannot recommend ' + q.place + '.', 'There is not much good to say about ' + q.place + '.', 'Nothing to praise: ' + q.clause(q.a, false, false) + '.'])),
      compare: (pos, q) => cap(q.place) + ' ' + (pos ? pick([q.is + ' better than ever', q.has + ' really improved', q.is + ' even better than last time', q.is + ' as good as people say']) : pick([q.is + ' not what ' + q.it + ' used to be', q.has + ' gone downhill', q.is + ' worse than last time', q.is + ' not as good as people say'])) + (r() < 0.6 ? ', and ' + q.clause(q.a, pos, false) + '.' : '.'),
      nothing: (pos, q) => (pos ? 'There was nothing wrong with ' + q.a[0] + ', and ' + q.clause(q.b, true, false) + '.' : 'There was nothing right about ' + q.a[0] + ', and ' + q.clause(q.b, false, false) + '.'),
    };
    const kinds = Object.keys(T), weight = { plain: 3, two: 2, negated: 2, but: 3, although: 2, overall: 3, expect: 1, recommend: 2, compare: 1, nothing: 1 };
    const bag = kinds.flatMap((k) => new Array(weight[k]).fill(k)), seen = new Set(), rows = [];
    for (const label of [1, 0]) {
      let got = 0, tries = 0;
      while (got < 300 && tries++ < 20000) {
        const kind = pick(bag), text = T[kind](label === 1, parts()).trim();
        if (seen.has(text)) continue;
        seen.add(text); rows.push([text, label, kind]); got++;
      }
    }
    return rows;
  }
  D.opinionRows = generate(20241);
  D.register('opinions', { title: 'Reviews with a turn', kind: 'text', source: 'Written for Nooral', licence: 'GPL-3.0-or-later (part of Nooral)',
    story: 'Short reviews of cafés, hotels, films, shops, repairs and things people bought. Both labels use the same places and words: negations, a turn after "but", mixed reviews with a verdict and unmet hopes make it harder than counting good and bad words.',
    rule: 'Made by ten sentence templates from invented places and aspects, each with its own positive and negative words; no review repeats. Rows shuffled by the seed.',
    size: D.opinionRows.length + ' reviews', features: ['text'], target: 'sentiment (0 negative, 1 positive)', balance: '300 positive, 300 negative' },
  (opt) => {
    const rows = D.rng(opt.seed).shuffle(D.opinionRows.slice());
    return { texts: rows.map((x) => x[0]), y: rows.map((x) => x[1]), kinds: rows.map((x) => x[2]), classes: ['negative', 'positive'], n: rows.length, target: { name: 'label', type: 'category' } };
  });
})(globalThis.Nooral = globalThis.Nooral || {});
