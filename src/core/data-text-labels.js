// Small labelled text sets written for Nooral: short messages marked spam or not, short reviews marked positive
// or negative, and sentences labelled by topic. Names, shops and places in them are invented.
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});
  const own = { source: 'Written for Nooral', licence: 'GPL-3.0-or-later (part of Nooral)', kind: 'text' };

  const SPAM = `Congratulations! You have been selected to receive a free holiday. Reply YES to claim now.
URGENT: your account will be closed today unless you confirm your details at the link below.
You are our lucky winner this week! Claim your cash prize before midnight.
Lose ten kilos in ten days with this one simple trick. Order now and get a second box free.
Final notice: a parcel is waiting for you. Pay the small delivery fee to release it.
Earn money from home! No experience needed. Thousands per week guaranteed.
Hot deal: designer watches at 90% off, today only, click to buy.
You have won a brand new phone. To collect, send your address and a small handling fee.
Cheap loans approved in minutes, no credit check, apply now!
Your mobile number has won the monthly draw. Call this number to claim.
Exclusive offer just for you: free gift card worth 500 when you sign up today.
Act fast! Limited places left in our money-making programme. Join now.
Dear customer, we noticed unusual activity. Verify your password immediately.
WIN WIN WIN! Text PRIZE now to enter the free draw for a new car.
Get rich quick with our secret investment plan. Double your money in a week.
Free entry to a competition for concert tickets. Text GO to join, standard rates apply.
Your subscription has expired. Renew now to avoid losing all your photos forever.
Miracle cream removes all wrinkles overnight. Buy three, get three free.
Congratulations, your email was chosen! Click here to receive your reward.
Last chance! Your free trial ends today, claim your bonus credits now.
Hello friend, I have an amazing business offer for you, reply for details.
You are pre-approved for a credit card with no fees ever. Apply in one click.
Hurry, only two hours left to claim your free voucher!
We tried to deliver your prize. Confirm your bank details to receive it.
Big savings on medicines, no prescription needed, order online today.
Make 300 a day answering simple surveys. Sign up free now.
Special promotion: buy one get two free on all items, today only!
Claim your tax refund now by entering your card number on our secure page.
Unlock the secret the experts do not want you to know. Click now.
Your phone may be infected. Download this cleaner immediately.
Free ringtones for your phone, text TONE to subscribe now.
You have unclaimed rewards waiting, log in now before they expire.
Meet singles in your area tonight, join for free.
Cash is waiting for you. Reply with your account number to transfer.
Exclusive VIP access to our sale, members save up to 80 percent today.
Win a dream holiday for two, just answer one easy question.
Instant approval! Borrow up to 5000 today with no questions asked.
Your package could not be delivered, click to reschedule and pay the fee.
Amazing weight loss pills, results guaranteed or your money back.
Congratulations, you qualify for a special government grant. Apply now.`;
  const HAM = `Are we still meeting for lunch at one?
Can you pick up some milk on your way home?
The meeting has moved to Thursday at ten, same room.
Happy birthday! Hope you have a lovely day.
I left my umbrella at your place, I will get it tomorrow.
Thanks for the help with the move, I owe you dinner.
The train is late again, I will be there in twenty minutes.
Did you finish the report? I can read it tonight if you want.
Mum says dinner is at seven, don't be late.
Can you send me the photos from the weekend?
The plumber is coming at four, can someone be home?
Good luck with your exam today, you will do great.
I found your keys, they were in the car.
Shall we take the dogs to the park after work?
The library book is due back on Friday.
Do you want to watch a film tonight or are you too tired?
I'm at the shop, do we need bread or eggs?
Your appointment with the dentist is on Monday at nine.
The match was cancelled because of the rain.
Thanks for the lovely evening, the soup was delicious.
I'll call you when I get to the station.
Please remember to water the plants while I'm away.
The bus timetable changes next week, check the new times.
Could you look at my bike? The brakes are making a noise.
We are out of coffee, I'll buy some tomorrow.
Great news, I got the job! Let's celebrate on Saturday.
The kids' school play starts at six, save me a seat.
Is the heating working again? It was cold this morning.
Can I borrow your ladder this weekend?
The cat has been sick, I'm taking her to the vet.
Lunch is in the fridge, just warm it up for two minutes.
I sent you the address of the restaurant, see you there.
Your parcel arrived, I put it in the hall.
Let's go for a walk by the river on Sunday.
My phone battery is low, I will text you later.
The class tomorrow is in room twelve instead of room four.
Did you remember to lock the back door?
I have two tickets for the concert, do you want one?
The garden looks lovely after the rain.
See you at the station at half past eight.
Can you bring a salad to the picnic?
I'm running a little late, start without me.
How was the doctor? Call me when you can.
The new neighbours seem nice, they brought us cake.
Please send me the recipe for those biscuits.
Football practice is cancelled tonight.
Your sister called, she will ring again after dinner.
The car is ready to collect from the garage.
I finished the book, you can have it back.
Thank you for the flowers, they made my day.`;
  const POS = `The soup was hot, fresh and full of flavour. I will come back.
Lovely staff and a very clean room. We slept really well.
The bike was fixed in an hour and it rides like new.
Great film, funny and warm, with a perfect ending.
The delivery came a day early and everything was well packed.
Best coffee in town, and the cakes are wonderful too.
The teacher explained everything clearly and patiently.
I love this jacket. It is warm, light and fits perfectly.
A beautiful walk with amazing views from the top.
The app is simple to use and does exactly what it says.
Friendly service and very fair prices. Highly recommended.
The concert was brilliant, the band played for two hours.
Our guide was excellent and knew so much about the city.
The bread is always fresh and the bakers are so kind.
Comfortable seats, quiet carriage and the train was on time.
The garden centre had everything I needed and gave good advice.
My daughter loved the museum, especially the dinosaur room.
Fast, polite and careful movers. Nothing was broken.
The pizza was crispy and generous. Great value.
This kettle boils quickly and looks lovely in the kitchen.
The hotel breakfast was delicious, with lots of choice.
A clever, exciting book that I could not put down.
The doctor listened carefully and really helped me.
Very happy with my new shoes, comfortable from the first day.
The swimming lessons gave my son so much confidence.
Wonderful little shop with helpful, cheerful staff.
The repair was quick and cheaper than I expected.
Bright, spacious flat in a quiet street. We loved it.
The play was moving and the actors were superb.
Easy to set up, works perfectly, and the battery lasts for days.`;
  const NEG = `The soup was cold and far too salty. Very disappointing.
Rude staff and a dirty room. We will not stay again.
The bike broke again a week after the repair.
Boring film, far too long, and the ending made no sense.
The delivery was late and the box was crushed.
Weak coffee and stale cakes. Not worth the price.
The teacher rushed through everything and ignored our questions.
The jacket fell apart after two washes. Poor quality.
The path was muddy, badly marked and we got lost.
The app keeps crashing and deletes my notes.
Slow service and overpriced food. Would not recommend.
The concert started an hour late and the sound was awful.
Our guide was late, bored and knew very little.
The bread was hard and the staff seemed annoyed with us.
Crowded, noisy train and it was forty minutes late.
The shop had nothing in stock and nobody could help.
The museum was closed when we arrived, despite the website.
The movers were careless and scratched our table.
The pizza was burnt underneath and arrived cold.
This kettle leaks and the handle gets far too hot.
Breakfast was tiny and the eggs were cold.
A dull, confusing book. I gave up halfway.
The doctor barely looked at me and I left with no answers.
These shoes gave me blisters on the first day.
The lessons were cancelled three times without warning.
Unfriendly shop and the prices are much too high.
The repair took three weeks and cost twice the quote.
Dark, damp flat with a broken heater. Avoid.
The play was badly acted and impossible to follow.
Hard to set up, the instructions are wrong and the battery dies in an hour.`;
  const TOPIC = {
    weather: `Heavy rain is expected this afternoon with strong winds along the coast.
Tomorrow will be sunny and warm, with a light breeze.
Frost is likely tonight, so cover any young plants.
A thick fog covered the valley until late morning.
The temperature dropped by ten degrees in an hour.
Snow fell on the hills overnight and the roads are icy.
It has not rained for three weeks and the grass is brown.
Thunderstorms may bring hail in the evening.
The wind has turned and the clouds are moving in from the west.
After the storm the sky cleared and a rainbow appeared.`,
    food: `Fry the onions slowly until they are soft and golden.
The soup needs more salt and a squeeze of lemon.
We baked bread this morning and the kitchen smells wonderful.
Add the rice, stir once, and let it simmer for twenty minutes.
These apples are sweet enough to eat without sugar.
The curry was spicy but the yoghurt cooled it down.
Whisk the eggs with a little milk before you pour them in.
Fresh tomatoes and basil make the simplest pasta sauce.
The cake sank in the middle because we opened the oven too early.
Roast the potatoes until they are crisp on the outside.`,
    sport: `The home team scored twice in the last ten minutes to win.
She ran the race in a new personal best time.
The goalkeeper saved a penalty in the final minute.
After a long rally he won the point with a perfect drop shot.
The swimmers dived in as soon as the whistle blew.
Our club lost the match but played much better than last week.
The cyclists climbed the mountain in heavy rain.
He trains every morning before school, even in winter.
The referee showed a yellow card for a late tackle.
The relay team dropped the baton on the final bend.`,
    travel: `Our flight was delayed, so we spent three hours at the airport.
The ferry crosses to the island twice a day in summer.
We rented bicycles and rode along the old canal.
The night train arrives in the capital just after dawn.
Remember to bring your passport and a copy of the booking.
The hostel was small but close to the station.
We missed the last bus and had to walk back to the village.
From the top of the tower you can see the whole harbour.
The road over the pass is closed until the snow melts.
We stayed in a cabin by the lake for a week.`,
  };
  const lines = (s) => s.split('\n').map((l) => l.trim()).filter(Boolean);
  D.textSets = {
    spam: [...lines(SPAM).map((t) => [t, 1]), ...lines(HAM).map((t) => [t, 0])],
    sentiment: [...lines(POS).map((t) => [t, 1]), ...lines(NEG).map((t) => [t, 0])],
    topics: Object.keys(TOPIC).flatMap((k, c) => lines(TOPIC[k]).map((t) => [t, c])),
  };
  const make = (key, classes) => (opt) => {
    const rows = D.rng(opt.seed).shuffle(D.textSets[key].slice());
    return { texts: rows.map((r) => r[0]), y: rows.map((r) => r[1]), classes, n: rows.length, target: { name: 'label', type: 'category' } };
  };
  const count = (key, c) => D.textSets[key].filter((r) => r[1] === c).length;
  D.register('spam', Object.assign({ title: 'Spam or not', story: 'Short phone and email messages: everyday notes between friends, family and colleagues, and unwanted offers, fake prizes and fake warnings (spam).', rule: 'Hand-written; rows shuffled by the seed.', size: D.textSets.spam.length + ' messages', features: ['text'], target: 'spam (0 not spam, 1 spam)', balance: count('spam', 1) + ' spam, ' + count('spam', 0) + ' not spam' }, own), make('spam', ['not spam', 'spam']));
  D.register('reviews', Object.assign({ title: 'Short reviews', story: 'One- or two-sentence reviews of cafés, shops, films, repairs and trips; each positive review has a negative twin about the same kind of thing.', rule: 'Hand-written in matched pairs; rows shuffled by the seed.', size: D.textSets.sentiment.length + ' reviews', features: ['text'], target: 'sentiment (0 negative, 1 positive)', balance: count('sentiment', 1) + ' positive, ' + count('sentiment', 0) + ' negative' }, own), make('sentiment', ['negative', 'positive']));
  D.register('topics', Object.assign({ title: 'Sentence topics', story: 'Sentences about the weather, food, sport and travel.', rule: 'Hand-written; rows shuffled by the seed.', size: D.textSets.topics.length + ' sentences', features: ['text'], target: 'topic (weather, food, sport, travel)', balance: '10 sentences per topic' }, own), make('topics', Object.keys(TOPIC)));
})(globalThis.Nooral = globalThis.Nooral || {});
