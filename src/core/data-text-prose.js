// An original prose and dialogue corpus for word embeddings, tokenizers and the tiny GPT. Every passage below
// was written for Nooral: small scenes in plain modern English, with simple, repeated vocabulary so that a small
// model trained in the browser can learn something visible from it.
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});
  D.prose = [
    { title: 'The night market', text: `Every Friday the car park behind the station turns into a night market. At six o'clock the first vans arrive, and by seven there are forty stalls under strings of yellow lights. There is a man who sells hot noodles from a pan as wide as a cart wheel. There is a woman who sells old radios, and she will tell you the story of each one if you stand still long enough. There is a stall of oranges, a stall of socks, a stall of knives that are sharpened while you wait, and a stall that sells nothing but buttons.

Lena goes every week with her grandfather. He walks slowly and stops at every stall, not to buy but to talk. He asks the noodle man about his daughter. He asks the radio woman whether the old blue set has been sold yet. It has not. It never is. Lena thinks the radio woman keeps it on the table only so that her grandfather has something to ask about.

"Why do you talk to everyone?" Lena asks him one night.

"Because they talk back," he says. "That is the whole secret of a market. The noodles are good, but the talking is better."

At ten o'clock the lights go off one string at a time. The vans fill up again. By eleven the car park is empty and grey, and you would never know that a small town had lived there for five hours.` },
    { title: 'The repair shop', text: `Tomas fixes bicycles in a shop the size of a kitchen. The walls are covered with wheels, and the floor is covered with a layer of black grease that no broom has ever beaten. He works with the radio on and a cup of tea going cold on the bench.

A girl came in on Monday with a bicycle that would not change gear. She had ridden it up the hill in the highest gear and her legs were still shaking.

"Can you fix it today?" she asked.

"I can look at it today," said Tomas. "Fixing is a different promise."

He turned the bicycle upside down and spun the back wheel. The chain jumped and clicked. He listened to it the way a doctor listens to a cough. Then he found the problem: a thin cable had stretched, so the gear lever pulled but the gears did not move. He loosened a small screw, pulled the cable tight and turned a little barrel at the end of it, half a turn, then another quarter. He spun the wheel again. The chain moved from gear to gear as smoothly as water.

"That is all?" said the girl.

"That is all today," said Tomas. "Next month the cable will stretch again, and you will come back, and I will do it again. Bicycles are like people. They need a little attention often, not a lot of attention once."

She paid him, and on the way out she rode up the hill in the lowest gear, and she did not have to stand on the pedals at all.` },
    { title: 'The weather station', text: `On the roof of the school there is a white box on legs. Inside it are thermometers, and next to it a little cup spins in the wind, and a funnel catches the rain. The science class reads the instruments every morning at nine and writes the numbers in a big book.

For three years nobody did anything with the numbers. Then a new teacher arrived and asked a simple question: can we guess tomorrow's weather from today's? The class made a table. One column was today's temperature, one was the wind, one was the pressure, one was whether it rained, and the last column was whether it rained the next day.

They tried a rule: if the pressure falls, it will rain tomorrow. The rule was right on most days, but it was wrong in a way that annoyed them. It missed the summer storms, which came on hot still afternoons when the pressure had hardly changed. So they added a second rule for hot days with no wind. Now the rules were right more often, but there were two of them, and they argued with each other on some days.

"This is what forecasting is," said the teacher. "A rule that is simple enough to understand and good enough to use. Then you check it, every day, against what really happens."

The class still reads the instruments at nine. But now, before they write the numbers down, someone always says what they think tomorrow will bring, and the book has a new column for whether they were right.` },
    { title: 'Two neighbours', text: `"Your hedge is growing over my fence again," said Mrs Okafor.

"It is not my hedge," said Mr Hale. "It was here before I was."

"It grows out of your garden."

"It grows out of the ground. The ground was here before both of us."

Mrs Okafor put down her basket. "Then the ground can cut it. Or you can. I would like to see the sun on my tomatoes before the end of the summer."

Mr Hale looked at the hedge. It was very tall. It had small white flowers in the spring and hard black berries in the autumn, and the birds lived in it all year round. He did not want to cut it. He also did not want to argue for another summer.

"What if we cut the top only?" he said. "Down to the height of the fence. The birds can keep the bottom."

"And who will carry the branches away?"

"I will cut. You will tell me when it is low enough. We will both carry."

They worked all Saturday morning. By lunch the hedge was the height of the fence, the tomatoes had sun, and there was a pile of branches as tall as a car. Mrs Okafor brought out two glasses of lemonade. They drank them standing on either side of the fence, and they found that they could see each other properly for the first time in years.` },
    { title: 'Learning to swim', text: `Priya was nine and could not swim. Her little brother could, and her best friend could, and that made it worse. Every Saturday her mother took her to the pool, and every Saturday she held on to the edge and kicked and did not let go.

The new coach was a tall woman with a voice that could be heard over the whole pool without shouting. She did not ask Priya to let go of the edge. She asked her to put her face in the water and blow bubbles, and count how long she could blow.

"Four," said Priya.

"Next week, five," said the coach.

The week after, it was six. Then the coach asked her to float on her back while she held her up with one hand under her shoulders. Priya looked at the roof of the pool, at the long white lights, and listened to the strange quiet of her ears under the water. "I am going to move my hand a little," said the coach. "You will not notice." Priya did not notice. When she sat up, the coach was standing two steps away with her hands behind her back.

"You were floating by yourself," she said. "For about ten seconds."

Priya did not believe her. Then she tried again, and this time she knew, and she stayed up for twenty. By the end of the summer she could swim a whole length. It was slow and it was not pretty, but she did not hold on to anything at all.` },
    { title: 'The lost dog', text: `The poster on the lamp post said LOST DOG in large letters, with a photo of a small brown dog with one white ear. Her name was Pepper, and she had run away from the park on Sunday afternoon. Underneath there was a phone number and the words: she is friendly but she is scared of bicycles.

Sam saw the poster on his way to school. He saw it again on his way home, and on the way home he also saw a small brown dog with one white ear sitting under the bench outside the library.

He did not run at her. He sat down on the bench, very slowly, and took out the end of his sandwich. The dog watched him. He put a piece of bread on the ground near his foot and looked the other way, at the road, as if he did not care about dogs at all. After a while he heard her move. After a longer while he felt her nose against his hand.

"Hello, Pepper," he said quietly.

Her tail moved, once. He called the number on the poster, and a woman answered on the first ring as if she had been holding the phone all day. Ten minutes later she came running along the street, and Pepper ran to meet her, and for a minute nobody said anything at all.

"How did you catch her?" the woman asked at last.

"I didn't," said Sam. "I just sat still until she decided to come."` },
    { title: 'The kitchen robot', text: `"I would like a cup of tea," said Grandmother.

The kitchen robot rolled to the cupboard, took out a cup, filled the kettle and waited. When the water boiled it poured it into the cup, and then it stopped.

"Is something wrong?" asked Grandmother.

"I have hot water in a cup," said the robot. "You asked for tea. I do not know where the tea is."

"In the tin with the red lid."

The robot found the tin with the red lid, took out a tea bag and put it in the cup. Then it stopped again.

"How long should it stay in the water?"

"Three minutes. And then milk, but only a little."

"How much is a little?"

Grandmother laughed. "Enough to make it the colour of a biscuit."

The robot waited three minutes exactly. It took out the bag. It poured the milk one drop at a time and compared the colour with a biscuit from the jar after every drop, until the two were the same. Then it brought the cup to the table.

Grandmother tasted it. "That is a very good cup of tea," she said.

"I have written down the rule," said the robot. "Tin with the red lid, three minutes, milk to the colour of a biscuit. Tomorrow I will not need to ask."

"Tomorrow," said Grandmother, "I might want coffee."

The robot was quiet for a moment. "Then tomorrow," it said, "I will ask again."` },
    { title: 'The river path', text: `There is a path along the river that goes from the old mill to the bridge and back. It is exactly four kilometres, and on most mornings a few people run it. They do not know each other's names, but they know each other's faces, and they nod as they pass.

There is the tall man in the green hat who runs very fast and always looks surprised to see anyone else. There are the two sisters who run side by side and talk the whole way, so that you hear them before you see them. There is the old man who walks, not runs, with a stick that he swings in time with his steps. And there is Ruth, who started running in January because she wanted to be able to climb the stairs at work without stopping.

In January she could run from the mill to the first bench, and then she walked. In February she could run to the second bench. In March, to the bridge. In April she ran the whole path for the first time, and when she got back to the mill the old man with the stick lifted it in the air like a flag.

"I have been counting your benches," he said. "That is all of them now."

"I did not know anyone was watching," said Ruth.

"Everybody watches," he said. "That is what the path is for."` },
    { title: 'The garden in the yard', text: `The yard behind the flats was concrete, with two washing lines and a broken chair. Nobody went there except to hang up sheets. Then one spring Ada, who lived on the third floor, carried down a bag of soil and an old sink and planted beans in it.

The beans grew up the washing line pole. The next week a man from the ground floor brought a box of tomato plants he had grown on his windowsill. A boy from the second floor painted the old chair yellow. Somebody left a watering can by the door with a note that said: for everyone.

By the summer there were twelve pots and four sinks and a bath full of potatoes. There were sunflowers taller than the boy. People who had lived in the same building for years without speaking now stood in the yard in the evening and argued about whether the tomatoes needed more water. They usually did.

In the autumn they had a meal in the yard with everything they had grown. It was not a large meal. There were more people than potatoes. But they put two tables together and brought chairs from their kitchens, and they stayed until it was dark and somebody had to fetch a lamp.

"Next year," said Ada, "we will plant more potatoes."

"Next year," said the man from the ground floor, "we will need a bigger yard."` },
    { title: 'Counting birds', text: `Once a year, on the first Sunday in May, the village counts its birds. Everyone who wants to help gets a card with pictures of thirty common birds and a pencil, and is sent to a different place: the church yard, the pond, the school field, the long hedge by the farm. For one hour you write down every bird you see and how many.

Mina was given the pond with her uncle. In the first ten minutes they counted two ducks, one heron standing as still as a post, and a lot of small brown birds that moved too fast to name.

"Write down what you are sure of," said her uncle. "If you are not sure, write that too. A wrong guess is worse than an honest question mark."

So Mina wrote: small brown bird, maybe a sparrow, five. Then the heron flew, very slowly, with its long legs hanging behind it, and she forgot to write anything at all because she was watching.

At the end of the day all the cards were collected in the village hall, and a woman with a calculator added them up. There were fewer swallows than last year and more pigeons. There were three herons, which was a record. And there were forty-one small brown birds marked with a question mark.

"That is a useful number too," said the woman with the calculator. "It tells us how much we still do not know."` },
  ];
  /** the hand-written stories, then the dialogues, separated by blank lines (the short corpus, about 17K characters) */
  D.proseText = () => [...D.prose.map((p) => p.text), ...(D.dialogues || []).map(D.dialogueText)].join('\n\n');
  /**
   * every piece of the long corpus (about 175K characters: these stories and dialogues and the parts in
   * data-text-more-*.js), stories and dialogues interleaved evenly so that any tenth of it mixes both
   */
  D.corpusPieces = () => {
    const st = [...D.prose, ...(D.moreProse || [])].map((p) => ({ title: p.title, text: p.text }));
    const dl = [...(D.dialogues || []), ...(D.moreDialogues || [])].map((d) => ({ title: d.title, text: D.dialogueText(d) }));
    const out = [];
    let i = 0, j = 0;
    while (i < st.length || j < dl.length) {
      if (j >= dl.length || (i < st.length && i * dl.length <= j * st.length)) out.push(st[i++]); else out.push(dl[j++]);
    }
    return out;
  };
  D.corpusText = () => D.corpusPieces().map((p) => p.text).join('\n\n');
  const words = (t) => t.split(/\s+/).filter(Boolean).length;
  D.register('stories', { title: 'Stories and dialogues', kind: 'text', story: 'Short scenes of everyday life in a small town and short conversations, written for Nooral, for training word embeddings, tokenizers and the tiny GPT.', rule: 'Hand-written text; no generator. Dialogues are written one "Speaker: line" per line; stories and dialogues alternate.', source: 'Written for Nooral', licence: 'GPL-3.0-or-later (part of Nooral)',
    size: D.corpusPieces().length + ' stories and dialogues, about ' + Math.round(words(D.corpusText()) / 1000) + ' thousand words', features: ['text'], target: 'none (next token)', balance: 'not a classification set' },
  () => { const ps = D.corpusPieces(); return { texts: ps.map((p) => p.text), titles: ps.map((p) => p.title), text: D.corpusText(), short: D.proseText() }; });
})(globalThis.Nooral = globalThis.Nooral || {});
