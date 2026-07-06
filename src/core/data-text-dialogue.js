// Original dialogues for the text corpus: short conversations written as speaker lines, so the tiny GPT can learn
// the shape of a question and an answer. Written for Nooral. (Loaded before data-text-prose.js, which joins both.)
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});
  D.dialogues = [
    { title: 'At the bakery', lines: [
      ['Customer', 'Good morning. Is the brown bread fresh today?'], ['Baker', 'It came out of the oven an hour ago. It is still warm.'],
      ['Customer', 'Then I will take one loaf, please. And do you have any of the small rolls?'], ['Baker', 'Only six left. The early people took the rest.'],
      ['Customer', 'Six is fine. My children will eat them before I get home.'], ['Baker', 'That is the best thing that can happen to a roll.'],
      ['Customer', 'How much is that?'], ['Baker', 'Four for the loaf and three for the rolls. Seven altogether.'],
      ['Customer', 'Here you are. Thank you.'], ['Baker', 'Thank you. See you tomorrow.']] },
    { title: 'Directions', lines: [
      ['Visitor', 'Excuse me, how do I get to the museum?'], ['Local', 'Go straight on to the traffic lights, then turn left.'],
      ['Visitor', 'Left at the lights. Is it far after that?'], ['Local', 'About five minutes. You will see a big stone building with a clock.'],
      ['Visitor', 'Is that the museum?'], ['Local', 'No, that is the library. The museum is the small red building next to it.'],
      ['Visitor', 'The small red one. I would have walked straight past it.'], ['Local', 'Everyone does. That is why I always mention the clock.'],
      ['Visitor', 'Thank you very much.'], ['Local', 'You are welcome. Enjoy the museum.']] },
    { title: 'The broken window', lines: [
      ['Teacher', 'Who was playing football near the window?'], ['Pupil', 'I was, but I did not kick the ball that hit it.'],
      ['Teacher', 'Who did?'], ['Pupil', 'Nobody kicked it on purpose. It bounced off the wall.'],
      ['Teacher', 'And who kicked it at the wall?'], ['Pupil', 'I did. But the wall was supposed to stop it.'],
      ['Teacher', 'Walls are very good at stopping balls in one direction.'], ['Pupil', 'I know that now.'],
      ['Teacher', 'Then you can help the caretaker sweep up the glass, and next time play on the field.'], ['Pupil', 'Yes. Sorry.']] },
    { title: 'Planning a trip', lines: [
      ['Maya', 'Shall we go to the coast on Saturday?'], ['Jon', 'It depends on the weather. What does the forecast say?'],
      ['Maya', 'Sunny in the morning and rain after three.'], ['Jon', 'Then we should leave early and come back early.'],
      ['Maya', 'The first train is at seven.'], ['Jon', 'Seven is very early for a Saturday.'],
      ['Maya', 'Do you want sun or do you want sleep?'], ['Jon', 'I want both, but I will choose the sun.'],
      ['Maya', 'Good. I will bring the sandwiches.'], ['Jon', 'I will bring the umbrella, just in case the forecast is wrong.']] },
    { title: 'A new phone', lines: [
      ['Grandson', 'This is the button to answer a call. You slide it to the right.'], ['Grandfather', 'To the right. And to finish the call?'],
      ['Grandson', 'Press the red circle.'], ['Grandfather', 'Red to stop, like a traffic light. That I can remember.'],
      ['Grandson', 'And this is the camera. Point it and press the white circle.'], ['Grandfather', 'It took a picture of my thumb.'],
      ['Grandson', 'Hold it at the edges, not in the middle.'], ['Grandfather', 'Now it took a picture of you. A good one.'],
      ['Grandson', 'You see? You are better at this than you think.'], ['Grandfather', 'I had a good teacher.']] },
    { title: 'The library card', lines: [
      ['Child', 'Can I take out five books?'], ['Librarian', 'You can take out eight, if you can carry them.'],
      ['Child', 'Eight! When do I have to bring them back?'], ['Librarian', 'In three weeks. The date is printed on your ticket.'],
      ['Child', 'What if I have not finished them?'], ['Librarian', 'Then you can ask for three more weeks, unless someone else is waiting.'],
      ['Child', 'What if someone is waiting?'], ['Librarian', 'Then you read faster.'],
      ['Child', 'I will start tonight.'], ['Librarian', 'Good plan. Start with the shortest one.']] },
    { title: 'Fixing a leak', lines: [
      ['Tenant', 'There is water coming from under the sink.'], ['Plumber', 'Is it dripping or running?'],
      ['Tenant', 'Dripping, about one drop every second.'], ['Plumber', 'Then it is probably the joint on the pipe. Have you turned off the water?'],
      ['Tenant', 'I do not know where to turn it off.'], ['Plumber', 'There is a tap under the sink, on the left. Turn it to the right until it stops.'],
      ['Tenant', 'It is stopped. The drip has stopped too.'], ['Plumber', 'Good. Put a bowl under it, and I will come at four.'],
      ['Tenant', 'Thank you. Four is fine.'], ['Plumber', 'See you then.']] },
    { title: 'Choosing a cake', lines: [
      ['Father', 'Which cake would you like for your birthday?'], ['Daughter', 'Chocolate, with strawberries on top.'],
      ['Father', 'Last year you wanted lemon.'], ['Daughter', 'Last year I was seven.'],
      ['Father', 'That is true. How many candles this year?'], ['Daughter', 'Eight, and one extra for luck.'],
      ['Father', 'Nine candles on a small cake is a lot of fire.'], ['Daughter', 'Then make a big cake.'],
      ['Father', 'A big chocolate cake with strawberries and nine candles.'], ['Daughter', 'Yes. That is exactly right.']] },
  ];
  /** a dialogue as text: one "Speaker: line" per line */
  D.dialogueText = (d) => d.lines.map(([who, s]) => who + ': ' + s).join('\n');
})(globalThis.Nooral = globalThis.Nooral || {});
