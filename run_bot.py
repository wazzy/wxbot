from wxpy import *
from agent import *


bot = Bot()
friends = bot.friends()


agent = PairMatchAgent()


@bot.register(friends)
def reply_my_friend(msg):
    if msg.text.startswith('debug:'):
        text = msg.text[6:]
        debug = True
    else:
        text = msg.text
        debug = False
    replys = agent.reply(text)
    certainty = replys[0]
    question, answer = replys[1]

    if debug:
        msg.chat.send('certainty: %s'%certainty)
        msg.chat.send('matched question: '+question)
        msg.chat.send('answer: ')
    else:
        msg.chat.send('msg: '+msg.text)
        msg.chat.send('debug: %s' % msg.text.startswith('debug:'))
    for line in answer:
        msg.chat.send(line)


embed()
