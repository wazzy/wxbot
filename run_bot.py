from wxpy import *
from agent import *


bot = Bot(console_qr=True)
friends = bot.friends()


agent = PairMatchAgent()
turing = Tuling(api_key='25da5f49d4ad44d48a477543f0b3f55e')

@bot.register(friends)
def reply_my_friend(msg):
    text = msg.text
    # print('text received: '+text)
    if text.startswith('小姐姐'):
        print('chitchat mode on')
        turing.do_reply(msg)
        return

    run = False

    if text.startswith('debug:'):
        text = text[6:]
        print('debug mode on')
        debug = True
    else:
        # print('debug mode off')
        debug = False

    if debug:
        run = True

    if text.startswith('小助手'):
        print('assistant mode on')
        text = text.split('小助手')[-1].strip()
        run = True

    if run:
        print('running')
        print('text: '+text)
        replys = agent.reply(text)
        certainty = replys[0]
        if not debug and certainty < 0.9:
            msg.chat.send('这个问题我不知道诶，问人事组的小姐姐们吧')
            return
        question, answer = replys[1]
        print('matched: '+question)
        print(answer)
        # print('finish')
        if debug:
            msg.chat.send('certainty: %s' % certainty)
            msg.chat.send('matched question: '+question)
            msg.chat.send('answer: ')

        for line in answer:
            msg.chat.send(line)


embed()

