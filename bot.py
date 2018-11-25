from wxpy import *
from agent import *


# agent = PairMatchAgent()
# turing = Tuling(api_key='25da5f49d4ad44d48a477543f0b3f55e')
# helper = ensure_one(bot.friends().search(helper_name))
# # list of msgs that are not answered yet
# msg_list = []
# # cache for multiple line answer from helper
# reply_cache = []

def preprocess_raw_text(raw_text):
    """return text and mode"""
    skipwords = [',', ':', '.', '!', '，', '：', '。', '！']
    if raw_text.startswith('小姐姐'):
        mode = 'chitchat'
        text = raw_text.strip().split('小姐姐')[-1]
    elif raw_text.startswith('debug'):
        mode = 'debug'
        text = raw_text.strip().split('debug')[-1]
    else:
        mode = 'qa'
        text = raw_text.strip()
    while text[0] in skipwords and len(text) > 0:
        text = text[1:]
    return text, mode


class WXBot(Bot):
    def __init__(self, helper_name='代号A', threshold=0.9, **kwargs):
        super().__init__(kwargs)
        # agent for qa task
        self.agent = PairMatchAgent()
        # agent for chitchat
        self.turing = Tuling(api_key='25da5f49d4ad44d48a477543f0b3f55e')
        # human assistant
        self.helper = ensure_one(bot.friends().search(helper_name))
        # list of msgs that are not answered yet
        self.msg_list = []
        # cache for multiple line answer from helper
        self.reply_cache = []
        # locked user whose further msg won't be replied or recorded
        self.locked_users = set()
        # qa task threshold for reply certainty
        self.threshold = threshold
        # asking
        self.asking = True

    def reply_helper(self, msg):
        if msg.text == 'end':
            msg.chat.send("收到～")
            for reply in self.reply_cache:
                self.msg_list[0].chat.send(reply)
            del self.msg_list[0]
            self.reply_cache = []
            self.locked_users.remove(self.msg_list[0].chat)
        else:
            self.reply_cache.append(msg.text)
            if len(self.msg_list) > 0:
                self.helper.send(self.msg_list[0].text)
            else:
                self.asking = False

    def reply_user(self, msg):
        if msg.chat in self.locked_users:
            msg.chat.send('我去问问题了，所有新消息概不回复～')
            return
        text, mode = preprocess_raw_text(msg.text)
        if mode == 'chitchat':
            self.turing.do_reply(msg)
        else:
            certainty, qa_pair = self.agent.predict(msg.text)
            q, ans = qa_pair
            if mode == 'debug':
                msg.chat.send('certainty: %s' % certainty)
                msg.chat.send('matched: %s' % q)
                msg.chat.send('answer: ')
                for line in ans:
                    msg.chat.send(line)
            else:
                if certainty < self.threshold:
                    msg.chat.send('这个问题我不太确定，我去问下哈')
                    self.locked_users.add(msg.chat)
                    self.msg_list.append(msg)
                    if not self.asking:
                        self.helper.send(msg.text)
                        self.asking = True
                else:
                    for line in ans:
                        msg.chat.send(line)


bot = WXBot(console_qr=True)
friends = bot.friends()


@bot.register(friends)
def reply_my_friend(msg):
    if msg.chat == bot.helper:
        bot.reply_helper(msg)
    else:
        bot.reply_user(msg)


# @bot.register(friends)
# def reply_my_friend1(msg):
#     text = msg.text
#     # print('text received: '+text)
#     if text.startswith('小姐姐'):
#         print('chitchat mode on')
#         turing.do_reply(msg)
#         return
#
#     run = False
#
#     if text.startswith('debug:'):
#         text = text[6:]
#         print('debug mode on')
#         debug = True
#     else:
#         # print('debug mode off')
#         debug = False
#
#     if debug:
#         run = True
#
#     if text.startswith('小助手'):
#         print('assistant mode on')
#         text = text.split('小助手')[-1].strip()
#         run = True
#
#     if run:
#         print('running')
#         print('text: ' + text)
#         replys = agent.reply(text)
#         certainty = replys[0]
#         if not debug and certainty < 0.9:
#             msg.chat.send('这个问题我不知道诶，问人事组的小姐姐们吧')
#             return
#         question, answer = replys[1]
#         print('matched: ' + question)
#         print(answer)
#         # print('finish')
#         if debug:
#             msg.chat.send('certainty: %s' % certainty)
#             msg.chat.send('matched question: ' + question)
#             msg.chat.send('answer: ')
#
#         for line in answer:
#             msg.chat.send(line)


embed()

