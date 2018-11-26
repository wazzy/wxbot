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
    skipwords = [',', ':', '.', '!', '，', '：', '。', '！', '?', '？']
    if raw_text.lower().startswith('godmode'):
        mode = 'god'
        text = raw_text.strip().split('godmode')[-1]
    elif raw_text.startswith('小哥哥') or len(raw_text) < 4:
        mode = 'chitchat'
        text = raw_text
    elif raw_text.lower().startswith('debug'):
        mode = 'debug'
        text = raw_text.strip().split('debug')[-1]
    else:
        mode = 'qa'
        text = raw_text.strip()
    while text[0] in skipwords and len(text) > 0:
        text = text[1:]
    return text, mode


class WXBot(Bot):
    def __init__(self, helper_name='代号H', god_name='代号M', threshold=0.9, **kwargs):
        super().__init__(**kwargs)
        # agent for qa task
        self.agent = PairMatchAgent()
        # agent for chitchat
        self.turing = Tuling(api_key='25da5f49d4ad44d48a477543f0b3f55e')
        # human assistant
        self.helper = ensure_one(self.friends().search(helper_name))
        # human master, won't be locked
        # self.master = ensure_one(self.friends().search(master_name))
        # list of msgs that are not answered yet
        self.msg_list = []
        # cache for multiple line answer from helper
        self.reply_cache = []
        # locked user whose further msg won't be replied if not in knowledge base but will be recorded
        self.locked_users = set()
        # qa task threshold for reply certainty
        self.threshold = threshold
        # asking
        self.asking = False
        # backend users
        god = self.friends().search(god_name)[0]
        self.gods = {god}

    def record(self, qa_pair, qa_file='qa_pairs'):
        qa_path = os.path.join(FLAGS.data_dir, qa_file)
        with open(qa_path, 'a') as fout:
            q, ans = qa_pair
            fout.write('Q:%s\n' % q)
            fout.write('A:')
            for a in ans:
                fout.write('%s\n' % a)
        return

    def remove_record(self, record, qa_file='qa_pairs'):
        qa_path = os.path.join(FLAGS.data_dir, qa_file)
        keeps = []
        keep = False
        ans = []
        content_q = ''
        with open(qa_path, 'r') as fout:
            for line in fout.readlines():
                if line.startswith('Q:'):
                    if len(ans) > 0 and content_q:
                        keeps.append((content_q, ans))
                        ans = []
                    content_q = line.strip().split('Q:')[-1]
                    if content_q != record:
                        keep = True
                    else:
                        keep = False
                else:
                    content_a = line.strip().split('A:')[-1]
                    if keep:
                        ans.append(content_a)

    def reply_god(self, msg):
        text, _ = preprocess_raw_text(msg.text)
        if text.startswith('add_god:'):
            new_god_name = text.split('add_god:')[-1]
            new_god = self.friends().search(new_god_name)[0]
            self.gods.add(new_god)
            msg.chat.send('new god %s is added' % new_god.name)
        elif text.startswith('remove_record:'):
            record = text.split('remove_record:')[-1]
            self.remove_record(record)
            msg.chat.send('record is removed')
        return

    def reply_helper(self, msg):
        if not self.asking:
            msg.chat.send('别说话，没问你问题呢~')
            return
        if msg.text.lower() == 'end':
            msg.chat.send("收到～")
            if len(self.reply_cache) > 0:
                self.msg_list[0].chat.send('来自人事小姐姐的回答：')
                self.msg_list[0].chat.send('问题： %s' % self.msg_list[0].text)
                self.msg_list[0].chat.send('答案：')
            for reply in self.reply_cache:
                self.msg_list[0].chat.send(reply)
            self.locked_users.remove(self.msg_list[0].chat)
            del self.msg_list[0]
            self.reply_cache = []
            if len(self.msg_list) > 0:
                self.helper.send('来自%s的消息：%s' % (self.msg_list[0].chat.name, self.msg_list[0].text))
            else:
                self.asking = False
        else:
            self.reply_cache.append(msg.text)

    def reply_user(self, msg):
        # if msg.chat in self.locked_users:
        #     msg.chat.send('我去问问题了，所有新消息概不回复～')
        #     return
        text, mode = preprocess_raw_text(msg.text.lower())
        if len(text) == 0:
            msg.chat.send('这让我怎么回答')
            return
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
                    if msg.chat not in self.locked_users:
                        msg.chat.send('这个问题我不太确定，我去问下哈')
                    self.locked_users.add(msg.chat)
                    self.msg_list.append(msg)
                    if not self.asking:
                        self.helper.send('来自%s的消息：%s' % (msg.chat.name, msg.text))
                        self.asking = True
                else:
                    for line in ans:
                        msg.chat.send(line)


bot = WXBot(console_qr=True)
friends = bot.friends()


@bot.register(friends)
def reply_my_friend(msg):
    if msg.text.startswith('godmode') and msg.chat in bot.gods:
        bot.reply_god(msg)
    elif msg.chat == bot.helper:
        print('reply helper')
        bot.reply_helper(msg)
    else:
        print('reply user')
        bot.reply_user(msg)


embed()

