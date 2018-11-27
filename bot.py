from wxpy import *
from agent import *
import time
import re

# agent = PairMatchAgent()
# turing = Tuling(api_key='25da5f49d4ad44d48a477543f0b3f55e')
# helper = ensure_one(bot.friends().search(helper_name))
# # list of msgs that are not answered yet
# msg_list = []
# # cache for multiple line answer from helper
# msg_cache = []

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


class Chat:
    def __init__(self, name):
        self.name = name

    def send(self, text):
        print('%s receive: %s' % (self.name, text))


class MSG:
    def __init__(self, type, chat_name, text=None):
        self.type = type
        self.text = text
        self.chat = Chat(chat_name)

    def get_file(self, save_path):
        print('file download in %s' % save_path)

    def forward(self, chat):
        chat.send(self.text)


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
        self.msg_cache = []
        # locked user whose further msg won't be replied if not in knowledge base but will be recorded
        self.locked_users = set()
        # qa task threshold for reply certainty
        self.threshold = threshold
        # asking
        self.asking = False
        # backend users
        god = self.friends().search(god_name)[0]
        self.gods = {god}
        # waiting record decision
        self.waiting = False
        # uncomment user message dict
        self.user_msg = {}

    def get_save_path(self):
        date = time.strftime("%D:%H:%M:%S ", time.localtime(time.time()))
        save_name = '_'.join(re.split('[/:]', date))
        save_path = os.path.join(FLAGS.data_dir+'/pictures', save_name)
        return save_path

    def add_record(self, qa_pair, qa_file='qa_pairs.txt'):
        qa_path = os.path.join(FLAGS.data_dir, qa_file)
        count = 0
        with open(qa_path, 'r') as fin:
            for line in fin.readlines():
                if line.startswith('Q:'):
                    count += 1
        # print('count finished')
        with open(qa_path, 'a') as fout:
            q, msgs = qa_pair
            fout.write('Q:%s\n' % q)
            fout.write('A:')
            for msg in msgs:
                if msg.type == 'Text':
                    fout.write('%s\n' % msg.text)
                elif msg.type == 'Picture':
                    save_path = self.get_save_path()
                    msg.get_file(save_path=save_path)
                    fout.write('PICTURE:%s\n' % save_path)
        # print('recorded')
        return count+1

    def remove_record(self, record, qa_file='qa_pairs.txt'):
        qa_path = os.path.join(FLAGS.data_dir, qa_file)
        keeps = []
        keep = False
        ans = []
        content_q = ''
        find = False
        with open(qa_path, 'r') as fin, open(qa_path+'.copy_last', 'w') as fout:
            for line in fin.readlines():
                fout.write(line)
                if line.startswith('Q:'):
                    if len(ans) > 0 and content_q:
                        keeps.append((content_q, ans))
                        ans = []
                    content_q = line.strip().split('Q:')[-1]
                    if content_q != record:
                        keep = True
                    else:
                        find = True
                        keep = False

                else:
                    content_a = line.strip().split('A:')[-1]
                    if keep:
                        ans.append(content_a)
                    else:
                        if content_a.startswith('PICTURE:'):
                            path = content_a.split('PICTURE:')[-1]
                            # print('os.remove(path=path)')
                            os.remove(path=path+' ')
            if len(ans) > 0 and content_q:
                keeps.append((content_q, ans))
        count = 0
        with open(qa_path, 'w') as fout:
            for q, ans in keeps:
                count += 1
                fout.write('Q:%s\n' % q)
                fout.write('A:')
                for a in ans:
                    fout.write('%s\n' % a)
        return find, count

    def reply_god(self, msg):
        text, _ = preprocess_raw_text(msg.text)
        msg.chat.send('yes, my lord ~')
        if text.startswith('add_god:'):
            new_god_name = text.split('add_god:')[-1]
            # print('add new god %s ' % new_god_name)
            new_god = self.friends().search(new_god_name)[0]
            self.gods.add(new_god)
            msg.chat.send('new god %s is added' % new_god.name)
        elif text.startswith('remove_record:'):
            record = text.split('remove_record:')[-1]
            find, count = self.remove_record(record)
            if find:
                msg.chat.send('record is removed. Current record count: %s' % count)
            else:
                msg.chat.send('record is not found. Current record count: %s' % count)

    def reply_record(self, msg):
        if msg.text.lower() == 'y' and len(self.msg_cache) > 0:
            count = self.add_record((self.msg_list[0].text, self.msg_cache))
            msg.chat.send('好的～ 当前已收录%s条问题' % count)
        elif msg.text.lower() == 'y':
            msg.chat.send('当前回复为空，无法收录')
        else:
            msg.chat.send('好的~')

    def transfer_to_user(self):
        if len(self.msg_cache) > 0:
            self.msg_list[0].chat.send('转自人事小姐姐的消息：')
            self.msg_list[0].chat.send('问题： %s' % self.msg_list[0].text)
            self.msg_list[0].chat.send('回复：')
        for msg in self.msg_cache:
            msg.forward(self.msg_list[0].chat)
        self.locked_users.remove(self.msg_list[0].chat)
        del self.msg_list[0]
        self.msg_cache = []

    def reply_helper(self, msg):
        if not self.asking:
            msg.chat.send('别说话，没问你问题呢~')
            return
        if msg.text.lower() == 'end':
            msg.chat.send("收到～")
            msg.chat.send("是否记录当前问答内容？（y/n）")
            self.waiting = True
        elif msg.text.lower() in ['y', 'n'] and self.waiting:
            print('receive helper recording decision')
            self.waiting = False
            self.reply_record(msg)
            self.transfer_to_user()
            if len(self.msg_list) > 0:
                self.helper.send('来自%s的消息：%s' % (self.msg_list[0].chat.name, self.msg_list[0].text))
            else:
                self.asking = False
        else:
            print('append msg from helper to cache')
            self.msg_cache.append(msg)

    def send_user_ans(self, user, ans):
        for line in ans:
            if line.startswith('PICTURE:'):
                path = line.split('PICTURE:')[-1]
                user.send_image(path+' ')
            else:
                user.send(line)

    def transfer_to_helper(self, msg):
        self.locked_users.add(msg.chat)
        self.msg_list.append(msg)
        if not self.asking:
            self.helper.send('来自%s的消息：%s' % (msg.chat.name, msg.text))
            self.asking = True

    def reply_user(self, msg):
        text, mode = preprocess_raw_text(msg.text.lower())
        if len(text) == 0:
            msg.chat.send('这让我怎么回答')
            return

        if msg.chat in self.user_msg and msg.text == '是':
            msg.chat.send('嘻嘻～')
            del self.user_msg[msg.chat]
            return
        elif msg.chat in self.user_msg and msg.text == '否':
            msg.chat.send('好吧，我再去问问人事小姐姐看')
            self.transfer_to_helper(msg)
            return
        elif msg.chat in self.user_msg:
            msg.chat.send('麻烦先评价一下哦，回复是或否')
            return
        if mode == 'chitchat':
            self.turing.do_reply(msg)
            return
        else:
            if msg.chat in self.locked_users:
                msg.chat.send('我去问问题的时候，所有新消息都不回复哦～')
                return
            certainty, qa_pair = self.agent.predict(msg.text)
            q, ans = qa_pair
            if mode == 'debug':
                msg.chat.send('certainty: %s' % certainty)
                msg.chat.send('matched: %s' % q)
                msg.chat.send('answer: ')
                self.send_user_ans(user=msg.chat, ans=ans)
            else:
                if certainty < self.threshold:
                    if msg.chat not in self.locked_users:
                        msg.chat.send('这个问题我不太确定，我去问下哈')
                    self.transfer_to_helper(msg)
                else:
                    self.send_user_ans(msg.chat, ans)
                    msg.chat.send('这个回答解决你的问题了吗？（是/否）')
                    self.user_msg[msg.chat] = msg


bot = WXBot(console_qr=True)
friends = bot.friends()


@bot.register(friends)
def reply_my_friend(msg):
    target = ensure_one(bot.friends().search('代号H'))
    # if msg.chat == target and msg.type == 'Picture':
    #     print('test forward')
    #     msg.forward(target)
    #     print('test record and send')
    #     qa_pair = ('图片', [msg])
    #     bot.add_record(qa_pair)
    # elif msg.chat == target and msg.text == '图片':
    #     certainty, qa_pair = bot.agent.predict(msg.text)
    #     q, ans = qa_pair
    #     msg.chat.send('certainty: %s' % certainty)
    #     msg.chat.send('matched: %s' % q)
    #     msg.chat.send('answer: ')
    #     bot.send_user_ans(user=msg.chat, ans=ans)

    if msg.text.startswith('godmode') and msg.chat in bot.gods:
        bot.reply_god(msg)
    elif msg.chat == bot.helper:
        print('reply helper')
        bot.reply_helper(msg)
    else:
        print('reply user')
        bot.reply_user(msg)


embed()

