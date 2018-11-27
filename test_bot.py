from bot import *


def test_reply_god():
    add_god_msg = MSG(type='Text', text='godmode:add_god:Jone', chat_name='A')
    add_msg = MSG(type='Text', text='abc', chat_name='B')
    remove_msg = MSG(type='Text', text='godmode:remove_record:abc', chat_name='C')
    bot.reply_god(add_god_msg)
    bot.add_record(('abc', [add_msg]))
    bot.reply_god(remove_msg)


def test_reply_record():
    y_msg = MSG(type='Text', text='y', chat_name='A')
    n_msg = MSG(type='Text', text='n', chat_name='B')
    msg = MSG(type='Text', text='abc', chat_name='C')
    bot.msg_cache = [msg]
    bot.msg_list = [msg]
    bot.reply_record(y_msg)
    bot.msg_cache = []
    bot.msg_list = []

    bot.reply_record(y_msg)

    bot.reply_record(n_msg)

    bot.remove_record('abc')


def test_transfer_to_user():
    help_msg = MSG(type='Text', text='a msg from helper', chat_name='helper')
    user_msg = MSG(type='Text', text='a msg from user', chat_name='user')
    bot.msg_cache = [help_msg]
    bot.msg_list = [user_msg]
    bot.locked_users = [user_msg.chat]
    bot.transfer_to_user()


def test_reply_helper():
    msg = MSG(type='Text', text='a msg from helper', chat_name='helper')
    bot.reply_helper(msg)

    bot.asking = True
    msg = MSG(type='Text', text='end', chat_name='helper')
    bot.reply_helper(msg)

    msg = MSG(type='Text', text='y', chat_name='helper')
    help_msg = MSG(type='Text', text='a msg from helper', chat_name='helper')
    userA_msg = MSG(type='Text', text='a msg from user', chat_name='userA')
    userB_msg = MSG(type='Text', text='a msg from user', chat_name='userB')
    bot.msg_cache = []
    bot.reply_helper(help_msg)
    # bot.msg_cache = [help_msg]
    bot.msg_list = [userA_msg, userB_msg]
    bot.reply_helper(msg)


if __name__ == '__main__':
    # print('test god')
    # test_reply_god()
    #
    # print('test reply record')
    # test_reply_record()
    #
    # print('test transfer')
    # test_transfer_to_user()

    print('test reply helper')
    test_reply_helper()

