import datetime

from telegram.ext import MessageHandler, Filters, CallbackContext
from telegram.ext import Updater
from telegram.ext import ContextTypes, MessageHandler, filters
from collections import namedtuple
import enum


class strings(enum.Enum):
    entry = "beni de gör be!"
    on_chat_anyway = "sen diyetini ödemişsin baba, konseydesin"
    chat_full = "konsey dolu hocam, maalesef."
    congrats = "aaa buyur ekrem abi, iyi oyunlar!\n"
    ask_why = "niye geldin la buraya"


token_file_string = "Telegram/token.txt"
chats_file_string = "Telegram/chats.txt"

Chat_Fields = ['ID', 'Name']
Chat = namedtuple('chat', Chat_Fields)


class Telegram():

    def set_chats_length(self, value):
        self._chats_length = value
        if self.chats_length_changed:
            self.chats_length_changed()

    chats_length = property(
        lambda self: self._chats_length,
        set_chats_length)

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Telegram, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.chats_length_changed = None
        self.load_token()
        self.load_chats()
        print("Telegram Bot Start")
        self.updater = Updater(token=self.token)
        self.updater.dispatcher.add_handler(MessageHandler(filters.Filters.text & ~filters.Filters.command, self.get_message))
        self.updater.start_polling()
        for chat in self.chats:
            try:
                self.updater.bot.getFile(self.updater.bot.get_user_profile_photos(chat.ID).photos[0][0].file_id).download(f'Telegram/Images/{chat.ID}.jpg')
            except:
                pass
        # self.send_each("hi")

    def get_message(self, update, _):
        print(update.message.chat.id, update.message.chat.username)
        if update.message.text == strings.entry.value:
            new_chat = Chat(update.message.chat.id, update.message.chat.username)
            if self.chats.__contains__(new_chat):
                answer = strings.on_chat_anyway.value
            elif len(self.chats) >= self.chats_limit:
                answer = strings.chat_full.value
            else:
                answer = strings.congrats.value
                self.chats.append(new_chat)
                self.save_chats()
                self.updater.bot.send_message(chat_id=self.chats[-1].ID, text=str(self.chats[-1]))
        else:
            answer = strings.ask_why.value

        self.updater.bot.send_message(chat_id=update.message.chat.id, text=answer)

    def send_each(self, message):
        for chat in self.chats:
            self.updater.bot.send_message(chat_id=chat.ID, text=message)

    def load_token(self):
        with open(token_file_string, "r") as token_file:
            self.token = token_file.read()

    def load_chats(self):
        self.chats = []
        with open(chats_file_string, "r") as chats_file:
            for line in chats_file.read().splitlines():
                values = line.split()
                self.chats.append(Chat(int(values[0]), values[1]))

        self.chats_limit = self.chats_length = len(self.chats)

    def save_chats(self):
        with open(chats_file_string, "w") as chats_file:
            chats_file.write("".join(f"{chat.ID} {chat.Name}\n" for chat in self.chats))

        self.chats_length = len(self.chats)

    def get_photos(self, user_id):
        user_photos = self.updater.bot.get_user_profile_photos(user_id)
        print(user_photos)
        user_photos = user_photos.photos
        photos_ids = []
        for photo in user_photos:
            photos_ids.append(photo[0].file_id)
        return photos_ids

    def answer(self, user_id):
        photos_ids = self.get_photos(user_id)
        for photo_id in photos_ids:
            self.updater.bot.send_photo(user_id, photo_id)
            file = self.updater.bot.get_file(photo_id)


if __name__ == '__main__':

    bot = Telegram()
    print(bot.chats)
    import time

    while True:
        time.sleep(1)
        print("sleep 1 sec")
        pass
