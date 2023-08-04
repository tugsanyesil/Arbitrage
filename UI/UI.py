from itertools import chain
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import requests
from kivy.app import App
from kivy.base import EventLoop
from kivy.metrics import dp
from kivy.properties import StringProperty, NumericProperty, BoundedNumericProperty, Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.bubble import BubbleButton, Bubble
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.rst import RstDocument
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
import mplfinance as mpf
import os
from os.path import dirname
from kivy.uix.dropdown import DropDown
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.relativelayout import MDRelativeLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.tab import MDTabsBase
from kivymd.icon_definitions import md_icons

import Telegram.telegram as telegram

from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen

from kivy.config import Config

from Arbitrage.arbitrage import Arbitrage
from Market.Binance.binancemarket import BinanceMarket

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
infos_file_string = "UI/UI_Infos.txt"

import networkx as nx


class ChatSlot(MDRelativeLayout):
    def __init__(self, chat):
        super(ChatSlot, self).__init__()
        if chat:
            self.image = f"Telegram/Images/{chat.ID}.jpg"
            self.text = f"{chat.ID}\n{chat.Name}"
        else:
            self.image = ''
            self.text = ''


class MatrixBox(MDGridLayout):

    def set_length(self, value):
        self._length = value
        self.length_changed()

    length = property(
        lambda self: self._length,
        set_length)

    def __init__(self, **kwargs):
        super(MatrixBox, self).__init__(**kwargs)
        self.length = 1

    def length_changed(self):
        self.clear_widgets()
        self.cols = self.rows = self.length
        [self.add_widget(MDLabel(halign="center", md_bg_color=(1, 1, 1, 1), size=(dp(100), dp(40)))) for j in range(self.length) for i in range(self.length)]

    def show(self, arbitrage):
        self.length = len(arbitrage.assets) + 1

        for i in range(self.length):
            for j in range(self.length):
                index = -1 - (i * self.length + j)

                if i == j:
                    self.children[index].text = "-"
                elif i == 0:
                    self.children[index].text = arbitrage.assets[j - 1]
                elif j == 0:
                    self.children[index].text = arbitrage.assets[i - 1]
                else:
                    self.children[index].text = "{:.10f}".format(arbitrage.price_matrix[i - 1, j - 1])
        self.children[-1].text = "from\\to"


class Root(MDBoxLayout):

    def __init__(self, **kwargs):
        self.telegram = telegram.Telegram()
        self.market = BinanceMarket()
        self.arbitrage = Arbitrage(self.market)

        super(Root, self).__init__(**kwargs)

    def chats_counter_changed(self):
        if self.telegram.chats_length > self.ids.chats_counter.count:
            self.ids.chats_counter.count = self.telegram.chats_length
            self.telegram.chats_limit = self.ids.chats_counter.count

        difference = self.ids.chats_counter.count - len(self.ids.chat_slots.children)
        if difference > 0:
            for i in range(difference):
                self.ids.chat_slots.add_widget(ChatSlot(None))
        elif difference < 0:
            self.ids.chat_slots.remove_widget(self.ids.chat_slots.children[0])

    def chats_length_changed(self):
        [self.ids.chat_slots.add_widget(ChatSlot(chat)) for chat in self.telegram.chats]

    def initialize_widgets(self, time):
        # print(self.ids)

        self.telegram.chats_length_changed = self.chats_length_changed
        self.telegram.chats_length_changed()
        self.ids.chats_counter.count = self.telegram.chats_limit

        assets = ["BTC", "ETH", "BNB", "BUSD"]
        self.arbitrage.initialize(assets=assets, base="BUSD", path_length=3, max_profit_length=3, min_profit=1)
        # self.arbitrage.set_matrix()
        self.ids.matrix_box.show(self.arbitrage)

        print(self.arbitrage.price_matrix)

        G = nx.DiGraph(directed=True)
        G.add_nodes_from(self.arbitrage.assets)

        for i in range(self.arbitrage.price_matrix.shape[0]):
            for j in range(self.arbitrage.price_matrix.shape[1]):
                if 0 != self.arbitrage.price_matrix[i, j]:
                    G.add_edge(self.arbitrage.assets[i], self.arbitrage.assets[j], weight=self.arbitrage.price_matrix[i, j])

        pos = nx.circular_layout(G)

        fig, ax = plt.subplots()
        nx.draw(G, pos, connectionstyle='arc3, rad = 0.2')
        edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)

        h = plt.gcf()
        self.ids.GraphBox.add_widget(FigureCanvasKivyAgg(h))


class BotApp(MDApp):

    def __init__(self):
        super(BotApp, self).__init__()

    def build(self):
        # Window.bind(mouse_pos=self.mouse_pos)
        Window.bind(on_request_close=self.on_request_close)
        return Builder.load_file("UI/UI.kv")

    def on_start(self):
        # for name_tab in list(md_icons.keys())[15:30]:
        #     self.root.ids.tabs.add_widget(Tab(icon=name_tab, title=name_tab))

        self.load_infos()
        Clock.schedule_once(self.root.initialize_widgets, 1)

    # def mouse_pos(self, window, mouse_pos):
    #     self.root.mouse_pos = mouse_pos

    def on_request_close(self, *args):
        self.save_infos()
        if hasattr(self.root.telegram, "updater"):
            self.root.telegram.updater.stop()

        return False

    def load_infos(self):
        self.infos = []
        with open(infos_file_string, "r") as infos_file:
            for line in infos_file.read().splitlines():
                self.infos.append(line)

        self.root.telegram.chats_limit = int(self.infos[0])

    def save_infos(self):
        with open(infos_file_string, "w") as infos_file:
            infos_file.write("".join(f"{info}\n" for info in self.infos))
