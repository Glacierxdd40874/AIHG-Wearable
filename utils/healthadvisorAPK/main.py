import os
import json
import requests
from datetime import datetime

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.utils import platform
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle

if platform == "android":
    from android.storage import app_storage_path
    app_path = app_storage_path()
else:
    app_path = os.getcwd()

HISTORY_PATH = os.path.join(app_path, "history.json")

def load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading history: {e}")
        return {}

def save_history(entry: str):
    hist = load_history()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hist[ts] = entry
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def clear_history():
    try:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)  # Delete the file to clear history
            print("History cache cleared.")
    except IOError as e:
        print(f"Error clearing history: {e}")

class ColoredProgressBar(BoxLayout):
    def __init__(self, label, value=0, max_value=5, color=(0, 1, 0, 1), **kwargs):
        super().__init__(orientation='horizontal', size_hint_y=None, height=30, padding=[5, 5, 5, 5], **kwargs)
        self.label = Label(text=label, size_hint_x=0.25, color=(0, 0, 0, 1))
        self.value_label = Label(text=str(value), size_hint_x=0.15, color=(0, 0, 0, 1))

        bar_wrapper = BoxLayout(size_hint_x=0.6)
        with bar_wrapper.canvas.before:
            Color(0.8, 0.8, 0.8, 1)
            self.rect = Rectangle(size=bar_wrapper.size, pos=bar_wrapper.pos)
        bar_wrapper.bind(pos=self._update_rect, size=self._update_rect)

        self.bar = ProgressBar(max=max_value, value=value)
        bar_wrapper.add_widget(self.bar)

        self.add_widget(self.label)
        self.add_widget(bar_wrapper)
        self.add_widget(self.value_label)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update_value(self, value):
        self.bar.value = value
        self.value_label.text = str(value)

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.input = TextInput(hint_text='Describe how you\'re feeling...', size_hint_y=None, height=100)
        layout.add_widget(self.input)

        analyze_btn = Button(text='Analyze', size_hint_y=None, height=40)
        analyze_btn.bind(on_press=self.on_analyze)
        layout.add_widget(analyze_btn)

        self.result_container = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.result_container.bind(minimum_height=self.result_container.setter('height'))

        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.result_container)
        layout.add_widget(scroll)

        history_btn = Button(text='View History', size_hint_y=None, height=40)
        history_btn.bind(on_press=self.go_to_history)
        layout.add_widget(history_btn)

        self.add_widget(layout)

        # Initialize progress bars
        self.bars = {
            'fatigue': ColoredProgressBar("Fatigue", 0, 5, (1, 0, 0, 1)),
            'mood': ColoredProgressBar("Mood", 0, 5, (1, 0.5, 0, 1)),
            'readiness': ColoredProgressBar("Readiness", 0, 10, (0, 1, 0, 1)),
            'soreness': ColoredProgressBar("Soreness", 0, 5, (0, 0, 1, 1)),
            'stress': ColoredProgressBar("Stress", 0, 5, (0.5, 0, 0.5, 1)),
            'sleep_duration': ColoredProgressBar("Sleep Duration", 0, 12, (0.2, 0.6, 0.8, 1)),
            'sleep_quality': ColoredProgressBar("Sleep Quality", 0, 5, (0.4, 0.9, 0.4, 1))
        }
        for bar in self.bars.values():
            self.result_container.add_widget(bar)

        # Initialize recommendation label (now placed under progress bars)
        self.advice_label = Label(
            text='',
            color=(0, 0, 0, 1),
            size_hint_y=None,
            text_size=(Window.width - 40, None),
            halign='left',
            valign='top',
            height=200
        )
        self.result_container.add_widget(self.advice_label)

    def on_analyze(self, instance):
        text = self.input.text.strip()
        if not text:
            self.advice_label.text = "Please enter a description."
            return

        try:
            SERVER_URL = "http://192.168.4.23:5000/analyze"
            resp = requests.post(SERVER_URL, json={"text": text}, timeout=15)

            if resp.status_code != 200:
                raise Exception(resp.json().get("error", "Unknown error"))

            data = resp.json()
            feats = data["features"]
            dur = data["sleep_duration"]
            qual = data["sleep_quality"]
            advice = data["advice"]

            self.bars['fatigue'].update_value(feats['fatigue'])
            self.bars['mood'].update_value(feats['mood'])
            self.bars['readiness'].update_value(feats['readiness'])
            self.bars['soreness'].update_value(feats['soreness'])
            self.bars['stress'].update_value(feats['stress'])
            self.bars['sleep_duration'].update_value(dur)
            self.bars['sleep_quality'].update_value(qual)

            self.advice_label.text = f"Recommendations:\n{advice}"
            save_history(data)

        except Exception as e:
            self.advice_label.text = f"Error: {e}"

    def go_to_history(self, instance):
        self.manager.get_screen('history').update_spinner()
        self.manager.current = 'history'

class HistoryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.spinner = Spinner(text='Select Date', size_hint_y=None, height=40)
        self.layout.add_widget(self.spinner)

        query_btn = Button(text='Query', size_hint_y=None, height=40)
        query_btn.bind(on_press=self.query_history)
        self.layout.add_widget(query_btn)

        # Clear Cache Button
        clear_btn = Button(text='Clear Cache', size_hint_y=None, height=40)
        clear_btn.bind(on_press=self.clear_cache)
        self.layout.add_widget(clear_btn)

        self.result_container = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.result_container.bind(minimum_height=self.result_container.setter('height'))

        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.result_container)
        self.layout.add_widget(scroll)

        back_btn = Button(text='Back to Main', size_hint_y=None, height=40)
        back_btn.bind(on_press=lambda x: setattr(self.manager, 'current', 'main'))
        self.layout.add_widget(back_btn)

        self.add_widget(self.layout)

        # Initialize progress bars
        self.bars = {
            'fatigue': ColoredProgressBar("Fatigue", 0, 5, (1, 0, 0, 1)),
            'mood': ColoredProgressBar("Mood", 0, 5, (1, 0.5, 0, 1)),
            'readiness': ColoredProgressBar("Readiness", 0, 10, (0, 1, 0, 1)),
            'soreness': ColoredProgressBar("Soreness", 0, 5, (0, 0, 1, 1)),
            'stress': ColoredProgressBar("Stress", 0, 5, (0.5, 0, 0.5, 1)),
            'sleep_duration': ColoredProgressBar("Sleep Duration", 0, 12, (0.2, 0.6, 0.8, 1)),
            'sleep_quality': ColoredProgressBar("Sleep Quality", 0, 5, (0.4, 0.9, 0.4, 1))
        }
        for bar in self.bars.values():
            self.result_container.add_widget(bar)

        # Initialize recommendation label (now placed under progress bars)
        self.advice_label = Label(
            text='',
            color=(0, 0, 0, 1),
            size_hint_y=None,
            text_size=(Window.width - 40, None),
            halign='left',
            valign='top',
            height=200
        )
        self.result_container.add_widget(self.advice_label)

    def update_spinner(self):
        history = load_history()
        self.spinner.values = list(sorted(history.keys(), reverse=True))

    def query_history(self, instance):
        selected = self.spinner.text
        if not selected or selected == 'Select Date':  # Avoid empty or placeholder selection
            self.advice_label.text = "Please select a valid date."
            return

        history = load_history()
        if selected in history:
            data = history[selected]
            feats = data["features"]
            dur = data["sleep_duration"]
            qual = data["sleep_quality"]
            advice = data["advice"]

            self.bars['fatigue'].update_value(feats['fatigue'])
            self.bars['mood'].update_value(feats['mood'])
            self.bars['readiness'].update_value(feats['readiness'])
            self.bars['soreness'].update_value(feats['soreness'])
            self.bars['stress'].update_value(feats['stress'])
            self.bars['sleep_duration'].update_value(dur)
            self.bars['sleep_quality'].update_value(qual)

            self.advice_label.text = f"Recommendations:\n{advice}"
        else:
            self.advice_label.text = "No data available for the selected date."

    def clear_cache(self, instance):
        clear_history()
        self.spinner.values = []  # Clear the spinner values
        self.advice_label.text = "Cache cleared. No history available."

class HealthAdvisorApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(HistoryScreen(name='history'))
        return sm

if __name__ == '__main__':
    HealthAdvisorApp().run()
