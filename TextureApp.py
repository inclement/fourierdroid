from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty, AliasProperty, StringProperty, DictProperty, BooleanProperty, StringProperty, OptionProperty
from kivy.graphics.texture import Texture

from random import randint

class TextureDemo(Widget):
    texture = ObjectProperty(None,allownone=True)
    buf = ListProperty([0 for i in xrange(100*100*3)])
    def __init__(self,*args):
        super(TextureDemo, self).__init__(*args)
        self.texture = Texture.create((100,100))
    def add_random_points(self,*args):
        size = 100*100*3
        buf = self.buf
        for i in range(20):
            change = randint(0,size-1)
            buf[change] = 255
        buf = ''.join(map(chr, buf))

        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
    def add_random_points_via_region(self,*args):
        region = self.texture.get_region(40,40,50,50)
        size = 50*50*3
        buf = self.buf
        for i in range(20):
            change = randint(0,size-1)
            buf[change] = 255
        buf = ''.join(map(chr, buf))
        region.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

class TextureApp(App):
    def build(self):
        demo = TextureDemo()
        rand_button = Button(text='add to texture',size_hint_y=None,height=(40,'sp'))
        rand_button.bind(on_press=demo.add_random_points)
        region_button = Button(text='add to region',size_hint_y=None,height=(40,'sp'))
        region_button.bind(on_press=demo.add_random_points_via_region)

        bl = BoxLayout(orientation='vertical')
        bl.add_widget(demo)
        bl.add_widget(rand_button)
        bl.add_widget(region_button)
        return bl

if __name__ == '__main__':
    TextureApp().run()
