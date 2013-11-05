from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty, AliasProperty, StringProperty, DictProperty, BooleanProperty, StringProperty, OptionProperty

from kivy.graphics.texture import Texture

import numpy as n
from colorsys import hsv_to_rgb

class FourierCanvas(Widget):
    square_x = NumericProperty(0)
    square_y = NumericProperty(0)
    square_width = NumericProperty(10)
    square_height = NumericProperty(10)

    texture = ObjectProperty(None,allownone=True)
    k_intensity_texture = ObjectProperty(None,allownone=True)
    k_phase_texture = ObjectProperty(None,allownone=True)
    k_modified = BooleanProperty(True)
    real_intensity_texture = ObjectProperty(None,allownone=True)
    real_phase_texture = ObjectProperty(None,allownone=True)

    xnum = NumericProperty(100)
    ynum = NumericProperty(100)

    mode = OptionProperty('k-space',options=['k-space','Fourier intensity','Fourier phase'])

    def __init__(self,*args):
        super(FourierCanvas,self).__init__(*args)
        self.make_arrays()
        self.make_textures()
        self.texture = self.k_intensity_texture

    def set_mode(self,mode):
        self.mode = mode

    def reset(self):
        self.make_arrays()
        self.make_textures()
        self.k_modified = True
        self.texture = self.k_intensity_texture
        self.mode = 'k-space'

    def make_arrays(self):
        xnum = self.xnum
        ynum = self.ynum
        self.karray = n.zeros((xnum,ynum),dtype=n.float64)
        self.rarray = n.zeros((xnum,ynum),dtype=n.complex)

    def make_textures(self):
        xnum = self.xnum
        ynum = self.ynum
        self.k_intensity_texture = Texture.create((xnum,ynum))
        self.k_phase_texture = Texture.create((xnum,ynum))
        self.real_intensity_texture = Texture.create((xnum,ynum))
        self.real_phase_texture = Texture.create((xnum,ynum))

    # If any array size properties change, update all of them and regenerate textures.
    def on_xnum(self,*args):
        self.make_arrays()
        self.make_textures()
        self.k_modified = True
    def on_ynum(self,*args):
        self.on_xnum()
    def on_size(self,*args):
        self.on_xnum()

    def on_mode(self,*args):
        if self.k_modified:
            self.create_fourier_textures()
            self.k_modified = False
        mode = self.mode
        print 'on_mode called',mode
        if mode == 'k-space':
            print 'k-space chosen'
            self.texture = self.k_intensity_texture
        elif mode == 'Fourier intensity':
            print 'fourier intensity chosen'
            self.texture = self.real_intensity_texture
        elif mode == 'Fourier phase':
            print 'fourier phase chosen'
            self.texture = self.real_phase_texture
        else:
            print 'else chosen'
            self.texture = self.k_intensity_texture

    def create_fourier_textures(self):
        xnum = self.xnum
        ynum = self.ynum
        
        rarray = n.fft.fftshift(n.fft.fft2(self.karray))
        self.rarray = rarray
    
        real_intensity_buffer = []
        real_phase_buffer = []
        maxabs = n.max(n.abs(rarray))
        for row in n.rot90(rarray,3):
            for col in row[::-1]:
                real_intensity_buffer.extend(intensity_to_rgb(n.abs(col)/maxabs))
                real_phase_buffer.extend(phase_to_rgb(n.arctan2(col.imag,col.real)))

        real_intensity_buffer = ''.join(map(chr,real_intensity_buffer))
        real_phase_buffer = ''.join(map(chr,real_phase_buffer))
        self.real_intensity_texture.blit_buffer(real_intensity_buffer,colorfmt='rgb',bufferfmt='ubyte')
        self.real_phase_texture.blit_buffer(real_phase_buffer,colorfmt='rgb',bufferfmt='ubyte')

    def refresh_k_intensity_texture(self):
        karray = self.karray
        buf = []
        for row in n.rot90(karray,3):
            for col in row[::-1]:
                buf.extend(intensity_to_rgb(n.abs(col)))
        print 'REFRESH blitting buf',len(buf),self.k_intensity_texture.size
        buf = ''.join(map(chr,buf))


        self.k_intensity_texture.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')
        #self.texture.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')

    def set_k_rectangle(self,x,y,val,size=(1,1)):
        print 'called set_k_rectangle',x,y,val,size
        self.k_modified = True
        self.karray[x:x+size[0],y:y+size[1]] = val

        print 'getting',x,y,size[0],size[0]
        #region = self.k_intensity_texture.get_region(x,y,int(size[0]),(size[1]))
        #region = self.k_intensity_texture.get_region(int(x),int(y),40,40)
        #region = self.texture.get_region(x,y,size[0],size[1])
        #region_size = region.size
        region_size = size
        region_num = region_size[0]*region_size[1]
        display_val = intensity_to_rgb(val)


        buf = []
        for i in range(region_num):
            buf.extend(display_val)

        buf = ''.join(map(chr,buf))

        print 'blitting buf', buf
        self.texture.blit_buffer(buf, size=size, pos=(x, y), colorfmt='rgb', bufferfmt='ubyte')
        #region.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')

    def coord_to_ind(self,pos):
        xpos,ypos = pos
        dx = self.square_width/float(self.xnum)
        dy = self.square_height/float(self.ynum)
        x = n.floor((xpos-self.square_x)/dx)
        y = n.floor((ypos-self.square_y)/dy)

        x = max(x,0)
        x = min(x,self.xnum-1)
        y = max(y,0)
        y = min(y,self.ynum-1)
            
        return (int(x),int(y))

    def on_touch_down(self,touch):
        if self.k_modified:
            self.on_mode()
        if self.mode == 'k-space':
            self.on_touch_move(touch)
    def on_touch_move(self,touch):
        if self.mode == 'k-space':
            if self.collide_point(*touch.pos):
                tx,ty = touch.pos
                if (tx > self.square_x and
                    tx < self.square_x + self.square_width and
                    ty > self.square_y and
                    ty < self.square_y + self.square_height):
                    ix,iy = self.coord_to_ind(touch.pos)
                    self.set_k_rectangle(ix, iy, 5.0, (5, 5))

class FourierScreen(BoxLayout):
    def initialise_drawer(self):
        canvas = FourierCanvas()
        resetbutton = Button(text='reset',size_hint_y=None,height=(40,'sp'))
        kbutton = Button(text='k-space',size_hint_y=None,height=(40,'sp'))
        phasebutton = Button(text='To phase',size_hint_y=None,height=(40,'sp'))
        intensitybutton = Button(text='To intensity',size_hint_y=None,height=(40,'sp'))

        self.add_widget(canvas)
        self.add_widget(resetbutton)
        self.add_widget(kbutton)
        self.add_widget(phasebutton)
        self.add_widget(intensitybutton)

        resetbutton.bind(on_press=lambda j: canvas.reset())
        kbutton.bind(on_press=lambda j: canvas.set_mode('k-space'))
        phasebutton.bind(on_press=lambda j: canvas.set_mode('Fourier phase'))
        intensitybutton.bind(on_press=lambda j: canvas.set_mode('Fourier intensity'))

def phase_to_rgb(phase):
    frac = (phase + n.pi) / (2*n.pi)
    val = int(frac*255)
    return [val,val,val]
    colour = list(hsv_to_rgb(frac,1,1))
    print phase,frac,colour
    return map(lambda j: int(j*255),colour)


def intensity_to_rgb(intensity):
    if n.isnan(intensity):
        return [0,0,0]
    val = min(int(intensity*255),255)
    return [val,val,val]
    
# vector_phase_to_rgb = n.vectorize(phase_to_rgb)
# vector_intensity_to_rgb = n.vectorize(intensity_to_rgb)
            
class FourierApp(App):
    def build(self):
        #fc = FourierCanvas()
        screen = FourierScreen()
        #drawer = FourierDrawer()
        screen.initialise_drawer()
        return screen #fc
            

if __name__ == "__main__":
    FourierApp().run()
