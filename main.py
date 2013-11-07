from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty, AliasProperty, StringProperty, DictProperty, BooleanProperty, StringProperty, OptionProperty

from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.context_instructions import Color
from kivy.clock import Clock

import numpy as n
from colorsys import hsv_to_rgb

from toast import toast

__version__ = '0.1'

class Fourier1DScreen(BoxLayout):
    pass

class Fourier1DCanvas(Widget):
    line_points = NumericProperty(200)

    real_line = ObjectProperty(None, allownone=True)
    imag_line = ObjectProperty(None, allownone=True)
    abs_line = ObjectProperty(None, allownone=True)

    real_scale = NumericProperty(1)
    imag_scale = NumericProperty(1)
    abs_scale = NumericProperty(1)

    editing = OptionProperty('real', options=['real', 'imag'])

    def __init__(self, *args, **kwargs):
        super(Fourier1DCanvas, self).__init__(*args, **kwargs)
        self.create_lines()
        self.function = None
        Clock.schedule_once(self.refresh_graphics, 0)
    def create_lines(self):
        if self.real_line is None:
            with self.canvas:
                Color(1, 0, 0)
                self.real_line = Line(points=self.get_line_basepoints(), width=1)
        if self.imag_line is None:
            with self.canvas:
                Color(0, 1, 0)
                self.imag_line = Line(points=self.get_line_basepoints(), width=1)
        if self.abs_line is None:
            with self.canvas:
                Color(0, 0, 1)
                self.abs_line = Line(points=self.get_line_basepoints(), width=1)
        
    def reset_graph_line(self):
        print self.real_line, self.imag_line, self.abs_line

        self.real_line.points = self.get_line_basepoints()
        self.imag_line.points = self.get_line_basepoints()
        self.abs_line.points = self.get_line_basepoints()

        arr = n.zeros(self.line_points, dtype=n.complex)
        print arr
        self.function = arr
        print self.function
        self.set_lines_from_function()

    def get_line_basepoints(self):
        xs = list(n.linspace(self.x, self.x+self.width, self.line_points))
        y = self.y + 0.5*self.height
        points = []
        for i in range(self.line_points):
            points.append(xs[i])
            points.append(y)
        return points

    def set_lines_from_function(self):
        print ' SET LINES', self.function, type(self.function)
        reals = self.function.real
        imags = self.function.imag
        abss = n.abs(self.function)
        print '...and'

        maxr = n.max(n.abs(reals))
        maxi = n.max(n.abs(imags))

        print 'yay'

        scale_max = max(maxr, maxi)
        if scale_max == 0:
            scale_max = 1
        abs_max = n.max(abss)
        if abs_max == 0:
            abs_max = 1

        print reals
        print imags

        print 'moo'

        for i in range(self.line_points):
            self.real_line.points[2*i+1] = self.y + 0.5*self.height*(1 + reals[i] / scale_max)
            self.imag_line.points[2*i+1] = self.y + 0.5*self.height*(1 + imags[i] / scale_max)
            self.abs_line.points[2*i+1] = self.y + self.height*(abss[i] / abs_max)

        print 'SET POINTS'
        print self.real_line.points
        print self.imag_line.points
        print self.abs_line.points

        self.refresh_graphics()

    def get_line_index(self, x):
        index = int((x - self.x) / self.width * self.line_points)
        return index

    def on_size(self, *args):
        self.reset_graph_line()
    def on_pos(self, *args):
        self.reset_graph_line()
    def on_line_points(self, *args):
        self.reset_graph_line()
    def reset(self, *args):
        self.reset_graph_line()

    def on_touch_down(self, touch):
        self.on_touch_move(touch)
    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        index = self.get_line_index(touch.x)
        old_index = self.get_line_index(touch.px)

        if old_index == index:
            if self.editing == 'real':
                self.real_line.points[2*index + 1] = touch.y
                self.function.real[index] = (touch.y - (self.y + 0.5*self.height)) / (0.5*self.height) * self.real_scale
            elif self.editing == 'imag':
                self.imag_line.points[2*index + 1] = touch.y
                self.function.imag[index] = (touch.y - (self.y + 0.5*self.height)) / (0.5*self.height) * self.imag_scale
            self.abs_line.points[2*index + 1] = self.y + (n.abs(self.function[index]) / self.abs_scale) * self.height
            print self.abs_line.points[2*index + 1]
        else:
            for cur_index in range(old_index, index, int(n.sign(index-old_index))):
                if self.editing == 'real':
                    self.real_line.points[2*cur_index + 1] = touch.y
                    self.function.real[cur_index] = (touch.y - (self.y + 0.5*self.height)) / (0.5*self.height) * self.real_scale
                if self.editing == 'imag':
                    self.imag_line.points[2*cur_index + 1] = touch.y
                    self.function.imag[cur_index] = (touch.y - (self.y + 0.5*self.height)) / (0.5*self.height) * self.imag_scale
                self.abs_line.points[2*cur_index + 1] = self.y + (n.abs(self.function[cur_index]) / self.abs_scale) * self.height
            print '...'
        self.refresh_graphics()

    def refresh_graphics(self, *args):
        self.real_line.points += [self.x+self.width, self.y]
        self.real_line.points = self.real_line.points[:-2]
        self.imag_line.points += [self.x+self.width, self.y]
        self.imag_line.points = self.imag_line.points[:-2]
        self.abs_line.points += [self.x+self.width, self.y]
        self.abs_line.points = self.abs_line.points[:-2]

    def perform_fourier_transform(self, inverse=False):
        if not inverse:
            self.function = n.fft.fftshift(n.fft.fft(self.function))
        else:
            self.function = n.fft.ifft(n.fft.ifftshift(self.function))
        self.set_lines_from_function()

            

class Fourier2DCanvas(Widget):
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

    def __init__(self, *args, **kwargs):
        super(Fourier2DCanvas, self).__init__(*args, **kwargs)
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
        if mode == 'k-space':
            self.texture = self.k_intensity_texture
        elif mode == 'Fourier intensity':
            self.texture = self.real_intensity_texture
        elif mode == 'Fourier phase':
            self.texture = self.real_phase_texture
        else:
            self.texture = self.k_intensity_texture

    def create_fourier_textures(self):
        xnum = self.xnum
        ynum = self.ynum
        
        rarray = n.fft.fftshift(n.fft.fft2(self.karray))
        self.rarray = rarray
        import cPickle
        with open('rarray.pickle','w') as fileh:
            cPickle.dump(self.rarray, fileh)
    
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
        buf = ''.join(map(chr,buf))


        self.k_intensity_texture.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')
        #self.texture.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')

    def set_k_rectangle(self,x,y,val,size=(1,1)):
        self.k_modified = True
        self.karray[x:x+size[0], y:y+size[1]] = val

        #region = self.k_intensity_texture.get_region(x,y,int(size[0]),(size[1]))
        #region = self.k_intensity_texture.get_region(int(x),int(y),40,40)
        region = self.texture.get_region(x-int(size[0]/2),
                                         y-int(size[1]/2),
                                         size[0],
                                         size[1])
        #region_size = region.size
        region_size = size
        region_num = region_size[0]*region_size[1]
        display_val = intensity_to_rgb(val)


        buf = []
        for i in range(region_num):
            buf.extend(display_val)

        buf = ''.join(map(chr,buf))

        #self.texture.blit_buffer(buf, size=size, pos=(x-2, y-2), colorfmt='rgb', bufferfmt='ubyte')
        region.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

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
        # if self.k_modified:
        #     self.on_mode()
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
                    self.set_k_rectangle(ix, iy, 5.0, (3, 3))
                    self.canvas.ask_update()

class Fourier2DScreen(BoxLayout):
    pass

def phase_to_rgb(phase):
    frac = (phase + n.pi) / (2*n.pi)
    rgb = [min(255, int(j*255)) for j in hsv_to_rgb(frac, 1, 1)]
    return rgb
    return [val,val,val]
    colour = list(hsv_to_rgb(frac,1,1))
    return map(lambda j: int(j*255),colour)


def intensity_to_rgb(intensity):
    if n.isnan(intensity):
        return [0,0,0]
    val = min(int(intensity*255),255)
    return [val,val,val]

    
class FourierApp(App):
    def build(self):
        return Fourier1DScreen()

if __name__ == "__main__":
    FourierApp().run()
