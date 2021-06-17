import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import time
from PIL import Image,ImageTk
import os


window = tk.Tk()
window.title('Impressionism Style Transfer')
window.geometry('900x600')
window.configure(background='white')
window.resizable(False, False)

photo = Image.open("C:\\Users\\xiongziyin\\DeskTop\\大一第二学期\\人工智能引论\\CV\\style transform\\mount.jpg")
photo = photo.resize((900,600))
img0 = ImageTk.PhotoImage(photo)
Lab= tk.Label(window, image=img0)
Lab.pack()

# 全局变量
global s
s=''
global target
target = ''
global order
order = ''
global Fpath
Fpath = ""
global vv
vv = tk.StringVar()

# 控制颜色和风格化程度
var11 = tk.IntVar()  # 定义var1和var2整型变量用来存放选择行为返回值
c1 = tk.Checkbutton(window, text='preserve color',  font = ('Arial', 10, 'bold'), bg='Lavender', variable=var11, onvalue=1, offvalue=0)
c1.place(x=390, y=220)


l2 = tk.Label(window, bg='WhiteSmoke', fg='black', width=20, text='choose your alpha')
l2.place(x=380, y=320)

def print_selection(v):
    l2.config(text='selected alpha = ' + v)

# 尺度滑条，长度200字符，从0开始10结束，以2为刻度，精度为0.01，触发调用print_selection函数
s = tk.Scale(window, label='drag to control alpha', bg='WhiteSmoke', from_=0, to=1, orient=tk.HORIZONTAL, length=200, showvalue=0, tickinterval=1,
             resolution=0.1, variable=vv, command=print_selection)
s.place(x=350, y=250)

def play():
    os.system('start output')

def choose(event):
   s = str(com.get())
   target = 'test_style\\' + s + '.jpg'
   alpha = vv.get()
   color_flag = var11.get()  # 记录选择
   if color_flag==0:
       order = 'python test.py --content_dir test_content --style ' + target + ' --alpha ' + alpha + ' --output output'
   else:
       order = 'python test.py --content_dir test_content --style ' + target + ' --alpha ' + alpha + ' --preserve_color' + ' --output output'
   os.system(order)  # 执行命令

var = tk.StringVar()
com = ttk.Combobox(window, textvariable = var)
com['value'] = ('1', '2', '3', '4', '5', '6', '7', '8', 'monet_1', 'monet_2', 'van_gogh')  # style img
com.place(x = 400,y = 180, width = 100,height = 25)
com.bind("<<ComboboxSelected>>", choose)

plays_btn = tk.Button(window, text='Show', command=play, font=('Arial', 16, 'bold'), bg='Thistle')  # 显示图片
plays_btn.place(x=400, y=380, width = 100,height = 50)

text1 = tk.Label(window, text = "choose the wanted style", font = ('Arial', 16, 'bold'), bg='Lavender')
text1.place(x = 325, y = 150)

text2 = tk.Label(window, text = "Impressionism Style Transfer", font = ('Comic Sans MS', '30', 'bold'), fg='MediumOrchid', bg='LightSteelBlue')
text2.place(x = 150, y = 50)

window.mainloop()
