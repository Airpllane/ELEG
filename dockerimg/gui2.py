import tkinter as tk
from tkinter import ttk
import loadrnn

#%%

def next_sentence(event):
    s_lbl_stc.delete(1.0, tk.END)
    s_lbl_stc.insert(tk.END, loadrnn.generate_sentence(s_scl_tmp.get()))

def next_exercise(event):
    #try:
    stc = False
    while not stc:
        stc, opts = loadrnn.generate_exercise(e_scl_tmp.get())
    
    e_lbl_stc.delete(1.0, tk.END)
    e_lbl_stc.insert(tk.END, stc)
    
    answers = f'{opts[0]}\n{opts[1][0]}\n{opts[1][1]}'

    e_msg_a1.delete(1.0, tk.END)
    e_msg_a1.insert(tk.END, answers)
    #except:
    #    e_lbl_stc.delete(1.0, tk.END)
    #    e_lbl_stc.insert(tk.END, 'Error')
    #pass

def save_exercise(event):
    answers = [i for i in e_msg_a1.get(1.0, tk.END).split('\n') if i]
    print(answers)
    with open('exc_gift.txt', 'a+', newline = '') as handle:
        handle.write('\n')
        handle.write(e_lbl_stc.get(1.0, tk.END))
        handle.write('{')
        handle.write('\n')
        handle.write('=' + answers[0])
        for i in range(1, len(answers)):
            handle.write('\n')
            handle.write('~' + answers[i])
        handle.write('\n')
        handle.write('}')

def good_sentence(event):
    with open('good.txt', 'a+') as handle:
        handle.write(s_lbl_stc.get(1.0, tk.END))

def bad_sentence(event):
    with open('bad.txt', 'a+') as handle:
        handle.write(s_lbl_stc.get(1.0, tk.END))

def switch_model(event):
    if mdvar.get() == 'Present simple':
        loadrnn.model = loadrnn.model_present
    elif mdvar.get() == 'Past simple':
        loadrnn.model = loadrnn.model_past
    else:
        raise NotImplementedError('Unknown model')

#%%


root = tk.Tk()
root.title('English Language Exercise Generator')
tabControl = ttk.Notebook(root)

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)

mdvar = tk.StringVar()
e_mdcmb = ttk.Combobox(tab2, textvariable = mdvar)
s_mdcmb = ttk.Combobox(tab1, textvariable = mdvar)

e_mdcmb['values'] = ['Present simple', 'Past simple']
s_mdcmb['values'] = ['Present simple', 'Past simple']

e_mdcmb['state'] = 'readonly'
s_mdcmb['state'] = 'readonly'

e_mdcmb.set('Present simple')
s_mdcmb.set('Present simple')

e_mdcmb.bind('<<ComboboxSelected>>', switch_model)
s_mdcmb.bind('<<ComboboxSelected>>', switch_model)

tabControl.add(tab2, text = 'Exercise')

e_lbl_stc = tk.Text(tab2, height = 5, width = 60)
e_lbl_tmp = tk.Label(tab2, text = 'Temperature:')
e_scl_tmp = tk.Scale(tab2, from_ = 0.2, to = 2, orient = tk.HORIZONTAL, resolution = 0.1)
e_msg_a1 = tk.Text(tab2, height = 3, width = 15)
#e_msg_a2 = tk.Message(tab2, text = 'Answers number 2')
#e_msg_a3 = tk.Message(tab2, text = 'Answers number 3')
e_btn_save = tk.Button(tab2, text = 'Save exercise')
e_btn_next = tk.Button(tab2, text = 'New exercise')

e_btn_save.bind('<Button-1>', save_exercise)
e_btn_next.bind('<Button-1>', next_exercise)

e_lbl_stc.grid(column = 0, row = 0, columnspan = 5, rowspan = 1, padx=15, pady=15)
e_lbl_tmp.grid(column = 0, row = 1, padx=15, pady=10, sticky='n')
e_scl_tmp.grid(column = 0, row = 1, padx=17, pady=10, sticky='s')
e_mdcmb.grid(column = 1, row = 1, padx=17, pady=30)
e_msg_a1.grid(column = 2, row = 1, padx=15, pady=15)
#e_msg_a2.grid(column = 3, row = 1, padx=15, pady=15)
#e_msg_a3.grid(column = 4, row = 1, padx=15, pady=15)
e_btn_save.grid(column = 0, row = 2, padx=18, pady=15)
e_btn_next.grid(column = 2, row = 2, padx=15, pady=15)

tabControl.add(tab1, text = 'Sentence')

s_lbl_stc = tk.Text(tab1, height = 5, width = 60)
s_lbl_tmp = tk.Label(tab1, text = 'Temperature:')
s_scl_tmp = tk.Scale(tab1, from_ = 0.2, to = 2, orient = tk.HORIZONTAL, resolution = 0.1)
s_btn_good = tk.Button(tab1, text = 'Good sentence')
s_btn_bad = tk.Button(tab1, text = 'Bad sentence')
s_btn_next = tk.Button(tab1, text = 'New sentence')

s_btn_next.bind('<Button-1>', next_sentence)
s_btn_good.bind('<Button-1>', good_sentence)
s_btn_bad.bind('<Button-1>', bad_sentence)

s_lbl_stc.grid(column = 0, row = 0, columnspan = 5, rowspan = 1, padx=15, pady=15)
s_lbl_tmp.grid(column = 0, row = 1, padx=15, pady=10, sticky='n')
s_scl_tmp.grid(column = 0, row = 1, padx=17, pady=10, sticky='s')
s_mdcmb.grid(column = 1, row = 1, padx=17, pady=30)
s_btn_good.grid(column = 0, row = 2, padx=15, pady=15)
s_btn_bad.grid(column = 1, row = 2, padx=15, pady=15)
s_btn_next.grid(column = 2, row = 2, padx=15, pady=15)


tabControl.pack(expand = 1, fill = 'both')

root.mainloop()