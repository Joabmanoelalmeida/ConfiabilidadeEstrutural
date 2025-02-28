from scipy.stats import tmean, tvar, tstd, norm
import tkinter as tk
from tkinter import ttk, filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk  

def calcular_skew(numeros, media, desvio_padrao):
    quantidade = len(numeros)
    skew = (1 / (quantidade * desvio_padrao**3)) * sum((x - media)**3 for x in numeros)
    return skew

def calcular_kurtosis(numeros, media, desvio_padrao):
    quantidade = len(numeros)
    kurt = (1 / (quantidade * desvio_padrao**4)) * sum((x - media)**4 for x in numeros)
    return kurt

def get_histogram_bins(numeros):
    return int(np.sqrt(len(numeros))) 

def calcular_pdf_normal(numeros):
    media = tmean(numeros)
    std_dev = tstd(numeros)
    x = np.linspace(min(numeros), max(numeros), 200)
    pdf = norm.pdf(x, loc=media, scale=std_dev)
    return x, pdf

def plot_histograms(numeros):
    global last_numbers
    last_numbers = numeros

    for widget in tab_hist.winfo_children():
        if widget == frame_controls_hist:
            continue
        widget.destroy()

    canvas_frame = ttk.Frame(tab_hist, padding=10, style="Card.TFrame")
    canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)

    placeholder = tk.Canvas(canvas_frame, background="gray")
    placeholder.pack(fill="both", expand=True)

    def draw_histogram():
        placeholder.destroy()
        bins = get_histogram_bins(numeros)  
        counts, bins_array = np.histogram(numeros, bins=bins)
        bin_widths = np.diff(bins_array)
        rel_freq = counts / counts.sum()
        density = rel_freq / bin_widths

        fig, ax = plt.subplots(figsize=(5, 7))
        ax.bar(bins_array[:-1], density, width=bin_widths, align='edge', edgecolor='black', color='#4C72B0')
        
        # Plot the normal PDF on top of the histogram
        x, pdf = calcular_pdf_normal(numeros)
        ax.plot(x, pdf, color='red', linewidth=2, label='PDF')
        ax.legend()

        ax.set_title("Histograma de Densidade de Probabilidade", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"Valor\nCestas: {len(bins_array)-1}", fontsize=10)  
        ax.set_ylabel("Densidade", fontsize=10)
        ax.set_xlim(bins_array[0], bins_array[-1])
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    canvas_frame.after(500, draw_histogram)

def process_numeros(numeros, descricao):
    media_calculada = tmean(numeros)
    variancia = tvar(numeros)
    desvio_padrao_calculado = tstd(numeros)
    skewness = calcular_skew(numeros, media_calculada, desvio_padrao_calculado)
    kurt = calcular_kurtosis(numeros, media_calculada, desvio_padrao_calculado)

    for item in tree.get_children():
        tree.delete(item)
    
    tree.insert('', 'end', values=(descricao, str(numeros)))
    tree.insert('', 'end', values=('Média (calculada)', f"{media_calculada:.2f}"))
    tree.insert('', 'end', values=('Variância', f"{variancia:.2f}"))
    tree.insert('', 'end', values=('Desvio Padrão (calculado)', f"{desvio_padrao_calculado:.2f}"))
    tree.insert('', 'end', values=('Coeficiente de Skewness', f"{skewness:.2f}"))
    tree.insert('', 'end', values=('Coeficiente de Kurtosis', f"{kurt:.2f}"))
    
    plot_histograms(numeros)

def gerar_numeros_aleatorios():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 0  
        desvio_informado = float(entry_desvio.get()) if entry_desvio.get() else 1
        
        numeros = norm.rvs(loc=media_informada, scale=desvio_informado, size=quantidade).tolist()
        process_numeros(numeros, "Números gerados")
        
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in tab_hist.winfo_children():
            widget.destroy()

def importar_txt():
    file_path = filedialog.askopenfilename(
        title="Importar arquivo TXT",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not file_path:
        return
    try:
        df = pd.read_csv(file_path, header=None)
        numeros = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().tolist()
        if not numeros:
            raise ValueError("Nenhum valor numérico encontrado no arquivo.")
        process_numeros(numeros, "Números importados (TXT)")
    except Exception as e:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', str(e)))
        for widget in tab_hist.winfo_children():
            widget.destroy()

def importar_excel():
    file_path = filedialog.askopenfilename(
        title="Importar arquivo Excel",
        filetypes=[("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")]
    )
    if not file_path:
        return
    try:
        df = pd.read_excel(file_path)
        # Procura a primeira coluna com dados numéricos
        numeros = None
        for col in df.columns:
            temp = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
            if temp:
                numeros = temp
                break
        if not numeros:
            raise ValueError("Nenhum valor numérico encontrado no arquivo.")
        process_numeros(numeros, "Números importados (Excel)")
    except Exception as e:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', str(e)))
        for widget in tab_hist.winfo_children():
            widget.destroy()

def plot_histograma_replot():
    global last_numbers
    for widget in frame_plot_hist.winfo_children():
        widget.destroy()
    
    if last_numbers is None:
        lbl = ttk.Label(frame_plot_hist, text="Nenhum conjunto de dados disponível.", foreground="red")
        lbl.pack()
        return
    
    try:
        bins = int(entry_bins_hist.get())
    except ValueError:
        bins = get_histogram_bins()
    
    counts, bins_array = np.histogram(last_numbers, bins=bins)
    bin_widths = np.diff(bins_array)
    rel_freq = counts / counts.sum()
    density = rel_freq / bin_widths

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bins_array[:-1], density, width=bin_widths, align='edge', edgecolor='black')
    ax.set_title("Histograma de Densidade de Probabilidade")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidade")
    ax.set_xlim(bins_array[0], bins_array[-1])
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=frame_plot_hist)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

root = tk.Tk()
root.title("JOAB MANOEL")
root.geometry("1500x700")

frame_import = ttk.Frame(root, padding="10")
frame_import.pack(side="top", fill="x")

image_path = os.path.join(os.getcwd(), "Imagens")
abrir_image_path = os.path.join(image_path, "Abrir.jpg")
txt_image_path = os.path.join(image_path, "txt.jpg")
excel_image_path = os.path.join(image_path, "excel.jpg")

img_abrir = Image.open(abrir_image_path)
img_abrir = img_abrir.resize((40, 40), Image.Resampling.LANCZOS)
photo_abrir = ImageTk.PhotoImage(img_abrir)

img_txt = Image.open(txt_image_path)
img_txt = img_txt.resize((40, 40), Image.Resampling.LANCZOS)
photo_txt = ImageTk.PhotoImage(img_txt)

img_excel = Image.open(excel_image_path)
img_excel = img_excel.resize((40, 40), Image.Resampling.LANCZOS)
photo_excel = ImageTk.PhotoImage(img_excel)

def show_import_options():
    if frame_import_options.winfo_ismapped():
        frame_import_options.pack_forget()
    else:
        frame_import_options.pack(side="left", padx=10)

frame_abrir = ttk.Frame(frame_import)
frame_abrir.pack(side="left", padx=10)

btn_abrir = ttk.Button(frame_abrir, image=photo_abrir, command=show_import_options)
btn_abrir.image = photo_abrir
btn_abrir.pack()

lbl_abrir = ttk.Label(frame_abrir, text="Abrir", font=("Segoe UI", 11))
lbl_abrir.pack()

frame_import_options = ttk.Frame(frame_import, padding="10")

btn_txt = ttk.Button(frame_import_options, image=photo_txt, command=importar_txt)
btn_txt.image = photo_txt
btn_txt.pack(side="left", padx=10)

btn_excel = ttk.Button(frame_import_options, image=photo_excel, command=importar_excel)
btn_excel.image = photo_excel
btn_excel.pack(side="left", padx=10)

style = ttk.Style(root)
style.theme_use("clam")
style.configure("TFrame", background="#f7f7f7")
style.configure("TLabel", font=("Segoe UI", 11), background="#f7f7f7", foreground="#333333")
style.configure("TEntry", font=("Segoe UI", 11))
style.configure("TButton", font=("Segoe UI", 11, "bold"), foreground="#ffffff", background="#0078d7")
style.map("TButton",
          background=[("active", "#005fa3"), ("disabled", "#d9d9d9")])

notebook_principal = ttk.Notebook(root)
notebook_principal.pack(fill="both", expand=True, padx=10, pady=10)

tab_parametros = ttk.Frame(notebook_principal)
notebook_principal.add(tab_parametros, text="Parâmetros da Amostra")

tab_hist = ttk.Frame(notebook_principal)
notebook_principal.add(tab_hist, text="Histograma x PDF")

# -- Aba de Parâmetros --
frame_controles = ttk.Frame(tab_parametros, padding="20", style="Card.TFrame")
frame_controles.pack(side="left", fill="y", padx=10, pady=10)

var_parametros = tk.BooleanVar(value=False)
def toggle_quantidade():
    if var_parametros.get():
        frame_quantidade.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 15))
    else:
        frame_quantidade.grid_forget()

check_parametros = ttk.Checkbutton(
    frame_controles,
    text="Gerar amostra com valores aleatórios (NORMAL)",
    variable=var_parametros,
    command=toggle_quantidade,
    style="Toggle.TCheckbutton"
)
check_parametros.grid(row=0, column=0, sticky="w", pady=(30, 15))

frame_quantidade = ttk.Frame(frame_controles)
label_quantidade = ttk.Label(frame_quantidade, text="Número de pontos amostral:", style="TLabel")
label_quantidade.grid(row=0, column=0, sticky="w", pady=(0, 5))
entry_quantidade = ttk.Entry(frame_quantidade, width=20, font=("Segoe UI", 11))
entry_quantidade.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=(0, 15))

label_media = ttk.Label(frame_quantidade, text="Média:", style="TLabel")
label_media.grid(row=1, column=0, sticky="w", pady=(0,5))
entry_media = ttk.Entry(frame_quantidade, width=20, font=("Segoe UI", 11))
entry_media.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(0,15))

label_desvio = ttk.Label(frame_quantidade, text="Desvio Padrão:", style="TLabel")
label_desvio.grid(row=2, column=0, sticky="w", pady=(0,5))
entry_desvio = ttk.Entry(frame_quantidade, width=20, font=("Segoe UI", 11))
entry_desvio.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(0,15))

button_calcular = ttk.Button(
    frame_controles,
    text="Calcular",
    command=gerar_numeros_aleatorios,
    style="Accent.TButton"
)
button_calcular.grid(row=3, column=0, sticky="ew", pady=(5, 15))

file_type_var = tk.StringVar(value="txt")

frame_resultados = ttk.Frame(tab_parametros, padding="10", style="Card.TFrame")
frame_resultados.pack(side="right", fill="both", expand=True, padx=10, pady=10)

colunas = ("Parâmetros", "Valor")
tree = ttk.Treeview(frame_resultados, columns=colunas, show="headings", height=10)
for col in colunas:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", stretch=True)
tree.pack(fill="both", expand=True, padx=10, pady=10)

style.configure("Modern.TFrame", background="#ffffff")
style.configure("Modern.TLabel", background="#ffffff", font=("Segoe UI", 11), foreground="#333333")
style.configure("Modern.TEntry", font=("Segoe UI", 11))
style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), background="#0078d7", foreground="#ffffff", padding=6)
style.map("Accent.TButton",
          background=[("active", "#005fa3")],
          foreground=[("active", "#ffffff")])

# -- Aba de Histograma --
frame_controls_hist = ttk.Frame(tab_hist, padding="20", style="TFrame")
frame_controls_hist.pack(side="left", fill="y", padx=10, pady=10)

var_bins_hist = tk.BooleanVar()
def toggle_bins_hist():
    if var_bins_hist.get():
        entry_bins_hist.grid(row=1, column=0, sticky="w", pady=(0, 15))
        button_plot_hist.grid(row=2, column=0, sticky="w", pady=(5, 15))
    else:
        entry_bins_hist.grid_remove()
        button_plot_hist.grid_remove()

check_bins_hist = ttk.Checkbutton(
    frame_controls_hist,
    text="Quantidade de cestas:",
    variable=var_bins_hist,
    command=toggle_bins_hist,
    style="Toggle.TCheckbutton"
)
check_bins_hist.grid(row=0, column=0, sticky="w", pady=(30, 15))

entry_bins_hist = ttk.Entry(frame_controls_hist, width=20, style="TEntry")
entry_bins_hist.grid(row=1, column=0, sticky="w", pady=(0, 15))
entry_bins_hist.grid_remove()

button_plot_hist = ttk.Button(
    frame_controls_hist,
    text="Atualizar Histograma",
    command=plot_histograma_replot,
    style="Accent.TButton"
)
button_plot_hist.grid(row=2, column=0, sticky="w", pady=(5, 15))
button_plot_hist.grid_remove()

frame_plot_hist = ttk.Frame(tab_hist, padding="20", style="Modern.TFrame")
frame_plot_hist.pack(side="right", fill="both", expand=True, padx=10, pady=10)

root.mainloop()
