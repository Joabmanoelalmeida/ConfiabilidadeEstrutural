from scipy.stats import tmean, tvar, tstd, norm, lognorm
import tkinter as tk
from tkinter import ttk, filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk  
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os

def calcular_skew(numeros, media, desvio_padrao):
    quantidade = len(numeros)
    skew = (1 / (quantidade * desvio_padrao**3)) * sum((x - media)**3 for x in numeros)
    return skew

def calcular_kurtosis(numeros, media, desvio_padrao):
    quantidade = len(numeros)
    kurt = (1 / (quantidade * desvio_padrao**4)) * sum((x - media)**4 for x in numeros)
    return kurt

def get_histogram_bins(numeros):
    n = len(numeros)
    if n == 0:
        return 1
    sigma = tstd(numeros)
    h = 3.5 * sigma / (n ** (1/3))
    data_range = max(numeros) - min(numeros)
    bins = int(np.ceil(data_range / h))
    return bins if bins > 0 else 1

def calcular_pdf_normal(numeros):
    media = tmean(numeros)
    std_dev = tstd(numeros)
    # Expande o intervalo para cobrir aproximadamente 99.99% da distribuição
    x = np.linspace(media - 4 * std_dev, media + 4 * std_dev, 200)
    pdf = norm.pdf(x, loc=media, scale=std_dev)
    return x, pdf

def calcular_pdf_lognormal(numeros):
    numeros_pos = [x for x in numeros if x > 0]
    if not numeros_pos:
        return None, None
    logs = np.log(numeros_pos)
    media_log = tmean(logs)
    desvio_log = tstd(logs)
    x = np.linspace(min(numeros_pos), max(numeros_pos), 200)
    pdf = lognorm.pdf(x, s=desvio_log, scale=np.exp(media_log))
    return x, pdf

def calcular_cdf_normal(numeros):
    media = tmean(numeros)
    std_dev = tstd(numeros)
    x = np.linspace(min(numeros), max(numeros), 200)
    cdf = norm.cdf(x, loc=media, scale=std_dev)
    return x, cdf

def calcular_cdf_lognormal(numeros):
    numeros_pos = [x for x in numeros if x > 0]
    if not numeros_pos:
        return None, None
    logs = np.log(numeros_pos)
    media_log = tmean(logs)
    desvio_log = tstd(logs)
    x = np.linspace(min(numeros_pos), max(numeros_pos), 200)
    cdf = lognorm.cdf(x, s=desvio_log, scale=np.exp(media_log))
    return x, cdf

def calcular_histograma_acumulativo(numeros, bins=None):
  
    if bins is None:
        bins = get_histogram_bins(numeros)
    
    hist, bin_edges = np.histogram(numeros, bins=bins)
    hist_rel = hist / len(numeros)
    cumulative = np.cumsum(hist_rel)
    # Preparando dados para o gráfico de degraus
    x = bin_edges
    y = np.hstack([0, cumulative])
    return x, y


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
        for widget in frame_plot_hist.winfo_children():
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
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def salvar_funcao():
    # Função auxiliar para realizar a quebra de linha com base na largura máxima permitida.
    def wrap_text(text, canvas, font_name, font_size, max_width):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            if canvas.stringWidth(test_line, font_name, font_size) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    # Permite salvar os resultados e gráficos gerados em um arquivo PDF, incluindo
    # os gráficos de PDF, CDF e histogramas da aba de Histograma.
    # Solicita que o usuário forneça o nome do arquivo PDF
    pdf_file = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        title="Salvar arquivo PDF",
        initialfile="resultados.pdf"
    )
    if not pdf_file:
        return

    try:
        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter
        y = height - 50

        # Cabeçalho do PDF com os resultados da TreeView
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Parâmetros da Amostra")
        y -= 30
        c.setFont("Helvetica", 12)
        max_line_width = width - 100  # margem de 50pt em cada lado

        for item in tree.get_children():
            parametros, valor = tree.item(item, "values")
            text = f"{parametros}: {valor}"
            # Quebra o texto se exceder a largura máxima definida
            linhas = wrap_text(text, c, "Helvetica", 12, max_line_width)
            for linha in linhas:
                c.drawString(50, y, linha)
                y -= 20
                # Verifica se precisa criar uma nova página
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 12)

        c.showPage()  # Inicia nova página para os gráficos

        # Se houver amostras, gera os gráficos PDF e CDF com os histogramas
        if last_numbers is not None:
            gráficos = []

            # Gráfico PDF com histograma e função de densidade (PDF)
            fig_pdf, ax_pdf = plt.subplots(figsize=(8, 6))
            try:
                bins = int(entry_bins_hist.get())
            except ValueError:
                bins = get_histogram_bins(last_numbers)
            counts, bins_array = np.histogram(last_numbers, bins=bins)
            bin_widths = np.diff(bins_array)
            rel_freq = counts / counts.sum()
            density = rel_freq / bin_widths
            ax_pdf.bar(bins_array[:-1], density, width=bin_widths, align='edge', edgecolor='black')
            x_pdf, pdf_values = calcular_pdf_normal(last_numbers)
            ax_pdf.plot(x_pdf, pdf_values, color='red', linewidth=2, label='PDF')
            ax_pdf.legend()
            ax_pdf.set_title("Histograma de Densidade de Probabilidade (PDF)")
            ax_pdf.set_xlabel(f"Valor\nCestas: {len(bins_array)-1}")
            ax_pdf.set_ylabel("Densidade")
            gráficos.append(fig_pdf)

            # Gráfico CDF com histograma acumulativo e função de distribuição (CDF)
            fig_cdf, ax_cdf = plt.subplots(figsize=(8, 6))
            try:
                bins = int(entry_bins_hist.get())
            except ValueError:
                bins = get_histogram_bins(last_numbers)
            x_cdf, cdf_values = calcular_cdf_normal(last_numbers)
            ax_cdf.plot(x_cdf, cdf_values, color='red', linewidth=2, label='CDF')
            x_hist, y_hist = calcular_histograma_acumulativo(last_numbers, bins)
            ax_cdf.step(x_hist, y_hist, where='post', color='blue', linewidth=2, label='Histograma Acumulativo')
            ax_cdf.legend()
            ax_cdf.set_title("Curva de Distribuição Acumulada (CDF) e Histograma Acumulativo")
            ax_cdf.set_xlabel("Valor")
            ax_cdf.set_ylabel("Probabilidade Acumulada")
            ax_cdf.grid(True, linestyle='--', alpha=0.6)
            gráficos.append(fig_cdf)

            # Para cada gráfico gerado, insere uma nova página no PDF
            for index, fig in enumerate(gráficos):
                temp_file = os.path.join(tempfile.gettempdir(), f"grafico_extra_{index}.png")
                fig.savefig(temp_file)
                img = ImageReader(temp_file)
                img_width = width - 100
                img_height = height - 200
                c.drawImage(img, 50, 100, width=img_width, height=img_height, preserveAspectRatio=True)
                c.showPage()
                os.remove(temp_file)
                plt.close(fig)

        c.save()
        print("Arquivo PDF salvo com sucesso!")
    except Exception as e:
        print("Erro ao salvar arquivo PDF:", e)

# A nova função unificada que plota PDF ou CDF na aba Histograma
def plot_histograma_replot():
    global last_numbers
    try:
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()
    except tk.TclError:
        return

    if last_numbers is None:
        lbl = ttk.Label(frame_plot_hist, text="Nenhum conjunto de dados disponível.", foreground="red")
        lbl.pack()
        return

    opcao = opcao_hist.get()
    fig, ax = plt.subplots(figsize=(8, 6))

    if opcao == "PDF":
        try:
            bins = int(entry_bins_hist.get())
        except ValueError:
            bins = get_histogram_bins(last_numbers)
        counts, bins_array = np.histogram(last_numbers, bins=bins)
        bin_widths = np.diff(bins_array)
        rel_freq = counts / counts.sum()
        density = rel_freq / bin_widths

        ax.bar(bins_array[:-1], density, width=bin_widths, align='edge', edgecolor='black')
        x, pdf = calcular_pdf_normal(last_numbers)
        ax.plot(x, pdf, color='red', linewidth=2, label='PDF')
        ax.legend()
        ax.set_title("Histograma de Densidade de Probabilidade (PDF)")
        ax.set_xlabel(f"Valor\nCestas: {len(bins_array)-1}")
        ax.set_ylabel("Densidade")
  
    else:  # opcao == "CDF"
        try:
            bins = int(entry_bins_hist.get())
        except ValueError:
            bins = get_histogram_bins(last_numbers)

        # Plot da CDF normal
        x, cdf_values = calcular_cdf_normal(last_numbers)
        ax.plot(x, cdf_values, color='red', linewidth=2, label='CDF')
        
        # Plot do histograma acumulativo
        x_hist, y_hist = calcular_histograma_acumulativo(last_numbers, bins)
        ax.step(x_hist, y_hist, where='post', color='blue', linewidth=2, label='Histograma Acumulativo')
        
        ax.legend()
        ax.set_title("Curva de Distribuição Acumulada (CDF) e Histograma Acumulativo")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Probabilidade Acumulada")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame_plot_hist)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def process_numeros(numeros, descricao):
    global last_numbers
    media_calculada = tmean(numeros)
    variancia = tvar(numeros)
    desvio_padrao_calculado = tstd(numeros)
    skewness = calcular_skew(numeros, media_calculada, desvio_padrao_calculado)
    kurt = calcular_kurtosis(numeros, media_calculada, desvio_padrao_calculado)

    for item in tree.get_children():
        tree.delete(item)
    
    tree.insert('', 'end', values=(descricao, str(numeros)))
    tree.insert('', 'end', values=('Média (calculada)', f"{media_calculada:.4f}"))
    tree.insert('', 'end', values=('Variância', f"{variancia:.4f}"))
    tree.insert('', 'end', values=('Desvio Padrão (calculado)', f"{desvio_padrao_calculado:.4f}"))
    tree.insert('', 'end', values=('Coeficiente de Skewness', f"{skewness:.4f}"))
    tree.insert('', 'end', values=('Coeficiente de Kurtosis', f"{kurt:.4f}"))
    
    last_numbers = numeros
    plot_histograma_replot()

def gerar_numeros_aleatorios():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 0  
        desvio_informado = float(entry_desvio.get()) if entry_desvio.get() else 1
        
        # Gerar amostra com distribuição normal
        numeros_normal = norm.rvs(loc=media_informada, scale=desvio_informado, size=quantidade).tolist()
        process_numeros(numeros_normal, "Números gerados (Normal)")
        
        # Gerar amostra com distribuição lognormal
        numeros_lognormal = lognorm.rvs(s=desvio_informado, scale=np.exp(media_informada), size=quantidade).tolist()
        
        # Acrescentar os resultados da distribuição lognormal sem limpar a árvore
        tree.insert('', 'end', values=("Números gerados (Lognormal)", str(numeros_lognormal)))
        
        media_log = tmean(numeros_lognormal)
        variancia_log = tvar(numeros_lognormal)
        desvio_log = tstd(numeros_lognormal)
        skewness_log = calcular_skew(numeros_lognormal, media_log, desvio_log)
        kurt_log = calcular_kurtosis(numeros_lognormal, media_log, desvio_log)
        
        tree.insert('', 'end', values=('Média (Lognormal)', f"{media_log:.4f}"))
        tree.insert('', 'end', values=('Variância (Lognormal)', f"{variancia_log:.4f}"))
        tree.insert('', 'end', values=('Desvio Padrão (Lognormal)', f"{desvio_log:.4f}"))
        tree.insert('', 'end', values=('Coeficiente de Skewness (Lognormal)', f"{skewness_log:.4f}"))
        tree.insert('', 'end', values=('Coeficiente de Kurtosis (Lognormal)', f"{kurt_log:.4f}"))
        
        # Exibe o gráfico baseado na amostra normal (a opção de plot é controlada pelos botões de opção)
        last_numbers = numeros_normal
        plot_histograma_replot()
        
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_normal():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 0  
        desvio_informado = float(entry_desvio.get()) if entry_desvio.get() else 1
        
        numeros_normal = norm.rvs(loc=media_informada, scale=desvio_informado, size=quantidade).tolist()
        process_numeros(numeros_normal, "Números gerados (Normal)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_lognormal():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 0  
        desvio_informado = float(entry_desvio.get()) if entry_desvio.get() else 1
        
        numeros_lognormal = lognorm.rvs(s=desvio_informado, scale=np.exp(media_informada), size=quantidade).tolist()
        process_numeros(numeros_lognormal, "Números gerados (Lognormal)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_selecionada():
    if dist_type.get() == "normal":
        gerar_amostra_normal()
    else:
        gerar_amostra_lognormal()

def plot_current():
    # Atualiza o gráfico conforme a opção selecionada
    if last_numbers is not None:
        plot_histograma_replot()

root = tk.Tk()
root.title("JOAB MANOEL")
root.geometry("1500x700")

frame_import = ttk.Frame(root, padding="10")
frame_import.pack(side="top", fill="x")

# Create a left frame that holds the "Abrir", import options and "Salvar" buttons
frame_open_side = ttk.Frame(frame_import)
frame_open_side.pack(side="left", padx=1)

image_path = os.path.join(os.getcwd(), "Imagens")
abrir_image_path = os.path.join(image_path, "Abrir.jpg")
txt_image_path = os.path.join(image_path, "txt.jpg")
excel_image_path = os.path.join(image_path, "excel.jpg")
salvar_image_path = os.path.join(image_path, "salvar.jpg")

img_abrir = Image.open(abrir_image_path)
img_abrir = img_abrir.resize((40, 40), Image.Resampling.LANCZOS)
photo_abrir = ImageTk.PhotoImage(img_abrir)

img_txt = Image.open(txt_image_path)
img_txt = img_txt.resize((40, 40), Image.Resampling.LANCZOS)
photo_txt = ImageTk.PhotoImage(img_txt)

img_excel = Image.open(excel_image_path)
img_excel = img_excel.resize((40, 40), Image.Resampling.LANCZOS)
photo_excel = ImageTk.PhotoImage(img_excel)

img_salvar = Image.open(salvar_image_path)
img_salvar = img_salvar.resize((40, 40), Image.Resampling.LANCZOS)
photo_salvar = ImageTk.PhotoImage(img_salvar)

def show_import_options():
    # Toggle the visibility of the import options. In addition, reposition the "Salvar" button:
    # Before clicking "Abrir", "Salvar" is packed on the right.
    # When clicking "Abrir", it is repacked to the left, next to the Excel button.
    if frame_import_options.winfo_ismapped():
        frame_import_options.pack_forget()
        frame_salvar.pack_forget()
        frame_salvar.pack(side="right", padx=10)
    else:
        frame_import_options.pack(side="left", padx=10)
        frame_salvar.pack_forget()
        frame_salvar.pack(side="left", padx=10)

# Frame for the "Abrir" button
frame_abrir = ttk.Frame(frame_open_side)
frame_abrir.pack(side="left", padx=10)

btn_abrir = ttk.Button(frame_abrir, image=photo_abrir, command=show_import_options)
btn_abrir.image = photo_abrir
btn_abrir.pack()

lbl_abrir = ttk.Label(frame_abrir, text="Abrir", font=("Segoe UI", 11))
lbl_abrir.pack()

# Frame for import options (TXT and Excel), initially not packed
frame_import_options = ttk.Frame(frame_open_side, padding="10")

btn_txt = ttk.Button(frame_import_options, image=photo_txt, command=importar_txt)
btn_txt.image = photo_txt
btn_txt.pack(side="left", padx=10)

btn_excel = ttk.Button(frame_import_options, image=photo_excel, command=importar_excel)
btn_excel.image = photo_excel
btn_excel.pack(side="left", padx=10)

# Frame for the "Salvar" button.
frame_salvar = ttk.Frame(frame_open_side)

btn_salvar = ttk.Button(frame_salvar, image=photo_salvar, command=salvar_funcao)
btn_salvar.image = photo_salvar
btn_salvar.pack()

lbl_salvar = ttk.Label(frame_salvar, text="Salvar", font=("Segoe UI", 11))
lbl_salvar.pack()

# Initially, the "Salvar" option appears before clicking "Abrir" by being packed on the right.
frame_salvar.pack(side="right", padx=10)

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
notebook_principal.add(tab_hist, text="Histograma")

# -- Aba de Parâmetros --
frame_controles = ttk.Frame(tab_parametros, padding="20", style="Card.TFrame")
frame_controles.pack(side="left", fill="y", padx=10, pady=10)

dist_type = tk.StringVar(value="normal")

radio_normal = ttk.Radiobutton(
    frame_controles,
    text="Normal",
    value="normal",
    variable=dist_type,
    style="TRadiobutton"
)
radio_normal.grid(row=1, column=0, sticky="w", pady=(0, 5))
radio_normal.grid_remove()

radio_lognormal = ttk.Radiobutton(
    frame_controles,
    text="Lognormal",
    value="lognormal",
    variable=dist_type,
    style="TRadiobutton"
)
radio_lognormal.grid(row=2, column=0, sticky="w", pady=(0, 15))
radio_lognormal.grid_remove()

var_parametros = tk.BooleanVar(value=False)
def toggle_quantidade():
    if var_parametros.get():
        frame_quantidade.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        radio_normal.grid()
        radio_lognormal.grid()
    else:
        frame_quantidade.grid_forget()
        radio_normal.grid_remove()
        radio_lognormal.grid_remove()

check_parametros = ttk.Checkbutton(
    frame_controles,
    text="Gerar amostra com valores aleatórios",
    variable=var_parametros,
    command=toggle_quantidade,
    style="Toggle.TCheckbutton"
)
check_parametros.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(15, 5))

frame_quantidade = ttk.Frame(frame_controles)
label_quantidade = ttk.Label(frame_quantidade, text="Nº pontos:", style="TLabel")
label_quantidade.grid(row=0, column=0, sticky="w", pady=(0, 3))
entry_quantidade = ttk.Entry(frame_quantidade, width=10, font=("Segoe UI", 9))
entry_quantidade.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))

label_media = ttk.Label(frame_quantidade, text="Média:", style="TLabel")
label_media.grid(row=1, column=0, sticky="w", pady=(0, 3))
entry_media = ttk.Entry(frame_quantidade, width=10, font=("Segoe UI", 9))
entry_media.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))

label_desvio = ttk.Label(frame_quantidade, text="Desvio Padrão:", style="TLabel")
label_desvio.grid(row=2, column=0, sticky="w", pady=(0, 3))
entry_desvio = ttk.Entry(frame_quantidade, width=10, font=("Segoe UI", 9))
entry_desvio.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))

def gerar_amostra_normal():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 0  
        desvio_informado = float(entry_desvio.get()) if entry_desvio.get() else 1
        
        numeros_normal = norm.rvs(loc=media_informada, scale=desvio_informado, size=quantidade).tolist()
        process_numeros(numeros_normal, "Números gerados (Normal)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_lognormal():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 0  
        desvio_informado = float(entry_desvio.get()) if entry_desvio.get() else 1
        
        numeros_lognormal = lognorm.rvs(s=desvio_informado, scale=np.exp(media_informada), size=quantidade).tolist()
        process_numeros(numeros_lognormal, "Números gerados (Lognormal)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_selecionada():
    if dist_type.get() == "normal":
        gerar_amostra_normal()
    else:
        gerar_amostra_lognormal()

button_calcular = ttk.Button(
    frame_controles,
    text="Calcular",
    command=gerar_amostra_selecionada,
    style="Accent.TButton"
)
button_calcular.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(5, 15))

frame_quantidade.grid_forget()

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
    text="Atualizar Gráfico",
    command=plot_histograma_replot,
    style="Accent.TButton"
)
button_plot_hist.grid(row=2, column=0, sticky="w", pady=(5, 15))
button_plot_hist.grid_remove()

# Novos botões de opção para escolher entre PDF e CDF
opcao_hist = tk.StringVar(value="PDF")
lbl_plot_option = ttk.Label(frame_controls_hist, text="Visualização:")
lbl_plot_option.grid(row=3, column=0, sticky="w", pady=(15, 5))

radio_pdf = ttk.Radiobutton(frame_controls_hist, text="PDF", variable=opcao_hist, value="PDF", command=plot_current)
radio_pdf.grid(row=4, column=0, sticky="w")

radio_cdf = ttk.Radiobutton(frame_controls_hist, text="CDF", variable=opcao_hist, value="CDF", command=plot_current)
radio_cdf.grid(row=5, column=0, sticky="w")

frame_plot_hist = ttk.Frame(tab_hist, padding="20", style="Modern.TFrame")
frame_plot_hist.pack(side="right", fill="both", expand=True, padx=10, pady=10)

last_numbers = None

root.mainloop()
