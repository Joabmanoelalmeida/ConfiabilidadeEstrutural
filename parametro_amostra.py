from scipy.stats import norm, lognorm, weibull_min
import numpy as np
from numpy import mean, var, std
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

last_numbers = None

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
    sigma = std(numeros)
    h = 3.5 * sigma / (n ** (1/3))
    data_range = max(numeros) - min(numeros)
    bins = int(np.ceil(data_range / h))
    return bins if bins > 0 else 1

def calcular_pdf_normal(numeros):
    media = mean(numeros)
    std_dev = std(numeros)
    x = np.linspace(media - 4 * std_dev, media + 4 * std_dev, 200)
    pdf = norm.pdf(x, loc=media, scale=std_dev)
    return x, pdf

def calcular_pdf_lognormal(numeros):
    # Seleciona apenas os valores positivos (a distribuição lognormal é definida para x > 0)
    numeros_pos = [x for x in numeros if x > 0]
    if not numeros_pos:
        return None, None
    logs = np.log(numeros_pos)
    mu = np.mean(logs)
    sigma = np.std(logs, ddof=0)
    x = np.linspace(min(numeros_pos), max(numeros_pos), 200)
    pdf = lognorm.pdf(x, s=sigma, loc=0, scale=np.exp(mu))
    return x, pdf

def calcular_cdf_normal(numeros):
    media = mean(numeros)
    std_dev = std(numeros)
    x = np.linspace(min(numeros), max(numeros), 200)
    cdf = norm.cdf(x, loc=media, scale=std_dev)
    return x, cdf

def calcular_cdf_lognormal(numeros):
    numeros_pos = [x for x in numeros if x > 0]
    if not numeros_pos:
        return None, None
    logs = np.log(numeros_pos)
    media = mean(numeros_pos)
    desv = std(numeros_pos)
    s = np.sqrt(np.log(1 + (desv / media)**2))
    media_log = mean(logs)
    x = np.linspace(min(numeros_pos), max(numeros_pos), 200)
    cdf = lognorm.cdf(x, s=s, scale=np.exp(media_log))
    return x, cdf

def calcular_pdf_exponencial(numeros):
    numeros_pos = [x for x in numeros if x >= 0]
    if not numeros_pos:
        return None, None
    scale = mean(numeros_pos)
    x = np.linspace(0, max(numeros_pos), 200)
    pdf = (1 / scale) * np.exp(-x / scale)
    return x, pdf

def calcular_cdf_exponencial(numeros):
    numeros_pos = [x for x in numeros if x >= 0]
    if not numeros_pos:
        return None, None
    scale = mean(numeros_pos)
    x = np.linspace(0, max(numeros_pos), 200)
    cdf = 1 - np.exp(-x / scale)
    return x, cdf

def calcular_pdf_weibull(numeros):
    numeros_pos = [x for x in numeros if x >= 0]
    if not numeros_pos:
        return None, None
    params = weibull_min.fit(numeros_pos, floc=0)
    c = params[0]
    scale = params[2]
    x = np.linspace(min(numeros_pos), max(numeros_pos), 200)
    pdf = weibull_min.pdf(x, c, loc=0, scale=scale)
    return x, pdf

def calcular_cdf_weibull(numeros):
    numeros_pos = [x for x in numeros if x >= 0]
    if not numeros_pos:
        return None, None
    params = weibull_min.fit(numeros_pos, floc=0)
    c = params[0]
    scale = params[2]
    x = np.linspace(min(numeros_pos), max(numeros_pos), 200)
    cdf = weibull_min.cdf(x, c, loc=0, scale=scale)
    return x, cdf

def calcular_histograma_acumulativo(numeros, bins=None):
    data = np.array(numeros)
    data = data[data > 0]
    if len(data) == 0:
        return None, None
    sorted_data = np.sort(data)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n + 1) / n
    return sorted_data, empirical_cdf

def covariancia(media, desvio_padrao):
    return desvio_padrao / media

def teste_kolmogorov_smirnov(numeros, modelo='normal'):
    if not numeros:
        return None

    data = np.array(numeros)
    if modelo in ['lognormal', 'exponencial', 'weibull']:
        data = data[data > 0]
        if len(data) == 0:
            return None

    sorted_data = np.sort(data)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n + 1) / n

    if modelo == 'normal':
        media = mean(data)
        std_dev = std(data)
        theoretical_cdf = norm.cdf(sorted_data, loc=media, scale=std_dev)
    elif modelo == 'lognormal':
        logs = np.log(data)
        media_log = mean(logs)
        # Utiliza o parâmetro s obtido a partir dos dados originais
        std_original = std(data)
        theoretical_cdf = lognorm.cdf(sorted_data, s=np.sqrt(np.log(1 + (std_original / mean(data)) ** 2)),
                                      scale=np.exp(media_log))
    elif modelo == 'exponencial':
        scale = mean(data)
        theoretical_cdf = 1 - np.exp(-sorted_data / scale)
    elif modelo == 'weibull':
        params = weibull_min.fit(data, floc=0)
        c = params[0]
        scale_fit = params[2]
        theoretical_cdf = weibull_min.cdf(sorted_data, c, loc=0, scale=scale_fit)
    differences = np.abs(empirical_cdf - theoretical_cdf)
    ks = np.max(differences)
    return ks

def update_ks_test():
    ks = teste_kolmogorov_smirnov(last_numbers, modelo=dist_type.get()) if last_numbers else None
    ks_display = f"{ks:.4f}" if ks is not None else "N/A"
    for item in tree_teste.get_children():
        tree_teste.delete(item)
    tree_teste.insert('', 'end', values=(ks_display, '0.136'))

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
        update_ks_test()
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
        numeros = None
        for col in df.columns:
            temp = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
            if temp:
                numeros = temp
                break
        if not numeros:
            raise ValueError("Nenhum valor numérico encontrado no arquivo.")
        process_numeros(numeros, "Números importados (Excel)")
        update_ks_test()
    except Exception as e:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', str(e)))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def salvar_funcao():
    def wrap_text(text, canvas_obj, font_name, font_size, max_width):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            if canvas_obj.stringWidth(test_line, font_name, font_size) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

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

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Parâmetros da Amostra")
        y -= 30
        c.setFont("Helvetica", 12)
        max_line_width = width - 100

        for item in tree.get_children():
            parametros, valor = tree.item(item, "values")
            text = f"{parametros}: {valor}"
            linhas = wrap_text(text, c, "Helvetica", 12, max_line_width)
            for linha in linhas:
                c.drawString(50, y, linha)
                y -= 20
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 12)

        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Resultado Teste de Aderência:")
        y -= 30
        c.setFont("Helvetica", 12)
        for item in tree_teste.get_children():
            ks, limite = tree_teste.item(item, "values")
            text = f"KS: {ks}, Limite: {limite}"
            linhas = wrap_text(text, c, "Helvetica", 12, max_line_width)
            for linha in linhas:
                c.drawString(50, y, linha)
                y -= 20
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 12)

        c.showPage()

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
            
            if dist_type.get() == "normal":
                x_pdf, pdf_values = calcular_pdf_normal(last_numbers)
            elif dist_type.get() == "lognormal":
                x_pdf, pdf_values = calcular_pdf_lognormal(last_numbers)
            elif dist_type.get() == "exponencial":
                x_pdf, pdf_values = calcular_pdf_exponencial(last_numbers)
            elif dist_type.get() == "weibull":
                x_pdf, pdf_values = calcular_pdf_weibull(last_numbers)
            ax_pdf.plot(x_pdf, pdf_values, color='red', linewidth=2, label='PDF')
            ax_pdf.legend()
            ax_pdf.set_title("Histograma de Densidade de Probabilidade (PDF)")
            ax_pdf.set_xlabel(f"Valor\nCestas: {len(bins_array)-1}")
            ax_pdf.set_ylabel("Densidade de probabilidade")
            gráficos.append(fig_pdf)

            # Gráfico CDF com histograma acumulativo e função de distribuição (CDF)
            fig_cdf, ax_cdf = plt.subplots(figsize=(8, 6))
            try:
                bins = int(entry_bins_hist.get())
            except ValueError:
                bins = get_histogram_bins(last_numbers)
            if dist_type.get() == "normal":
                x_cdf, cdf_values = calcular_cdf_normal(last_numbers)
            elif dist_type.get() == "lognormal":
                x_cdf, cdf_values = calcular_cdf_lognormal(last_numbers)
            elif dist_type.get() == "exponencial":
                x_cdf, cdf_values = calcular_cdf_exponencial(last_numbers)
            elif dist_type.get() == "weibull":
                x_cdf, cdf_values = calcular_cdf_weibull(last_numbers)
            ax_cdf.plot(x_cdf, cdf_values, color='red', linewidth=2, label='CDF')
            x_hist, y_hist = calcular_histograma_acumulativo(last_numbers, bins)
            ax_cdf.step(x_hist, y_hist, where='post', color='blue', linewidth=2, label='Histograma Acumulativo')
            
            # Cálculo dos pontos de maior diferença (KS)
            data = np.array(last_numbers)
            sorted_data = np.sort(data)
            n = len(sorted_data)
            empirical_cdf = np.arange(1, n + 1) / n
            if dist_type.get() == "normal":
                theoretical_cdf = norm.cdf(sorted_data, loc=mean(data), scale=std(data))
            elif dist_type.get() == "lognormal":
                logs = np.log(sorted_data)
                theoretical_cdf = lognorm.cdf(sorted_data, s=np.sqrt(np.log(1 + (std(data)/mean(data))**2)),
                                              scale=np.exp(mean(logs)))
            elif dist_type.get() == "exponencial":
                scale = mean(data)
                theoretical_cdf = 1 - np.exp(-sorted_data / scale)
            elif dist_type.get() == "weibull":
                params = weibull_min.fit(data, floc=0)
                c_val = params[0]
                scale_fit = params[2]
                theoretical_cdf = weibull_min.cdf(sorted_data, c_val, loc=0, scale=scale_fit)
            differences = np.abs(empirical_cdf - theoretical_cdf)
            idx_max = np.argmax(differences)
            x_max = sorted_data[idx_max]
            emp_val = empirical_cdf[idx_max]
            theo_val = theoretical_cdf[idx_max]
            ax_cdf.plot([x_max], [emp_val], marker='o', markersize=8, color='green', label='Empírica KS')
            ax_cdf.plot([x_max], [theo_val], marker='o', markersize=8, color='purple', label='Teórica KS')
            ax_cdf.plot([x_max, x_max], [emp_val, theo_val], color='black', linestyle='--', linewidth=2)

            ax_cdf.legend()
            ax_cdf.set_title("Curva de Distribuição Acumulada (CDF) e Histograma Acumulativo")
            ax_cdf.set_xlabel("Valor")
            ax_cdf.set_ylabel("Probabilidade Acumulada")
            ax_cdf.grid(True, linestyle='--', alpha=0.6)
            gráficos.append(fig_cdf)

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
        if dist_type.get() == "normal":
            x, pdf = calcular_pdf_normal(last_numbers)
        elif dist_type.get() == "lognormal":
            x, pdf = calcular_pdf_lognormal(last_numbers)
        elif dist_type.get() == "exponencial":
            x, pdf = calcular_pdf_exponencial(last_numbers)
        elif dist_type.get() == "weibull":
            x, pdf = calcular_pdf_weibull(last_numbers)
        ax.plot(x, pdf, color='red', linewidth=2, label='PDF')
        ax.legend()
        ax.set_title("Histograma de Densidade de Probabilidade (PDF)")
        ax.set_xlabel(f"Valor\nCestas: {len(bins_array)-1}")
        ax.set_ylabel("Densidade")
        
        update_ks_test()

    else:  # CDF
        try:
            bins = int(entry_bins_hist.get())
        except ValueError:
            bins = get_histogram_bins(last_numbers)
        if dist_type.get() == "normal":
            x, cdf_values = calcular_cdf_normal(last_numbers)
        elif dist_type.get() == "lognormal":
            x, cdf_values = calcular_cdf_lognormal(last_numbers)
        elif dist_type.get() == "exponencial":
            x, cdf_values = calcular_cdf_exponencial(last_numbers)
        elif dist_type.get() == "weibull":
            x, cdf_values = calcular_cdf_weibull(last_numbers)
        ax.plot(x, cdf_values, color='red', linewidth=2, label='CDF')
        x_hist, y_hist = calcular_histograma_acumulativo(last_numbers, bins)
        ax.step(x_hist, y_hist, where='post', color='blue', linewidth=2, label='Histograma Acumulativo')
        
        data = np.array(last_numbers)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        empirical_cdf = np.arange(1, n + 1) / n
        if dist_type.get() == "normal":
            theoretical_cdf = norm.cdf(sorted_data, loc=mean(data), scale=std(data))
        elif dist_type.get() == "lognormal":
            logs = np.log(sorted_data)
            theoretical_cdf = lognorm.cdf(sorted_data, s=np.sqrt(np.log(1 + (std(data)/mean(data))**2)),
                                          scale=np.exp(mean(logs)))
        elif dist_type.get() == "exponencial":
            scale = mean(data)
            theoretical_cdf = 1 - np.exp(-sorted_data / scale)
        elif dist_type.get() == "weibull":
            params = weibull_min.fit(data, floc=0)
            c_val = params[0]
            scale_fit = params[2]
            theoretical_cdf = weibull_min.cdf(sorted_data, c_val, loc=0, scale=scale_fit)
        differences = np.abs(empirical_cdf - theoretical_cdf)
        idx_max = np.argmax(differences)
        x_max = sorted_data[idx_max]
        emp_val = empirical_cdf[idx_max]
        theo_val = theoretical_cdf[idx_max]
        ax.plot([x_max], [emp_val], marker='o', markersize=8, color='green', label='Empírica KS')
        ax.plot([x_max], [theo_val], marker='o', markersize=8, color='purple', label='Teórica KS')
        ax.plot([x_max, x_max], [emp_val, theo_val], color='black', linestyle='--', linewidth=2)
        
        ax.legend()
        ax.set_title("Curva de Distribuição Acumulada (CDF) e Histograma Acumulativo")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Probabilidade Acumulada")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=frame_plot_hist)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(fill="both", expand=True)

def process_numeros(numeros, descricao):
    global last_numbers
    media_calculada = mean(numeros)
    variancia = var(numeros)
    desvio_padrao_calculado = std(numeros)
    skewness = calcular_skew(numeros, media_calculada, desvio_padrao_calculado)
    kurt = calcular_kurtosis(numeros, media_calculada, desvio_padrao_calculado)
    cov = covariancia(media_calculada, desvio_padrao_calculado)

    for item in tree.get_children():
        tree.delete(item)
    
    tree.insert('', 'end', values=(descricao, str(numeros)))
    tree.insert('', 'end', values=('Média', f"{media_calculada:.4f}"))
    tree.insert('', 'end', values=('Variância', f"{variancia:.4f}"))
    tree.insert('', 'end', values=('Desvio Padrão', f"{desvio_padrao_calculado:.4f}"))
    tree.insert('', 'end', values=('Covariância', f"{cov:.4f}"))
    tree.insert('', 'end', values=('Coeficiente de Skewness', f"{skewness:.4f}"))
    tree.insert('', 'end', values=('Coeficiente de Kurtosis', f"{kurt:.4f}"))
    
    last_numbers = numeros
    plot_histograma_replot()

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

def gerar_amostra_exponencial():
    try:
        quantidade = int(entry_quantidade.get())
        media_informada = float(entry_media.get()) if entry_media.get() else 1  
        numeros_exponencial = np.random.exponential(scale=media_informada, size=quantidade).tolist()
        process_numeros(numeros_exponencial, "Números gerados (Exponencial)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_weibull():
    try:
        quantidade = int(entry_quantidade.get())
        shape = float(entry_desvio.get()) if entry_desvio.get() else 1  
        scale = float(entry_media.get()) if entry_media.get() else 1  
        numeros_weibull = (scale * np.random.weibull(shape, quantidade)).tolist()
        process_numeros(numeros_weibull, "Números gerados (Weibull)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_selecionada():
    if dist_type.get() == "normal":
        gerar_amostra_normal()
    elif dist_type.get() == "lognormal":
        gerar_amostra_lognormal()
    elif dist_type.get() == "exponencial":
        gerar_amostra_exponencial()
    elif dist_type.get() == "weibull":
        gerar_amostra_weibull()

def plot_current():
    if last_numbers is not None:
        plot_histograma_replot()

root = tk.Tk()
root.title("JOAB MANOEL")
root.geometry("1500x700")

frame_import = ttk.Frame(root, padding="10")
frame_import.pack(side="top", fill="x")

# frame "Abrir" e "Salvar" 
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
    if frame_import_options.winfo_ismapped():
        frame_import_options.pack_forget()
        frame_salvar.pack_forget()
        frame_salvar.pack(side="right", padx=10)
    else:
        frame_import_options.pack(side="left", padx=10)
        frame_salvar.pack_forget()
        frame_salvar.pack(side="left", padx=10)

# Abrir
frame_abrir = ttk.Frame(frame_open_side)
frame_abrir.pack(side="left", padx=10)

btn_abrir = ttk.Button(frame_abrir, image=photo_abrir, command=show_import_options)
btn_abrir.image = photo_abrir
btn_abrir.pack()

lbl_abrir = ttk.Label(frame_abrir, text="Abrir", font=("Segoe UI", 11))
lbl_abrir.pack()

# TXT e Excel
frame_import_options = ttk.Frame(frame_open_side, padding="10")

btn_txt = ttk.Button(frame_import_options, image=photo_txt, command=importar_txt)
btn_txt.image = photo_txt
btn_txt.pack(side="left", padx=10)

btn_excel = ttk.Button(frame_import_options, image=photo_excel, command=importar_excel)
btn_excel.image = photo_excel
btn_excel.pack(side="left", padx=10)

# "Salvar"
frame_salvar = ttk.Frame(frame_open_side)

btn_salvar = ttk.Button(frame_salvar, image=photo_salvar, command=salvar_funcao)
btn_salvar.image = photo_salvar
btn_salvar.pack()

lbl_salvar = ttk.Label(frame_salvar, text="Salvar", font=("Segoe UI", 11))
lbl_salvar.pack()


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

radio_lognormal = ttk.Radiobutton(
    frame_controles,
    text="Lognormal",
    value="lognormal",
    variable=dist_type,
    style="TRadiobutton"
)
radio_lognormal.grid(row=2, column=0, sticky="w", pady=(0, 5))

radio_exponencial = ttk.Radiobutton(
    frame_controles,
    text="Exponencial",
    value="exponencial",
    variable=dist_type,
    style="TRadiobutton"
)
radio_exponencial.grid(row=3, column=0, sticky="w", pady=(0, 5))

radio_weibull = ttk.Radiobutton(
    frame_controles,
    text="Weibull",
    value="weibull",
    variable=dist_type,
    style="TRadiobutton"
)
radio_weibull.grid(row=4, column=0, sticky="w", pady=(0, 15))

var_parametros = tk.BooleanVar(value=False)
def toggle_quantidade():
    if var_parametros.get():
        frame_quantidade.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 15))
    else:
        frame_quantidade.grid_forget()

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

label_media = ttk.Label(frame_quantidade, text="Média/Scale:", style="TLabel")
label_media.grid(row=1, column=0, sticky="w", pady=(0, 3))
entry_media = ttk.Entry(frame_quantidade, width=10, font=("Segoe UI", 9))
entry_media.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))

label_desvio = ttk.Label(frame_quantidade, text="Desvio/Shape:", style="TLabel")
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

def gerar_amostra_exponencial():
    try:
        quantidade = int(entry_quantidade.get())
        # Na distribuição exponencial, a "média" é o parâmetro scale (λ = 1/scale)
        media_informada = float(entry_media.get()) if entry_media.get() else 1  
        numeros_exponencial = np.random.exponential(scale=media_informada, size=quantidade).tolist()
        process_numeros(numeros_exponencial, "Números gerados (Exponencial)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_weibull():
    try:
        quantidade = int(entry_quantidade.get())
        shape = float(entry_desvio.get()) if entry_desvio.get() else 1  
        scale = float(entry_media.get()) if entry_media.get() else 1  
        numeros_weibull = (scale * np.random.weibull(shape, quantidade)).tolist()
        process_numeros(numeros_weibull, "Números gerados (Weibull)")
    except ValueError:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert('', 'end', values=('Erro', 'Insira números válidos.'))
        for widget in frame_plot_hist.winfo_children():
            widget.destroy()

def gerar_amostra_selecionada():
    if dist_type.get() == "normal":
        gerar_amostra_normal()
    elif dist_type.get() == "lognormal":
        gerar_amostra_lognormal()
    elif dist_type.get() == "exponencial":
        gerar_amostra_exponencial()
    elif dist_type.get() == "weibull":
        gerar_amostra_weibull()

button_calcular = ttk.Button(
    frame_controles,
    text="Calcular",
    command=gerar_amostra_selecionada,
    style="Accent.TButton"
)
button_calcular.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 15))

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
    else:
        entry_bins_hist.grid_remove()

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

opcao_hist = tk.StringVar(value="PDF")
lbl_plot_option = ttk.Label(frame_controls_hist, text="Visualização:")
lbl_plot_option.grid(row=3, column=0, sticky="w", pady=(15, 5))

radio_pdf = ttk.Radiobutton(frame_controls_hist, text="PDF", variable=opcao_hist, value="PDF", command=plot_current)
radio_pdf.grid(row=4, column=0, sticky="w")

radio_cdf = ttk.Radiobutton(frame_controls_hist, text="CDF", variable=opcao_hist, value="CDF", command=plot_current)
radio_cdf.grid(row=5, column=0, sticky="w")

frame_plot_hist = ttk.Frame(tab_hist, padding="20", style="Modern.TFrame")
frame_plot_hist.pack(side="right", fill="both", expand=True, padx=10, pady=10)

style = ttk.Style()
style.configure("Centered.TLabelframe.Label", anchor="center")
frame_teste = ttk.LabelFrame(frame_controls_hist, text="Resultado\nTeste de aderência:", style="Centered.TLabelframe")
frame_teste.grid(row=6, column=0, pady=(15, 5), sticky="ew")

tree_teste = ttk.Treeview(frame_teste, columns=("KS", "Limite"), show="headings", height=2)
tree_teste.heading("KS", text="KS")
tree_teste.column("KS", anchor="center", width=150)
tree_teste.heading("Limite", text="Limite")
tree_teste.column("Limite", anchor="center", width=150)
tree_teste.grid(row=0, column=0, padx=5, pady=5)

def update_ks_test():
    ks = teste_kolmogorov_smirnov(last_numbers, modelo=dist_type.get()) if last_numbers else None
    ks_display = f"{ks:.4f}" if ks is not None else "N/A"
    for item in tree_teste.get_children():
        tree_teste.delete(item)
    tree_teste.insert('', 'end', values=(ks_display, '0.136'))

root.mainloop()
