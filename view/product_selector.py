"""
Interface de seleção de produtos
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import json

# Definir fonte padrão que suporta acentos
DEFAULT_FONT = ('DejaVu Sans', 10)
DEFAULT_FONT_BOLD = ('DejaVu Sans', 10, 'bold')
DEFAULT_FONT_LARGE = ('DejaVu Sans', 14)
DEFAULT_FONT_LARGE_BOLD = ('DejaVu Sans', 14, 'bold')
DEFAULT_FONT_TITLE = ('DejaVu Sans', 20, 'bold')

def get_products():
    """Obtém os produtos cadastrados""" 
    import os
    # Caminho relativo ao diretório atual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir um nível para chegar ao diretório raiz do projeto
    project_root = os.path.dirname(current_dir)
    products_path = os.path.join(project_root, 'resources', 'products', 'products.json')
    
    try:
        with open(products_path, 'r') as file:
            data = json.load(file)
            # Se for dict indexado por class_id, converter para lista ordenada
            if isinstance(data, dict):
                try:
                    # Ordena por chave numérica
                    ordered_items = sorted(
                        ((int(k), v) for k, v in data.items()), key=lambda x: x[0]
                    )
                    return [v for _, v in ordered_items]
                except Exception:
                    # Fallback: apenas valores
                    return list(data.values())
            return data
    except FileNotFoundError:
        # Se o arquivo não existir, retornar lista vazia
        return []

class ProductSelector:
    def __init__(self, parent_window, on_product_selected=None):
        self.parent = parent_window
        self.on_product_selected = on_product_selected
        
        self.create_selector_window()
    
    def create_selector_window(self):
        """Cria a janela de seleção de produtos"""
        self.selector_window = tk.Toplevel(self.parent)
        self.selector_window.title("Selecionar Produto")
        self.selector_window.geometry("900x700")
        self.selector_window.configure(bg='#2c3e50')
        
        # Título
        title_label = tk.Label(
            self.selector_window,
            text="Selecionar Produto",
            font=DEFAULT_FONT_TITLE,
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Frame para botões de ação
        action_frame = tk.Frame(self.selector_window, bg='#2c3e50')
        action_frame.pack(pady=10)
        
        # Botão para recarregar
        reload_button = tk.Button(
            action_frame,
            text="RECARREGAR",
            font=DEFAULT_FONT_BOLD,
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            command=self.load_products,
            relief='flat',
            cursor='hand2'
        )
        reload_button.pack(side='left', padx=10)
        
        # Frame para produtos
        self.products_frame = tk.Frame(self.selector_window, bg='#2c3e50')
        self.products_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Carregar produtos
        self.load_products()
    
    def load_products(self):
        """Carrega e exibe os produtos"""
        # Limpar frame
        for widget in self.products_frame.winfo_children():
            widget.destroy()
        
        products = get_products()
        
        if not products:
            # Mostrar mensagem se não há produtos
            no_products_label = tk.Label(
                self.products_frame,
                text="Nenhum produto cadastrado.\nClique em 'ADICIONAR PRODUTO' para começar.",
                font=DEFAULT_FONT_LARGE,
                fg='#bdc3c7',
                bg='#2c3e50',
                justify='center'
            )
            no_products_label.pack(expand=True)
            return
        
        # Organizar produtos em grid
        for i, product in enumerate(products):
            row = i // 2
            col = i % 2
            
            self.create_product_button(product, row, col)
    
    def create_product_button(self, product, row, col):
        """Cria botão para um produto"""
        # Frame do produto
        product_frame = tk.Frame(
            self.products_frame,
            bg='#34495e',
            relief='raised',
            bd=2,
            width=300,
            height=400
        )
        product_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        product_frame.grid_propagate(False)  # Manter tamanho fixo
        
        # Configurar grid
        self.products_frame.grid_rowconfigure(row, weight=1)
        self.products_frame.grid_columnconfigure(col, weight=1)
        
        # Imagem do produto
        try:
            # Caminho relativo ao diretório atual
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            img_path = os.path.join(project_root, product["img_path"])
            
            # Carregar e redimensionar imagem
            img = Image.open(img_path)
            img = img.resize((80, 80), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Label para imagem
            img_label = tk.Label(
                product_frame,
                image=photo,
                bg='#34495e'
            )
            img_label.image = photo  # Manter referência
            img_label.pack(pady=5)
        except Exception as e:
            print(f"Erro ao carregar imagem {product['img_path']}: {e}")
            # Label de erro se imagem não carregar
            error_label = tk.Label(
                product_frame,
                text="Imagem não encontrada",
                font=DEFAULT_FONT,
                fg='#e74c3c',
                bg='#34495e'
            )
            error_label.pack(pady=10)
        
        # Nome do produto
        name_label = tk.Label(
            product_frame,
            text=product["name"],
            font=DEFAULT_FONT_LARGE_BOLD,
            fg='white',
            bg='#34495e',
            wraplength=250
        )
        name_label.pack(pady=2)
        
        # Preço
        price_label = tk.Label(
            product_frame,
            text=product["price"],
            font=DEFAULT_FONT_LARGE,
            fg='#f39c12',
            bg='#34495e'
        )
        price_label.pack(pady=2)
        
        # Botão selecionar
        select_button = tk.Button(
            product_frame,
            text="SELECIONAR",
            font=DEFAULT_FONT_BOLD,
            bg='#27ae60',
            fg='white',
            padx=15,
            pady=8,
            command=lambda p=product: self.select_product(p),
            relief='flat',
            cursor='hand2'
        )
        select_button.pack(pady=5)
    
    def select_product(self, product):
        """Seleciona um produto"""
        if self.on_product_selected:
            self.on_product_selected(product)
        
        self.selector_window.destroy()
    