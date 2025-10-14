"""
Interface de seleção de produtos
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
from product_manager import ProductManager

class ProductSelector:
    def __init__(self, parent_window, on_product_selected=None):
        self.parent = parent_window
        self.on_product_selected = on_product_selected
        self.product_manager = ProductManager()
        
        self.create_selector_window()
    
    def create_selector_window(self):
        """Cria a janela de seleção de produtos"""
        self.selector_window = tk.Toplevel(self.parent)
        self.selector_window.title("Selecionar Produto")
        self.selector_window.geometry("800x600")
        self.selector_window.configure(bg='#2c3e50')
        
        # Título
        title_label = tk.Label(
            self.selector_window,
            text="Selecionar Produto",
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Frame para botões de ação
        action_frame = tk.Frame(self.selector_window, bg='#2c3e50')
        action_frame.pack(pady=10)
        
        # Botão para adicionar produto
        add_button = tk.Button(
            action_frame,
            text="ADICIONAR PRODUTO",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            command=self.open_add_product_dialog,
            relief='flat',
            cursor='hand2'
        )
        add_button.pack(side='left', padx=10)
        
        # Botão para recarregar
        reload_button = tk.Button(
            action_frame,
            text="RECARREGAR",
            font=('Arial', 12, 'bold'),
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
        
        products = self.product_manager.get_products()
        
        if not products:
            # Mostrar mensagem se não há produtos
            no_products_label = tk.Label(
                self.products_frame,
                text="Nenhum produto cadastrado.\nClique em 'ADICIONAR PRODUTO' para começar.",
                font=('Arial', 14),
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
            bd=2
        )
        product_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        # Configurar grid
        self.products_frame.grid_rowconfigure(row, weight=1)
        self.products_frame.grid_columnconfigure(col, weight=1)
        
        # Nome do produto
        name_label = tk.Label(
            product_frame,
            text=product["name"],
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#34495e'
        )
        name_label.pack(pady=10)
        
        # Preço
        price_label = tk.Label(
            product_frame,
            text=product["price"],
            font=('Arial', 12),
            fg='#f39c12',
            bg='#34495e'
        )
        price_label.pack(pady=5)
        
        # Botão selecionar
        select_button = tk.Button(
            product_frame,
            text="SELECIONAR",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            command=lambda p=product: self.select_product(p),
            relief='flat',
            cursor='hand2'
        )
        select_button.pack(pady=10)
    
    def select_product(self, product):
        """Seleciona um produto"""
        if self.on_product_selected:
            self.on_product_selected(product)
        
        self.selector_window.destroy()
    
    def open_add_product_dialog(self):
        """Abre diálogo para adicionar produto"""
        dialog = AddProductDialog(self.selector_window, self.product_manager)
        self.selector_window.wait_window(dialog.dialog)
        self.load_products()  # Recarregar lista após adicionar

class AddProductDialog:
    def __init__(self, parent, product_manager):
        self.parent = parent
        self.product_manager = product_manager
        self.selected_image_path = None
        
        self.create_dialog()
    
    def create_dialog(self):
        """Cria o diálogo de adicionar produto"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Adicionar Produto")
        self.dialog.geometry("500x400")
        self.dialog.configure(bg='#2c3e50')
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Centralizar na tela
        self.dialog.geometry("+%d+%d" % (self.parent.winfo_rootx() + 50, self.parent.winfo_rooty() + 50))
        
        # Título
        title_label = tk.Label(
            self.dialog,
            text="Adicionar Novo Produto",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Frame principal
        main_frame = tk.Frame(self.dialog, bg='#2c3e50')
        main_frame.pack(pady=20, padx=30, fill='both', expand=True)
        
        # Nome do produto
        name_label = tk.Label(
            main_frame,
            text="Nome do Produto:",
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        name_label.pack(anchor='w', pady=(0, 5))
        
        self.name_entry = tk.Entry(
            main_frame,
            font=('Arial', 12),
            width=40
        )
        self.name_entry.pack(pady=(0, 15))
        
        # Preço
        price_label = tk.Label(
            main_frame,
            text="Preço:",
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        price_label.pack(anchor='w', pady=(0, 5))
        
        self.price_entry = tk.Entry(
            main_frame,
            font=('Arial', 12),
            width=40
        )
        self.price_entry.pack(pady=(0, 15))
        
        # Imagem
        image_label = tk.Label(
            main_frame,
            text="Imagem do Produto:",
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        image_label.pack(anchor='w', pady=(0, 5))
        
        # Frame para seleção de imagem
        image_frame = tk.Frame(main_frame, bg='#2c3e50')
        image_frame.pack(fill='x', pady=(0, 15))
        
        self.image_path_label = tk.Label(
            image_frame,
            text="Nenhuma imagem selecionada",
            font=('Arial', 10),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        self.image_path_label.pack(side='left', fill='x', expand=True)
        
        select_image_button = tk.Button(
            image_frame,
            text="SELECIONAR",
            font=('Arial', 10, 'bold'),
            bg='#3498db',
            fg='white',
            padx=15,
            pady=5,
            command=self.select_image,
            relief='flat',
            cursor='hand2'
        )
        select_image_button.pack(side='right')
        
        # Botões
        buttons_frame = tk.Frame(main_frame, bg='#2c3e50')
        buttons_frame.pack(pady=20)
        
        save_button = tk.Button(
            buttons_frame,
            text="SALVAR",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=30,
            pady=10,
            command=self.save_product,
            relief='flat',
            cursor='hand2'
        )
        save_button.pack(side='left', padx=10)
        
        cancel_button = tk.Button(
            buttons_frame,
            text="CANCELAR",
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=30,
            pady=10,
            command=self.dialog.destroy,
            relief='flat',
            cursor='hand2'
        )
        cancel_button.pack(side='left', padx=10)
    
    def select_image(self):
        """Seleciona imagem do produto"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem do Produto",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            filename = os.path.basename(file_path)
            self.image_path_label.config(text=f"Imagem: {filename}")
    
    def save_product(self):
        """Salva o produto"""
        name = self.name_entry.get().strip()
        price = self.price_entry.get().strip()
        
        if not name:
            messagebox.showerror("Erro", "Por favor, digite o nome do produto.")
            return
        
        if not price:
            messagebox.showerror("Erro", "Por favor, digite o preço do produto.")
            return
        
        if not self.selected_image_path:
            messagebox.showerror("Erro", "Por favor, selecione uma imagem para o produto.")
            return
        
        # Adicionar produto
        success = self.product_manager.add_product(name, price, self.selected_image_path)
        
        if success:
            messagebox.showinfo("Sucesso", f"Produto '{name}' adicionado com sucesso!")
            self.dialog.destroy()
        else:
            messagebox.showerror("Erro", "Erro ao adicionar produto. Verifique os dados e tente novamente.")
