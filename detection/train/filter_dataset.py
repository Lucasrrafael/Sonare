import os
import shutil
from pathlib import Path

def filter_yolo_dataset(source_dir, dest_dir, classes_to_remove):
    """
    Cria uma cópia do dataset YOLO removendo classes específicas e ajustando os índices
    
    Args:
        source_dir: Diretório do dataset original
        dest_dir: Diretório do dataset filtrado
        classes_to_remove: Lista de índices de classes a remover (ex: [0, 1])
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Criar diretório de destino
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Criando cópia filtrada do dataset...")
    print(f"Origem: {source_path}")
    print(f"Destino: {dest_path}")
    print(f"Removendo classes: {classes_to_remove}\n")
    
    # Ler data.yaml original
    data_yaml_path = source_path / "data.yaml"
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Processar data.yaml
    new_yaml_lines = []
    original_names = []
    nc_original = 0
    
    for line in lines:
        if line.strip().startswith('nc:'):
            nc_original = int(line.split(':')[1].strip())
        elif line.strip().startswith('names:'):
            # Extrair nomes das classes
            names_str = line.split(':', 1)[1].strip()
            # Remover [ ] e aspas
            names_str = names_str.strip('[]')
            original_names = [name.strip().strip("'\"") for name in names_str.split(',')]
    
    print(f"Classes originais ({nc_original}): {original_names}")
    
    # Filtrar classes
    new_names = [name for i, name in enumerate(original_names) if i not in classes_to_remove]
    nc_new = len(new_names)
    
    print(f"Classes após filtro ({nc_new}): {new_names}\n")
    
    # Criar mapeamento de índices antigos para novos
    index_mapping = {}
    new_index = 0
    for old_index in range(nc_original):
        if old_index not in classes_to_remove:
            index_mapping[old_index] = new_index
            new_index += 1
    
    print(f"Mapeamento de índices: {index_mapping}\n")
    
    # Escrever novo data.yaml
    new_yaml_path = dest_path / "data.yaml"
    with open(new_yaml_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().startswith('nc:'):
                f.write(f"nc: {nc_new}\n")
            elif line.strip().startswith('names:'):
                f.write(f"names: {new_names}\n")
            else:
                f.write(line)
    
    print(f"✓ Novo data.yaml criado")
    
    # Processar diretórios de imagens e labels
    for split in ['train', 'test', 'valid', 'val']:
        img_dir = source_path / split / 'images'
        label_dir = source_path / split / 'labels'
        
        if not img_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"Processando {split.upper()}...")
        print(f"{'='*60}")
        
        # Criar diretórios de destino
        dest_img_dir = dest_path / split / 'images'
        dest_label_dir = dest_path / split / 'labels'
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Processar cada arquivo de imagem
        img_files = list(img_dir.glob('*.*'))
        total_images = len(img_files)
        images_copied = 0
        annotations_removed = 0
        annotations_kept = 0
        
        for img_file in img_files:
            # Nome base sem extensão
            base_name = img_file.stem
            
            # Caminho do arquivo de label correspondente
            label_file = label_dir / f"{base_name}.txt"
            
            if not label_file.exists():
                # Copiar imagem sem label
                shutil.copy2(img_file, dest_img_dir / img_file.name)
                images_copied += 1
                continue
            
            # Processar arquivo de label
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                
                # Pular se for uma classe a ser removida
                if class_id in classes_to_remove:
                    annotations_removed += 1
                    continue
                
                # Ajustar índice da classe
                if class_id in index_mapping:
                    new_class_id = index_mapping[class_id]
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                    annotations_kept += 1
            
            # Copiar imagem e label apenas se houver anotações restantes
            if new_lines:
                shutil.copy2(img_file, dest_img_dir / img_file.name)
                
                dest_label_file = dest_label_dir / f"{base_name}.txt"
                with open(dest_label_file, 'w') as f:
                    f.writelines(new_lines)
                
                images_copied += 1
        
        print(f"Imagens processadas: {total_images}")
        print(f"Imagens copiadas: {images_copied}")
        print(f"Anotações removidas: {annotations_removed}")
        print(f"Anotações mantidas: {annotations_kept}")
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset filtrado criado com sucesso!")
    print(f"Localização: {dest_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    # Configuração
    source_dataset = "C:/Users/lucas/OneDrive/Área de Trabalho/freiburg-groceries.v4i.yolov8"
    dest_dataset = "C:/Users/lucas/OneDrive/Área de Trabalho/freiburg-groceries-filtered"
    
    # Classes a remover (atum=0, cafe=1)
    classes_to_remove = [0, 1]
    
    # Executar filtro
    filter_yolo_dataset(source_dataset, dest_dataset, classes_to_remove)
    
    print("\n\n🔍 Para usar o novo dataset, atualize seu script de treinamento:")
    print(f'data="C:/Users/lucas/OneDrive/Área de Trabalho/freiburg-groceries-filtered/data.yaml"')

