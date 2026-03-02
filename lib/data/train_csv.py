import os
import csv

# Configuration
root_dir = 'data/train'
mapping_file = 'data/class-mapping.txt'
output_csv = os.path.join('data', 'train_corrected.csv') 

def generate_dataset_csv(root_dir, mapping_file, output_csv):
    # 1. Charger le mapping {classe: score}
    class_to_value = {}
    with open(mapping_file, 'r') as f:
        n_line = 1
        for line in f:
            line = line.rstrip("\n")
            species, value = line, n_line
            class_to_value[species.strip()] = value
            n_line += 1

    # 2. Parcourir les dossiers et créer les lignes du CSV
    data_rows = []
    
    # On liste les sous-dossiers (les classes)
    for species_name in os.listdir(root_dir):
        species_path = os.path.join(root_dir, species_name)
        
        if os.path.isdir(species_path):
            label = class_to_value.get(species_name)
            
            if label is None:
                print(f"No label found for the species '{species_name}'")
                continue

            # Lister les images dans le dossier de la classe
            for img_name in os.listdir(species_path):
                if img_name.lower().endswith(('.jpg')):
                    # Chemin relatif : train/classe_1/img1.jpg
                    relative_path = os.path.join('train', species_path, img_name)
                    data_rows.append([relative_path, label])

    # 3. Écriture du fichier CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label']) 
        writer.writerows(data_rows)

    print(f"Done ! {len(data_rows)} lines written in {output_csv}")

# Lancer la génération
generate_dataset_csv(root_dir, mapping_file, output_csv)
