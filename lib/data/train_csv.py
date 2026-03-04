import os
import csv

# Configuration
root_dir = 'data/train'
mapping_file = 'data/class-mapping.txt'
output_csv = os.path.join('data', 'train_corrected.csv') 

to_delete = [
"train/Andrena leucophaea/4347cb0b05c0c6c69e82aac788046352e0536aba.jpg",
"train/Andrena plana/da76db25fdbde04ec4e8693d75618ec5137798df.jpg",
"train/Andrena hesperia/827baffdcb5edb79881ad546804f51aab81312c5.jpg",
"train/Andrena hesperia/d6216915cf80f4d07054a210d04a9ad24a7ffd73.jpg"
]

def generate_dataset_csv(root_dir, mapping_file, output_csv):
    # 1. Charger le mapping {classe: score}
    class_to_value = {}
    with open(mapping_file, 'r') as f:
        for n_line, line in enumerate (f):
            line = line.rstrip("\n")
            species, value = line, n_line
            class_to_value[species.strip()] = value
            

    # 2. Parcourir les dossiers et créer les lignes du CSV
    data_rows = []
    
    # On liste les sous-dossiers (les classes)
    species_list = sorted(os.listdir(root_dir))

    for species_name in species_list:
        species_path = os.path.join(root_dir, species_name)
        
        if os.path.isdir(species_path):
            label = class_to_value.get(species_name)
            
            if label is None:
                print(f"No label found for the species '{species_name}'")
                continue

            # Lister les images dans le dossier de la classe
            img_list = sorted(os.listdir(species_path))

            for img_name in img_list:
                if img_name.lower().endswith(('.jpg')):
                    # Chemin relatif : train/classe_1/img1.jpg
                    relative_path = os.path.join('train', species_name, img_name)
                    data_rows.append([relative_path, label])

                    # --- NOUVELLE LOGIQUE DE SUPPRESSION ---
                    if relative_path in to_delete:
                            full_path = os.path.join('data', relative_path)
                            if os.path.exists(full_path):
                                os.remove(full_path)
                                print(f"🗑️ Supprimé physiquement : {relative_path}")
                            else:
                                print(f"🚫 Ignoré (non ajouté au CSV) : {relative_path}")
                            continue # On passe à l'image suivante, on ne l'ajoute pas au CSV
                        # ---------------------------------------

    # 3. Écriture du fichier CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label']) 
        writer.writerows(data_rows)

    print(f"Done ! {len(data_rows)} lines written in {output_csv}")

# Lancer la génération
generate_dataset_csv(root_dir, mapping_file, output_csv)
