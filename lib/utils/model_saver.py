import datetime
import os
import json
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import numpy as np


from lib.data.dataset import BeeDataset
from torch.utils.data import DataLoader
import pandas as pd


class ModelSaver:
    def __init__(self, model, username="bizuuuuth"):
        self.model = model
        self.folder_path = self.create_model_folder(model, username)


    def save_training_config(self, model, optimizer, BATCH_SIZE, NUM_EPOCHS, LR, DEVICE, scheduler=None, criterion=None):
        summary = {}

        # ===== MODEL =====
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        summary["model"] = {
            "architecture": type(model).__name__,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

        # ===== TRAINING CONFIG =====
        summary["training_config"] = {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LR,
            "device": str(DEVICE),
            "dropout": getattr(model, "dropout", "N/A"),  # Si le modèle a un attribut dropout, on le récupère
        }

        # ===== OPTIMIZER =====
        first_group = optimizer.param_groups[0]

        optim_keys = ["lr", "weight_decay", "momentum", "betas", "eps"]
        optimizer_info = {
            "type": type(optimizer).__name__,
        }

        for key in optim_keys:
            if key in first_group:
                optimizer_info[key] = first_group[key]

        summary["optimizer"] = optimizer_info

        # ===== SCHEDULER =====
        if scheduler is not None:
            scheduler_info = {
                "type": type(scheduler).__name__,
            }

            # Récupération simple des attributs principaux
            possible_attrs = [
                "T_max",
                "eta_min",
                "gamma",
                "step_size",
                "patience",
                "factor",
                "total_steps",
                "base_lrs",
            ]

            for attr in possible_attrs:
                if hasattr(scheduler, attr):
                    scheduler_info[attr] = getattr(scheduler, attr)

            summary["scheduler"] = scheduler_info
        else:
            summary["scheduler"] = "None (Constant LR)"

        # ===== CRITERION =====
        if criterion is not None:
            criterion_info = {
                "type": type(criterion).__name__,
            }

            if hasattr(criterion, "label_smoothing"):
                criterion_info["label_smoothing"] = criterion.label_smoothing

            if hasattr(criterion, "weight") and criterion.weight is not None:
                criterion_info["class_weights"] = True

            summary["criterion"] = criterion_info

        json_path = os.path.join(self.folder_path, "training_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        

    def create_model_folder(self, model, username="bizuuuuth"):
        model_name = type(model).__name__
        hour_min = datetime.now().strftime("%H%M")
        folder_name = f"{model_name}_{username}_{hour_min}"
        folder_path = os.path.join("saved_models", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def load_model(self, model, model_name="best_model.pth", device="cpu"):
        model_path = os.path.join(self.folder_path, model_name)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.to(device)
        model.eval()
        return model
    
    def submission(self, model, batch_size=32, transform=None, 
               model_name="best_model.pth", device="cpu"):
    
        model = self.load_model(model, model_name, device)

        test_dataset = BeeDataset(train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_ids, all_preds = [], []

        with torch.no_grad():
            for imgs, ids in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_ids.extend(
                    [int(x) if isinstance(x, torch.Tensor) else x for x in ids]
                )

        submission_df = pd.DataFrame({
            "id": all_ids,
            "label": all_preds
        })

        save_path = os.path.join(self.folder_path, "submission.csv")
        submission_df.to_csv(save_path, index=False)

        print(f"Submission saved to {save_path}")

    def evaluate(self, model, batch_size=32, transform=None,
                model_name="best_model.pth", device="cpu",
                num_classes=None):

        model = self.load_model(model, model_name, device)

        val_dataset = BeeDataset(train=False, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        if num_classes is None:
            num_classes = max(max(all_labels), max(all_preds)) + 1

        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm)

        cm_path = os.path.join(self.folder_path, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=False)

        # ===== Metrics =====
        metrics = []

        total = cm.sum()

        for cls in range(num_classes):
            TP = cm[cls, cls]
            FP = cm[:, cls].sum() - TP
            FN = cm[cls, :].sum() - TP
            TN = total - (TP + FP + FN)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy_cls = (TP + TN) / total

            metrics.append({
                "class": cls,
                "accuracy": accuracy_cls,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        metrics_df = pd.DataFrame(metrics)

        metrics_path = os.path.join(self.folder_path, "metrics_per_class.csv")
        metrics_df.to_csv(metrics_path, index=False)

        print(f"Confusion matrix saved to {cm_path}")
        print(f"Metrics saved to {metrics_path}")

        return cm_df, metrics_df
    

    def compute_metrics(val_labels, val_preds, num_classes):
        cm = confusion_matrix(val_labels, val_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, zero_division=0
        )

        total = cm.sum()
        accuracy_per_class = []

        for cls in range(num_classes):
            TP = cm[cls, cls]
            FP = cm[:, cls].sum() - TP
            FN = cm[cls, :].sum() - TP
            TN = total - (TP + FP + FN)
            accuracy_per_class.append((TP + TN) / total)

        metrics_df = pd.DataFrame({
            "class": list(range(num_classes)),
            "accuracy": accuracy_per_class,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        return cm, metrics_df

    def save_model(self, model, name="best_model.pth"):
        path = os.path.join(self.folder_path, name)
        torch.save(model.state_dict(), path)
        return path


    def save_confusion_matrix(self, cm, name="confusion_matrix.csv"):
        path = os.path.join(self.folder_path, name)
        pd.DataFrame(cm).to_csv(path, index=False)


    def save_metrics(self, metrics_df, name="metrics_per_class.csv"):
        path = os.path.join(self.folder_path, name)
        metrics_df.to_csv(path, index=False)


    def save_training_log(self, log_row, fieldnames, name="training_log.csv"):
        path = os.path.join(self.folder_path, name)

        file_exists = os.path.isfile(path)

        with open(path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)