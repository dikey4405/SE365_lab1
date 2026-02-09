import os
import pandas as pd
from train import run_training
from utils import plot_history, plot_confusion_matrix

if os.path.exists("/kaggle/working"):
    DATA_DIR = "/kaggle/working/data"
else:
    DATA_DIR = "./data"
configs = [
    # Exp-0: Baseline (Adam, No Regularization)
    {
        'name': 'Exp0_Baseline_v1',
        'optimizer': 'adam',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 15,
        'dropout': 0.0,
        'weight_decay': 0.0,
        'augmentation': False,
        'early_stopping': False
    },

    # Exp-0: Baseline (SGD, No Regularization)
    {
        'name': 'Exp0_Baseline_v2',
        'optimizer': 'sgd',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 15,
        'dropout': 0.0,
        'weight_decay': 0.0,
        'augmentation': False,
        'early_stopping': False
    },
    
    # Exp-1: Weight Decay (L2)
    {
        'name': 'Exp1_WeightDecay',
        'optimizer': 'adam',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 15,
        'dropout': 0.0,
        'weight_decay': 1e-3,
        'augmentation': False,
        'early_stopping': False
    },

    # Exp-2: Dropout
    {
        'name': 'Exp2_Dropout',
        'optimizer': 'adam',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 15,
        'dropout': 0.5,       
        'weight_decay': 0.0,
        'augmentation': False,
        'early_stopping': False
    },

    # Exp-3: Data Augmentation
    {
        'name': 'Exp3_Augmentation',
        'optimizer': 'adam',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 15,
        'dropout': 0.0,
        'weight_decay': 0.0,
        'augmentation': True, 
        'early_stopping': False
    },
    
    #Exp-4: Early Stopping (kết hợp tất cả)
    {
        'name': 'Exp4_FullReg_EarlyStop',
        'optimizer': 'adam',
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 50,        
        'dropout': 0.3,
        'weight_decay': 1e-4,
        'augmentation': True,
        'early_stopping': True, 
        'patience': 5
    }
]

def main():
    results = []

    selected_experiments = configs 

    for config in selected_experiments:
        print(f"\n{'='*40}")
        print(f"STARTING: {config['name']}")
        print(f"{'='*40}")
        
        try:
            history, metrics, classes = run_training(config, DATA_DIR)
            
            print(f"\n--- RESULT: {config['name']} ---")
            print(f"Accuracy:  {metrics['acc']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            
            plot_history(history, config['name'])
            plot_confusion_matrix(metrics['labels'], metrics['preds'], classes, config['name'])
            
            res_entry = {
                'Experiment': config['name'],
                'Accuracy': metrics['acc'],
                'F1-Score': metrics['f1'],
                'Val Loss': history['val_loss'][-1]
            }
            results.append(res_entry)

        except Exception as e:
            print(f"ERROR in {config['name']}: {e}")

    if len(results) > 0:
        print("\n\n=== FINAL COMPARISON TABLE ===")
        df = pd.DataFrame(results)
        print(df)

if __name__ == "__main__":

    main()





