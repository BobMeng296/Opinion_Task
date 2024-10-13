import json
import matplotlib.pyplot as plt
import os
import numpy as np

# File paths
file_paths = [
    './performance_improved/performance_BERT_BiLSTM_MultiHead.json',
    './performance_improved/performance_BERT_BiLSTM_NER_Sentiment.json',
    './performance_improved/performance_BERT_BiLSTM.json',
    './performance_improved/performance_Hierarchial-LSTM-BERT.json',
    './performance_improved/performance_MultiHead_Hierarchial.json' 
]

# Directory to save the plots
save_dir = './compare/'
os.makedirs(save_dir, exist_ok=True)

# Loading the JSON data from each file
model_data = {}
for file_path in file_paths:
    print(f"Attempting to load file: {file_path}")  # Debug output
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            model_name = data['model_name']
            
            if 'MultiHead_Hierarchial' in file_path or model_name == 'MultiHead_Hierarchial':
                model_name = 'MultiHead_Hierarchical' 
            elif 'Hierarchial-LSTM-BERT' in file_path:
                model_name = 'Hierarchical_BERT_BiLSTM'
            
            model_data[model_name] = data
            print(f"Loaded data for model: {model_name}")  # Debug output
    else:
        print(f"File not found: {file_path}")  # Debug output

print("Loaded models:", list(model_data.keys()))  # Debug output

def add_comparison_note(ax, note):
    ax.text(0.5, -0.15, note, ha='center', va='center', transform=ax.transAxes, wrap=True, fontsize=10)

# Function to plot training and validation losses
def plot_losses(model_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    for model_name, data in model_data.items():
        ax.plot(data['train_losses'], label=f'{model_name} Train Loss')
        ax.plot(data['val_losses'], label=f'{model_name} Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Losses Comparison')
    ax.legend()
    ax.grid(True)
    
    comparison_note = "Purpose: Compare convergence speed and generalization ability across models. \
                       Look for models with low and stable validation loss."
    add_comparison_note(ax, comparison_note)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_validation_losses.png'))
    plt.close()

# Function to plot accuracies
def plot_accuracies(model_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['ner_accuracy', 'sentiment_accuracy', 'combined_accuracy']
    x = range(len(metrics))
    width = 0.2
    for i, (model_name, data) in enumerate(model_data.items()):
        values = [data['metrics'][metric] for metric in metrics]
        ax.bar([xi + i*width for xi in x], values, width, label=model_name, alpha=0.7)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Models')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(['NER Accuracy', 'Sentiment Accuracy', 'Combined Accuracy'])
    ax.legend()
    ax.grid(True)
    
    comparison_note = "Purpose: Compare overall performance across tasks. \
                       Higher combined accuracy indicates better multi-task learning."
    add_comparison_note(ax, comparison_note)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'))
    plt.close()

# Function to plot NER performance metrics
def plot_ner_performance(model_data):
    metrics = ['precision', 'recall', 'f1-score']
    categories = ['PART', 'PRODUCT', 'micro avg', 'macro avg', 'weighted avg']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        for model_name, data in model_data.items():
            values = [data['metrics']['ner_report'][category][metric] for category in categories]
            ax.plot(categories, values, marker='o', label=model_name)
        ax.set_xlabel('NER Categories')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'NER {metric.capitalize()} Comparison Across Models')
        ax.legend()
        ax.grid(True)
        
        comparison_note = f"Purpose: Compare {metric} across NER categories. \
                           Look for models that perform well on both PART and PRODUCT consistently."
        add_comparison_note(ax, comparison_note)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ner_{metric}_comparison.png'))
        plt.close()

# Function to plot Sentiment Analysis performance metrics
def plot_sentiment_performance(model_data):
    metrics = ['precision', 'recall', 'f1-score']
    categories = ['negative', 'neutral', 'positive', 'macro avg', 'weighted avg']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        for model_name, data in model_data.items():
            values = [data['metrics']['sentiment_report'][category][metric] for category in categories]
            ax.plot(categories, values, marker='o', label=model_name)
        ax.set_xlabel('Sentiment Classes')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Sentiment Analysis {metric.capitalize()} Comparison Across Models')
        ax.legend()
        ax.grid(True)
        
        comparison_note = f"Purpose: Compare {metric} across sentiment classes. \
                           Look for models that perform well across all sentiment categories, especially on neutral class."
        add_comparison_note(ax, comparison_note)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sentiment_{metric}_comparison.png'))
        plt.close()

def plot_training_efficiency(model_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    for model_name, data in model_data.items():
        ax.plot(range(1, len(data['train_losses'])+1), data['train_losses'], label=f'{model_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Efficiency Comparison')
    ax.legend()
    ax.grid(True)
    
    comparison_note = "Purpose: Compare how quickly each model converges during training. \
                       Steeper slopes indicate faster learning."
    add_comparison_note(ax, comparison_note)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_efficiency_comparison.png'))
    plt.close()

def plot_ner_category_performance(model_data):
    categories = ['PART', 'PRODUCT']
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(categories))
    width = 0.35
    
    for i, (model_name, data) in enumerate(model_data.items()):
        f1_scores = [data['metrics']['ner_report'][category]['f1-score'] for category in categories]
        ax.bar(x + i*width/len(model_data), f1_scores, width/len(model_data), label=model_name)
    
    ax.set_ylabel('F1-Score')
    ax.set_title('NER Performance by Category')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(categories)
    ax.legend()
    
    comparison_note = "Purpose: Compare how well each model performs on different NER categories. \
                       Look for models that perform consistently well across categories."
    add_comparison_note(ax, comparison_note)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ner_category_performance.png'))
    plt.close()


def plot_sentiment_category_performance(model_data):
    categories = ['negative', 'neutral', 'positive']
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(categories))
    width = 0.35
    
    for i, (model_name, data) in enumerate(model_data.items()):
        f1_scores = [data['metrics']['sentiment_report'][category]['f1-score'] for category in categories]
        ax.bar(x + i*width/len(model_data), f1_scores, width/len(model_data), label=model_name)
    
    ax.set_ylabel('F1-Score')
    ax.set_title('Sentiment Analysis Performance by Category')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(categories)
    ax.legend()
    
    comparison_note = "Purpose: Compare how well each model performs on different sentiment categories. \
                       Look for models that perform well across all sentiment types, especially on the challenging neutral class."
    add_comparison_note(ax, comparison_note)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sentiment_category_performance.png'))
    plt.close()
def plot_ner_specific_metrics(model_data):
    metrics = ['precision', 'recall', 'f1-score']
    categories = ['PART', 'PRODUCT']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.15
        
        for i, (model_name, data) in enumerate(model_data.items()):
            values = [data['metrics']['ner_report'][category][metric] for category in categories]
            ax.bar(x + i*width, values, width, label=model_name)
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'NER {metric.capitalize()} Comparison by Category')
        ax.set_xticks(x + width * (len(model_data) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ner_{metric}_by_category.png'))
        plt.close()

def plot_sentiment_specific_metrics(model_data):
    metrics = ['precision', 'recall', 'f1-score']
    categories = ['negative', 'neutral', 'positive']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.15
        
        for i, (model_name, data) in enumerate(model_data.items()):
            values = [data['metrics']['sentiment_report'][category][metric] for category in categories]
            ax.bar(x + i*width, values, width, label=model_name)
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Sentiment Analysis {metric.capitalize()} Comparison by Category')
        ax.set_xticks(x + width * (len(model_data) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sentiment_{metric}_by_category.png'))
        plt.close()

def plot_combined_performance(model_data):
    metrics = ['ner_accuracy', 'sentiment_accuracy', 'combined_accuracy']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (model_name, data) in enumerate(model_data.items()):
        values = [data['metrics'][metric] for metric in metrics]
        ax.bar(x + i*width, values, width, label=model_name)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Combined Performance Comparison')
    ax.set_xticks(x + width * (len(model_data) - 1) / 2)
    ax.set_xticklabels(['NER Accuracy', 'Sentiment Accuracy', 'Combined Accuracy'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_performance.png'))
    plt.close()

def plot_training_convergence(model_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, data in model_data.items():
        ax.plot(data['train_losses'], label=f'{model_name} (Train)')
        ax.plot(data['val_losses'], label=f'{model_name} (Val)', linestyle='--')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Convergence')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_convergence.png'))
    plt.close()


if __name__ == "__main__":
    
    plot_losses(model_data)
    plot_accuracies(model_data)
    plot_ner_performance(model_data)
    plot_sentiment_performance(model_data)
    plot_training_efficiency(model_data)
    plot_ner_category_performance(model_data)
    plot_sentiment_category_performance(model_data)
    plot_ner_specific_metrics(model_data)
    plot_sentiment_specific_metrics(model_data)
    plot_combined_performance(model_data)
    plot_training_convergence(model_data)

    
    print("All plots have been generated and saved in the 'compare' directory.")