import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from seqeval.metrics import classification_report as ner_classification_report
from sklearn.metrics import classification_report as sentiment_classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

class BERTOpinionMiningDataset(Dataset):
    @classmethod
    def create_split_datasets(cls, csv_file, tokenizer, max_length=128, test_size=0.3, random_state=42):
        df = pd.read_csv(csv_file)
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        train_dataset = cls(train_df, tokenizer, max_length)
        val_dataset = cls(val_df, tokenizer, max_length)
        
        return train_dataset, val_dataset

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.tag_vocab = {'O': 0, 'B-PRODUCT': 1, 'I-PRODUCT': 2, 'B-PART': 3, 'I-PART': 4}
        self.idx2tag = {v: k for k, v in self.tag_vocab.items()}
        self.sentiment_vocab = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['processed_text_ner']
        ner_labels = self.data.iloc[idx]['ner_labels'].split()
        sentiment = self.data.iloc[idx]['class']

        encoding = self.tokenizer(text, 
                                  return_offsets_mapping=True, 
                                  max_length=self.max_length, 
                                  truncation=True, 
                                  padding='max_length')
        
        aligned_labels = self.align_labels_with_tokens(ner_labels, encoding)

        # debug to check lables alignment
        print(f"Original Text: {text}")
        print(f"Tokens: {self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])}")
        print(f"NER Labels: {[self.idx2tag[label] for label in aligned_labels]}")
        print("=" * 100)

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'ner_labels': torch.tensor(aligned_labels),
            'sentiment_label': torch.tensor(self.sentiment_vocab[sentiment])
        }

    def align_labels_with_tokens(self, labels, encoding):
        aligned_labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(self.tag_vocab['O'])
            elif word_idx != previous_word_idx:
                try:
                    aligned_labels.append(self.tag_vocab[labels[word_idx]])
                except IndexError:
                    aligned_labels.append(self.tag_vocab['O'])
            else:
                prev_label = aligned_labels[-1]
                if prev_label % 2 == 1:  
                    aligned_labels.append(prev_label + 1)  
                else:
                    aligned_labels.append(prev_label)
            
            previous_word_idx = word_idx

        # debug to check lables alignment
        print(f"Original labels: {labels}")
        print(f"Aligned labels: {[self.idx2tag[label] for label in aligned_labels]}")
        print(f"Word IDs: {word_ids}")
        
        return aligned_labels
    
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_hidden = torch.bmm(attention_weights.unsqueeze(1), hidden_states)
        return attended_hidden.squeeze(1)
    
class HierarchicalBERTBiLSTMNERSentimentModel(nn.Module):
    def __init__(self, num_ner_labels, num_sentiment_labels, hidden_size=256, shared_lstm_layers=2, task_specific_lstm_layers=1):
        super(HierarchicalBERTBiLSTMNERSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        
        # Shared BiLSTM layer
        self.shared_lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=shared_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if shared_lstm_layers > 1 else 0
        )
        
        # NER-specific BiLSTM layer
        self.ner_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=task_specific_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if task_specific_lstm_layers > 1 else 0
        )
        
        # Sentiment-specific BiLSTM layer
        self.sentiment_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=task_specific_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if task_specific_lstm_layers > 1 else 0
        )
        
        self.ner_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.sentiment_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.ner_classifier = nn.Linear(hidden_size * 2, num_ner_labels)
        self.sentiment_classifier = nn.Linear(hidden_size * 2, num_sentiment_labels)
        
        self.ner_residual = nn.Linear(self.bert.config.hidden_size, num_ner_labels)
        self.sentiment_residual = nn.Linear(self.bert.config.hidden_size, num_sentiment_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Shared BiLSTM
        shared_lstm_output, _ = self.shared_lstm(sequence_output)
        
        # NER task
        ner_lstm_output, _ = self.ner_lstm(shared_lstm_output)
        ner_attention_weights = self.ner_attention(ner_lstm_output).squeeze(-1)
        ner_attention_weights = torch.softmax(ner_attention_weights, dim=1)
        # ner_attended = torch.bmm(ner_attention_weights.unsqueeze(1), ner_lstm_output).squeeze(1)
        ner_output = self.ner_classifier(self.dropout(ner_lstm_output))
        ner_output += self.ner_residual(sequence_output)
        
        # Sentiment task
        sentiment_lstm_output, _ = self.sentiment_lstm(shared_lstm_output)
        sentiment_attention_weights = self.sentiment_attention(sentiment_lstm_output).squeeze(-1)
        sentiment_attention_weights = torch.softmax(sentiment_attention_weights, dim=1)
        sentiment_attended = torch.bmm(sentiment_attention_weights.unsqueeze(1), sentiment_lstm_output).squeeze(1)
        sentiment_output = self.sentiment_classifier(self.dropout(sentiment_attended))
        sentiment_output += self.sentiment_residual(sequence_output[:, 0, :])

        return ner_output, sentiment_output
    
class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, ner_weight=0.5, sentiment_weight=0.5, part_weight=2.0):
        super(WeightedMultiTaskLoss, self).__init__()
        self.ner_weight = ner_weight
        self.sentiment_weight = sentiment_weight
        self.part_weight = part_weight
        self.ner_criterion = nn.CrossEntropyLoss(reduction='none')
        self.sentiment_criterion = nn.CrossEntropyLoss()

    def forward(self, ner_output, ner_labels, sentiment_output, sentiment_labels):
        ner_loss = self.ner_criterion(ner_output.view(-1, ner_output.shape[-1]), ner_labels.view(-1))
        
        ner_weights = torch.ones_like(ner_labels, dtype=torch.float)
        ner_weights[ner_labels == 3] = self.part_weight  # B-PART
        ner_weights[ner_labels == 4] = self.part_weight  # I-PART
        
        ner_loss = (ner_loss * ner_weights.view(-1)).mean()
        
        sentiment_loss = self.sentiment_criterion(sentiment_output, sentiment_labels)
        return self.ner_weight * ner_loss + self.sentiment_weight * sentiment_loss


def train(model, train_loader, val_loader, criterion, epochs=10, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            optimizer.zero_grad()
            ner_output, sentiment_output = model(input_ids, attention_mask)
            
            loss = criterion(ner_output, ner_labels, sentiment_output, sentiment_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        train_losses.append(total_loss / len(train_loader))
        val_loss = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def evaluate(model, data_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            ner_output, sentiment_output = model(input_ids, attention_mask)
            
            loss = criterion(ner_output, ner_labels, sentiment_output, sentiment_labels)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def predict(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            ner_output, sentiment_output = model(input_ids, attention_mask)
            
            ner_preds = torch.argmax(ner_output, dim=2)
            sentiment_preds = torch.argmax(sentiment_output, dim=1)
            
            for i in range(ner_preds.shape[0]):
                ner_pred = ner_preds[i][attention_mask[i] == 1][1:-1].cpu().tolist()
                ner_true = ner_labels[i][attention_mask[i] == 1][1:-1].cpu().tolist()
                sentiment_pred = sentiment_preds[i].item()
                sentiment_true = sentiment_labels[i].item()
                
                all_results.append({
                    'ner_pred': ner_pred,
                    'ner_true': ner_true,
                    'sentiment_pred': sentiment_pred,
                    'sentiment_true': sentiment_true
                })
    
    return all_results

def calculate_metrics(results, idx2tag, idx2sentiment):
    ner_correct = 0
    sentiment_correct = 0
    combined_correct = 0
    total_samples = len(results)
    
    ner_preds, ner_trues = [], []
    sentiment_preds, sentiment_trues = [], []
    
    for result in results:
        ner_pred = [idx2tag[idx] for idx in result['ner_pred']]
        ner_true = [idx2tag[idx] for idx in result['ner_true']]
        sentiment_pred = idx2sentiment[result['sentiment_pred']]
        sentiment_true = idx2sentiment[result['sentiment_true']]
        
        ner_preds.append(ner_pred)
        ner_trues.append(ner_true)
        sentiment_preds.append(sentiment_pred)
        sentiment_trues.append(sentiment_true)
        
        if ner_pred == ner_true:
            ner_correct += 1
        if sentiment_pred == sentiment_true:
            sentiment_correct += 1
        if ner_pred == ner_true and sentiment_pred == sentiment_true:
            combined_correct += 1
    
    ner_accuracy = ner_correct / total_samples
    sentiment_accuracy = sentiment_correct / total_samples
    combined_accuracy = combined_correct / total_samples
    
    ner_report = ner_classification_report(ner_trues, ner_preds, output_dict=True, zero_division=0)
    sentiment_report = sentiment_classification_report(sentiment_trues, sentiment_preds, output_dict=True, zero_division=0)
    sentiment_cm = confusion_matrix(sentiment_trues, sentiment_preds)
    
    return {
        'ner_accuracy': ner_accuracy,
        'sentiment_accuracy': sentiment_accuracy,
        'combined_accuracy': combined_accuracy,
        'ner_report': ner_report,
        'sentiment_report': sentiment_report,
        'sentiment_confusion_matrix': sentiment_cm.tolist()
    }

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'./performance_improved/{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_metrics_comparison(metrics, title):
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(f'./performance_improved/{title.lower().replace(" ", "_")}.png')
    plt.close()

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj


def save_performance_metrics(model_name, train_losses, val_losses, metrics, file_path):
    output = {
        "model_name": model_name,
        "train_losses": convert_numpy(train_losses),
        "val_losses": convert_numpy(val_losses),
        "metrics": convert_numpy(metrics)
    }
    
    with open(file_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    train_dataset, val_dataset = BERTOpinionMiningDataset.create_split_datasets(
        './data/data_all_processed.csv', 
        tokenizer,
        max_length=128,
        test_size=0.3,
        random_state=42
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = HierarchicalBERTBiLSTMNERSentimentModel(
                num_ner_labels=5, 
                num_sentiment_labels=3, 
                hidden_size=256, 
                shared_lstm_layers=2, 
                task_specific_lstm_layers=1
            )

    ner_weight = 0.8
    sentiment_weight = 0.2
    part_weight = 3.5  
    learning_rate = 1e-5

   
    criterion = WeightedMultiTaskLoss(ner_weight=ner_weight, sentiment_weight=sentiment_weight, part_weight=part_weight)
    train_losses, val_losses = train(model, train_loader, val_loader, 
                                    criterion=criterion, epochs=10, lr=learning_rate)

    val_results = predict(model, val_loader)

    idx2tag = {v: k for k, v in train_dataset.tag_vocab.items()}
    idx2sentiment = {v: k for k, v in train_dataset.sentiment_vocab.items()}
    metrics = calculate_metrics(val_results, idx2tag, idx2sentiment)

    plot_confusion_matrix(
        np.array(metrics['sentiment_confusion_matrix']),
        classes=list(idx2sentiment.values()),
        title='Hierarchial-LSTM BERT Sentiment Analysis Confusion Matrix'
    )

    plot_metrics_comparison(
        {
            'NER Accuracy': metrics['ner_accuracy'],
            'Sentiment Accuracy': metrics['sentiment_accuracy'],
            'Combined Accuracy': metrics['combined_accuracy']
        },
        title='Hierarchial-LSTM BERT Performance Comparison'
    )

    save_performance_metrics(
        "BERT_BiLSTM_NER_Sentiment",
        train_losses,
        val_losses,
        metrics,
        "./performance_improved/performance_Hierarchial-LSTM-BERT.json"
    )

    print("Performance metrics saved to 'performance_BERT_BiLSTM.json'")

    print("\nAccuracies:")
    print(f"NER Accuracy: {metrics['ner_accuracy']:.4f}")
    print(f"Sentiment Accuracy: {metrics['sentiment_accuracy']:.4f}")
    print(f"Combined Accuracy: {metrics['combined_accuracy']:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Hierarchial-LSTM BERT Training Progress')
    plt.savefig('./performance_improved/training_progress_Hierarchial-LSTM BERT.png')
    plt.show()

   
    print("\nVisualization of a few examples:")
    for i in range(min(5, len(val_dataset))):  
        sample = val_dataset[i]
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
        ner_labels = [idx2tag[label.item()] for label in sample['ner_labels']]
        sentiment = idx2sentiment[sample['sentiment_label'].item()]
        
        print(f"\nExample {i+1}:")
        print("Tokens:", tokens)
        print("NER Labels:", ner_labels)
        print("Sentiment:", sentiment)
        print("-" * 50)

    print("\nTraining and evaluation completed.")