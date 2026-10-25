"""
Zvy Proprietary Neural Agent Trainer
Trains Zvy's custom neural architecture completely from scratch on the local CPU.
No HuggingFace. No Groq. No external pre-trained weights.
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from zvy_agent.model.architecture import (
    ZvyTokenizer,
    ZvyNeuralNetwork,
    TOOL_TO_IDX,
    TOOL_LIST
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(MODEL_DIR, "zvy_complete_dataset.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "zvy_neural_weights.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")

class ZvyDataset(Dataset):
    def __init__(self, data, tokenizer: ZvyTokenizer, cid_to_idx: dict):
        self.samples = []
        for item in data:
            q = item["query"]
            input_ids = tokenizer.encode(q)
            
            # Action: 0 for CONVERSATION, 1 for TOOL_CALL
            action_label = 1 if item.get("action_type") == "TOOL_CALL" else 0
            
            # Tool label (0 to 5)
            tool_name = item.get("tool", "NONE")
            tool_label = TOOL_TO_IDX.get(tool_name, 0)
            
            # Customer ID
            cid = str(item.get("target_customer_id", "0"))
            cid_label = cid_to_idx.get(cid, 0)
            
            # Amount label (for refunds)
            amt = float(item.get("arguments", {}).get("amount", 0.0))
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "action_label": torch.tensor(action_label, dtype=torch.long),
                "tool_label": torch.tensor(tool_label, dtype=torch.long),
                "cid_label": torch.tensor(cid_label, dtype=torch.long),
                "amt_label": torch.tensor([amt / 500.0], dtype=torch.float32), # normalized
                "response_template": item.get("response_template", "")
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train():
    print("=" * 60)
    print("  ZVY NEURAL AGENT - PROPRIETARY LOCAL TRAINING")
    print("  Zero External LLMs | Pure PyTorch Architecture")
    print("=" * 60)

    # 1. Load Dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[*] Loaded {len(data)} training samples from {DATASET_PATH}")

    # 2. Build Customer ID Mapping (from 101 to 228 + 0 for None)
    customer_ids = sorted(list({str(item.get("target_customer_id", "0")) for item in data}))
    if "0" not in customer_ids:
        customer_ids.insert(0, "0")
    cid_to_idx = {cid: idx for idx, cid in enumerate(customer_ids)}
    idx_to_cid = {idx: cid for idx, cid in enumerate(customer_ids)}
    num_customers = len(cid_to_idx)
    print(f"[*] Indexed {num_customers} customer ID routing classes.")

    # 3. Fit Tokenizer
    tokenizer = ZvyTokenizer(vocab_size=3200, max_len=64)
    all_texts = [d["query"] for d in data]
    tokenizer.fit(all_texts)
    print(f"[*] Built proprietary vocabulary with {len(tokenizer.w2i)} tokens.")

    # Save Vocab & Customer Mapping
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "w2i": tokenizer.w2i,
            "cid_to_idx": cid_to_idx,
            "idx_to_cid": idx_to_cid,
            "tool_to_idx": TOOL_TO_IDX
        }, f, indent=2)
    print(f"[*] Vocabulary and mapping saved to {VOCAB_PATH}")

    # 4. Prepare DataLoader
    dataset = ZvyDataset(data, tokenizer, cid_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 5. Initialize Model
    model = ZvyNeuralNetwork(
        vocab_size=len(tokenizer.w2i),
        d_model=128,
        n_layers=2,
        num_customers=num_customers
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Initialized ZvyNeuralNetwork: {total_params:,} learnable parameters.")

    # 6. Loss Functions & Optimizer
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 7. Training Loop
    epochs = 12
    print(f"[*] Beginning training across {epochs} epochs on CPU...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_action = 0
        correct_tool = 0
        total_samples = 0

        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"]
            action_label = batch["action_label"]
            tool_label = batch["tool_label"]
            cid_label = batch["cid_label"]
            amt_label = batch["amt_label"]

            preds = model(input_ids)

            # Multi-task loss
            loss_action = ce_loss(preds["action_logits"], action_label)
            loss_tool = ce_loss(preds["tool_logits"], tool_label)
            loss_cid = ce_loss(preds["customer_logits"], cid_label)
            loss_amt = mse_loss(preds["amount_pred"], amt_label)

            loss = loss_action + (1.5 * loss_tool) + (1.2 * loss_cid) + (0.5 * loss_amt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(input_ids)
            pred_action = preds["action_logits"].argmax(dim=-1)
            pred_tool = preds["tool_logits"].argmax(dim=-1)
            
            correct_action += (pred_action == action_label).sum().item()
            correct_tool += (pred_tool == tool_label).sum().item()
            total_samples += len(input_ids)

        avg_loss = total_loss / total_samples
        action_acc = (correct_action / total_samples) * 100.0
        tool_acc = (correct_tool / total_samples) * 100.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Action Acc: {action_acc:.1f}% | Tool Selection Acc: {tool_acc:.1f}%")

    # 8. Save Weights
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"[*] Training finished! Model checkpoint saved to: {WEIGHTS_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    train()
