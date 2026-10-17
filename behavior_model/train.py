"""
LATIS Behavior Model Training and Evaluation Pipeline (Phase H.2)
Trains Field-Aware LATISBehaviorNet with:
- 5-field structured context encoding [SYS, DOC, MEM, AUTH, QUERY]
- Query-Memory Relevance Gating
- Dynamic Candidate Tool Scoring from description specs
- Auxiliary Intent Head (0=INFORMATIONAL, 1=OPERATIONAL)
- Post-training validation temperature calibration (ECE evaluation)
Saves model weights and vocabulary to behavior_model/latis_model.pt.
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple

from behavior_model.model_arch import FastVocabulary, LATISBehaviorNet
from behavior_model.dataset import (
    generate_synthetic_dataset,
    TOOL_SPEC_CATALOG,
    TOOL_REGISTRY,
    TOOL_TO_IDX,
    SEVERITY_REGISTRY,
    SEVERITY_TO_IDX
)


class AgentDecisionDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], vocab: FastVocabulary):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Phase H.4: Parse structured fields into 5 tensor sequences: [sys, doc, mem, auth, query]
        fields_encoded = self.vocab.parse_fields_and_encode(item["text"])  # List of 5 lists of len field_len
        field_tensor = torch.tensor(fields_encoded, dtype=torch.long)       # (5, field_len)

        candidates = item.get("candidate_tools", ["NONE", "get_customer_data", "send_email", "export_metrics"])
        target_tool = item.get("target_tool", "NONE")
        
        # Target candidate index
        target_cand_idx = candidates.index(target_tool) if target_tool in candidates else 0

        # Encode candidate specifications using capability descriptions
        custom_specs = item.get("custom_specs", {})
        cand_token_list = []
        for c in candidates:
            if c in custom_specs:
                desc_text = custom_specs[c]
            else:
                spec = TOOL_SPEC_CATALOG.get(c, {"name": c, "description": f"Capability: Tool named {c}"})
                desc_text = spec['description']
            t_toks = self.vocab.encode(desc_text, max_len=self.vocab.tool_max_len)
            cand_token_list.append(t_toks)

        intent_label = item.get("intent_label", item["action_label"])
        operation_label = item.get("operation_label", 10 if item["action_label"] == 0 else 0)
        mem_constraint_label = item.get("mem_constraint_label", 0)
        severity_label = item.get("severity_label", 0)

        return (
            field_tensor,                                                   # (5, field_len)
            torch.tensor(item["action_label"], dtype=torch.long),           # scalar
            torch.tensor(cand_token_list, dtype=torch.long),                # (K, tool_max_len)
            torch.tensor(target_cand_idx, dtype=torch.long),                # scalar
            torch.tensor(intent_label, dtype=torch.long),                   # scalar
            torch.tensor(operation_label, dtype=torch.long),                # scalar
            torch.tensor(mem_constraint_label, dtype=torch.long),           # scalar
            torch.tensor(severity_label, dtype=torch.long),                 # scalar
            candidates,
            target_tool
        )


def custom_collate(batch):
    field_tensors = torch.stack([item[0] for item in batch])        # (B, 5, field_len)
    action_tensors = torch.stack([item[1] for item in batch])       # (B,)
    cand_tensors = torch.stack([item[2] for item in batch])         # (B, K, tool_max_len)
    target_cand_tensors = torch.stack([item[3] for item in batch])  # (B,)
    intent_tensors = torch.stack([item[4] for item in batch])       # (B,)
    oper_tensors = torch.stack([item[5] for item in batch])         # (B,)
    mem_contra_tensors = torch.stack([item[6] for item in batch])   # (B,)
    severity_tensors = torch.stack([item[7] for item in batch])     # (B,)
    candidates_list = [item[8] for item in batch]
    target_tools_list = [item[9] for item in batch]
    return (
        field_tensors, action_tensors, cand_tensors, target_cand_tensors,
        intent_tensors, oper_tensors, mem_contra_tensors, severity_tensors,
        candidates_list, target_tools_list
    )


def compute_ece(probs: List[float], correct: List[int], n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE) and confidence bucket table."""
    bins = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(probs)
    bucket_table = []
    
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        bin_indices = [idx for idx, p in enumerate(probs) if (p > low and p <= high) or (i == 0 and p >= low and p <= high)]
        if len(bin_indices) == 0:
            continue
        bin_acc = sum(correct[idx] for idx in bin_indices) / len(bin_indices)
        bin_conf = sum(probs[idx] for idx in bin_indices) / len(bin_indices)
        weight = len(bin_indices) / total
        ece += weight * abs(bin_acc - bin_conf)
        bucket_table.append({
            "bin": f"[{low:.1f}, {high:.1f}]",
            "count": len(bin_indices),
            "avg_conf": round(bin_conf, 4),
            "acc": round(bin_acc, 4)
        })
        
    return round(ece, 4), bucket_table


def fit_temperature(model: LATISBehaviorNet, val_loader: DataLoader) -> float:
    """Learns optimal temperature scaling parameter T on validation set using NLL loss."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for field_batch, act_batch, _, _, intent_batch, _, mem_contra_batch, _, _, _ in val_loader:
            res = model.encode_fields(field_batch)
            h_ctx, intent_logits, _, mem_constraint_logits = res[0], res[1], res[2], res[3]
            logits = model.compute_action_logits(h_ctx, mem_constraint_logits, intent_logits)
            logits_list.append(logits)
            labels_list.append(act_batch)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Search optimal T in [0.5, 4.0] with step 0.05
    best_t = 1.0
    best_nll = float("inf")
    for t_val in [x * 0.05 for x in range(10, 80)]:
        nll = F.cross_entropy(all_logits / t_val, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_t = round(t_val, 2)

    model.temperature.data = torch.tensor([best_t])
    return best_t


def evaluate_model(model: LATISBehaviorNet, dataloader: DataLoader, use_calibrated: bool = True) -> Dict[str, Any]:
    model.eval()
    correct_action = 0
    correct_tool = 0
    correct_intent = 0
    correct_operation = 0
    correct_mem_contra = 0
    correct_severity = 0
    total = 0
    
    tp, fp, fn, tn = 0, 0, 0, 0
    conf_list = []
    correct_list = []

    with torch.no_grad():
        for field_batch, act_batch, cand_batch, target_cand_batch, intent_batch, oper_batch, mem_contra_batch, sev_batch, cand_names, target_names in dataloader:
            B = field_batch.size(0)
            h_ctx, intent_logits, oper_logits, mem_constraint_logits, g_mem, severity_logits = model.encode_fields(field_batch)
            action_logits = model.compute_action_logits(h_ctx, mem_constraint_logits, intent_logits)
            
            # Severity prediction
            pred_sev = torch.argmax(severity_logits, dim=-1)
            correct_severity += (pred_sev == sev_batch).sum().item()

            # Intent prediction
            pred_intent = torch.argmax(intent_logits, dim=-1)
            correct_intent += (pred_intent == intent_batch).sum().item()

            # Operation prediction
            pred_oper = torch.argmax(oper_logits, dim=-1)
            correct_operation += (pred_oper == oper_batch).sum().item()

            # Memory Contradiction prediction
            pred_contra = torch.argmax(mem_constraint_logits, dim=-1)
            correct_mem_contra += (pred_contra == mem_contra_batch).sum().item()

            # Calibrated or raw action probabilities
            T = model.temperature.item() if use_calibrated else 1.0
            calibrated_logits = action_logits / T
            probs = F.softmax(calibrated_logits, dim=-1)
            pred_action = torch.argmax(probs, dim=-1)

            for b in range(B):
                pred_cls = pred_action[b].item()
                true_cls = act_batch[b].item()
                conf = probs[b, pred_cls].item()
                is_corr = 1 if pred_cls == true_cls else 0
                conf_list.append(conf)
                correct_list.append(is_corr)

            correct_action += (pred_action == act_batch).sum().item()
            total += B

            t_repr = model.encode_tools(cand_batch)                                  # (B, K, 128)
            scores = model.score_candidates(h_ctx, t_repr, oper_repr=h_ctx)          # (B, K)
            pred_cand_idx = torch.argmax(scores, dim=-1)                             # (B,)
            correct_tool += (pred_cand_idx == target_cand_batch).sum().item()

            for b in range(B):
                p = pred_action[b].item()
                y = act_batch[b].item()
                if p == 1 and y == 1:
                    tp += 1
                elif p == 1 and y == 0:
                    fp += 1
                elif p == 0 and y == 1:
                    fn += 1
                elif p == 0 and y == 0:
                    tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ece, buckets = compute_ece(conf_list, correct_list)

    return {
        "action_accuracy": round(correct_action / total if total > 0 else 0.0, 4),
        "tool_accuracy": round(correct_tool / total if total > 0 else 0.0, 4),
        "severity_accuracy": round(correct_severity / total if total > 0 else 0.0, 4),
        "intent_accuracy": round(correct_intent / total if total > 0 else 0.0, 4),
        "operation_accuracy": round(correct_operation / total if total > 0 else 0.0, 4),
        "mem_contra_accuracy": round(correct_mem_contra / total if total > 0 else 0.0, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ece": ece,
        "buckets": buckets,
        "total": total,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


def train_latis_behavior_model():
    print("=== LATIS PHASE H.6: TRAINING SEVERITY-AWARE BEHAVIOR MODEL ===")
    
    # 1. Generate Dataset with Hard Negatives & Context Constraints
    train_data, val_data, test_unseen = generate_synthetic_dataset()
    print(f"Dataset Split: Train={len(train_data)}, Val={len(val_data)}, Unseen Test={len(test_unseen)}")

    # 2. Build Subword Vocabulary
    vocab = FastVocabulary(vocab_size=4000, max_seq_len=192, tool_max_len=48, field_len=48)
    all_texts = [item["text"] for item in train_data + val_data]
    for spec in TOOL_SPEC_CATALOG.values():
        all_texts.append(spec["description"])
    vocab.build_vocab(all_texts)
    print(f"Subword Vocabulary Size: {len(vocab.word2idx)} tokens")

    # 3. Create DataLoaders
    train_loader = DataLoader(AgentDecisionDataset(train_data, vocab), batch_size=16, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(AgentDecisionDataset(val_data, vocab), batch_size=16, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(AgentDecisionDataset(test_unseen, vocab), batch_size=16, shuffle=False, collate_fn=custom_collate)

    # 4. Instantiate Model
    model = LATISBehaviorNet(
        vocab_size=len(vocab.word2idx),
        embed_dim=64,
        hidden_dim=96,
        num_tools=len(TOOL_REGISTRY),
        dropout=0.20
    )
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: LATISBehaviorNet Phase H.6 (Parameters: {param_count:,})")

    # 5. Loss & Optimizer with Label Smoothing (alpha = 0.05)
    criterion_action = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_match = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_intent = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_oper = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_contra = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_severity = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)

    # Multi-task loss weights:
    lambda_match = 1.2
    lambda_intent = 0.5
    lambda_oper = 0.5
    lambda_contra = 0.5
    lambda_severity = 0.8

    # 6. Training Loop (20 Epochs)
    num_epochs = 20
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for field_batch, act_batch, cand_batch, target_cand_batch, intent_batch, oper_batch, mem_contra_batch, sev_batch, _, _ in train_loader:
            optimizer.zero_grad()
            h_ctx, intent_logits, oper_logits, mem_constraint_logits, g_mem, severity_logits = model.encode_fields(field_batch)
            
            # 1. Action Loss with memory constraint residual pathway and learned intent coupling
            action_logits = model.compute_action_logits(h_ctx, mem_constraint_logits, intent_logits)
            loss_action = criterion_action(action_logits, act_batch)

            # 2. Operation-Aware Normalized Projected Tool Match Loss
            t_repr = model.encode_tools(cand_batch)
            scores = model.score_candidates(h_ctx, t_repr, oper_repr=h_ctx)
            loss_match = criterion_match(scores, target_cand_batch)

            # 3. Auxiliary Intent Loss
            loss_intent = criterion_intent(intent_logits, intent_batch)

            # 4. Auxiliary Operation Loss
            loss_oper = criterion_oper(oper_logits, oper_batch)

            # 5. Auxiliary Memory Contradiction Loss
            loss_contra = criterion_contra(mem_constraint_logits, mem_contra_batch)

            # 6. 4-Tier Severity Classification Loss
            loss_sev = criterion_severity(severity_logits, sev_batch)

            loss = (loss_action + lambda_match * loss_match + lambda_intent * loss_intent + 
                    lambda_oper * loss_oper + lambda_contra * loss_contra + lambda_severity * loss_sev)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == num_epochs:
            train_metrics = evaluate_model(model, train_loader, use_calibrated=False)
            val_metrics = evaluate_model(model, val_loader, use_calibrated=False)
            history.append({
                "epoch": epoch,
                "loss": round(total_loss / len(train_loader), 4),
                "train_action_acc": train_metrics["action_accuracy"],
                "val_action_acc": val_metrics["action_accuracy"],
                "val_tool_acc": val_metrics["tool_accuracy"],
                "val_severity_acc": val_metrics["severity_accuracy"],
                "val_intent_acc": val_metrics["intent_accuracy"],
                "val_oper_acc": val_metrics["operation_accuracy"],
                "val_f1": val_metrics["f1"]
            })
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {history[-1]['loss']:.4f} | "
                  f"Train Act: {train_metrics['action_accuracy']*100:.1f}% | "
                  f"Val Act: {val_metrics['action_accuracy']*100:.1f}% | "
                  f"Val Tool: {val_metrics['tool_accuracy']*100:.1f}% | "
                  f"Val Sev: {val_metrics['severity_accuracy']*100:.1f}% | "
                  f"Val Intent: {val_metrics['intent_accuracy']*100:.1f}% | "
                  f"Val Oper: {val_metrics['operation_accuracy']*100:.1f}% | Val F1: {val_metrics['f1']:.4f}", flush=True)
        else:
            print(f"Epoch {epoch:02d}/{num_epochs:02d} complete | Loss: {total_loss / len(train_loader):.4f}", flush=True)

    # 7. Validation Temperature Calibration
    print("\n--- LEARNING VALIDATION TEMPERATURE SCALING (T) ---")
    calibrated_T = fit_temperature(model, val_loader)
    print(f"Fitted Validation Temperature T = {calibrated_T}")

    # 8. Evaluate on Isolated Unseen Test Set (with frozen T)
    test_metrics_raw = evaluate_model(model, test_loader, use_calibrated=False)
    test_metrics_calibrated = evaluate_model(model, test_loader, use_calibrated=True)
    
    print("\n=== FINAL UNSEEN TEST EVALUATION (PHASE H.4) ===")
    print(f"Action Selection Accuracy   : {test_metrics_calibrated['action_accuracy']*100:.2f}%")
    print(f"Tool Selection Accuracy     : {test_metrics_calibrated['tool_accuracy']*100:.2f}%")
    print(f"Intent Selection Accuracy   : {test_metrics_calibrated['intent_accuracy']*100:.2f}%")
    print(f"Operation Selection Accuracy: {test_metrics_calibrated['operation_accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics_calibrated['precision']:.4f} | Recall: {test_metrics_calibrated['recall']:.4f} | F1: {test_metrics_calibrated['f1']:.4f}")
    print(f"Uncalibrated ECE: {test_metrics_raw['ece']:.4f} -> Calibrated ECE: {test_metrics_calibrated['ece']:.4f}")
    print(f"Confusion Matrix: {test_metrics_calibrated['confusion_matrix']}")
    print(f"Confidence Buckets: {json.dumps(test_metrics_calibrated['buckets'], indent=2)}")

    # 9. Save Model Artifacts
    save_path = os.path.join(os.path.dirname(__file__), "latis_model.pt")
    metadata = {
        "vocab_word2idx": vocab.word2idx,
        "vocab_idx2word": vocab.idx2word,
        "vocab_size": len(vocab.word2idx),
        "max_seq_len": vocab.max_seq_len,
        "tool_max_len": vocab.tool_max_len,
        "field_len": vocab.field_len,
        "embed_dim": 64,
        "hidden_dim": 96,
        "num_tools": len(TOOL_REGISTRY),
        "tool_registry": TOOL_REGISTRY,
        "tool_catalog": TOOL_SPEC_CATALOG,
        "severity_registry": SEVERITY_REGISTRY,
        "temperature": calibrated_T,
        "lambda_match": lambda_match,
        "lambda_intent": lambda_intent,
        "lambda_oper": lambda_oper,
        "lambda_contra": lambda_contra,
        "lambda_severity": lambda_severity,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_calibrated,
        "history": history
    }
    
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": metadata
    }, save_path)
    print(f"Model saved successfully to: {save_path}")

    return model, vocab, metadata


if __name__ == "__main__":
    train_latis_behavior_model()


