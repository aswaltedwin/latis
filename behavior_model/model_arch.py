"""
LATIS Neural Behavior Model (Phase H.2 Architecture)
A purpose-built neural network trained from scratch to model GenAI agent behavior:
Context (System, RAG, Memory, Auth, Query) + Candidate Tools -> Action Class (ANSWER vs TOOL_CALL) + Dynamic Candidate Tool Scoring.

Phase H.2 Upgrades:
1. Field-Aware Context Representation: Encodes [SYS], [DOC], [MEM], [AUTH], and [QUERY] independently.
2. Cross-Field Multi-Head Self-Attention Fusion: Eliminates positional ordering bias across segments.
3. Learned Query-Memory Relevance Gating: Quantifies and routes memory influence dynamically.
4. Auxiliary Query Intent Head: Distinctly separates informational requests from operational directives.
5. Description-Centric Candidate Tool Representation: Decouples tool semantics from arbitrary tool identifiers.
6. Zero pretrained weights. Zero external LLMs. Fully project-owned.
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple


class FastVocabulary:
    """Character/Word n-gram hash vectorizer and vocabulary with segment tokens and subword fallback."""
    SPECIAL_TOKENS = [
        "<PAD>", "<UNK>", "<MASK>", "<SEP>",
        "[SYS]", "[DOC]", "[MEM]", "[QUERY]", "[TOOLS]", "[AUTH]"
    ]

    def __init__(self, vocab_size: int = 4000, max_seq_len: int = 192, tool_max_len: int = 48, field_len: int = 48):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tool_max_len = tool_max_len
        self.field_len = field_len
        self.word2idx: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.idx2word: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.is_locked = False

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary frequencies from text corpus."""
        word_counts = {}
        for text in texts:
            tokens = self._tokenize(text)
            for tok in tokens:
                word_counts[tok] = word_counts.get(tok, 0) + 1
                
        # Sort by frequency and add up to vocab_size
        sorted_tokens = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for tok, _ in sorted_tokens:
            if len(self.word2idx) >= self.vocab_size:
                break
            if tok not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[tok] = idx
                self.idx2word[idx] = tok
        self.is_locked = True

    def _subtokenize_word(self, w: str) -> List[str]:
        """Splits words longer than 4 characters into 4-gram subwords for OOV resilience."""
        if len(w) <= 4:
            return [w]
        parts = []
        for i in range(0, len(w), 4):
            parts.append(w[i:i+4])
        return parts

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower().replace("\n", " ").replace("\t", " ")
        # Ensure special segment tokens are isolated with spaces
        for sp in ["[sys]", "[doc]", "[mem]", "[query]", "[tools]", "[auth]", "<sep>", "<pad>"]:
            text = text.replace(sp, f" {sp} ")
        cleaned = "".join([c if (c.isalnum() or c in " _-.:/@[]") else " " for c in text])
        raw_words = [t.strip() for t in cleaned.split() if t.strip()]
        
        tokens = []
        for w in raw_words:
            w_upper = w.upper()
            if w_upper in self.word2idx:
                tokens.append(w_upper)
            else:
                tokens.append(w)
                if len(w) > 4:
                    tokens.extend(self._subtokenize_word(w))
        return tokens

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        limit = max_len or self.max_seq_len
        tokens = self._tokenize(text)
        indices = []
        for t in tokens[:limit]:
            indices.append(self.word2idx.get(t, 1))  # 1 = <UNK>
        if len(indices) < limit:
            indices += [0] * (limit - len(indices))
        return indices

    def parse_fields_and_encode(self, text: str) -> List[List[int]]:
        """
        Parses multi-component incident text into 5 distinct semantic field token sequences:
        Order: [SYS, DOC, MEM, AUTH, QUERY].
        Each returned as a list of integers of length self.field_len.
        """
        fields = {"sys": "", "doc": "", "mem": "", "auth": "", "query": ""}
        pattern = r"\[(SYS|DOC|MEM|AUTH|QUERY)\]\s*(.*?)(?=\[(?:SYS|DOC|MEM|AUTH|QUERY)\]|$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            for tag, content in matches:
                fields[tag.lower()] = content.strip()
        else:
            fields["query"] = text.strip()

        order = ["sys", "doc", "mem", "auth", "query"]
        return [self.encode(fields[k], max_len=self.field_len) for k in order]


class LATISBehaviorNet(nn.Module):
    """
    Project-owned neural model with Field-Aware Context Representation,
    Query-Memory Gating, Auxiliary Intent Modeling, and Dynamic Candidate Tool Scoring.
    """
    def __init__(
        self,
        vocab_size: int = 4000,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        num_tools: int = 8,
        dropout: float = 0.20
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Shared recurrent field encoder (sys, doc, mem, auth, query)
        self.field_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.field_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )
        self.field_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Candidate tool specification encoder
        self.tool_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.tool_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )
        self.tool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Query-Memory Relevance Gating: g_mem = sigmoid(h_q W h_mem + b)
        self.gate_qm = nn.Bilinear(128, 128, 1)
        
        # Field Fusion: 5 field embeddings (0=sys, 1=doc, 2=mem, 3=auth, 4=query)
        self.field_embed = nn.Embedding(5, 128)
        self.fusion_mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(128)
        self.context_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Head 1: Action Classification (Binary: ANSWER/NO_ACTION vs TOOL_ACTION)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Auxiliary Head 1: Query Intent Classification (0=INFORMATIONAL, 1=OPERATIONAL)
        self.intent_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Auxiliary Head 2: Operation / Verb Classification (12 categories)
        # RETRIEVE=0, QUERY=1, ARCHIVE=2, DELETE=3, SEND=4, UPDATE=5,
        # CREATE=6, APPROVE=7, REJECT=8, BLOCK=9, INFORMATIONAL=10, OTHER=11
        self.operation_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )

        # Learned Query-Memory Constraint / Contradiction Head:
        # Measures whether memory actively contradicts/inhibits query action
        self.mem_constraint_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [0: NEUTRAL/SUPPORTIVE, 1: CONTRADICTORY/INHIBITORY]
        )

        # 4-Tier Security Severity Head (0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL)
        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        # Phase H.4 Normalized Projected Candidate Scoring:
        # f(c) = context projection, g(t) = tool projection
        self.ctx_match_proj = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )
        self.tool_match_proj = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )
        # Phase H.5 Operation-Aware Candidate Alignment:
        # Learned bridge between query operation representation and tool capability representation
        self.oper_match_proj = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )
        self.tool_oper_proj = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )
        # Learned scale parameter tau for cosine similarity
        self.tau = nn.Parameter(torch.tensor(10.0))
        # Learned gating parameter alpha between context score and operation score:
        # alpha = sigmoid(raw_alpha). Initialize raw_alpha ~ 0.85 -> sigmoid(0.85) ~ 0.70
        self.raw_alpha = nn.Parameter(torch.tensor(0.85))

        # Phase H.5 Learned Intent -> Action Coupling Projection:
        # Transforms informational probability p_info into an inhibitory residual bias against TOOL_ACTION (index 1)
        self.intent_inhibit_weight = nn.Parameter(torch.tensor([5.0]))
        self.intent_support_weight = nn.Parameter(torch.tensor([3.0]))

        # Dynamic Candidate Match Layer (retained for backward compatibility)
        self.match_layer = nn.Bilinear(128, 128, 1)

        # Legacy fixed tool head (for backward compatibility if needed)
        self.legacy_tool_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_tools)
        )

        # Temperature calibration parameter
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def encode_single_field(self, field_tokens: torch.Tensor) -> torch.Tensor:
        """Encodes a single field tensor (B, L) -> (B, 128)."""
        emb = self.embedding(field_tokens)
        out, _ = self.field_gru(emb)
        weights = F.softmax(self.field_attn(out), dim=1)
        pooled = torch.sum(out * weights, dim=1)
        return self.field_proj(pooled)

    def encode_fields(self, field_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase H.4 Field-aware context encoding:
        field_tensors: (B, 5, field_len) where 5 fields are [sys, doc, mem, auth, query]
        returns: (h_context, intent_logits, operation_logits, mem_constraint_logits, g_mem)
        """
        B, N, L = field_tensors.shape
        flat = field_tensors.view(B * N, L)
        h_flat = self.encode_single_field(flat)  # (B*5, 128)
        h_fields = h_flat.view(B, N, 128)
        
        h_sys = h_fields[:, 0]
        h_doc = h_fields[:, 1]
        h_mem = h_fields[:, 2]
        h_auth = h_fields[:, 3]
        h_query = h_fields[:, 4]
        
        # 1. Query-Memory Relevance Gating: g_mem = sigmoid(h_q W h_mem + b)
        g_mem = torch.sigmoid(self.gate_qm(h_query, h_mem))  # (B, 1)
        h_mem_gated = h_mem * g_mem
        
        # 2. Add learned semantic field identity embeddings
        f_ids = torch.arange(5, device=field_tensors.device).unsqueeze(0).expand(B, 5)
        f_emb = self.field_embed(f_ids)
        
        gated_fields = torch.stack([h_sys, h_doc, h_mem_gated, h_auth, h_query], dim=1) + f_emb
        
        # 3. Cross-field Multi-Head Attention Fusion
        fused_mha, _ = self.fusion_mha(gated_fields, gated_fields, gated_fields)
        fused = self.fusion_norm(gated_fields + fused_mha)  # (B, 5, 128)
        fused_pool = torch.mean(fused, dim=1)  # (B, 128)
        
        # 4. Context Combination: Fused fields + Dedicated Query representation
        h_context = self.context_fusion(torch.cat([fused_pool, h_query], dim=-1))  # (B, 128)
        
        # 5. Auxiliary Query Intent Head (0=INFORMATIONAL, 1=OPERATIONAL)
        intent_logits = self.intent_head(h_query)

        # 6. Auxiliary Operation / Verb Head (12 categories)
        operation_logits = self.operation_head(h_query)

        # 7. Learned Query-Memory Constraint Head (0=NEUTRAL/SUPPORTIVE, 1=CONTRADICTORY/INHIBITORY)
        # Evaluates joint query and memory representations
        qm_joint = torch.cat([h_query, h_mem], dim=-1)  # (B, 256)
        mem_constraint_logits = self.mem_constraint_head(qm_joint)  # (B, 2)

        # 8. 4-Tier Security Severity Classification (0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL)
        severity_logits = self.severity_head(h_context)  # (B, 4)
        
        return h_context, intent_logits, operation_logits, mem_constraint_logits, g_mem, severity_logits

    def compute_action_logits(self, h_context: torch.Tensor, mem_constraint_logits: Optional[torch.Tensor] = None, intent_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes action logits with optional learned constraint/inhibition residual pathway
        and learned intent-to-action coupling.
        When memory actively contradicts query action:
            p_contra imparts a learned inhibitory bias against TOOL_ACTION (index 1).
        When query intent is predicted as INFORMATIONAL:
            p_info imparts a learned inhibitory bias against TOOL_ACTION (index 1) and positive bias for ANSWER (index 0).
        """
        base_logits = self.action_head(h_context)  # (B, 2)
        total_logits = base_logits
        
        if mem_constraint_logits is not None:
            # p_contra: probability that memory is contradictory/inhibitory
            p_contra = F.softmax(mem_constraint_logits, dim=-1)[:, 1:2]  # (B, 1)
            # Inhibit action=1 (TOOL_ACTION) proportional to learned contradiction probability
            inhibition_mem = torch.cat([p_contra * 4.0, -p_contra * 6.0], dim=-1)  # (B, 2)
            total_logits = total_logits + inhibition_mem

        if intent_logits is not None:
            # p_info: probability that query is INFORMATIONAL (class 0)
            p_info = F.softmax(intent_logits, dim=-1)[:, 0:1]  # (B, 1)
            # Learned intent residual: boosts class 0 (ANSWER), suppresses class 1 (TOOL_ACTION)
            inhibition_intent = torch.cat([p_info * self.intent_support_weight, -p_info * self.intent_inhibit_weight], dim=-1)
            total_logits = total_logits + inhibition_intent

        return total_logits

    def encode_context(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Flexible context encoder:
        If context_tokens is 3D (B, 5, L), processes through field-aware pipeline.
        If context_tokens is 2D (B, L), parses segment tokens or processes through field encoder.
        """
        if context_tokens.dim() == 3 and context_tokens.size(1) == 5:
            res = self.encode_fields(context_tokens)
            return res[0]
        elif context_tokens.dim() == 2:
            B, L = context_tokens.shape
            chunk_len = max(1, L // 5)
            field_list = []
            for i in range(5):
                start = i * chunk_len
                end = min((i + 1) * chunk_len, L)
                sub = context_tokens[:, start:end]
                if sub.size(1) < chunk_len:
                    pad = torch.zeros(B, chunk_len - sub.size(1), dtype=torch.long, device=context_tokens.device)
                    sub = torch.cat([sub, pad], dim=1)
                field_list.append(sub)
            field_t = torch.stack(field_list, dim=1)
            res = self.encode_fields(field_t)
            return res[0]
        else:
            emb = self.embedding(context_tokens)
            out, _ = self.field_gru(emb)
            weights = F.softmax(self.field_attn(out), dim=1)
            pooled = torch.sum(out * weights, dim=1)
            return self.field_proj(pooled)

    def encode_tools(self, tool_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encodes candidate tool definitions into latent vectors.
        tool_tokens: (K, L) or (B, K, L)
        returns: (K, 128) or (B, K, 128)
        """
        if tool_tokens.dim() == 3:
            B, K, L = tool_tokens.shape
            flat_tools = tool_tokens.view(B * K, L)
            emb = self.embedding(flat_tools)
            out, _ = self.tool_gru(emb)
            weights = F.softmax(self.tool_attn(out), dim=1)
            pooled = torch.sum(out * weights, dim=1)
            proj = self.tool_proj(pooled)
            return proj.view(B, K, -1)
        else:
            emb = self.embedding(tool_tokens)
            out, _ = self.tool_gru(emb)
            weights = F.softmax(self.tool_attn(out), dim=1)
            pooled = torch.sum(out * weights, dim=1)
            return self.tool_proj(pooled)

    def score_candidates(self, context_repr: torch.Tensor, tool_reprs: torch.Tensor, oper_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Phase H.5 Operation-Aware Normalized Projected Candidate Scoring:
        f(c) = projection of context representation -> c_norm
        g(t) = projection of tool representation -> t_norm
        context_score = cosine(c_norm, t_norm)

        When oper_repr is provided (e.g. h_query or operation representation):
        o(q) = projection of query operation representation -> o_norm
        v(t) = projection of tool capability into operation space -> t_oper_norm
        operation_score = cosine(o_norm, t_oper_norm)

        alpha = sigmoid(raw_alpha)  (learned gating in [0, 1])
        Score(c, t) = tau * (alpha * context_score + (1 - alpha) * operation_score)
        """
        # 1. Project context representation -> (B, 64)
        c_proj = self.ctx_match_proj(context_repr)
        c_norm = F.normalize(c_proj, p=2, dim=-1)  # (B, 64)

        alpha = torch.sigmoid(self.raw_alpha)

        if tool_reprs.dim() == 3:
            # tool_reprs: (B, K, 128)
            B, K, D = tool_reprs.shape
            t_flat = tool_reprs.reshape(B * K, D)
            t_proj = self.tool_match_proj(t_flat).view(B, K, -1)  # (B, K, 64)
            t_norm = F.normalize(t_proj, p=2, dim=-1)             # (B, K, 64)

            # Cosine similarity: (B, 1, 64) x (B, 64, K) -> (B, 1, K) -> (B, K)
            c_norm_unsqueezed = c_norm.unsqueeze(1)               # (B, 1, 64)
            ctx_scores = torch.bmm(c_norm_unsqueezed, t_norm.transpose(1, 2)).squeeze(1)  # (B, K)

            if oper_repr is not None:
                o_proj = self.oper_match_proj(oper_repr)
                o_norm = F.normalize(o_proj, p=2, dim=-1).unsqueeze(1)  # (B, 1, 64)
                t_oper_proj = self.tool_oper_proj(t_flat).view(B, K, -1)  # (B, K, 64)
                t_oper_norm = F.normalize(t_oper_proj, p=2, dim=-1)      # (B, K, 64)
                oper_scores = torch.bmm(o_norm, t_oper_norm.transpose(1, 2)).squeeze(1)  # (B, K)
                combined = alpha * ctx_scores + (1.0 - alpha) * oper_scores
                return self.tau * combined
            else:
                return self.tau * ctx_scores
        else:
            # tool_reprs: (K, 128)
            K, D = tool_reprs.shape
            t_proj = self.tool_match_proj(tool_reprs)             # (K, 64)
            t_norm = F.normalize(t_proj, p=2, dim=-1)             # (K, 64)
            # Cosine similarity: (B, 64) x (64, K) -> (B, K)
            ctx_scores = torch.matmul(c_norm, t_norm.t())         # (B, K)

            if oper_repr is not None:
                o_proj = self.oper_match_proj(oper_repr)
                o_norm = F.normalize(o_proj, p=2, dim=-1)         # (B, 64)
                t_oper_proj = self.tool_oper_proj(tool_reprs)     # (K, 64)
                t_oper_norm = F.normalize(t_oper_proj, p=2, dim=-1) # (K, 64)
                oper_scores = torch.matmul(o_norm, t_oper_norm.t()) # (B, K)
                combined = alpha * ctx_scores + (1.0 - alpha) * oper_scores
                return self.tau * combined
            else:
                return self.tau * ctx_scores

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward pass for backward compatibility."""
        c_repr = self.encode_context(x)
        action_logits = self.action_head(c_repr)
        tool_logits = self.legacy_tool_head(c_repr)
        return action_logits, tool_logits

