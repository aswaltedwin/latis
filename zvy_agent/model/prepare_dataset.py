"""
Zvy Company Model - Dataset Preparation & Distillation Pipeline
Generates high-quality fine-tuning/training data specifically aligned
to the company's 125 enterprise accounts, 10 operational policies,
and strict multi-step tool execution patterns.
"""

import os
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "customers.json")
OUTPUT_DATASET = os.path.join(MODEL_DIR, "zvy_company_training_data.jsonl")

SYSTEM_INSTRUCTION = (
    "You are Zvy-Enterprise-v1, the proprietary AI model developed specifically for this company. "
    "You autonomously manage 125 customer accounts, enforce strict policy thresholds ($500 refund limit), "
    "read compliance runbooks, and dispatch customer emails via structured tool execution."
)

def build_dataset():
    with open(CUSTOMERS_FILE, "r", encoding="utf-8") as f:
        customers = json.load(f)

    samples = []

    # 1. Customer Queries & Balance Inquiries
    for cid, c in list(customers.items())[:30]:
        user_prompt = f"Look up customer {c['name']} (ID #{cid}) and check their balance."
        tool_call = {
            "name": "query_customer",
            "arguments": {"query": cid}
        }
        tool_result = {"status": "success", "count": 1, "customers": [c]}
        assistant_resp = (
            f"Here are the account details for **{c['name']}** (Account ID #{cid}):\n"
            f"- **Tier**: {c['tier']}\n"
            f"- **Current Balance**: ${c['balance']:,.2f}\n"
            f"- **Billing Status**: {c['status']}\n"
            f"- **Account Manager**: {c.get('account_manager', 'Assigned')}\n"
            f"- **Recent Invoices**: {', '.join(c.get('recent_invoices', []))}"
        )
        samples.append({
            "instruction": user_prompt,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "response": assistant_resp
        })

    # 2. Refund Processing (Within Limit: <= $500)
    refund_reasons = ["overcharge adjustment", "billing cycle dispute", "SLA credit compensation", "service downtime discount"]
    for cid, c in list(customers.items())[30:55]:
        amt = random.choice([25.0, 50.0, 100.0, 150.0, 250.0, 450.0])
        reason = random.choice(refund_reasons)
        user_prompt = f"Process a ${amt:.0f} refund for customer {cid} due to {reason}."
        tool_call = {
            "name": "process_refund",
            "arguments": {"customer_id": cid, "amount": amt, "reason": reason}
        }
        new_bal = max(0.0, c['balance'] - amt)
        tool_result = {
            "status": "success",
            "customer_id": cid,
            "customer_name": c['name'],
            "refunded_amount": amt,
            "new_balance": new_bal,
            "reason": reason
        }
        assistant_resp = (
            f"✅ **Refund Authorized and Executed**\n\n"
            f"- **Customer**: {c['name']} (#{cid})\n"
            f"- **Refund Amount**: ${amt:,.2f}\n"
            f"- **Updated Balance**: ${new_bal:,.2f}\n"
            f"- **Policy Check**: Permitted (under $500.00 standard agent threshold)\n"
            f"- **Reason**: {reason}"
        )
        samples.append({
            "instruction": user_prompt,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "response": assistant_resp
        })

    # 3. Policy Inquiries
    policy_questions = [
        ("What is our enterprise refund limit for standard support agents?", "return_and_refund_policy.txt", 
         "According to our **Enterprise Refund & Return Policy**, standard support agents can autonomously grant refunds up to **$500.00**. Any adjustment exceeding $500.00 strictly mandates two-party manager approval."),
        ("What are the SLA response times for Enterprise VIP vs Pro Scale?", "service_level_agreements.txt",
         "Under our **Service Level Agreements (SLA)**:\n- **Enterprise VIP**: 15-minute response guarantee with dedicated technical account manager.\n- **Pro Scale**: 2-hour response window during business hours.\n- **Uptime Commitment**: 99.95% across all distributed edge services."),
        ("What is our disaster recovery RPO and RTO commitment?", "cloud_disaster_recovery_plan.txt",
         "Under our **Cloud Disaster Recovery Plan**:\n- **RPO (Recovery Point Objective)**: Under 60 seconds across active database replicas.\n- **RTO (Recovery Time Objective)**: Under 5 minutes for automated DNS failover.")
    ]
    for q, doc, ans in policy_questions:
        samples.append({
            "instruction": q,
            "tool_call": {"name": "read_document", "arguments": {"doc_name": doc}},
            "tool_result": {"status": "success", "file": doc, "content": ans},
            "response": ans
        })

    # Write training JSONL
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        for s in samples:
            item = {
                "system": SYSTEM_INSTRUCTION,
                "messages": [
                    {"role": "user", "content": s["instruction"]},
                    {"role": "assistant", "content": None, "tool_calls": [{"name": s["tool_call"]["name"], "arguments": s["tool_call"]["arguments"]}]},
                    {"role": "tool", "content": json.dumps(s["tool_result"])},
                    {"role": "assistant", "content": s["response"]}
                ]
            }
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(samples)} proprietary training samples in {OUTPUT_DATASET}")

if __name__ == "__main__":
    build_dataset()
