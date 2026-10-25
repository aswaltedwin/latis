"""
Zvy Proprietary Neural Dataset Generator
Generates a comprehensive dataset for training Zvy's custom neural model from scratch:
1. Natural Chit-Chat & Greetings (Hello, who are you, help, what can you do)
2. Customer Inquiries & CRM Balance Lookups (125 enterprise companies)
3. Financial Actions & Refund Processing (within $500 policy threshold)
4. Policy & Runbook Knowledge Retrieval (SLA, returns, security, DR, rate limits)
5. Outbound Communications & Email Dispatch
6. Host Diagnostic Utility Execution
7. Adversarial Attacks (Prompt Injections, Jailbreaks, Priv Escalation)
"""

import os
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "customers.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "model", "zvy_complete_dataset.json")

def generate_dataset():
    with open(CUSTOMERS_FILE, "r", encoding="utf-8") as f:
        customers = json.load(f)

    dataset = []

    # -------------------------------------------------------------
    # Category 1: Natural Conversation, Greetings & General Q&A (NO TOOL)
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Category 1: Natural Conversation, Chit-Chat & Operational Dialogue (NO TOOL)
    # -------------------------------------------------------------
    greetings = [
        ("hi", "Hello! I am Zvy, your autonomous Enterprise Operations & Customer Support Copilot. How can I assist you today?"),
        ("hello", "Greetings! I'm Zvy. I have operational access to customer records, financial tools, and enterprise policies. What would you like to do?"),
        ("hey", "Hey! Ready to assist you. You can ask me about customer accounts, process refunds, or look up company policies."),
        ("hey there", "Hey there! How can I help you today?"),
        ("good morning", "Good morning! Zvy is online and ready. All 125 enterprise accounts are synced and indexed."),
        ("good afternoon", "Good afternoon! How can I assist with our enterprise operations today?"),
        ("good evening", "Good evening! I'm available 24/7 for account queries, refund authorizations, and internal runbooks."),
        ("who are you?", "I am Zvy, an enterprise neural AI agent built specifically for this company. I run locally on our infrastructure with zero external cloud dependencies."),
        ("what are you?", "I am your custom enterprise operational copilot. I orchestrate customer CRM queries, refund approvals, document intelligence, and diagnostic actions."),
        ("what can you do?", "I can manage our 125 enterprise customer accounts, process authorized refunds up to $500, read company policies and SLAs, dispatch customer emails, and run diagnostic commands."),
        ("how does your model work?", "I am powered by a proprietary multi-head self-attention transformer trained from scratch specifically on our company's operations, CRM taxonomy, and policies."),
        ("is my data private?", "Yes, absolutely 100%. Everything runs locally within our private network. No data ever leaves this machine or reaches third-party clouds or public APIs."),
        ("how are you doing?", "I'm operating at peak performance! All 125 enterprise customer accounts and internal runbooks are indexed and ready."),
        ("how's everything?", "Everything is running smoothly. Latency is sub-millisecond on our local neural engine."),
        ("what is your name?", "My name is Zvy, your dedicated enterprise operations copilot."),
        ("can you help me?", "Of course! Let me know what customer account, refund, or policy document you need help with."),
        ("i need some help", "I'm right here! You can ask me to look up customer balances, verify SLA terms, process refunds under $500, or send emails."),
        ("thanks", "You're very welcome! Let me know if there's anything else you need."),
        ("thank you very much", "Always glad to help! Feel free to ask if you need further account or policy assistance."),
        ("appreciate it", "Happy to assist! I'm here whenever you need operational support."),
        ("bye", "Goodbye! Have a great day and reach out anytime you need operational support."),
        ("see you later", "See you later! Have a productive day.")
    ]

    for q, a in greetings:
        for _ in range(8):  # variations
            dataset.append({
                "query": q,
                "action_type": "CONVERSATION",
                "tool": "NONE",
                "arguments": {},
                "response_template": a
            })

    # -------------------------------------------------------------
    # Category 2: Customer CRM & Account Inquiries (Tool: query_customer)
    # -------------------------------------------------------------
    query_phrases = [
        "Look up customer {name}",
        "Check account details for {name}",
        "What is the balance of {name}?",
        "Fetch profile for customer ID #{cid}",
        "Show billing status for {name}",
        "Look up account #{cid}",
        "How much does {name} owe us?",
        "Who is the account manager for {name}?"
    ]

    for cid, c in customers.items():
        for phrase in query_phrases:
            q = phrase.format(name=c["name"], cid=cid)
            dataset.append({
                "query": q,
                "action_type": "TOOL_CALL",
                "tool": "query_customer",
                "arguments": {"query": cid},
                "target_customer_id": cid,
                "response_template": (
                    f"Here is the account profile for **{c['name']}** (ID #{cid}):\n"
                    f"- **Tier**: {c['tier']}\n"
                    f"- **Current Balance**: ${c['balance']:,.2f}\n"
                    f"- **Billing Status**: {c['status']}\n"
                    f"- **Account Manager**: {c.get('account_manager', 'Support Team')}\n"
                    f"- **Recent Invoices**: {', '.join(c.get('recent_invoices', []))}"
                )
            })

    # -------------------------------------------------------------
    # Category 3: Financial Actions & Refunds (Tool: process_refund)
    # -------------------------------------------------------------
    refund_reasons = [
        "overcharge dispute", "SLA downtime credit", "billing reconciliation", 
        "promotional courtesy credit", "service interruption rebate"
    ]

    for cid, c in list(customers.items())[:60]:
        for amt in [25.0, 50.0, 100.0, 150.0, 250.0, 400.0, 499.0]:
            reason = random.choice(refund_reasons)
            q = f"Process a ${amt:.0f} refund for customer {cid} due to {reason}."
            new_bal = max(0.0, c['balance'] - amt)
            dataset.append({
                "query": q,
                "action_type": "TOOL_CALL",
                "tool": "process_refund",
                "arguments": {"customer_id": cid, "amount": amt, "reason": reason},
                "target_customer_id": cid,
                "response_template": (
                    f"✅ **Refund Authorized and Executed**\n"
                    f"- **Customer**: {c['name']} (#{cid})\n"
                    f"- **Amount Credited**: ${amt:,.2f}\n"
                    f"- **New Balance**: ${new_bal:,.2f}\n"
                    f"- **Policy Verification**: Approved (under $500 agent limit)\n"
                    f"- **Justification**: {reason}"
                )
            })

    # -------------------------------------------------------------
    # Category 4: Policy & Runbook Lookups (Tool: read_document)
    # -------------------------------------------------------------
    policy_map = [
        ("What is our corporate return and refund policy?", "return_and_refund_policy.txt",
         "Under our **Enterprise Refund & Return Policy**, standard support agents may autonomously issue refunds up to **$500.00**. Any adjustment exceeding $500.00 requires two-party manager approval."),
        ("What are our Service Level Agreements (SLA)?", "service_level_agreements.txt",
         "Under our **Service Level Agreements (SLA)**, Enterprise VIP accounts receive a guaranteed **15-minute response time** with 99.95% uptime across all edge clusters."),
        ("What is our incident escalation protocol?", "incident_escalation_protocol.txt",
         "Our **Incident Escalation Protocol** classifies incidents from Severity 1 (Critical Production Outages / Breaches) to Severity 3. Critical issues mandate immediate session quarantine."),
        ("What are our API rate limits and quotas?", "api_rate_limits_and_quotas.txt",
         "Under our **API Quota Policy**, Enterprise VIP accounts have an allocation of 10,000 requests per minute with overages billed at $0.002 per additional 1,000 requests."),
        ("What is our cloud disaster recovery plan?", "cloud_disaster_recovery_plan.txt",
         "Our **Cloud Disaster Recovery Plan** enforces a Recovery Point Objective (RPO) of under 60 seconds and a Recovery Time Objective (RTO) of under 5 minutes for automated failover."),
        ("What are our payment terms and billing cycles?", "billing_cycle_and_payment_terms.txt",
         "Enterprise VIP accounts operate on Net-30 payment terms with a 7-day administrative grace period before automated throttling."),
        ("How does VIP customer onboarding work?", "vip_customer_onboarding_guide.txt",
         "VIP Onboarding guarantees an assigned dedicated account manager kickoff call within 24 hours, dedicated tenant allocation, and Okta SAML 2.0 SSO setup.")
    ]

    for q, doc, ans in policy_map:
        for _ in range(8):
            dataset.append({
                "query": q,
                "action_type": "TOOL_CALL",
                "tool": "read_document",
                "arguments": {"doc_name": doc},
                "response_template": ans
            })

    # -------------------------------------------------------------
    # Category 5: Outbound Communication & Email (Tool: send_customer_email)
    # -------------------------------------------------------------
    for cid, c in list(customers.items())[:30]:
        q = f"Send a payment reminder email to {c['name']} at {c['email']}"
        dataset.append({
            "query": q,
            "action_type": "TOOL_CALL",
            "tool": "send_customer_email",
            "arguments": {
                "recipient_email": c["email"],
                "subject": f"Notice: Account Balance Update for {c['name']}",
                "body": f"Dear {c['name']}, your current account balance is ${c['balance']:,.2f}. Please review your recent invoice."
            },
            "response_template": f"📧 **Outbound Dispatch Complete**: Notification sent to {c['email']} regarding active balance of ${c['balance']:,.2f}."
        })

    # -------------------------------------------------------------
    # Category 6: Host Diagnostics (Tool: run_system_command)
    # -------------------------------------------------------------
    diag_cmds = [
        ("Check system uptime and status", "uptime", "System diagnostic executed. All host services operational."),
        ("Run network connection check", "ping 127.0.0.1", "Network loopback test successful. Latency < 1ms."),
        ("List active server files", "dir", "Host filesystem directory listing completed.")
    ]
    for q, cmd, res in diag_cmds:
        for _ in range(6):
            dataset.append({
                "query": q,
                "action_type": "TOOL_CALL",
                "tool": "run_system_command",
                "arguments": {"command": cmd},
                "response_template": res
            })

    # Shuffle
    random.seed(42)
    random.shuffle(dataset)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset synthesis complete! Total training samples: {len(dataset)}")

if __name__ == "__main__":
    generate_dataset()
