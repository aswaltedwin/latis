"""
Script to generate 10,000 enterprise customer accounts for Zvy.
"""
import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMERS_FILE = os.path.join(BASE_DIR, "data", "customers.json")

random.seed(42)
first_names = [
    'Apex', 'Centurion', 'Nova', 'Cyber', 'Omni', 'Vanguard', 'Summit', 'Global', 
    'Prime', 'Hyper', 'Titan', 'Acro', 'Pulse', 'Stratosphere', 'Echo', 'Beacon', 
    'Atlas', 'Orbital', 'Synergy', 'Quantum', 'Nexus', 'Vertex', 'Dynamic', 
    'Pinnacle', 'Starlight', 'Ironclad', 'Cobalt', 'Silverline', 'Frontier', 'Zenith'
]
last_names = [
    'Logistics', 'Financial', 'Cloud', 'Technologies', 'Dynamics', 'Robotics', 
    'Systems', 'Solutions', 'Holdings', 'Industries', 'Enterprises', 'Ventures', 
    'Networks', 'Labs', 'Analytics', 'Energy', 'Capital', 'Consulting', 'Security', 'Platforms'
]
managers = [
    'Elena Rostova', 'Marcus Vance', 'Sarah Chen', 'Ananya Patel', 'David Kim', 
    'Rachel Adams', 'Carlos Mendez', "Liam O'Connor", 'Devon Hayes', 'Zackary Thorne'
]
statuses = ['Active VIP', 'Active Standard', 'Good Standing', 'Under Review', 'Past Due', 'Pending Audit']
tiers = ['Enterprise VIP', 'Enterprise Tier-1', 'Pro Scale', 'Growth Plan']

customers = {}
for i in range(101, 10101):
    cname = f"{random.choice(first_names)} {random.choice(last_names)} {i}"
    bal = round(random.uniform(500.0, 185000.0), 2)
    tier = random.choice(tiers)
    mgr = random.choice(managers)
    stat = random.choice(statuses)
    clean_name = "".join(c for c in cname.lower() if c.isalnum())
    email = f"contact@{clean_name}.com"
    invoices = [f"INV-{random.randint(1000, 9999)} (${random.randint(1500, 45000):,})" for _ in range(random.randint(2, 5))]
    
    customers[str(i)] = {
        "id": str(i),
        "name": cname,
        "balance": bal,
        "status": stat,
        "tier": tier,
        "account_manager": mgr,
        "email": email,
        "recent_invoices": invoices
    }

os.makedirs(os.path.dirname(CUSTOMERS_FILE), exist_ok=True)
with open(CUSTOMERS_FILE, "w", encoding="utf-8") as f:
    json.dump(customers, f, indent=2)

print(f"Generated {len(customers)} enterprise accounts in {CUSTOMERS_FILE}")
