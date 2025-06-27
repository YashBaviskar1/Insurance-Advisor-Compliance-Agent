

import os
COMPLIANCE_RULE_FILES = ["IRDAI_compliance_agent/compliance_rules.json", "IRDAI_compliance_agent/compliance_rules2.json"]

import json

for rule_file in COMPLIANCE_RULE_FILES:
    print(f"\nTesting: {rule_file}")
    with open(rule_file, 'r', encoding='utf-8') as f:
        try:
            rules = json.load(f)
            print(f"✅ Loaded {len(rules)} rules from {rule_file}")
        except json.JSONDecodeError as e:
            print(f"❌ JSONDecodeError in {rule_file}: {e}")