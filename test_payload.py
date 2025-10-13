#!/usr/bin/env python3
import random
from training_new import _generate_structured_payload

# Test the structured payload generation
rng = random.Random(12345)
print("Testing Structured Payload Generation:")
print("=" * 50)

for i in range(5):
    payload = _generate_structured_payload(rng)
    payload_str = payload.decode('utf-8', errors='replace')
    print(f'Payload {i+1}: {payload_str}')
    print(f'Length: {len(payload)} bytes')
    print(f'Structure check: ISRC={payload_str[:16].startswith("ISRC")}, ISFR={"ISFR" in payload_str[:32]}, N={"N" in payload_str}, D={"D" in payload_str}')
    print()
