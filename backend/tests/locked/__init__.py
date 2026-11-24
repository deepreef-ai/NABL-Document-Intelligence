"""LOCKED regression suite.

Each test here guards a property a human reviewer would insist on for a
lab-report attestation product. These files are hashed into SUITE.lock; changing
one without `scripts/lock_tests.py --update --reason "..."` fails
test_lock_integrity.py. Change them only with a deliberate, reviewed reason.
"""
