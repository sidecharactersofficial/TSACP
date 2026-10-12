import sys

with open("experiments/exp_transformer/run.py", "r") as f:
    content = f.read()

# Replace the conditions list to only include Undefended and Basis
old_cond = """    conditions = [
        ("Undefended", base_model, False),
        ("TASCP (Naive)", naive, False),
        ("TASCP (Full Adaptive)", full_adaptive, True),
        ("TASCP (Ablation 1: Grad Mag)", grad_mag, True),
        ("TASCP (Ablation 2: Basis)", basis_c, True),
        ("TASCP (Ablation 3: Proj Only)", proj_only, True)
    ]"""

new_cond = """    conditions = [
        ("Undefended", base_model, False),
        ("TASCP (Ablation 2: Basis)", basis_c, True)
    ]"""

content = content.replace(old_cond, new_cond)

# We want to keep track of the final x_adv for Undefended and Basis. run.py already does this!
# `saved_adv[name] = x_adv`
# We just need to add the verification at the end of the script!

verification_code = """
    print("\\n=== BIT-FOR-BIT TENSOR VERIFICATION ===")
    u_flags = success_flags_list_for_undefended  # wait, I need to get the success flags
"""

# Let's write the whole patched run_exp_transformer in a new script that just imports everything from run.py
