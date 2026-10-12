# Reproducibility targets for the TASCP artifact.
#
# Default Python: `python3`. CPU is sufficient for every target except the
# Transformer architecture comparison (exp22). Runtimes are approximate and
# measured on the reference machine (see LIMITATIONS.md).
#
# Fresh outputs are written under results_fresh/ and never overwrite the
# canonical archive in results/ (see doc/adr/ADR-004).

PY ?= python3
FRESH := results_fresh

.PHONY: help setup verify verify-env test traceability smoke \
        reproduce-foundational reproduce-adaptive reproduce-ablations \
        reproduce-blackbox reproduce-hybrid reproduce-disambiguation reproduce-multiseed \
        reproduce-stationarity reproduce-refmath reproduce-offline reproduce-hp-sweep \
        reproduce-family reproduce-triggers reproduce-arch reproduce-audit \
        reproduce-microbench reproduce-transfer reproduce-transformer \
        reproduce-performance reproduce-all clean

help:
	@echo "Targets:"
	@echo "  make setup                     install pinned deps (see REPRODUCE.md)"
	@echo "  make verify                    checkpoint + dataset SHA-256 check"
	@echo "  make verify-env                check interpreter/deps vs requirements.txt"
	@echo "  make test                      run the test suite"
	@echo "  make traceability              check archived results carry traceability records"
	@echo "  make smoke                     verify-env + verify + traceability + tests"
	@echo "  make reproduce-foundational    C1-C3: report exp1-3 as not implemented (see OPEN_SCIENCE.md)"
	@echo "  make reproduce-adaptive        C4: adaptive BPDA white-box attack (exp5)"
	@echo "  make reproduce-ablations       C4: adaptive ablations (exp8)"
	@echo "  make reproduce-blackbox        C5: NES/HSJ black-box attacks (exp9)"
	@echo "  make reproduce-hybrid          C6: hybrid detection + robustness (exp10/exp11)"
	@echo "  make reproduce-disambiguation  C6: CICIDS control-budget disambiguation (exp12)"
	@echo "  make reproduce-multiseed       C7: 5-seed white-box replication (exp13)"
	@echo "  make reproduce-stationarity    C8: verdict stationarity (exp14)"
	@echo "  make reproduce-refmath         C9: reference-math validation (exp16)"
	@echo "  make reproduce-offline         C10: offline attacker + scope (exp15)"
	@echo "  make reproduce-hp-sweep        C10: gamma x tau x window sweep (exp17)"
	@echo "  make reproduce-family          C11: transformation family + coupling (exp19/exp20)"
	@echo "  make reproduce-triggers        C12: trigger x attack-family matrix (exp21)"
	@echo "  make reproduce-arch            C13: architecture split (exp22)"
	@echo "  make reproduce-audit           C14: defense-class audit (exp23)"
	@echo "  make reproduce-microbench      C15: in-process Rust hot-path bench (exp18)"
	@echo "  make reproduce-transfer        C16: NF-ToN-IoT-v2 third-dataset transfer (exp24)"
	@echo "  make reproduce-transformer     C13: Transformer architecture comparison (exp_transformer)"
	@echo "  make reproduce-performance     C15: FPR + latency (exp6/exp7)"
	@echo "  make reproduce-all             run all of the above"
	@echo "  make clean                     remove fresh outputs (preserves archive)"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

verify-env:
	$(PY) scripts/verify_environment.py

verify:
	$(PY) verification/verify_manifest.py
	@if [ -d data/processed ]; then \
		$(PY) verification/verify_datasets.py; \
	else \
		echo "verify_datasets: SKIP (data/processed absent; see dataset_links.md)"; \
	fi

test:
	$(PY) -m pytest tests/ -q

traceability:
	$(PY) scripts/check_traceability.py

smoke: verify-env verify traceability test
	@echo "smoke: PASS (environment + integrity + traceability + tests)"

# --- Claim I: mechanism behaves as specified ------------------------------
# exp1-exp3 are placeholders: the artifact ships no implementation or archived
# output for the momentum-separation / gradient-correlation / query-budget
# claims. They exit 2 if invoked directly so orchestration cannot mistake a
# no-op for success. See doc/OPEN_SCIENCE.md.
reproduce-foundational:
	@echo "reproduce-foundational: SKIPPED — exp1/exp2/exp3 are not implemented"
	@echo "  (claims C1-C3 are not independently reproducible in this artifact;"
	@echo "   see doc/OPEN_SCIENCE.md). Their reproducible component — the"
	@echo "   DACM/TASCP/undefended white-box comparison — is covered by exp5."

# --- Claim II: no security boundary ---------------------------------------
reproduce-adaptive:
	$(PY) -m experiments.exp5_adaptive_bpda_attack.run

reproduce-ablations:
	$(PY) -m experiments.exp8_adaptive_ablations.run

reproduce-blackbox:
	$(PY) -m experiments.exp9_blackbox_attack.run
	$(PY) -m experiments.exp9_blackbox_attack.run_cicids2017
	$(PY) -m experiments.exp9_blackbox_attack.run_multiseed

reproduce-hybrid:
	$(PY) -m experiments.exp10_hybrid_detection_sim.run
	$(PY) -m experiments.exp11_detector_robustness.run

reproduce-disambiguation:
	$(PY) -m experiments.exp12_cicids_disambiguation.run cicids2017

reproduce-multiseed:
	$(PY) -m experiments.exp13_whitebox_multiseed.run unsw_nb15
	$(PY) -m experiments.exp13_whitebox_multiseed.run cicids2017
	@echo "Compare a fresh run with: $(PY) verification/compare_results.py results/logs/exp13_whitebox_multiseed/multiseed_unsw_nb15.json <fresh>.json"

reproduce-stationarity:
	$(PY) -m experiments.exp14_stationarity.run

reproduce-refmath:
	$(PY) -m experiments.exp16_refmath_validation.run

reproduce-offline:
	$(PY) -m experiments.exp15_offline_and_scope.run

reproduce-hp-sweep:
	$(PY) -m experiments.exp17_hp_sweep.run unsw_nb15

reproduce-family:
	$(PY) -m experiments.exp19_trajectory_family.run unsw_nb15
	$(PY) -m experiments.exp20_coupling_audit.run unsw_nb15

reproduce-triggers:
	$(PY) -m experiments.exp21_trigger_matrix.run unsw_nb15

reproduce-arch:
	$(PY) -m experiments.exp22_arch_split.run unsw_nb15

reproduce-audit:
	$(PY) -m experiments.exp23_defense_audit.run unsw_nb15

# --- Claim III: deployable and still pointless ----------------------------
reproduce-microbench:
	$(PY) -m experiments.exp18_rust_bench.run

reproduce-transfer:
	$(PY) -m experiments.exp24_ton_iot_transfer.run

reproduce-transformer:
	$(PY) -m experiments.exp_transformer.run

reproduce-performance:
	$(PY) -m experiments.exp6_false_positive_rate.run
	$(PY) -m experiments.exp7_latency_benchmark.run

reproduce-all: reproduce-foundational reproduce-adaptive \
	reproduce-ablations reproduce-hybrid reproduce-disambiguation \
	reproduce-multiseed reproduce-stationarity reproduce-refmath \
	reproduce-offline reproduce-hp-sweep reproduce-family reproduce-triggers \
	reproduce-arch reproduce-audit reproduce-microbench reproduce-transfer \
	reproduce-transformer reproduce-performance
	@echo "reproduce-all: complete (black-box sweeps run via reproduce-blackbox)"

clean:
	rm -rf $(FRESH) results/*.fresh.* results/foolbox/*.fresh* __pycache__
