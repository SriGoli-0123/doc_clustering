# CCST reproducible workflow.  `make all` runs the full study; `make quick` is a ~10-minute smoke test.
PY      ?= .venv/bin/python
JOBS    ?= 4
CCST    := $(PY) -m ccst --jobs $(JOBS)
# Written by `synthetic` / `synthetic-v2`; later stages only need them to exist.
SYN     := data/cache/synthetic_state.pkl
SYN2    := data/cache/synthetic_v2_state.pkl

.PHONY: setup test download synthetic synthetic-v2 analysis analysis-v2 tune tiebreak extraction-check \
        candidates bertopic real corpus scaling report all quick clean

setup:            ## create the virtualenv and install everything
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
	$(PY) -m spacy download en_core_web_sm

test:             ## unit + property tests (CCST properties from proposal Sec. 5)
	$(PY) -m pytest -q

download:         ## PubMed (10 topics x 500), arXiv (10 categories x 500), SciFact, ManConCorpus
	$(CCST) download

# ---- controlled benchmarks ---------------------------------------------------------------------------------
synthetic:        ## benchmark v1 (clean prose): CCST vs 14 baselines vs 15 ablations
	$(CCST) synthetic

synthetic-v2:     ## benchmark v2 (pre-registered realistic difficulty, docs/benchmark_v2_preregistration.md)
	$(CCST) synthetic --realism

$(SYN):
	$(CCST) synthetic

$(SYN2):
	$(CCST) synthetic --realism

analysis: | $(SYN)    ## v1: alignment, certificates, analogy retrieval, budgeted review, signed clustering
	$(CCST) synthetic-analysis

analysis-v2: | $(SYN2)  ## the same analyses on benchmark v2
	$(CCST) synthetic-analysis --realism

tune: | $(SYN) $(SYN2)  ## item 2: nested-CV hyper-parameter selection (docs/tuning_protocol.md)
	$(CCST) tune

tiebreak: | $(SYN) $(SYN2)  ## item 3: sign-aware tie-breaking (docs/tiebreak_protocol.md)
	$(CCST) tiebreak

extraction-check: ## item 4: extraction accuracy on the held-out phrasing set
	$(CCST) extraction-check

# ---- real data ---------------------------------------------------------------------------------------------
real: | $(SYN)        ## SciFact + ManConCorpus (needs the synthetic model for the zero-shot analysis)
	$(CCST) real-pairs

corpus: | $(SYN)      ## RQ5 topics / clustering / summaries / t-SNE maps + zero-shot discovery on the 2k subset
	$(CCST) corpus

bertopic:         ## item 6: official BERTopic baseline (docs/bertopic_protocol.md)
	$(CCST) bertopic

scaling:          ## RQ4 runtime scaling and candidate recall
	$(CCST) scaling

candidates:       ## item 5: learned combined candidate index (docs/candidate_recall_protocol.md)
	$(CCST) candidates

report:           ## merge every REPORT.md into results/SUMMARY.md
	$(CCST) report

all: test download synthetic analysis synthetic-v2 analysis-v2 tune tiebreak extraction-check real corpus \
     bertopic scaling candidates report

quick:            ## fast end-to-end smoke test (no SPECTER / NLI downloads)
	$(CCST) synthetic --quick --no-specter --no-nli
	$(CCST) synthetic-analysis --quick --no-specter

clean:
	rm -rf results data/cache
