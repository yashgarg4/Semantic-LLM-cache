# semcache — common tasks. Activate the venv first:
#   Windows:  .venv\Scripts\activate
#   Unix:     source .venv/bin/activate
.PHONY: help install demo demo-wrap proxy dashboard test

help:
	@echo "make install    - install pinned deps + editable package"
	@echo "make demo       - run the Phase 1/2 core cache demo"
	@echo "make demo-wrap  - run the Phase 3 @cached decorator demo"
	@echo "make proxy      - run the OpenAI-compatible proxy (also serves /metrics)"
	@echo "make dashboard  - run the Streamlit metrics dashboard"
	@echo "make test       - run the test suite"

install:
	pip install -r requirements.txt
	pip install -e .

demo:
	python examples/demo_basic.py

demo-wrap:
	python examples/demo_wrap_agent.py

proxy:
	uvicorn server.proxy:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q
