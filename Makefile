UV ?= uv
PYTEST_ARGS ?=
ENV ?= dungeon

# Allow "make play racer" by treating the second goal as ENV and trimming extras
ifeq ($(firstword $(MAKECMDGOALS)),play)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    ENV := $(word 2,$(MAKECMDGOALS))
    override MAKECMDGOALS := play
  endif
endif

.PHONY: help install test build publish clean

help:
	@echo "Available targets:"
	@echo "  install  - Sync project dependencies with uv"
	@echo "  test     - Run pytest suite (use PYTEST_ARGS=... to filter)"
	@echo "  build    - Build distributions with uv"
	@echo "  publish  - Publish package via uv"
	@echo "  play     - Launch interactive pygame demo (ENV=dungeon|racer|hacking)"
	@echo "  clean    - Remove build/test artifacts"

install:
	$(UV) sync

test:
	$(UV) run pytest $(PYTEST_ARGS)

build:
	$(UV) build

publish:
	$(UV) publish

play:
	$(UV) run python examples/play_env_pygame.py --env $(ENV)

clean:
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache

