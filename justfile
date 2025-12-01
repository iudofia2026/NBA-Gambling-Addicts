# NBA Betting Predictions - Task Runner
# ======================================
# Simple commands for development and testing
# Install just: https://github.com/casey/just

# Show available commands
default:
    @just --list

# Development Commands
# ====================

# Set up development environment
setup:
    python -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -r tests/requirements-test.txt
    @echo "Setup complete! Activate with: source .venv/bin/activate"

# Run development server (if applicable)
dev:
    @echo "Development mode - no server needed for this project"
    @echo "Run 'just test-watch' to run tests in watch mode"

# Testing Commands
# ================

# Run all tests
test:
    pytest tests -v

# Run tests with coverage
test-cov:
    pytest tests -v --cov=src --cov-report=html --cov-report=term-missing
    @echo "Coverage report generated in htmlcov/index.html"

# Run only fast tests (skip slow integration tests)
test-fast:
    pytest tests -v -m "not slow" --durations=10

# Run only unit tests
test-unit:
    pytest tests/unit -v

# Run only integration tests
test-integration:
    pytest tests/integration -v

# Run only validation tests
test-validation:
    pytest tests/validation -v

# Run tests in watch mode (requires pytest-watch)
test-watch:
    pytest-watch tests -v -m "not slow"

# Run specific test file
test-file FILE:
    pytest {{FILE}} -v

# Run tests matching pattern
test-pattern PATTERN:
    pytest tests -v -k {{PATTERN}}

# Run tests in parallel (requires pytest-xdist)
test-parallel:
    pytest tests -v -n auto

# Quality Checks
# ==============

# Run all quality checks
quality: lint format-check type-check
    @echo "All quality checks passed!"

# Lint code with flake8
lint:
    flake8 src tests --max-line-length=100 --exclude=.venv

# Check code formatting
format-check:
    black --check src tests
    isort --check-only src tests

# Format code
format:
    black src tests
    isort src tests

# Type check with mypy
type-check:
    mypy src --ignore-missing-imports

# Security check
security:
    bandit -r src -ll

# Model Commands
# ==============

# Train ML models
train:
    cd src && python ml_models.py

# Generate predictions (requires ODDS_API_KEY)
predict:
    cd src && python daily_predictions.py

# Test API connection
test-api:
    cd src && python odds_api_client.py

# Data Commands
# =============

# Clean generated files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name ".coverage" -delete
    find . -type f -name "test_log.txt" -delete
    @echo "Cleaned up temporary files"

# Clean all (including models and data)
clean-all: clean
    rm -rf models/*.pkl
    rm -rf data/processed/daily_predictions_*.csv
    @echo "Cleaned all generated files"

# CI/CD Commands
# ==============

# Run full CI pipeline locally
ci: quality test-cov
    @echo "CI pipeline completed successfully!"

# Quick pre-commit checks (fast tests only)
pre-commit: format lint test-fast
    @echo "Pre-commit checks passed!"

# Production deployment check
deploy-check: quality test
    @echo "Deployment checks passed!"

# Monitoring Commands
# ===================

# Check test performance
benchmark:
    pytest tests -v --benchmark-only

# View coverage report
coverage:
    open htmlcov/index.html || xdg-open htmlcov/index.html

# Documentation Commands
# ======================

# Generate API documentation (requires pdoc)
docs:
    pdoc --html --output-dir docs src --force
    @echo "Documentation generated in docs/"

# View test report
report:
    @echo "Opening test coverage report..."
    open htmlcov/index.html || xdg-open htmlcov/index.html

# Utility Commands
# ================

# Install pre-commit hooks
install-hooks:
    @echo "Installing pre-commit hooks..."
    @echo "#!/bin/bash" > .git/hooks/pre-commit
    @echo "just pre-commit" >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo "Pre-commit hooks installed!"

# Check dependencies for security issues
check-deps:
    pip-audit

# Update dependencies
update-deps:
    pip-compile requirements.txt --upgrade
    pip-compile tests/requirements-test.txt --upgrade
