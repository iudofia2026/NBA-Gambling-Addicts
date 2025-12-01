#!/bin/bash
# NBA Betting Predictions - Test Runner
# ======================================
# Simple script to run tests with common configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}==>${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install with: pip install -r tests/requirements-test.txt"
    exit 1
fi

# Help message
show_help() {
    cat << EOF
NBA Betting Predictions - Test Runner

Usage: ./run_tests.sh [OPTION]

Options:
    all         Run all tests with coverage (default)
    fast        Run only fast tests (skip slow integration tests)
    unit        Run only unit tests
    integration Run only integration tests
    validation  Run only validation tests
    coverage    Run all tests and generate HTML coverage report
    watch       Run tests in watch mode (requires pytest-watch)
    ci          Run full CI pipeline (linting + tests)
    help        Show this help message

Examples:
    ./run_tests.sh              # Run all tests
    ./run_tests.sh fast         # Run fast tests only
    ./run_tests.sh coverage     # Generate coverage report
    ./run_tests.sh ci           # Run full CI pipeline

EOF
}

# Run all tests
run_all() {
    print_msg "Running all tests..."
    pytest tests -v --tb=short
}

# Run fast tests only
run_fast() {
    print_msg "Running fast tests (excluding slow integration tests)..."
    pytest tests -v -m "not slow" --durations=10
}

# Run unit tests
run_unit() {
    print_msg "Running unit tests..."
    pytest tests/unit -v
}

# Run integration tests
run_integration() {
    print_msg "Running integration tests..."
    pytest tests/integration -v
}

# Run validation tests
run_validation() {
    print_msg "Running validation tests..."
    pytest tests/validation -v
}

# Run with coverage
run_coverage() {
    print_msg "Running tests with coverage..."
    pytest tests -v --cov=src --cov-report=html --cov-report=term-missing
    print_msg "Coverage report generated at: htmlcov/index.html"

    # Try to open coverage report
    if command -v open &> /dev/null; then
        open htmlcov/index.html
    elif command -v xdg-open &> /dev/null; then
        xdg-open htmlcov/index.html
    fi
}

# Run in watch mode
run_watch() {
    if ! command -v pytest-watch &> /dev/null; then
        print_error "pytest-watch not installed. Install with: pip install pytest-watch"
        exit 1
    fi
    print_msg "Running tests in watch mode (press Ctrl+C to stop)..."
    pytest-watch tests -v -m "not slow"
}

# Run full CI pipeline
run_ci() {
    print_msg "Running full CI pipeline..."

    # Check code formatting
    if command -v black &> /dev/null; then
        print_msg "Checking code formatting..."
        black --check src tests || {
            print_warning "Code formatting issues found. Run: black src tests"
        }
    fi

    # Linting
    if command -v flake8 &> /dev/null; then
        print_msg "Linting code..."
        flake8 src tests --max-line-length=100 --exclude=.venv || {
            print_warning "Linting issues found"
        }
    fi

    # Type checking
    if command -v mypy &> /dev/null; then
        print_msg "Type checking..."
        mypy src --ignore-missing-imports || {
            print_warning "Type checking found issues"
        }
    fi

    # Run tests with coverage
    print_msg "Running tests with coverage..."
    pytest tests -v --cov=src --cov-report=term-missing --cov-report=html

    print_msg "${GREEN}CI pipeline complete!${NC}"
}

# Main script logic
case "${1:-all}" in
    all)
        run_all
        ;;
    fast)
        run_fast
        ;;
    unit)
        run_unit
        ;;
    integration)
        run_integration
        ;;
    validation)
        run_validation
        ;;
    coverage)
        run_coverage
        ;;
    watch)
        run_watch
        ;;
    ci)
        run_ci
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

exit 0
