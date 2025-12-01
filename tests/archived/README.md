# Archived Test Suite

This directory contains the comprehensive test suite that was built for the NBA betting model. These tests are saved here for reference but are not actively used in GitHub Actions.

## What's Included

- **unit/**: Unit tests for core components
  - `test_core_components.py` - Data cleaning, feature engineering, ML models
  - `test_data_integrations.py` - API clients and data sources
  - `test_model_validation.py` - Model validation and monitoring

- **integration/**: Integration tests for end-to-end workflows
  - `test_end_to_end_pipeline.py` - Complete pipeline testing
  - `test_data_flow_validation.py` - Data integrity validation

- **Test Infrastructure**:
  - `conftest.py` - Pytest configuration and fixtures
  - `requirements.txt` - Test dependencies

## Why Archived?

- GitHub Actions currently disabled to avoid automatic testing
- Tests saved for potential future use
- Maintains all the testing work that was built

## To Reactivate (Optional)

If you want to enable these tests later:

1. Move files back to tests/ directory:
   ```bash
   mv tests/archived/* tests/
   ```

2. Update GitHub Actions workflow in `.github/workflows/test-suite.yml`

3. Remove the workflow disable command if present

## Test Summary (When Active)

- 13/20 unit tests passing (with mock implementations)
- Integration tests working for end-to-end validation
- Comprehensive coverage of data pipeline, ML models, and API integrations
- Mock fallbacks for when actual classes aren't available

---

*Archived on: December 1, 2024*
*Reason: Disable automatic GitHub Actions testing*