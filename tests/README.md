# Chapter 5 Testing and Verification

FinPersona is verified with a `pytest` suite that combines module tests and
integrated system tests. The suite uses small deterministic data frames and
mocked model stages, so it can run quickly during development without relying
on GPU training jobs or large production datasets.

## 5.1 Testing Setup

- Test runner: `pytest`
- Test location: `tests/`
- Environment: local Python environment for the FinPersona repository
- Data strategy: synthetic in-memory data for unit tests, temporary files for
  output verification, and monkeypatched dependencies for integration tests
- Command: `python -m pytest tests`

## 5.2 Testing Plan and Strategy

The strategy is layered. Unit tests validate deterministic functions in each
module, while integration tests verify that modules communicate correctly,
proxy failures are handled, launcher configuration is complete, and batch
recommendation scoring produces the expected output schema.

## 5.2.1 Module Testing

Covered examples:

- Credit risk target generation for hard, soft, weighted, and tiered labels
- Credit evaluation metrics and F-beta threshold selection
- Churn model comparison sorting by ROC AUC
- Recommendation preprocessing for missing values and one-hot encoding
- Recommendation threshold optimization and multilabel metric reporting

## 5.2.2 Integration Testing

Covered examples:

- Gateway proxy behavior when a backend module is unavailable
- Unified launcher server list, ports, and script existence
- Recommendation batch scoring pipeline with mocked feature engineering and
  model prediction stages, verifying customer-level output and saved CSVs

## 5.3 Testing Schedule

- During development: run affected tests after each module change
- Before integration: run all module tests
- Before delivery: run the full `tests/` suite and save the result summary

## 5.4 Comparative Results to Previous Work

| Area | Previous manual check | Current automated check |
| --- | --- | --- |
| Credit labels | Inspect CSV/report output | Deterministic target-generation tests |
| Churn model selection | Read printed comparison table | Assertion that best ROC AUC is ranked first |
| Recommendation outputs | Manually open generated CSV | Pipeline test verifies schema and saved file |
| UI gateway | Browser-only smoke test | Proxy failure test verifies 502 JSON response |

These tests provide repeatable evidence that the implemented modules and the
integrated system satisfy the expected project outcomes.
