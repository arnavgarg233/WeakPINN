# Recovery artifacts

Frozen protocols, receipts, logs and per-arm results behind the solar numbers in
the paper. All values are recorded here; the paper is where they are interpreted.

    receipts/    completion, screening, selection and data-integrity receipts,
                 plus the corrected test scores as JSON and CSV
    protocols/   protocol documents frozen before the runs they govern
    logs/        per-arm training logs for the matched continuation
    results_v5/  per-arm, per-checkpoint result files, EMA and raw weights

## Two results, two partitions

The paper reports two solar numbers that answer different questions on different
data. They are not comparable and are never subtracted from one another.

**Recipe (chronological test).** 40,000 data-only steps, then an integrated
weak-moment continuation, scored at step 44,000. Grand-mean TSS 0.7951 against
0.7130 for the data-only arm, a difference of +0.0822. Scores in
`receipts/corrected_test_results.json`. Three test evaluations in total, one per
seed, with thresholds fit on validation only and the selection receipt frozen
beforehand (`receipts/checkpoint_selection.json`).

**Attribution (validation).** A matched continuation from one shared
step-40,000 checkpoint to a fixed step-46,000 endpoint, changing only the
physics term, gives +0.0022 against an advancement threshold of 0.009 fixed
before the runs. The threshold was not cleared, so the protocol stopped and the
test set was never opened for these arms. See
`receipts/v3_offline_completion_receipt_v2.json`, whose `split_estimand_map`
states the boundary explicitly. That receipt supersedes an earlier one which
mislabelled a validation anchor as a test quantity; it records the superseded
file's hash and changes no formal value.

## Note on paths

Absolute filesystem paths in three protocol files were replaced with
placeholders (`python`, `<REPO_ROOT>`, `<HOME>`). No other content was altered,
and none of those files is referenced in any receipt's `source_hashes`.
