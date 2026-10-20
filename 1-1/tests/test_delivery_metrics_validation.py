from Code.scripts.validate_delivery_metrics import edit_counts, normalize_text, stratified_items


def test_edit_counts_separates_cjk_and_other_target_edits() -> None:
    total, cjk_edits, other_edits = edit_counts("中文A1", "中文B")

    assert total == 2
    assert cjk_edits == 0
    assert other_edits == 2


def test_metric_sampling_is_deterministic_and_spread_out() -> None:
    values = list(range(10))

    assert stratified_items(values, 4) == [0, 3, 6, 9]
    assert normalize_text(" A b\n1 ") == "Ab1"
