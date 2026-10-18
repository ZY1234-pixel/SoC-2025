from pathlib import Path

from docflow.utils.result_layout import ResultRunLayout


def test_render_outputs_live_directly_in_run_directory(tmp_path) -> None:
    run = ResultRunLayout.create(tmp_path, run_id="run_test")
    sample = run.create_sample(Path("academic_paper_01.jpg"))

    assert sample.pdf_path == run.run_dir / "academic_paper_01.pdf"
    assert sample.txt_path == run.run_dir / "academic_paper_01.txt"
    assert sample.html_path == run.run_dir / "academic_paper_01.html"
    assert sample.rtf_path == run.run_dir / "academic_paper_01.rtf"
    assert sample.xlsx_path == run.run_dir / "academic_paper_01.xlsx"
    assert sample.docx_path == run.run_dir / "academic_paper_01.docx"
    assert sample.recognition_path == run.run_dir / "stages" / "academic_paper_01.recognition.json"
    assert sample.assets_dir == run.run_dir / "assets" / "academic_paper_01"
    assert sample.table_dir == run.run_dir / "assets" / "tables" / "academic_paper_01"
    assert not (run.run_dir / "academic_paper_01").exists()
