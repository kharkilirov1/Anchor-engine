from scripts.run_qwen_long_retention_compare import analyze_keywords


def test_analyze_keywords_penalizes_repetitive_loops() -> None:
    repeated = "FastAPI is good. " * 20
    grounded = "FastAPI uses async handlers, Pydantic models, and dependency injection."

    repeated_analysis = analyze_keywords(repeated, ["fastapi", "async"], ["django"])
    grounded_analysis = analyze_keywords(grounded, ["fastapi", "async"], ["django"])

    assert repeated_analysis["lexical_score"] >= grounded_analysis["lexical_score"]
    assert repeated_analysis["quality_score"] < grounded_analysis["quality_score"]
    assert repeated_analysis["degeneracy_penalty"] > grounded_analysis["degeneracy_penalty"]
