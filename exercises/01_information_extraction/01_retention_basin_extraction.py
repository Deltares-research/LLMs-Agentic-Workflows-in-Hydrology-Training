"""Extract structured water-retention-basin records from a hydrology PDF.

Uses DLLMForge InfoExtractor and Azure OpenAI. Run from the repo root:

    uv run python exercises/01_information_extraction/01_retention_basin_extraction.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from dllmforge.IE_agent_document_processor import DocumentProcessor, ProcessedDocument
from dllmforge.IE_agent_extractor import InfoExtractor
from dllmforge.langchain_api import LangchainAPI

EXERCISE_DIR = Path(__file__).resolve().parent
if str(EXERCISE_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISE_DIR))

from schema import WaterRetentionBasinDocument  # noqa: E402

DEFAULT_PDF = EXERCISE_DIR / "data" / "pdfs" / "nhess-15-365-2015.pdf"
DEFAULT_OUTPUT_DIR = EXERCISE_DIR / "output"

SYSTEM_PROMPT = """Extract every water retention basin, polder, or multi-purpose reservoir
mentioned in the document into the provided schema.

Rules:
- One list entry per distinct site. Do not merge two dams or basins (e.g. keep Eupen and La Gileppe separate).
- Only extract values that are explicitly stated. Use null if a field is missing or uncertain.
- Convert all volumes to cubic metres (m³). 1 hm³ = 1e6 m³, 1 Mm³ = 1e6 m³.
- river_name is the local river; river_basin is the main basin (e.g. Meuse).
- basin_control_type is 'controlled' (gates / operating rules) or 'uncontrolled' (fixed sill / spillway).
- Set source_page from the [PAGE N] marker in the text only. Do not guess and do not use journal page numbers.
- Copy source_quote verbatim from the document (one sentence or a short paragraph). Do not paraphrase.
- Leave processor metadata to the caller; do not invent source_file or text_length.

Return JSON only, matching the schema keys.
"""

# Approximate figures from Bruwier et al. 2015 (NHESS) for a live sanity check.
GOLD_VALUES = [
    {
        "name": "Eupen dam",
        "country": "Belgium",
        "river_name": "Vesdre",
        "river_basin": "Meuse",
        "dam_height_m": 50.0,
        "storage_volume_total_m3": 25_000_000.0,
        "flood_water_storage_volume_m3": 3_000_000.0,
        "basin_control_type": "controlled",
    },
    {
        "name": "La Gileppe dam",
        "country": "Belgium",
        "river_name": "Vesdre",
        "river_basin": "Meuse",
        "dam_height_m": 50.0,
        "storage_volume_total_m3": 25_000_000.0,
        "flood_water_storage_volume_m3": 3_000_000.0,
        "basin_control_type": "controlled",
    },
]


def pdf_to_text_with_page_markers(path: Path) -> ProcessedDocument:
    """Load a PDF as text with [PAGE N] markers so the LLM can cite PDF pages."""
    import pymupdf as fitz

    path = Path(path)
    doc = fitz.open(str(path))
    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        parts.append(f"\n[PAGE {i}]\n{page.get_text()}")
    text = "".join(parts)
    page_count = len(doc)
    doc.close()
    return ProcessedDocument(
        content=text,
        content_type="text",
        metadata={
            "source_file": str(path),
            "file_type": path.suffix.lower(),
            "text_length": len(text),
            "page_count": page_count,
        },
    )


def init_azure_llm() -> LangchainAPI:
    load_dotenv()
    if not os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_BASE"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["AZURE_OPENAI_API_BASE"]
    missing = [
        name
        for name in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
        )
        if not os.getenv(name)
    ]
    if missing or not (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")):
        raise SystemExit(
            "Missing Azure OpenAI settings in .env: "
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_API_BASE), "
            "AZURE_OPENAI_DEPLOYMENT_NAME. Do not paste keys into the notebook or this script."
        )
    return LangchainAPI(model_provider="azure-openai", temperature=0.1)


def make_extractor(llm_api: LangchainAPI, output_dir: Path) -> InfoExtractor:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_processor = DocumentProcessor(
        input_dir=str(EXERCISE_DIR / "data" / "pdfs"),
        file_pattern="*.pdf",
        output_type="text",
        output_dir=str(output_dir),
    )
    return InfoExtractor(
        output_schema=WaterRetentionBasinDocument,
        llm_api=llm_api,
        system_prompt=SYSTEM_PROMPT,
        chunk_size=80000,
        chunk_overlap=10000,
        doc_processor=doc_processor,
        document_output_type="text",
    )


def compare_to_gold(results: list[dict]) -> None:
    basins: list[dict] = []
    for result in results:
        basins.extend(result.get("basins") or [])
    print("\nGold-value check (approximate; from Bruwier et al. 2015):")
    print(f"{'expected':<22} {'extracted name':<28} {'country':<12} {'river':<12} {'type':<12}")
    for gold in GOLD_VALUES:
        needle = gold["name"].split()[0].lower()
        match = next(
            (b for b in basins if needle in (b.get("name") or "").lower()),
            {},
        )
        print(
            f"{gold['name']:<22} "
            f"{str(match.get('name') or 'NOT FOUND'):<28} "
            f"{str(match.get('country') or ''):<12} "
            f"{str(match.get('river_name') or ''):<12} "
            f"{str(match.get('basin_control_type') or ''):<12}"
        )


def extract_pdf(pdf_path: Path, output_dir: Path) -> list[dict]:
    llm_api = init_azure_llm()
    extractor = make_extractor(llm_api, output_dir)
    processed = pdf_to_text_with_page_markers(pdf_path)
    print(
        f"Processed {pdf_path.name}: "
        f"{processed.metadata['page_count']} pages, "
        f"{processed.metadata['text_length']} characters"
    )
    results = extractor.process_document(processed)
    output_path = output_dir / f"{pdf_path.stem}_extracted.json"
    extractor.save_results(results, output_path)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str)[:4000])
    if len(json.dumps(results, default=str)) > 4000:
        print("... (truncated in stdout; full JSON is in the output file)")
    compare_to_gold(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract water retention basins from a PDF with DLLMForge + Azure GPT."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to extract (default: {DEFAULT_PDF.name})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extracted JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")
    extract_pdf(pdf_path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
