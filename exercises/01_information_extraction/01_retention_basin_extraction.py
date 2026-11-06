"""Extract structured water-retention-basin records from a hydrology PDF.

Same workflow as 01_retention_basin_extraction.ipynb (one PDF, Docling text path).
Run from the repo root:

    uv run python exercises/01_information_extraction/01_retention_basin_extraction.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from dllmforge.IE_agent_extractor_docling import DoclingDocumentProcessor, DoclingInfoExtractor
from dllmforge.langchain_api import LangchainAPI

EXERCISE_DIR = Path(__file__).resolve().parent
if str(EXERCISE_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISE_DIR))

from schema import WaterRetentionBasin, WaterRetentionBasinDocument  # noqa: E402

DEFAULT_PDF = EXERCISE_DIR / "data" / "pdfs" / "nhess-15-365-2015.pdf"
DEFAULT_OUTPUT_DIR = EXERCISE_DIR / "output"

SYSTEM_PROMPT = """Extract every water retention basin, polder, or multi-purpose reservoir
mentioned in the document into the provided schema.

Rules:
- One list entry per distinct site. Do not merge two dams or basins (e.g. keep Eupen and La Gileppe separate).
- Only extract values that are explicitly stated. Use null if a field is missing or uncertain.
- Convert all volumes to cubic metres (m³). 1 hm³ = 1e6 m³, 1 Mm³ = 1e6 m³.
- river_name is the local river; river_basin is the main basin (e.g. Meuse).
- Prefer markdown table values over rounded prose when they conflict.
- Expand table scientific notation to a number (25 × 10^6 m³ → 25000000).
- Copy source_quote from one table row or one sentence (not several blocks concatenated). Do not paraphrase.
- Do not set source_page; it is filled after extraction from the quote.

Return JSON only, matching the schema keys.
"""

GOLD_VALUES = [
    {
        "name": "Eupen dam",
        "country": "Belgium",
        "river_name": "Vesdre",
        "river_basin": "Meuse",
        "basin_control_type": "controlled",
    },
    {
        "name": "La Gileppe dam",
        "country": "Belgium",
        "river_name": "Vesdre",
        "river_basin": "Meuse",
        "basin_control_type": "controlled",
    },
]


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


def inspect_schema() -> None:
    print("\nWaterRetentionBasin fields:")
    for name, info in WaterRetentionBasin.model_fields.items():
        print(f"  - {name}: {info.annotation}")


def inspect_results(results: list[dict]) -> None:
    basins: list[dict] = []
    for result in results:
        print("Processor metadata:", result.get("metadata"))
        basins.extend(result.get("basins") or [])

    print(f"\nExtracted {len(basins)} basin(s)")
    for basin in basins:
        print("-" * 60)
        print(f"name:         {basin.get('name')}")
        print(f"country:      {basin.get('country')}")
        print(f"river_name:   {basin.get('river_name')}")
        print(f"river_basin:  {basin.get('river_basin')}")
        print(f"control:      {basin.get('basin_control_type')}")
        print(f"total m3:     {basin.get('storage_volume_total_m3')}")
        print(f"flood m3:     {basin.get('flood_water_storage_volume_m3')}")
        print(f"source_page:  {basin.get('source_page')}")
        print(f"quote_ok:     {basin.get('quote_verified')}")
        print(f"source_quote: {basin.get('source_quote')}")

    print("\nGold-value check:")
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
    # Step 1
    llm_api = init_azure_llm()
    print("Azure OpenAI ready (", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), ")")

    # Step 2
    inspect_schema()

    # Step 3
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_processor = DoclingDocumentProcessor(
        input_dir=str(EXERCISE_DIR / "data" / "pdfs"),
        file_pattern="*.pdf",
        output_dir=str(output_dir),
    )
    processed = doc_processor.process_document(pdf_path)
    if processed is None:
        raise SystemExit(f"Docling failed to process {pdf_path}")
    print(f"File: {pdf_path.name}")
    print(
        f"Pages: {processed.metadata.get('page_count')}, "
        f"characters: {processed.metadata.get('text_length')}, "
        f"tables: {processed.metadata.get('num_tables')}"
    )
    print("--- first 800 characters ---")
    print(processed.content[:800])
    table_at = processed.content.find("[TABLE:")
    if table_at >= 0:
        print("--- Docling table preview ---")
        print(processed.content[table_at:table_at + 600])

    # Step 4
    extractor = DoclingInfoExtractor(
        output_schema=WaterRetentionBasinDocument,
        llm_api=llm_api,
        system_prompt=SYSTEM_PROMPT,
        chunk_size=80000,
        chunk_overlap=10000,
        doc_processor=doc_processor,
    )
    print("Extractor ready")

    # Step 5
    results = extractor.process_document(processed)
    output_path = output_dir / f"{pdf_path.stem}_extracted.json"
    extractor.save_results(results, output_path)
    print(f"Saved {len(results)} result object(s) to {output_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    # Step 6
    inspect_results(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract water retention basins from one PDF with DLLMForge Docling + Azure GPT."
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
