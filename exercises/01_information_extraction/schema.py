"""Pydantic schema for water retention basin extraction (training exercise 1).

Identity and engineering fields are extracted by the LLM. Document-level
processor metadata (source_file, file_type, text_length) is appended by
DLLMForge after validation — do not put those keys on this schema.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class WaterRetentionBasin(BaseModel):
    """One flood-retention basin, polder, or multi-purpose reservoir."""

    name: Optional[str] = Field(
        None,
        description="Official name of the basin, dam, or reservoir (e.g. 'Eupen dam').",
    )
    alternative_names: Optional[List[str]] = Field(
        None, description="Other names used in the document for the same site."
    )
    country: Optional[str] = Field(None, description="Country where the basin is located.")
    location: Optional[str] = Field(
        None,
        description="Town, valley, or site description (e.g. '3 km upstream of Eupen').",
    )
    river_name: Optional[str] = Field(
        None, description="River on which the basin sits (e.g. 'Vesdre')."
    )
    river_basin: Optional[str] = Field(
        None, description="Main river basin (e.g. 'Meuse')."
    )
    catchment: Optional[str] = Field(
        None, description="Subcatchment if stated (e.g. 'Vesdre catchment')."
    )

    flood_water_storage_volume_m3: Optional[float] = Field(
        None,
        description=(
            "Additional flood or free storage in cubic metres (m³). "
            "Convert hm³ or Mm³ to m³ (1 hm³ = 1e6 m³, 1 Mm³ = 1e6 m³). "
            "ONLY extract if explicitly associated with this basin."
        ),
    )
    storage_volume_total_m3: Optional[float] = Field(
        None,
        description=(
            "Total reported storage volume in cubic metres (m³). "
            "Convert hm³ or Mm³ to m³. ONLY extract if explicitly associated with this basin."
        ),
    )
    basin_control_type: Optional[str] = Field(
        None,
        description=(
            "'controlled' if inflow/outflow is regulated by gates or operating rules; "
            "'uncontrolled' if water enters over a fixed sill, spillway, or dam crest."
        ),
    )
    dam_height_m: Optional[float] = Field(
        None, description="Height of the dam or embankment in metres (m)."
    )
    sill_level_m: Optional[float] = Field(
        None,
        description=(
            "Elevation of the sill or spillway crest in metres (m). "
            "Uncontrolled basins; leave null for controlled basins unless stated."
        ),
    )
    operating_rules: Optional[str] = Field(
        None,
        description=(
            "Operating rules such as discharge or water-level conditions that trigger "
            "filling or release. Controlled basins; leave null if not stated."
        ),
    )
    activation_discharge_m3_s: Optional[float] = Field(
        None,
        description=(
            "River discharge threshold in m³/s at which the basin starts filling. "
            "Controlled basins; leave null if not stated."
        ),
    )
    activation_water_level_m: Optional[float] = Field(
        None,
        description=(
            "Water-level threshold in metres that triggers filling. "
            "Controlled basins; leave null if not stated."
        ),
    )
    inlet_cross_section_m2: Optional[float] = Field(
        None, description="Inlet cross-section in m². Null if not stated."
    )
    inlet_capacity_m3_s: Optional[float] = Field(
        None, description="Maximum inlet capacity in m³/s. Null if not stated."
    )

    source_page: Optional[int] = Field(
        None,
        description=(
            "PDF page number from the [PAGE N] marker in the extracted text. "
            "Do not guess. Use the marker index, not journal pagination."
        ),
    )
    source_quote: Optional[str] = Field(
        None,
        description=(
            "Verbatim sentence or short paragraph copied from the document "
            "(not paraphrased) that supports this basin record."
        ),
    )


class WaterRetentionBasinDocument(BaseModel):
    """All basins mentioned in one source document."""

    source_title: Optional[str] = Field(None, description="Title of the source paper.")
    basins: List[WaterRetentionBasin] = Field(
        default_factory=list,
        description=(
            "One entry per distinct basin, dam, or reservoir. "
            "Do not merge two sites (e.g. keep Eupen and La Gileppe separate)."
        ),
    )
