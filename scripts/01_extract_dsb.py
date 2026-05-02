#!/usr/bin/env python3
"""
extract_dsb.py

Textbook-faithful DSB-resolved SDD parser with correct complexity classification
based on Huang et al. (2015) / MCDS definitions:

    DSB   (DSB0)  : Exactly one pair of opposite-strand backbone breaks within
                    the DSB clustering window (10 bp by default).  No additional
                    backbone breaks nearby and the DSB is not within the window
                    of any other DSB in the same damage site.

    DSB+          : A single DSB that has one or more *extra* backbone breaks
                    (on either strand) within the clustering window, but whose
                    centroid is NOT within the clustering window of any other
                    DSB in the same site.

    DSB++ (DSB++) : Two or more DSBs whose centroids are within the clustering
                    window of each other (i.e. more than one DSB within 10 bp).
                    ALL such DSBs in the group are promoted to DSB++.

Key corrections over the previous version
------------------------------------------
1.  DSB++ was systematically missed.  When bipartite matching produces 2+ DSBs
    from one SDD site, the old code excluded each other's backbone breaks from
    the additional_count calculation, so both would be labelled DSB0.  The fix
    is a post-reconstruction pass: for every pair of DSBs in a site, if their
    centroids are ≤ dsb_distance apart both are marked DSB++.

2.  The DSB+ upper boundary of "additional_count ≤ 2" was arbitrary.  DSB+ is
    now any DSB that (a) is not DSB++ and (b) has ≥1 extra backbone break in
    its cluster window.

3.  Base damages are correctly excluded from complexity classification (they
    only contribute to n_base_damages_in_cluster for downstream use).

Author: (your name)
Date:   February 2026

USAGE
-----
  python 01_extract_dsb.py --particle helium --let 30.0 --run 05
  python 01_extract_dsb.py --particle carbon --let 40.9 --run 12
  python 01_extract_dsb.py --particle electron --let 0.2 --run 01 --quiet
  python 01_extract_dsb.py --sdd custom_file.sdd --particle proton --let 4.65 --run 03

The --let argument must match the LET string as it appears in the TOPAS
OutputPrefix (e.g. "30.0" for helium dSOBP, "4.65" for proton pSOBP).
Auto-detection constructs the expected SDD filename as:
  {particle}_{let}_21.0_{run:02d}_DNADamage_sdd.txt
and searches for it in the current working directory.  When multiple SDD
files are present, the script will find exactly the right one without
requiring manual isolation first.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import re
import json
import csv
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

# ============================================================================
# Constants
# ============================================================================

HEADER_END_MARKER = "***EndOfHeader***"
DEFAULT_NUCLEUS_RADIUS_UM = 4.65
BP_PER_NM = 1.0 / 0.334          # ≈ 2.99 bp per nm

# SDD Field-7 strand codes
STRAND_BACKBONE_5TO3 = 1          # 5'→3' backbone  (strand 1)
STRAND_BASES_5TO3    = 2          # 5'→3' bases
STRAND_BASES_3TO5    = 3          # 3'→5' bases
STRAND_BACKBONE_3TO5 = 4          # 3'→5' backbone  (strand 2)

BACKBONE_STRANDS = (STRAND_BACKBONE_5TO3, STRAND_BACKBONE_3TO5)
BASE_STRANDS     = (STRAND_BASES_5TO3,    STRAND_BASES_3TO5)

# Field-7 damage-type codes
DAMAGE_SUBTHRESHOLD = 0
DAMAGE_DIRECT       = 1
DAMAGE_INDIRECT     = 2
DAMAGE_MULTIPLE     = 3


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BreakEvent:
    """Single damage entry from Field 7 of an SDD data line."""
    strand:      int
    position:    float   # bp position (1-indexed, relative to site)
    damage_type: int

    @property
    def is_backbone(self) -> bool:
        return self.strand in BACKBONE_STRANDS

    @property
    def is_base(self) -> bool:
        return self.strand in BASE_STRANDS

    @property
    def is_actual_damage(self) -> bool:
        """True for any damage above the sub-threshold level."""
        return self.damage_type > DAMAGE_SUBTHRESHOLD

    @property
    def is_direct(self) -> bool:
        return self.damage_type == DAMAGE_DIRECT

    @property
    def is_indirect(self) -> bool:
        return self.damage_type == DAMAGE_INDIRECT


@dataclass
class DamageSite:
    """Parsed SDD data line (one clustered damage site)."""
    # Identification
    classification: int
    event_id:       int
    line_number:    int = 0
    raw_line:       str = ""

    # Spatial coordinates (µm) – Field 2
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

    # Chromosome info – Fields 3 & 4
    chromosome:          Optional[int]   = None
    chromatid:           Optional[int]   = None
    arm:                 Optional[int]   = None
    chromosome_position: Optional[float] = None

    # Cause summary – Field 5
    cause_code: Optional[int] = None
    n_direct:   Optional[int] = None
    n_indirect: Optional[int] = None

    # Damage summary – Field 6
    n_base_damage: Optional[int]  = None
    n_ssb_total:   Optional[int]  = None
    has_dsb_flag:  Optional[bool] = None

    # Detailed breaks – Field 7
    break_events: List[BreakEvent] = field(default_factory=list)

    # Reconstructed DSBs (populated after reconstruction)
    reconstructed_dsbs: List["ReconstructedDSB"] = field(default_factory=list)


@dataclass
class ReconstructedDSB:
    """A single DSB reconstructed from a matched backbone-break pair."""
    # Identification
    global_id:          int = -1
    site_index:         int = -1
    dsb_index_in_site:  int = 0

    # Break positions (bp, relative to site)
    bp_pos_strand1: float = 0.0   # position on 5'→3' backbone
    bp_pos_strand2: float = 0.0   # position on 3'→5' backbone
    centroid_bp:    float = 0.0

    # Spatial coordinates (µm, inherited from site)
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    z_um: Optional[float] = None

    # Complexity metrics
    n_additional_backbone_breaks: int = 0   # unmatched extra backbone breaks
    n_base_damages_in_cluster:    int = 0
    complexity: str = "DSB"                 # "DSB", "DSB+", or "DSB++"

    # Direct / indirect fractions (over all backbone breaks in the cluster)
    direct_fraction:   float = 0.0
    indirect_fraction: float = 0.0

    # Chromosome info (inherited from site)
    chromosome:          Optional[int]   = None
    chromosome_position: Optional[float] = None

    # Event info
    event_id:   int            = 0
    cause_code: Optional[int]  = None


# ============================================================================
# Hopcroft-Karp Maximum Bipartite Matching
# ============================================================================

class HopcroftKarp:
    """
    Maximum bipartite matching via the Hopcroft-Karp algorithm.
    Time complexity: O(E · √V).

    Used to find the largest set of non-overlapping DSB pairs from backbone
    breaks on opposing strands within the clustering window.
    """

    def __init__(self,
                 left_nodes:  List[int],
                 right_nodes: List[int],
                 edges:       List[Tuple[int, int]]) -> None:
        self.left  = set(left_nodes)
        self.right = set(right_nodes)

        self.adj: Dict[int, List[int]] = {u: [] for u in self.left}
        for u, v in edges:
            if u in self.left and v in self.right:
                self.adj[u].append(v)

        self.pair_u: Dict[int, Optional[int]] = {u: None for u in self.left}
        self.pair_v: Dict[int, Optional[int]] = {v: None for v in self.right}
        self.dist:   Dict[int, float]         = {}

    def _bfs(self) -> bool:
        from collections import deque
        queue: deque[int] = deque()
        for u in self.left:
            if self.pair_u[u] is None:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = float("inf")
        found = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                pv = self.pair_v[v]
                if pv is None:
                    found = True
                elif self.dist.get(pv, float("inf")) == float("inf"):
                    self.dist[pv] = self.dist[u] + 1
                    queue.append(pv)
        return found

    def _dfs(self, u: int) -> bool:
        for v in self.adj[u]:
            pv = self.pair_v[v]
            if pv is None or (
                self.dist.get(pv, float("inf")) == self.dist[u] + 1
                and self._dfs(pv)
            ):
                self.pair_u[u] = v
                self.pair_v[v] = u
                return True
        self.dist[u] = float("inf")
        return False

    def maximum_matching(self) -> List[Tuple[int, int]]:
        while self._bfs():
            for u in self.left:
                if self.pair_u[u] is None:
                    self._dfs(u)
        return [(u, v) for u, v in self.pair_u.items() if v is not None]


# ============================================================================
# SDD File Loading & Header Parsing
# ============================================================================

def load_sdd_file(path: Path,
                  verbose: bool = False) -> Tuple[List[str], List[str]]:
    """Split an SDD file into its header lines and data lines."""
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    header_lines: List[str] = []
    data_lines:   List[str] = []
    header_done = False

    for ln in lines:
        if not header_done:
            header_lines.append(ln)
            if HEADER_END_MARKER in ln:
                header_done = True
        else:
            if ln.strip():
                data_lines.append(ln)

    if verbose:
        print(f"[INFO] Loaded {len(header_lines)} header lines, "
              f"{len(data_lines)} data lines")

    return header_lines, data_lines


def parse_header(lines: List[str],
                 verbose: bool = False) -> Dict[str, Any]:
    """
    Parse SDD header fields relevant to DSB reconstruction:
      - nucleus radius (from Volumes)
      - data entries mask (Field 25)
      - damage definition distances (Field 22)
      - chromosome sizes (Field 15)
    """
    info: Dict[str, Any] = {
        "nucleus_radius":     None,
        "data_entries_mask":  None,
        "damage_definition":  None,
        "chromosome_sizes":   None,
        "dsb_distance_bp":    10.0,   # default per SDD spec
        "base_grouping_bp":   10.0,
    }

    for line in lines:
        if not line.strip():
            continue

        parts = line.split(",", 1)
        key   = parts[0].strip().lower()
        value = parts[1].strip().rstrip(";") if len(parts) > 1 else ""

        # ---- Volumes → nucleus radius ----
        if key == "volumes":
            floats = []
            for tok in re.split(r"[,/]", value):
                try:
                    floats.append(float(tok.strip()))
                except ValueError:
                    pass
            candidates = [f for f in floats if 0.5 < f < 100]
            if candidates:
                info["nucleus_radius"] = max(set(candidates),
                                             key=candidates.count)

        # ---- Data entries mask (Field 25) ----
        elif key.startswith("data entries"):
            tokens = [t.strip() for t in value.split(",") if t.strip()]
            mask: List[bool] = []
            for t in tokens:
                if   t.lower() in ("true",  "t", "yes", "1"):
                    mask.append(True)
                elif t.lower() in ("false", "f", "no",  "0"):
                    mask.append(False)
                else:
                    try:
                        mask.append(bool(int(t)))
                    except ValueError:
                        mask.append(False)
            while len(mask) < 14:
                mask.append(False)
            info["data_entries_mask"] = mask[:14]

        # ---- Damage definition (Field 22) ----
        elif key.startswith("damage definition"):
            tokens = [t.strip() for t in value.split(",") if t.strip()]
            try:
                dd = {
                    "direct_indirect_flag": int(tokens[0])   if len(tokens) > 0 else 0,
                    "units_flag":           int(tokens[1])   if len(tokens) > 1 else 0,
                    "dsb_distance":         float(tokens[2]) if len(tokens) > 2 else 10.0,
                    "base_grouping":        float(tokens[3]) if len(tokens) > 3 else 0.0,
                }
                info["damage_definition"] = dd
                if dd["units_flag"] == 0:       # units in bp
                    info["dsb_distance_bp"]  = dd["dsb_distance"]
                    info["base_grouping_bp"] = dd["base_grouping"]
                else:                           # units in nm → convert to bp
                    info["dsb_distance_bp"]  = dd["dsb_distance"]  * BP_PER_NM
                    info["base_grouping_bp"] = dd["base_grouping"] * BP_PER_NM
            except (ValueError, IndexError):
                pass

        # ---- Chromosome sizes (Field 15) ----
        elif key.startswith("chromosome sizes"):
            tokens = [t.strip() for t in value.split(",") if t.strip()]
            try:
                n = int(tokens[0])
                sizes = [float(t) for t in tokens[1: n + 1]]
                info["chromosome_sizes"] = {"n": n, "sizes_mbp": sizes}
            except (ValueError, IndexError):
                pass

    if verbose:
        print(f"[INFO] Header: nucleus_radius={info['nucleus_radius']} µm, "
              f"dsb_distance={info['dsb_distance_bp']:.1f} bp, "
              f"base_grouping={info['base_grouping_bp']:.1f} bp")

    return info


# ============================================================================
# Data-Line Parsing
# ============================================================================

def _field_segment_index(field_number: int,
                          mask: Optional[List[bool]]) -> Optional[int]:
    """
    Return the segment index for a field given the Data Entries mask.
    Returns None if the field is not present in the file.
    """
    if mask is None or not (1 <= field_number <= len(mask)):
        return None
    if not mask[field_number - 1]:
        return None
    return int(sum(mask[: field_number - 1]))


def _parse_field7(segment: str) -> List[BreakEvent]:
    """Parse a Field-7 segment into a list of BreakEvent objects."""
    events: List[BreakEvent] = []
    for triplet in segment.split("/"):
        parts = [p.strip() for p in triplet.split(",") if p.strip()]
        if len(parts) >= 3:
            try:
                events.append(BreakEvent(
                    strand      = int(parts[0]),
                    position    = float(parts[1]),
                    damage_type = int(parts[2]),
                ))
            except ValueError:
                continue
    return events


def parse_data_line(line:        str,
                    line_number: int,
                    header_info: Dict[str, Any]) -> Optional[DamageSite]:
    """Parse a single SDD data line into a DamageSite object."""
    raw = line.strip()
    if not raw or raw.startswith(";"):
        return None

    segments = [s.strip() for s in raw.split(";") if s.strip()]
    if not segments:
        return None

    mask = header_info.get("data_entries_mask")

    # Field 1: classification flag + optional event ID
    m = re.match(r"\s*(\d+)\s*(?:,\s*(\d+))?", segments[0])
    if not m:
        return None

    site = DamageSite(
        classification = int(m.group(1)),
        event_id       = int(m.group(2)) if m.group(2) else 0,
        line_number    = line_number,
        raw_line       = raw,
    )

    # Field 2: spatial coordinates
    idx = _field_segment_index(2, mask)
    if idx is not None and idx < len(segments):
        triplets = segments[idx].split("/")
        if triplets:
            coords = [p.strip() for p in triplets[0].split(",") if p.strip()]
            if len(coords) >= 3:
                try:
                    site.x = float(coords[0])
                    site.y = float(coords[1])
                    site.z = float(coords[2])
                except ValueError:
                    pass

    # Field 3: chromosome IDs
    idx = _field_segment_index(3, mask)
    if idx is not None and idx < len(segments):
        parts = [p.strip() for p in segments[idx].split(",") if p.strip()]
        try:
            if len(parts) >= 2:
                site.chromosome = int(parts[1])
            if len(parts) >= 3:
                site.chromatid  = int(parts[2])
            if len(parts) >= 4:
                site.arm        = int(parts[3])
        except ValueError:
            pass

    # Field 4: chromosome position
    idx = _field_segment_index(4, mask)
    if idx is not None and idx < len(segments):
        try:
            site.chromosome_position = float(segments[idx].strip())
        except ValueError:
            pass

    # Field 5: cause
    idx = _field_segment_index(5, mask)
    if idx is not None and idx < len(segments):
        parts = [p.strip() for p in segments[idx].split(",") if p.strip()]
        try:
            if len(parts) >= 1:
                site.cause_code = int(parts[0])
            if len(parts) >= 2:
                site.n_direct   = int(parts[1])
            if len(parts) >= 3:
                site.n_indirect = int(parts[2])
        except ValueError:
            pass

    # Field 6: damage summary
    idx = _field_segment_index(6, mask)
    if idx is not None and idx < len(segments):
        parts = [p.strip() for p in segments[idx].split(",") if p.strip()]
        try:
            if len(parts) >= 1:
                site.n_base_damage = int(parts[0])
            if len(parts) >= 2:
                site.n_ssb_total   = int(parts[1])
            if len(parts) >= 3:
                site.has_dsb_flag  = (int(parts[2]) == 1)
        except ValueError:
            pass

    # Field 7: full break specification
    idx = _field_segment_index(7, mask)
    if idx is not None and idx < len(segments):
        site.break_events = _parse_field7(segments[idx])

    return site


# ============================================================================
# DSB Reconstruction
# ============================================================================

def _reconstruct_raw_dsbs(site:        DamageSite,
                           site_index:  int,
                           dsb_distance: float,
                           base_grouping: float) -> List[ReconstructedDSB]:
    """
    Step 1 – Bipartite matching.

    Pair backbone breaks on strand 1 (5'→3') with backbone breaks on strand 4
    (3'→5') that lie within `dsb_distance` bp.  Maximum matching maximises the
    number of DSBs extracted from the site.

    After pairing, count:
      - additional backbone breaks (not belonging to ANY matched pair) within
        `dsb_distance` of this DSB's centroid  →  used for DSB+ classification
      - base damages within `base_grouping` of the centroid  →  informational

    Returns a list of ReconstructedDSB objects with preliminary complexity
    labels.  DSB++ promotion is done in a separate post-processing pass so that
    pairs of close DSBs can be identified correctly.
    """
    # Separate actual backbone and base damage events
    s1_breaks:    List[Tuple[int, BreakEvent]] = []   # strand 1 backbone
    s2_breaks:    List[Tuple[int, BreakEvent]] = []   # strand 4 backbone
    base_damages: List[Tuple[int, BreakEvent]] = []

    for i, ev in enumerate(site.break_events):
        if not ev.is_actual_damage:
            continue
        if ev.strand == STRAND_BACKBONE_5TO3:
            s1_breaks.append((i, ev))
        elif ev.strand == STRAND_BACKBONE_3TO5:
            s2_breaks.append((i, ev))
        elif ev.is_base:
            base_damages.append((i, ev))

    if not s1_breaks or not s2_breaks:
        return []

    # Build bipartite graph edges where |pos1 − pos2| ≤ dsb_distance
    left_nodes  = [i for i, _ in s1_breaks]
    right_nodes = [i for i, _ in s2_breaks]
    edges: List[Tuple[int, int]] = []
    for i1, ev1 in s1_breaks:
        for i2, ev2 in s2_breaks:
            if abs(ev1.position - ev2.position) <= dsb_distance:
                edges.append((i1, i2))

    if not edges:
        return []

    matcher  = HopcroftKarp(left_nodes, right_nodes, edges)
    matching = matcher.maximum_matching()
    if not matching:
        return []

    # Index all break events for fast lookup
    break_lookup: Dict[int, BreakEvent] = {
        i: ev for i, ev in enumerate(site.break_events)
    }

    # Set of all backbone indices that are part of a matched DSB pair
    matched_indices: set[int] = set()
    for i1, i2 in matching:
        matched_indices.add(i1)
        matched_indices.add(i2)

    # Build DSB objects
    dsbs: List[ReconstructedDSB] = []

    for dsb_idx, (idx_s1, idx_s2) in enumerate(sorted(matching)):
        ev1      = break_lookup[idx_s1]
        ev2      = break_lookup[idx_s2]
        centroid = (ev1.position + ev2.position) / 2.0

        # ---- Additional backbone breaks (not in ANY matched pair) ----
        # These are unmatched breaks near this DSB that contribute to DSB+.
        additional_count = 0
        cluster_backbone_evs = [ev1, ev2]

        for i, ev in s1_breaks + s2_breaks:
            if i in matched_indices:
                continue          # part of a matched pair; handled as a DSB
            if abs(ev.position - centroid) <= dsb_distance:
                additional_count += 1
                cluster_backbone_evs.append(ev)

        # ---- Base damages within base_grouping of centroid ----
        effective_base_dist = base_grouping if base_grouping > 0 else dsb_distance
        base_count = sum(
            1 for _, ev in base_damages
            if abs(ev.position - centroid) <= effective_base_dist
        )

        # ---- Direct / indirect fractions over cluster backbone breaks ----
        n_direct   = sum(1 for ev in cluster_backbone_evs if ev.is_direct)
        n_indirect = sum(1 for ev in cluster_backbone_evs if ev.is_indirect)
        n_total    = len(cluster_backbone_evs)
        direct_frac   = n_direct   / n_total if n_total > 0 else 0.0
        indirect_frac = n_indirect / n_total if n_total > 0 else 0.0

        # ---- Preliminary complexity (DSB++ promotion happens later) ----
        if additional_count == 0:
            complexity = "DSB"
        else:
            complexity = "DSB+"

        dsbs.append(ReconstructedDSB(
            site_index             = site_index,
            dsb_index_in_site      = dsb_idx,
            bp_pos_strand1         = ev1.position,
            bp_pos_strand2         = ev2.position,
            centroid_bp            = centroid,
            x_um                   = site.x,
            y_um                   = site.y,
            z_um                   = site.z,
            n_additional_backbone_breaks = additional_count,
            n_base_damages_in_cluster    = base_count,
            complexity             = complexity,
            direct_fraction        = round(direct_frac,   4),
            indirect_fraction      = round(indirect_frac, 4),
            chromosome             = site.chromosome,
            chromosome_position    = site.chromosome_position,
            event_id               = site.event_id,
            cause_code             = site.cause_code,
        ))

    return dsbs


def _promote_dsb_plusplus(dsbs:         List[ReconstructedDSB],
                           dsb_distance: float) -> None:
    """
    Step 2 – DSB++ promotion (in-place).

    Per Huang et al. (2015) / MCDS: a DSB++ is defined as *more than one DSB
    within 10 base pairs*.  After bipartite matching produces individual DSB
    objects, we check every pair within the same damage site.  If two DSBs have
    centroids separated by ≤ dsb_distance bp they both become DSB++.

    NOTE: centroids are used as a practical proxy for "separation between DSBs".
    Using the minimum distance between the two DSBs' constituent backbone breaks
    would be marginally more precise but requires storing break positions per
    DSB, which centroid-based comparison already approximates well within the
    ~10 bp window.
    """
    n = len(dsbs)
    if n < 2:
        return

    promote: set[int] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if abs(dsbs[i].centroid_bp - dsbs[j].centroid_bp) <= dsb_distance:
                promote.add(i)
                promote.add(j)

    for idx in promote:
        dsbs[idx].complexity = "DSB++"


def reconstruct_dsbs_for_site(site:        DamageSite,
                               site_index:  int,
                               header_info: Dict[str, Any]) -> List[ReconstructedDSB]:
    """
    Full DSB reconstruction pipeline for a single SDD damage site:
      1. Bipartite matching  →  raw DSB objects with DSB / DSB+ labels
      2. DSB++ promotion     →  pairs of close DSBs relabelled in-place
    """
    dsb_dist     = header_info.get("dsb_distance_bp",  10.0)
    base_grouping = header_info.get("base_grouping_bp", 10.0)

    dsbs = _reconstruct_raw_dsbs(site, site_index, dsb_dist, base_grouping)
    _promote_dsb_plusplus(dsbs, dsb_dist)

    return dsbs


# ============================================================================
# Main Extraction Pipeline
# ============================================================================

def extract_dsbs(sdd_path: Path,
                 verbose:  bool = False,
                 no_scale: bool = False
                 ) -> Tuple[List[ReconstructedDSB],
                            List[DamageSite],
                            Dict[str, Any],
                            bool]:
    """
    Full pipeline: load → parse header → parse data lines →
                   (optionally scale coordinates) → reconstruct DSBs.

    Returns
    -------
    dsbs        : flat list of all reconstructed DSBs
    sites       : list of all parsed DamageSite objects
    header_info : dict of parsed header values
    scaled      : True if coordinates were rescaled from normalised units
    """
    header_lines, data_lines = load_sdd_file(sdd_path, verbose=verbose)
    header_info = parse_header(header_lines, verbose=verbose)

    # Parse all data lines
    sites: List[DamageSite] = []
    for i, line in enumerate(data_lines):
        site = parse_data_line(line, i, header_info)
        if site is not None:
            sites.append(site)

    if verbose:
        print(f"[INFO] Parsed {len(sites)} damage sites")

    # Optionally rescale normalised coordinates (max |coord| < 0.6 → normalised)
    scaled = False
    if not no_scale and sites:
        coords = [
            (s.x, s.y, s.z) for s in sites
            if s.x is not None and s.y is not None and s.z is not None
        ]
        if coords:
            max_abs = max(max(abs(c) for c in xyz) for xyz in coords)
            radius  = header_info.get("nucleus_radius") or DEFAULT_NUCLEUS_RADIUS_UM
            if max_abs < 0.6:
                if verbose:
                    print(f"[INFO] Rescaling normalised coordinates × {radius} µm")
                for s in sites:
                    if s.x is not None:
                        s.x *= radius
                        s.y *= radius
                        s.z *= radius
                scaled = True

    # Reconstruct DSBs site by site
    all_dsbs: List[ReconstructedDSB] = []
    for site_idx, site in enumerate(sites):
        dsbs = reconstruct_dsbs_for_site(site, site_idx, header_info)
        site.reconstructed_dsbs = dsbs
        all_dsbs.extend(dsbs)

    # Assign global sequential IDs
    for gid, dsb in enumerate(all_dsbs):
        dsb.global_id = gid

    if verbose:
        n_with_dsb = sum(1 for s in sites if s.reconstructed_dsbs)
        print(f"[INFO] Reconstructed {len(all_dsbs)} DSBs from {n_with_dsb} sites")
        counts = Counter(d.complexity for d in all_dsbs)
        for label in ("DSB", "DSB+", "DSB++"):
            n   = counts.get(label, 0)
            pct = n / len(all_dsbs) * 100 if all_dsbs else 0.0
            print(f"       {label:<6}: {n:6d}  ({pct:.1f}%)")

    return all_dsbs, sites, header_info, scaled


# ============================================================================
# Output Functions
# ============================================================================

def write_dsb_csv(dsbs:    List[ReconstructedDSB],
                  outpath: Path,
                  verbose: bool = False) -> None:
    """Write reconstructed DSBs to a CSV file."""
    CAUSE_MAP = {0: "DIRECT", 1: "INDIRECT", 2: "MIXED", 3: "CHARGE_MIGRATION"}

    fieldnames = [
        "dsb_id", "site_index", "event_id",
        "x_um", "y_um", "z_um",
        "complexity",
        "n_additional_backbone", "n_base_damage_in_cluster",
        "has_base_damage_in_cluster",
        "cause",
        "direct_fraction", "indirect_fraction",
        "chromosome", "chromosome_position",
        "centroid_bp",
    ]

    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for dsb in dsbs:
            cause_name = (CAUSE_MAP.get(dsb.cause_code, "UNKNOWN")
                          if dsb.cause_code is not None else "UNKNOWN")
            writer.writerow({
                "dsb_id":                    dsb.global_id,
                "site_index":                dsb.site_index,
                "event_id":                  dsb.event_id,
                "x_um":                      dsb.x_um if dsb.x_um is not None else "",
                "y_um":                      dsb.y_um if dsb.y_um is not None else "",
                "z_um":                      dsb.z_um if dsb.z_um is not None else "",
                "complexity":                dsb.complexity,
                "n_additional_backbone":     dsb.n_additional_backbone_breaks,
                "n_base_damage_in_cluster":  dsb.n_base_damages_in_cluster,
                "has_base_damage_in_cluster": 1 if dsb.n_base_damages_in_cluster > 0 else 0,
                "cause":                     cause_name,
                "direct_fraction":           dsb.direct_fraction,
                "indirect_fraction":         dsb.indirect_fraction,
                "chromosome":                dsb.chromosome if dsb.chromosome is not None else -1,
                "chromosome_position":       (dsb.chromosome_position
                                              if dsb.chromosome_position is not None else ""),
                "centroid_bp":               dsb.centroid_bp,
            })

    if verbose:
        print(f"[DONE] Saved {len(dsbs)} DSBs → {outpath}")


def write_summary_json(dsbs:        List[ReconstructedDSB],
                       sites:       List[DamageSite],
                       header_info: Dict[str, Any],
                       outpath:     Path,
                       verbose:     bool = False) -> None:
    """Write summary statistics to a JSON file."""
    counts = Counter(d.complexity for d in dsbs)
    dsbs_per_site_hist = Counter(len(s.reconstructed_dsbs) for s in sites)

    summary = {
        "header_info": {
            "nucleus_radius_um": header_info.get("nucleus_radius"),
            "dsb_distance_bp":   header_info.get("dsb_distance_bp"),
            "base_grouping_bp":  header_info.get("base_grouping_bp"),
        },
        "totals": {
            "n_damage_sites":       len(sites),
            "n_sites_with_dsb":     sum(1 for s in sites if s.reconstructed_dsbs),
            "n_reconstructed_dsbs": len(dsbs),
        },
        "complexity_distribution": {
            "DSB":   counts.get("DSB",   0),
            "DSB+":  counts.get("DSB+",  0),
            "DSB++": counts.get("DSB++", 0),
        },
        "dsbs_per_site_histogram": {
            str(k): v for k, v in sorted(dsbs_per_site_hist.items())
        },
        "direct_indirect_summary": {
            "mean_direct_fraction":
                float(np.mean([d.direct_fraction   for d in dsbs])) if dsbs else 0.0,
            "mean_indirect_fraction":
                float(np.mean([d.indirect_fraction for d in dsbs])) if dsbs else 0.0,
        },
    }

    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print(f"[DONE] Saved summary → {outpath}")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Textbook-faithful DSB reconstruction from SDD files "
                    "(Huang et al. 2015 / MCDS definitions).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sdd", default=None,
        help="Input SDD file.  Auto-detected if omitted — uses --particle, "
             "--let, and --run to construct the expected filename "
             "'{particle}_{let}_21.0_{run:02d}_DNADamage_sdd.txt'.",
    )
    parser.add_argument(
        "--particle", required=True,
        help="Particle name (e.g. 'helium', 'proton', 'carbon', 'electron'). "
             "Combined with --let to form the full run prefix.",
    )
    parser.add_argument(
        "--let", required=True,
        help="LET value as it appears in the TOPAS OutputPrefix "
             "(e.g. '30.0', '4.65', '0.2'). "
             "Used in both the SDD input search and the output file names.",
    )
    parser.add_argument(
        "--run", type=int, required=True,
        help="Run number (e.g. 5 → zero-padded to '05').  "
             "Used in both the SDD input search and the output file names.",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Disable automatic rescaling of normalised coordinates.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress informational output.",
    )

    args    = parser.parse_args()
    verbose = not args.quiet

    # ---- Build the canonical run prefix ----
    # Matches the TOPAS OutputPrefix format: {particle}_{let}_21.0_{run:02d}
    # e.g. helium_30.0_21.0_05
    particle = args.particle.lower()
    let_str  = args.let          # keep as-is to preserve decimal (e.g. "30.0")
    run_no   = f"{args.run:02d}" # 1 → "01", 10 → "10", 50 → "50"
    run_prefix = f"{particle}_{let_str}_21.0_{run_no}"

    # ---- Locate SDD file ----
    if args.sdd:
        sdd_path = Path(args.sdd)
        if not sdd_path.exists():
            print(f"[ERROR] SDD file not found: {args.sdd}")
            sys.exit(1)
    else:
        # Primary target: the exact filename that TOPAS produces for this run.
        # Format: {particle}_{let}_21.0_{run:02d}_DNADamage_sdd.txt
        exact_name = f"{run_prefix}_DNADamage_sdd.txt"
        exact_path = Path(".") / exact_name

        if exact_path.exists():
            sdd_path = exact_path
            if verbose:
                print(f"[INFO] Located SDD by prefix: {sdd_path}")
        else:
            # Fallback: broad glob, but only as a diagnostic — report what
            # was found so the user knows what prefix to use with --sdd.
            candidates = (
                list(Path(".").glob("*DNADamage_sdd.txt"))
                + list(Path(".").glob("*.sdd"))
            )
            if len(candidates) == 0:
                print(
                    f"[ERROR] SDD file not found.\n"
                    f"  Expected: {exact_name}\n"
                    f"  No SDD files present in the current directory."
                )
                sys.exit(1)
            elif len(candidates) == 1:
                sdd_path = candidates[0]
                if verbose:
                    print(
                        f"[WARNING] Expected '{exact_name}' not found; "
                        f"falling back to the only SDD present: {sdd_path}\n"
                        f"  Verify this file corresponds to run {run_no}."
                    )
            else:
                print(
                    f"[ERROR] Expected SDD not found and multiple candidates present.\n"
                    f"  Expected : {exact_name}\n"
                    f"  Found    : {[str(c) for c in candidates]}\n"
                    f"  Use --sdd to specify the correct file explicitly."
                )
                sys.exit(1)

    # ---- Extract ----
    dsbs, sites, header_info, scaled = extract_dsbs(
        sdd_path, verbose=verbose, no_scale=args.no_scale
    )

    # ---- Write outputs ----
    # Output names match the directory structure expected by scripts 02–04:
    #   {particle}_{let}_21.0_{run:02d}_dsb_complexity.csv
    #   {particle}_{let}_21.0_{run:02d}_complexity_summary.json
    csv_path  = Path(f"{run_prefix}_dsb_complexity.csv")
    json_path = Path(f"{run_prefix}_complexity_summary.json")

    write_dsb_csv(dsbs, csv_path, verbose=verbose)
    write_summary_json(dsbs, sites, header_info, json_path, verbose=verbose)

    if verbose:
        print(f"\n[DONE] Outputs: {csv_path}, {json_path}")


if __name__ == "__main__":
    main()