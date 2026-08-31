import json
from io import BytesIO
from pathlib import Path
from html import escape

import pandas as pd
import streamlit as st
import re
import json
from xml.etree import ElementTree as ET
from collections import defaultdict, Counter
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

NS = {"r": "http://developer.cognos.com/schemas/report/17.5/"}


def _local(tag):
    """Strip namespace from an ElementTree tag, e.g. '{ns}expression' -> 'expression'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_all(root, tagname):
    """Find all descendant elements with a given local tag name, namespace-agnostic."""
    return [el for el in root.iter() if _local(el.tag) == tagname]


def _find_child(el, tagname):
    for child in el:
        if _local(child.tag) == tagname:
            return child
    return None


def _text_of(el):
    return (el.text or "").strip() if el is not None else ""


_FULL_MODEL_PATH_RE = re.compile(r"^\[C\]\.\[([^\]]+)\]\.\[([^\]]+)\]\.\[([^\]]+)\]$")


def parse_model_expression(expr):
    """
    Parse a Cognos model-path expression like:
      [C].[Sample_data_module].[sheet1].[Revenue]
    into (module, table, column). Requires the ENTIRE expression to be a clean
    bracketed path (nothing else around it) — this distinguishes a plain field
    reference from a calculated/macro expression that merely CONTAINS bracket
    fragments (e.g. a dynamically-built path via string concatenation).
    Returns (None, None, None) if it doesn't match.
    """
    m = _FULL_MODEL_PATH_RE.match(expr.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def classify_cognos_expression(expr):
    """
    Cognos equivalent of classify_formula(). Categories chosen to mirror
    WebFOCUS categories where a real equivalent exists, plus new categories
    for patterns that have no WebFOCUS analog.
    """
    e = expr.strip()
    types = []

    has_prompt_call = re.search(r"\bprompt\s*\(", e, re.IGNORECASE) is not None
    is_macro_wrapped = e.startswith("#")  # Cognos macro syntax: # ... #
    builds_path_dynamically = has_prompt_call and re.search(r"[\"'].*\+.*prompt", e, re.IGNORECASE)

    if builds_path_dynamically:
        types.append("Dynamic Model Path (Prompt-Built)")
    elif has_prompt_call and is_macro_wrapped:
        types.append("Parameter-Driven Metric Swap")
    elif has_prompt_call:
        types.append("Parameter Reference")

    if re.search(r"\bParamDisplayValue\s*\(", e):
        types.append("Prompt Display Value")
    if re.search(r"\bif\s*\(.*\)\s*then\s*\(", e, re.IGNORECASE):
        types.append("Conditional IF")
    if not has_prompt_call and re.search(r"[-+*/]", e):
        types.append("Arithmetic")

    module, table, column = parse_model_expression(e)
    if module:
        types.append("Direct Model Reference")

    if not types:
        types.append("Other/Review")
    return " + ".join(types)


def extract_queries(root):
    """
    Extract every <query> block: its dataItems (fields/calcs), its detailFilters,
    and whether it targets the model directly.
    This is the WebFOCUS TABLE FILE + DEFINE + WHERE equivalent, all in one block,
    since Cognos groups them per-query rather than per-report.
    """
    queries = []
    for q in _iter_all(root, "query"):
        qname = q.get("name", "")
        data_items = []

        selection = _find_child(q, "selection")
        if selection is not None:
            for di in selection:
                tag = _local(di.tag)
                if tag not in ("dataItem", "dataItemListSummary"):
                    continue
                name = di.get("name", "")
                aggregate = di.get("aggregate", "")
                expr_el = _find_child(di, "expression")
                expr = _text_of(expr_el)
                module, table, column = parse_model_expression(expr)
                data_items.append({
                    "query": qname,
                    "field": name,
                    "aggregate": aggregate,
                    "expression": expr,
                    "module": module,
                    "table": table,
                    "column": column,
                    "is_calculated": module is None and expr != "",
                    "expression_type": classify_cognos_expression(expr) if expr else "",
                })

        filters = []
        for df in _iter_all(q, "detailFilter"):
            fexpr = _find_child(df, "filterExpression")
            filters.append({
                "query": qname,
                "use": df.get("use", "required"),
                "expression": _text_of(fexpr),
                "is_prompted": "?" in _text_of(fexpr),
            })

        queries.append({"name": qname, "data_items": data_items, "filters": filters})
    return queries


def extract_prompts(root):
    """
    Extract <selectValue parameter="..."> elements — the Cognos prompt/parameter
    equivalent of WebFOCUS &&PARAM. Captures default value and option list when present.
    """
    prompts = []
    for sv in _iter_all(root, "selectValue"):
        param = sv.get("parameter", "")
        if not param:
            continue
        options = [opt.get("useValue", "") for opt in _iter_all(sv, "selectOption")]
        default_el = None
        for d in _iter_all(sv, "defaultSimpleSelection"):
            default_el = d
            break
        prompts.append({
            "parameter": param,
            "ref_query": sv.get("refQuery", ""),
            "options": options,
            "default": _text_of(default_el),
            "auto_submit": sv.get("autoSubmit", "false"),
        })
    return prompts


def extract_drillthroughs(root, report_name):
    """
    Extract <reportDrill> / <drillTarget> definitions — the Cognos equivalent of
    WebFOCUS extract_drilldowns(), except fully structured: target report path
    AND explicit source->target parameter mapping are both declared, not guessed.
    """
    drills = []
    for rd in _iter_all(root, "reportDrill"):
        drill_name = rd.get("name", "")
        target = _find_child(rd, "drillTarget")
        if target is None:
            continue
        report_path_el = _find_child(target, "reportPath")
        target_path = report_path_el.get("path", "") if report_path_el is not None else ""
        target_report_name = ""
        m = re.search(r"report\[@name='([^']*)'\]", target_path)
        if m:
            target_report_name = m.group(1)

        param_links = []
        for link in _iter_all(target, "drillLink"):
            src_ctx = _find_child(link, "drillSourceContext")
            tgt_ctx = _find_child(link, "drillTargetContext")
            src_param = _find_child(src_ctx, "parameterContext") if src_ctx is not None else None
            tgt_param = _find_child(tgt_ctx, "parameterContext") if tgt_ctx is not None else None
            param_links.append({
                "source_parameter": src_param.get("parameter", "") if src_param is not None else "",
                "target_parameter": tgt_param.get("parameter", "") if tgt_param is not None else "",
            })

        drills.append({
            "source_report": report_name,
            "drill_name": drill_name,
            "target_report": target_report_name,
            "target_path": target_path,
            "parameter_links": param_links,
            "opens_new_window": target.get("showInNewWindow", "false"),
        })
    return drills


def extract_visualizations(root):
    """
    Extract <vizControl> elements — charts/maps/word clouds etc. No WebFOCUS
    equivalent; captures type, referenced data store, and slot->column mapping
    so the AI build plan can suggest a matching Power BI visual type.
    """
    vizzes = []
    for vc in _iter_all(root, "vizControl"):
        viz_name = vc.get("name", "")
        viz_type = vc.get("type", "")
        slots = []
        for ds in _iter_all(vc, "vcDataSet"):
            ref_store = ds.get("refDataStore", "")
            for slot in _iter_all(ds, "vcSlotData"):
                slot_id = slot.get("idSlot", "")
                cols = [c.get("refDsColumn", "") for c in _iter_all(slot, "vcSlotDsColumn")]
                slots.append({"data_store": ref_store, "slot": slot_id, "columns": cols})
        vizzes.append({"name": viz_name, "type": viz_type, "slots": slots})
    return vizzes


def extract_data_stores(root):
    """
    Extract <reportDataStore> elements — maps a visualization's data feed back
    to the query + specific columns it uses, with a role (indexed=category/BY,
    value=measure/SUM). This is the closest Cognos equivalent to WebFOCUS
    by_real/by_calc vs sum_real/sum_calc classification.
    """
    stores = []
    for store in _iter_all(root, "reportDataStore"):
        store_name = store.get("name", "")
        list_query = None
        for q in _iter_all(store, "dsV5ListQuery"):
            list_query = q.get("refQuery", "")
            break
        columns = []
        for item in _iter_all(store, "dsV5DataItem"):
            columns.append({
                "field": item.get("refDataItem", ""),
                "role": item.get("dsColumnType", ""),  # 'indexed' = category, 'value' = measure
            })
        stores.append({"name": store_name, "query": list_query, "columns": columns})
    return stores


def extract_custom_controls(root):
    """
    Extract <customControl> elements — JS-based widgets with no WebFOCUS or
    standard Power BI equivalent. Always flagged for manual review.
    """
    controls = []
    for cc in _iter_all(root, "customControl"):
        controls.append({
            "name": cc.get("name", ""),
            "description": cc.get("description", ""),
            "path": cc.get("path", ""),
        })
    return controls


def extract_model_path(root):
    """Extract the <modelPath> — the single most important field: which Data
    Module / Framework Manager package this report depends on."""
    el = _iter_all(root, "modelPath")
    if not el:
        return ""
    return _text_of(el[0])


def parse_cognos_report(xml_text):
    """
    Main entry point — mirrors parse_fex(xml_text) -> dict shape.
    """
    root = ET.fromstring(xml_text)

    report_name_el = _iter_all(root, "reportName")
    report_name = _text_of(report_name_el[0]) if report_name_el else ""

    queries = extract_queries(root)

    all_fields = []
    all_filters = []
    for q in queries:
        all_fields.extend(q["data_items"])
        all_filters.extend(q["filters"])

    result = {
        "report_name": report_name,
        "model_path": extract_model_path(root),
        "queries": queries,
        "fields": all_fields,
        "filters": all_filters,
        "prompts": extract_prompts(root),
        "drillthroughs": extract_drillthroughs(root, report_name),
        "visualizations": extract_visualizations(root),
        "data_stores": extract_data_stores(root),
        "custom_controls": extract_custom_controls(root),
    }

    # Derived: real vs calculated fields (WebFOCUS source_fields vs define_fields split)
    result["source_fields"] = [f for f in all_fields if f["module"] is not None]
    result["calculated_fields"] = [f for f in all_fields if f["module"] is None and f["expression"]]

    # Derived: which modules/tables this report actually touches (real_sources equivalent)
    tables_touched = set()
    for f in result["source_fields"]:
        tables_touched.add(f"{f['module']}.{f['table']}")
    result["source_tables"] = sorted(tables_touched)

    return result


# ============================================================================
# DATA MODULE PARSER
# Cognos Data Modules are delivered as base64(gzip(JSON)) inside a field
# called "smartsData" — this is the semantic-layer / model equivalent of
# WebFOCUS's MAS/ACX metadata, except it's the PRIMARY source of truth here,
# not an optional bolt-on. Multiple smartsModule entries can exist per
# Data Module — each one typically wraps one physical table/sheet, plus
# occasionally an index-only entry that references all tables together
# (useful for detecting declared cross-table relationships, when present).
# ============================================================================

import base64
import gzip


def decode_smarts_data(smarts_data_b64):
    """Decode one smartsData field: base64 -> gzip -> JSON dict."""
    compressed = base64.b64decode(smarts_data_b64)
    decompressed = gzip.decompress(compressed)
    return json.loads(decompressed)


def extract_table_schema(decoded_module):
    """
    Extract a clean {table_id: {column_id: {type, usage, aggregation}}} schema
    from one decoded smartsModule, stripping the NLP/semantic-search metadata
    (labels, tokens, entities, concepts) that Cognos Assistant uses but that
    is irrelevant to migration.
    """
    tables = {}
    for ds in decoded_module.get("datasets", []):
        table_id = ds.get("id", "")
        columns = {}
        for item in ds.get("item", []):
            col = item.get("column")
            if not col:
                continue
            col_id = col.get("id", "")
            columns[col_id] = {
                "datatype": col.get("dataType", {}).get("type", ""),
                "usage": col.get("usage", ""),  # IDENTIFIER, FACT, ATTRIBUTE
                "default_aggregation": col.get("defaultAggregation", ""),
            }
        if columns:  # only record tables that actually have columns (skip index-only entries)
            tables[table_id] = {
                "name": ds.get("name", ""),
                "columns": columns,
            }
    return tables


def find_relationship_candidates(decoded_module):
    """
    Recursively scans a decoded Data Module for any key that looks like it
    declares a relationship/join between tables (e.g. 'relationships',
    'relationship', 'joins', 'links'). Cognos doesn't use one fixed key name
    across all versions/module types, so this is a best-effort structural
    scan rather than a lookup on one known field.

    Returns a list of {"path": "dotted.path.to.key", "key": key_name, "value": raw_value}
    for every match found, regardless of shape - callers should inspect
    the raw value themselves, since the internal schema of a real relationship
    entry hasn't been confirmed against a live example yet.
    """
    RELATIONSHIP_KEY_PATTERNS = ("relationship", "join")
    matches = []

    def walk(obj, path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                if any(pat in key_lower for pat in RELATIONSHIP_KEY_PATTERNS):
                    matches.append({"path": f"{path}.{key}" if path else key, "key": key, "value": value})
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                walk(item, f"{path}[{idx}]")

    walk(decoded_module, "")
    return matches


def parse_cognos_data_module(smarts_module_entries):
    """
    Given a list of raw {'smartsData': b64, 'id': ...} entries (as returned by
    /bi/v1/objects/{id}/smarts_modules), decode all of them and merge into one
    combined schema: {module_name, tables: {table_id: {name, columns}}, relationships}.

    'relationships' is populated by find_relationship_candidates() across every
    decoded module. In the one real Data Module tested so far (4 standalone
    uploaded-Excel tables, no declared joins), this correctly comes back empty -
    that's the expected result for a flat, join-free model, not a parser failure.
    If you load a Data Module that DOES declare relationships, this list will
    contain the raw matched entries for manual inspection, since the exact
    field-level schema of a populated relationship entry hasn't been confirmed
    against a real example yet.
    """
    module_name = ""
    all_tables = {}
    all_relationships = []
    for entry in smarts_module_entries:
        decoded = decode_smarts_data(entry["smartsData"])
        module_name = decoded.get("name", module_name)
        tables = extract_table_schema(decoded)
        all_tables.update(tables)
        all_relationships.extend(find_relationship_candidates(decoded))
    return {"module_name": module_name, "tables": all_tables, "relationships": all_relationships}


def validate_report_against_model(parsed_report, data_module_schema):
    """
    Cross-checks every source field a report references against the real
    Data Module schema. Mirrors validate_plan_references() from KashMap's
    AI build-plan validator — same spirit, applied earlier in the pipeline.
    Returns a list of issues (empty list = fully verified).
    """
    issues = []
    tables = data_module_schema.get("tables", {})

    for field in parsed_report.get("source_fields", []):
        table_id = field["table"]
        col_id = field["column"]
        if table_id not in tables:
            issues.append(f"Field '{field['field']}' references unknown table '{table_id}'")
            continue
        if col_id not in tables[table_id]["columns"]:
            issues.append(
                f"Field '{field['field']}' references unknown column "
                f"'{col_id}' in table '{table_id}'"
            )

    return issues




# ============================================================================
# END OF COGNOS PARSER
# ============================================================================

from collections import Counter, defaultdict


def _upper_set_cognos(values):
    return frozenset(str(v).strip().upper() for v in values if str(v or '').strip())


def compute_cognos_profile(parsed):
    """Signature of a report's real structure, used for duplicate comparison.
    Mirrors compute_fex_profile() from KashMap."""
    model_path = (parsed.get('model_path') or '').strip().upper()
    source_tables = _upper_set_cognos(parsed.get('source_tables', []))
    all_fields = _upper_set_cognos(f['field'] for f in parsed.get('fields', []))
    filters = frozenset(
        (f['query'].upper(), f['use'].upper(), f['expression'].strip().upper())
        for f in parsed.get('filters', [])
    )
    prompts = _upper_set_cognos(p['parameter'] for p in parsed.get('prompts', []))
    drillthrough_targets = _upper_set_cognos(d['target_report'] for d in parsed.get('drillthroughs', []))

    if not model_path and not all_fields:
        return None

    return {
        'model_path': model_path,
        'source_tables': source_tables,
        'all_fields': all_fields,
        'filters': filters,
        'prompts': prompts,
        'drillthrough_targets': drillthrough_targets,
    }


def exact_duplicate_key_cognos(profile):
    if profile is None:
        return None
    return (
        profile['model_path'], profile['source_tables'], profile['all_fields'],
        profile['filters'], profile['prompts'], profile['drillthrough_targets'],
    )


def field_similarity_cognos(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def profile_difference_summary_cognos(base, other):
    diffs = []
    for label, key in [('fields', 'all_fields'), ('filters', 'filters'),
                        ('prompts', 'prompts'), ('drillthroughs', 'drillthrough_targets')]:
        if base[key] != other[key]:
            diffs.append(label)
    return ', '.join(diffs) if diffs else 'No meaningful differences'


def build_cognos_duplicate_analysis(parsed_by_report):
    """parsed_by_report: {report_name: (parsed, validation_issues)}
    Mirrors build_duplicate_analysis() from KashMap."""
    records = []
    exact_groups = defaultdict(list)
    table_groups = defaultdict(list)

    for report_name, (parsed, _issues) in parsed_by_report.items():
        profile = compute_cognos_profile(parsed)
        record = {
            'report_name': report_name, 'profile': profile,
            'category': 'Unparsed' if profile is None else 'Unique',
            'group_id': '', 'group_size': 1,
            'exact_matches': [], 'near_matches': [], 'same_source_matches': [],
            'best_similarity': 0.0, 'difference_summary': '',
        }
        records.append(record)
        if profile is not None:
            exact_groups[exact_duplicate_key_cognos(profile)].append(record)
            table_key = (profile['model_path'], profile['source_tables'])
            table_groups[table_key].append(record)

    group_counter = 1
    for members in exact_groups.values():
        if len(members) <= 1:
            continue
        group_id = f'Exact {group_counter}'
        group_counter += 1
        names = [m['report_name'] for m in members]
        for record in members:
            record['category'] = 'Exact Duplicate'
            record['group_id'] = group_id
            record['group_size'] = len(members)
            record['exact_matches'] = [n for n in names if n != record['report_name']]
            record['best_similarity'] = 1.0
            record['difference_summary'] = 'Exact same model, tables, fields, filters, prompts, and drillthroughs'

    for table_key, members in table_groups.items():
        if (not table_key[0] and not table_key[1]) or len(members) <= 1:
            continue
        for record in members:
            if record['category'] == 'Exact Duplicate':
                continue
            best_near, same_source, best_similarity, diff_notes = [], [], 0.0, []
            for other in members:
                if other is record:
                    continue
                sim = field_similarity_cognos(record['profile']['all_fields'], other['profile']['all_fields'])
                best_similarity = max(best_similarity, sim)
                if sim >= 0.80:
                    if (record['profile']['filters'] != other['profile']['filters']
                            or record['profile']['prompts'] != other['profile']['prompts']
                            or record['profile']['drillthrough_targets'] != other['profile']['drillthrough_targets']):
                        best_near.append(other['report_name'])
                        diff_notes.append(profile_difference_summary_cognos(record['profile'], other['profile']))
                    else:
                        same_source.append(other['report_name'])
                else:
                    same_source.append(other['report_name'])
            if best_near:
                record['category'] = 'Near Duplicate'
                record['near_matches'] = sorted(set(best_near))
                record['group_id'] = f'Near {abs(hash(table_key)) % 100000}'
                record['group_size'] = len(best_near) + 1
                record['difference_summary'] = '; '.join(sorted(set(diff_notes))) or '80%+ same fields with logic differences'
            elif same_source:
                record['category'] = 'Same Data Source'
                record['same_source_matches'] = sorted(set(same_source))
                record['group_id'] = f'Source {abs(hash(table_key)) % 100000}'
                record['group_size'] = len(same_source) + 1
                record['difference_summary'] = 'Same model/tables with different fields or report logic'
            record['best_similarity'] = max(record['best_similarity'], best_similarity)

    for record in records:
        if record['category'] == 'Unique':
            record['difference_summary'] = 'No meaningful match found'
        elif record['category'] == 'Unparsed':
            record['difference_summary'] = 'Could not parse enough model/fields'
    return records


def cognos_duplicate_counts(records):
    counts = Counter(r['category'] for r in records)
    return {
        'exact': counts.get('Exact Duplicate', 0), 'near': counts.get('Near Duplicate', 0),
        'same_source': counts.get('Same Data Source', 0), 'unique': counts.get('Unique', 0),
        'unparsed': counts.get('Unparsed', 0),
    }


def build_duplicate_df(records):
    rows = []
    for r in records:
        rows.append({
            'Report': r['report_name'], 'Duplicate Type': r['category'],
            'Group ID': r['group_id'], 'Group Size': r['group_size'],
            'Confidence': f"{r['best_similarity']:.0%}" if r['best_similarity'] else '',
            'Matched With': ', '.join(r['exact_matches'] + r['near_matches'] + r['same_source_matches']),
            'Difference Summary': r['difference_summary'],
        })
    return pd.DataFrame(rows)


CognosClassification = Literal["confirmed", "rule_based", "inferred", "suggested", "manual_review"]

CognosPowerBILayer = Literal[
    "Power Query", "Semantic Model", "DAX Measure", "Calculated Column",
    "Report Filter", "Slicer", "Visual", "Drill-through", "Manual Review",
]


class CognosPlanStep(BaseModel):
    step_number: int = Field(ge=1)
    title: str
    action: str
    power_bi_layer: CognosPowerBILayer
    classification: CognosClassification
    reason: str
    referenced_sources: list[str] = Field(default_factory=list)
    referenced_fields: list[str] = Field(default_factory=list)
    confidence: int = Field(ge=0, le=100)
    manual_validation: Optional[str] = None


class CognosReportSummary(BaseModel):
    report_name: str
    likely_purpose: str
    report_type: str
    complexity_level: Literal["Very Small", "Small", "Medium", "Large", "Very Complex"]
    overall_confidence: int = Field(ge=0, le=100)


class CognosBuildPlan(BaseModel):
    report_summary: CognosReportSummary
    implementation_plan: list[CognosPlanStep]
    calculation_plan: list[CognosPlanStep] = Field(default_factory=list)
    filter_prompt_plan: list[CognosPlanStep] = Field(default_factory=list)
    drillthrough_plan: list[CognosPlanStep] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)
    manual_review_items: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


COGNOS_AI_BUILD_PLAN_SYSTEM_PROMPT = """You are a senior BI migration engineer converting one parsed IBM Cognos report to Power BI.
The input contains Cognos-report-derived logic: queries, fields (source and calculated), detail filters,
prompts, drillthroughs, visualizations, and custom controls. Field references are already resolved to
[Module].[Table].[Column] paths where possible; unresolved_field_issues (if present) lists fields that
could NOT be verified against the real Data Module schema - treat those as higher-risk manual review.

Rules:
1. Use only tables, fields, filters, prompts, and drillthroughs present in the input. Never invent one.
2. Cognos has no HOLD/staging chain - all shaping happens at the query or model level. Do not describe
   staging steps that aren't in the input.
3. Label every implementation step as one of: confirmed, rule_based, inferred, suggested, manual_review.
4. Every step must include referenced_sources (table names) and referenced_fields drawn only from the input.
5. Custom controls (JS widgets) have no Power BI equivalent - always classify them as manual_review.
6. Confidence (0-100) must reflect how directly the input supports the recommendation.
7. Return ONLY strict JSON matching the required schema - no markdown fences, no commentary outside JSON.

Return JSON with exactly these keys:
{
  "report_summary": {"report_name": "", "likely_purpose": "", "report_type": "",
                      "complexity_level": "Very Small|Small|Medium|Large|Very Complex",
                      "overall_confidence": 0},
  "implementation_plan": [{"step_number": 1, "title": "", "action": "",
                            "power_bi_layer": "Power Query|Semantic Model|DAX Measure|Calculated Column|Report Filter|Slicer|Visual|Drill-through|Manual Review",
                            "classification": "confirmed|rule_based|inferred|suggested|manual_review",
                            "reason": "", "referenced_sources": [], "referenced_fields": [],
                            "confidence": 0, "manual_validation": null}],
  "calculation_plan": [... same PlanStep shape ..., must cover every calculated field in the input],
  "filter_prompt_plan": [... same PlanStep shape ..., must cover every filter and prompt],
  "drillthrough_plan": [... same PlanStep shape ..., must cover every drillthrough],
  "validation_checks": ["..."],
  "manual_review_items": ["..."],
  "limitations": ["..."]
}

Every custom control in the input must produce exactly one manual_review step in implementation_plan.
Every entry in unresolved_field_issues must appear in manual_review_items.
"""

_COGNOS_VALID_LAYERS = {
    "Power Query", "Semantic Model", "DAX Measure", "Calculated Column",
    "Report Filter", "Slicer", "Visual", "Drill-through", "Manual Review",
}
_COGNOS_LAYER_PRECEDENCE = [
    "Power Query", "Semantic Model", "DAX Measure", "Calculated Column",
    "Report Filter", "Slicer", "Visual", "Drill-through", "Manual Review",
]


def _coerce_cognos_layer(value):
    if value in _COGNOS_VALID_LAYERS:
        return value
    parts = re.split(r'[|/,]', str(value))
    parts = [p.strip() for p in parts if p.strip() in _COGNOS_VALID_LAYERS]
    if parts:
        for preferred in _COGNOS_LAYER_PRECEDENCE:
            if preferred in parts:
                return preferred
        return parts[0]
    return "Manual Review"


def normalize_cognos_layers(raw):
    for key in ['implementation_plan', 'calculation_plan', 'filter_prompt_plan', 'drillthrough_plan']:
        steps = raw.get(key)
        if not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict) and 'power_bi_layer' in step:
                step['power_bi_layer'] = _coerce_cognos_layer(step['power_bi_layer'])
    return raw


def parse_llm_json(text):
    text = (text or '').strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def build_cognos_ai_payload(report_name, parsed, validation_issues):
    """Compact payload for one AI call, mirroring build_ai_build_plan_payload()
    from KashMap but shaped around Cognos structures (queries/prompts/drillthroughs
    instead of HOLD/JOIN)."""
    shown_fields = sorted({f['field'] for f in parsed['fields']})
    return {
        'report_name': report_name,
        'model_path': parsed['model_path'],
        'source_tables': parsed['source_tables'],
        'fields': [
            {'field': f['field'], 'query': f['query'], 'is_calculated': f['is_calculated'],
             'expression_type': f['expression_type'], 'expression': f['expression'][:200]}
            for f in parsed['fields']
        ][:60],
        'filters': [
            {'query': f['query'], 'use': f['use'], 'expression': f['expression'][:200]}
            for f in parsed['filters']
        ][:30],
        'prompts': [
            {'parameter': p['parameter'], 'options': p['options'], 'default': p['default']}
            for p in parsed['prompts']
        ],
        'drillthroughs': [
            {'drill_name': d['drill_name'], 'target_report': d['target_report'],
             'parameter_links': d['parameter_links']}
            for d in parsed['drillthroughs']
        ],
        'visualizations': [{'name': v['name'], 'type': v['type']} for v in parsed['visualizations']],
        'custom_controls': [{'name': c['name'], 'path': c['path']} for c in parsed['custom_controls']],
        'unresolved_field_issues': validation_issues,
        'shown_fields': shown_fields[:60],
    }


def strip_unverified_cognos_steps(plan, allowed_sources, allowed_fields):
    """Mirrors strip_unverified_steps() from KashMap: downgrades any step
    referencing unknown sources/fields to manual_review instead of trusting it."""
    def clean(section):
        cleaned = []
        for step in section:
            unknown_sources = set(step.referenced_sources) - allowed_sources
            unknown_fields = set(step.referenced_fields) - allowed_fields
            if unknown_sources or unknown_fields:
                step.classification = "manual_review"
                parts = []
                if unknown_sources:
                    parts.append(f"unknown source(s) {sorted(unknown_sources)}")
                if unknown_fields:
                    parts.append(f"unknown field(s) {sorted(unknown_fields)}")
                step.manual_validation = "AI recommendation flagged: referenced " + " and ".join(parts) + " not present in the parsed report."
                step.confidence = min(step.confidence, 40)
            cleaned.append(step)
        return cleaned

    plan.implementation_plan = clean(plan.implementation_plan)
    plan.calculation_plan = clean(plan.calculation_plan)
    plan.filter_prompt_plan = clean(plan.filter_prompt_plan)
    plan.drillthrough_plan = clean(plan.drillthrough_plan)
    return plan


def generate_cognos_ai_build_plan(client, model, payload):
    """Single on-demand API call with automatic retry on truncation.
    Returns (CognosBuildPlan | None, error_str | None). Mirrors
    generate_ai_build_plan() from KashMap."""
    user_prompt = json.dumps(payload, ensure_ascii=False)
    output_budget = 3500
    max_budget_cap = 12000

    while True:
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": COGNOS_AI_BUILD_PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_output_tokens=output_budget,
            )
        except Exception as e:
            return None, f'LLM error: {e}'

        raw_text = response.output_text or ''
        was_truncated = getattr(response, 'status', None) == 'incomplete'
        raw = parse_llm_json(raw_text)

        if not raw and was_truncated and output_budget < max_budget_cap:
            output_budget = min(output_budget * 2, max_budget_cap)
            continue
        if not raw:
            reason = ' (response was truncated)' if was_truncated else ''
            return None, f'Model did not return parseable JSON{reason}. Raw (truncated): {raw_text[:500]}'

        raw = normalize_cognos_layers(raw)
        try:
            plan = CognosBuildPlan.model_validate(raw)
        except ValidationError as ve:
            return None, f'AI response failed schema validation: {ve}'

        allowed_sources = set(payload.get('source_tables', []))
        allowed_fields = set(payload.get('shown_fields', []))
        for f in payload.get('fields', []):
            allowed_fields.add(f['field'])

        plan = strip_unverified_cognos_steps(plan, allowed_sources, allowed_fields)
        return plan, None


def get_openai_client():
    if OpenAI is None:
        return None
    try:
        api_key = st.secrets.get('OPENAI_API_KEY', '')
    except FileNotFoundError:
        return None
    if not api_key or api_key == 'paste_your_openai_api_key_here':
        return None
    return OpenAI(api_key=api_key)


def _render_cognos_plan_step(step):
    level_map = {'confirmed': 'ok', 'rule_based': 'ok', 'inferred': 'medium', 'suggested': 'medium', 'manual_review': 'high'}
    level = level_map.get(step.classification, 'medium')
    with st.container(border=True):
        st.markdown(f"**{step.title}** — {step.power_bi_layer}")
        st.markdown(
            f"{badge(step.classification.replace('_', ' ').title(), level)} "
            f"{badge(f'Confidence: {step.confidence}%')}",
            unsafe_allow_html=True,
        )
        st.write(step.action)
        st.caption(f"Why: {step.reason}")
        if step.referenced_sources or step.referenced_fields:
            st.caption(f"References: {', '.join(step.referenced_sources + step.referenced_fields)}")
        if step.manual_validation:
            st.warning(step.manual_validation)


def render_cognos_ai_build_plan(report_name, parsed, validation_issues):
    client = get_openai_client()
    if client is None:
        st.warning("OpenAI is not configured. Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` to enable the AI build plan.")
        return

    report_key = re.sub(r'[^A-Za-z0-9_]+', '_', report_name)
    cache = st.session_state.setdefault('cognos_ai_build_plan_cache', {})

    clicked = st.button(
        "Generate AI build plan" if report_key not in cache else "Regenerate AI build plan",
        key=f"cognos_ai_btn_{report_key}",
    )
    if clicked:
        payload = build_cognos_ai_payload(report_name, parsed, validation_issues)
        with st.spinner("Calling OpenAI..."):
            plan, error = generate_cognos_ai_build_plan(client, "gpt-4.1-mini", payload)
        cache[report_key] = (plan, error)

    cached = cache.get(report_key)
    if cached is None:
        st.info("Click the button above to generate the AI-assisted build plan.")
        return
    plan, error = cached
    if error:
        st.error(error)
        return

    summary = plan.report_summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Complexity", summary.complexity_level)
    c2.metric("Overall confidence", f"{summary.overall_confidence}%")
    c3.metric("Manual review items", len(plan.manual_review_items))
    st.write(f"**Likely purpose:** {summary.likely_purpose}")
    st.write(f"**Report type:** {summary.report_type}")

    tabs = st.tabs(["Implementation", "Calculations", "Filters/Prompts", "Drillthroughs", "Validation", "Manual Review"])
    with tabs[0]:
        if not plan.implementation_plan:
            st.info("No implementation steps returned.")
        for step in sorted(plan.implementation_plan, key=lambda s: s.step_number):
            _render_cognos_plan_step(step)
    with tabs[1]:
        if not plan.calculation_plan:
            st.info("No calculation steps returned.")
        for step in plan.calculation_plan:
            _render_cognos_plan_step(step)
    with tabs[2]:
        if not plan.filter_prompt_plan:
            st.info("No filter/prompt steps returned.")
        for step in plan.filter_prompt_plan:
            _render_cognos_plan_step(step)
    with tabs[3]:
        if not plan.drillthrough_plan:
            st.info("No drillthrough steps returned.")
        for step in plan.drillthrough_plan:
            _render_cognos_plan_step(step)
    with tabs[4]:
        if not plan.validation_checks:
            st.info("No validation checks returned.")
        for i, check in enumerate(plan.validation_checks, 1):
            st.write(f"{i}. {check}")
    with tabs[5]:
        if not plan.manual_review_items:
            st.success("No manual review items flagged.")
        for i, item in enumerate(plan.manual_review_items, 1):
            st.warning(f"{i}. {item}")
        if plan.limitations:
            st.caption("Limitations:")
            for lim in plan.limitations:
                st.caption(f"• {lim}")


st.set_page_config(page_title="CognosMap Migration Workspace", layout="wide")

st.markdown(
    """
<style>
:root {
    --kash-bg: var(--background-color, #FFFFFF);
    --kash-section: var(--secondary-background-color, #FBFAF8);
    --kash-card: var(--background-color, #FFFFFF);
    --kash-navy: var(--text-color, #1F2933);
    --kash-deep: var(--text-color, #111827);
    --kash-blue: #2563EB;
    --kash-blue-soft: var(--secondary-background-color, #EAF2FF);
    --kash-gold: #FF5A1F;
    --kash-green: #2E7D32;
    --kash-orange: #FF5A1F;
    --kash-orange-dark: #E94D13;
    --kash-orange-soft: var(--secondary-background-color, #FFF3EC);
    --kash-red: #B42318;
    --kash-border: color-mix(in srgb, var(--text-color, #1F2933) 18%, transparent);
    --kash-muted: color-mix(in srgb, var(--text-color, #1F2933) 55%, transparent);
}

.upload-card, .upload-card-small,
.wizard-card, .workspace-card,
.report-header, .kpi-strip {
    background: rgba(127, 127, 127, 0.08) !important;
    border-color: var(--kash-border) !important;
}

.main .block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}
.kash-hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    border: 1px solid #FFE0D2;
    background: linear-gradient(135deg, #FFFFFF 0%, #FFFFFF 58%, #FFF7F2 100%);
    border-radius: 8px;
    padding: 30px 34px;
    margin-bottom: 20px;
    box-shadow: 0 18px 44px rgba(16, 24, 40, 0.08);
}
.kash-hero::before {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 56%;
    z-index: 0;
    opacity: 0.72;
    pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg width='760' height='220' viewBox='0 0 760 220' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%23ff5a1f' stroke-width='1.35' stroke-linecap='round'%3E%3Cpath opacity='.28' d='M18 178 C110 40 226 34 330 104 S520 184 732 42'/%3E%3Cpath opacity='.18' d='M52 22 C142 116 238 134 372 72 S598 8 736 132'/%3E%3Cpath opacity='.22' d='M112 212 C204 92 310 82 418 142 S586 202 750 94'/%3E%3Cpath opacity='.16' d='M0 92 L126 46 L238 128 L364 56 L498 144 L646 84 L760 124'/%3E%3Cpath opacity='.14' d='M84 168 L210 84 L338 162 L466 82 L608 152 L724 50'/%3E%3C/g%3E%3Cg fill='%23111827'%3E%3Ccircle opacity='.45' cx='126' cy='46' r='3'/%3E%3Ccircle opacity='.35' cx='238' cy='128' r='2.5'/%3E%3Ccircle opacity='.4' cx='364' cy='56' r='3'/%3E%3Ccircle opacity='.35' cx='498' cy='144' r='2.5'/%3E%3Ccircle opacity='.42' cx='646' cy='84' r='3'/%3E%3C/g%3E%3Cg fill='%23ff5a1f'%3E%3Ccircle opacity='.26' cx='210' cy='84' r='4'/%3E%3Ccircle opacity='.24' cx='418' cy='142' r='4'/%3E%3Ccircle opacity='.22' cx='608' cy='152' r='4'/%3E%3C/g%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: cover;
    background-position: right center;
}
.kash-title {
    position: relative;
    z-index: 1;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--kash-orange);
    margin: 0;
    letter-spacing: 0;
}
.kash-title-kash {
    color: #111827;
}
.kash-title-map {
    color: var(--kash-orange);
}
.kash-subtitle {
    position: relative;
    z-index: 1;
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
    margin-top: 8px;
    max-width: 980px;
    line-height: 1.55;
}
.kash-subtitle-accent {
    color: var(--kash-orange);
}
.section-label {
    font-size: 1.2rem;
    font-weight: 800;
    margin: 18px 0 8px 0;
}

.upload-card {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    min-height: 100%;
}

.upload-card-small {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    margin-bottom: 16px;
}

.kash-table-wrap {
    border: 1px solid var(--kash-border);
    border-radius: 14px;
    overflow: auto;
    max-height: 640px;
    margin: 14px 0 22px 0;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
    background: rgba(127, 127, 127, 0.08);
}

.kash-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
}

.kash-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.kash-table thead th {
    background: linear-gradient(180deg, #2B2B2B 0%, #1F2933 100%);
    color: #FFFFFF;
    font-weight: 800;
    text-align: left;
    padding: 13px 15px;
    border-right: 1px solid rgba(255, 255, 255, 0.12);
    border-bottom: 3px solid #FF5A1F;
    white-space: nowrap;
    letter-spacing: 0.01em;
}

.kash-table thead th:last-child {
    border-right: none;
}

.kash-table tbody td {
    padding: 12px 15px;
    border-bottom: 1px solid rgba(127, 127, 127, 0.2);
    border-right: 1px solid rgba(127, 127, 127, 0.12);
    vertical-align: top;
}

.kash-table tbody td:last-child {
    border-right: none;
}

.kash-table tbody tr:nth-child(even) {
    background: rgba(127, 127, 127, 0.06);
}

.kash-table tbody tr:hover {
    background: rgba(255, 90, 31, 0.12);
}

.kash-table tbody tr:last-child td {
    border-bottom: none;
}

.kash-table td:first-child {
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--kash-border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(16, 24, 40, 0.06);
    margin: 14px 0 22px 0;
}

.card-title {
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 6px;
}

.card-subtitle {
    font-size: 0.86rem;
    opacity: 0.7;
    margin-bottom: 14px;
}

.badge-required {
    width: fit-content;
    margin-left: auto;
    font-size: 0.72rem;
    font-weight: 700;
    color: #B42318;
    background: #FFF1F0;
    border: 1px solid #FFE1DF;
    border-radius: 999px;
    padding: 6px 10px;
    text-align: center;
}

.badge-optional {
    width: fit-content;
    margin-left: auto;
    font-size: 0.72rem;
    font-weight: 700;
    opacity: 0.75;
    background: rgba(127, 127, 127, 0.12);
    border: 1px solid var(--kash-border);
    border-radius: 999px;
    padding: 6px 10px;
    text-align: center;
}

.file-count-box {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 12px;
    padding: 10px 12px;
    margin-top: 10px;
    font-size: 0.88rem;
    font-weight: 600;
}

.wizard-card {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 8px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.wizard-card h3 {
    font-size: 1.15rem;
    margin: 0 0 10px 0;
}
.wizard-card p {
    line-height: 1.45;
    margin: 4px 0 10px 0;
}

div[data-testid="stExpander"]:hover {
    border-left: 4px solid var(--kash-orange);
}
.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background-color: var(--kash-blue-soft);
    color: var(--kash-navy);
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-high {
    background-color: var(--kash-navy);
    color: #FFFFFF;
}
.badge-medium {
    background-color: #FEF3C7;
    color: #92400E;
}
.badge-low {
    background-color: #DCFCE7;
    color: #166534;
}
.badge-ok {
    background-color: #DCFCE7;
    color: #166534;
}
.badge-review {
    background-color: #FEF3C7;
    color: #92400E;
}
.badge-error {
    background-color: #FEE2E2;
    color: var(--kash-red);
}
.workspace-card {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.055);
    margin-bottom: 14px;
}
.report-header {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 8px;
    padding: 22px 24px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    margin: 12px 0 16px 0;
}
.report-header h2 {
    font-size: 1.7rem;
    margin: 0 0 8px 0;
    letter-spacing: 0;
}
.report-header p {
    margin: 0 0 12px 0;
    line-height: 1.45;
}
.kpi-strip {
    background: rgba(127, 127, 127, 0.08);
    border: 1px solid var(--kash-border);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid var(--kash-border);
    font-weight: 700;
}
.stButton > button[kind="primary"] {
    background-color: #FF5A1F !important;
    border-color: #FF5A1F !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--kash-orange-dark) !important;
    border-color: var(--kash-orange-dark) !important;
    color: white !important;
}

@media (max-width: 700px) {
    .kash-title {
        font-size: 2rem;
    }
    .kash-hero {
        padding: 24px;
    }
}
</style>
    """,
    unsafe_allow_html=True,
)


def badge(label, level=''):
    css_class = 'badge'
    if level:
        css_class += f' badge-{level}'
    return f'<span class="{css_class}">{label}</span>'


def show_table_or_info(df, message, large=False, height=None):
    """Same styled-table pattern as KashMap: small tables get the orange
    kash-table HTML treatment, large ones stay as normal Streamlit dataframes.
    height defaults to None (auto-size) rather than the string "content" -
    some Streamlit versions require height to be an int or None, not a string."""
    if df is None or df.empty:
        st.info(message)
        return

    if large or len(df) > 150:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
        return

    html = df.to_html(index=False, escape=True, classes="kash-table", border=0)
    st.markdown(f'<div class="kash-table-wrap">{html}</div>', unsafe_allow_html=True)


st.markdown(
    """
    <div class="kash-hero">
      <div class="kash-title"><span class="kash-title-kash">Cognos</span><span class="kash-title-map">Map</span></div>
      <div class="kash-subtitle">Break Down <span class="kash-subtitle-accent">IBM Cognos</span> Complexity for Faster <span class="kash-subtitle-accent">Power BI</span> Migration</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to use CognosMap", expanded=True):
    st.markdown(
        """
        1. Upload one or more Cognos report `.xml` files.
        2. Optional: upload the Data Module `.json` response (from `/smarts_modules`) to validate fields against the real model.
        3. Click **Analyze Files**.
        4. Review the overview first, then open each report in the inspector below.
        5. Download the combined Excel workbook when you're ready.
        """
    )

st.markdown('<div class="section-label">Upload Files</div>', unsafe_allow_html=True)

upload_col, side_col = st.columns([1, 1], gap="large")

with upload_col:
    upload_card = st.container(border=True)
    with upload_card:
        title_col, badge_col = st.columns([0.82, 0.18])
        with title_col:
            st.markdown(
                """
                <div class="card-title">Upload Cognos Report XML</div>
                <div class="card-subtitle">
                    Upload one or more report specification .xml files.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with badge_col:
            st.markdown('<div class="badge-required">Required</div>', unsafe_allow_html=True)

        uploaded_reports = st.file_uploader(
            "Upload Cognos report XML files",
            type=["xml"],
            accept_multiple_files=True,
            key="report_uploader",
            label_visibility="collapsed",
        )
        if uploaded_reports:
            total_size = sum(getattr(f, "size", 0) for f in uploaded_reports)
            st.markdown(
                f"""
                <div class="file-count-box">
                    {len(uploaded_reports)} report file(s) uploaded · {total_size / 1024:.1f} KB total
                </div>
                """,
                unsafe_allow_html=True,
            )

with side_col:
    dm_card = st.container(border=True)
    with dm_card:
        title_col, badge_col = st.columns([0.78, 0.22])
        with title_col:
            st.markdown(
                """
                <div class="card-title">Data Module JSON</div>
                <div class="card-subtitle">
                    Optional. Upload this to validate report fields against the real model schema.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with badge_col:
            st.markdown('<div class="badge-optional">Optional</div>', unsafe_allow_html=True)

        uploaded_data_module = st.file_uploader(
            "Upload Data Module JSON (optional)",
            type=["json"],
            key="data_module_uploader",
            label_visibility="collapsed",
        )
        if uploaded_data_module:
            st.markdown(
                f"""
                <div class="file-count-box">
                    Data Module uploaded · {uploaded_data_module.size / 1024:.1f} KB
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    output_card = st.container(border=True)
    with output_card:
        st.markdown(
            """
            <div class="card-title">Output Settings</div>
            <div class="card-subtitle">Name of the generated Excel migration workbook.</div>
            """,
            unsafe_allow_html=True,
        )
        output_name = st.text_input(
            "Output file name",
            value="CognosMap_Migration_Output.xlsx",
            key="output_file_name",
            label_visibility="collapsed",
        )

submit = st.button("Analyze Files", type="primary", use_container_width=True)


def build_overview_row(report_name, parsed, validation_issues):
    return {
        "Report Name": parsed["report_name"] or report_name,
        "Model Path": parsed["model_path"],
        "Source Tables": ", ".join(parsed["source_tables"]),
        "Field Count": len(parsed["fields"]),
        "Source Field Count": len(parsed["source_fields"]),
        "Calculated Field Count": len(parsed["calculated_fields"]),
        "Filter Count": len(parsed["filters"]),
        "Prompt Count": len(parsed["prompts"]),
        "Drillthrough Count": len(parsed["drillthroughs"]),
        "Visualization Count": len(parsed["visualizations"]),
        "Custom Control Count": len(parsed["custom_controls"]),
        "Validation Status": "Verified" if not validation_issues else f"{len(validation_issues)} issue(s)",
    }


def build_fields_df(report_name, parsed):
    rows = []
    for f in parsed["fields"]:
        rows.append({
            "Report": report_name, "Query": f["query"], "Field": f["field"],
            "Aggregate": f["aggregate"], "Module": f["module"] or "", "Table": f["table"] or "",
            "Column": f["column"] or "", "Expression": f["expression"],
            "Is Calculated": "Yes" if f["is_calculated"] else "No",
            "Expression Type": f["expression_type"],
        })
    return pd.DataFrame(rows)


def build_filters_df(report_name, parsed):
    rows = []
    for f in parsed["filters"]:
        rows.append({
            "Report": report_name, "Query": f["query"], "Use": f["use"],
            "Expression": f["expression"], "Is Prompted": "Yes" if f["is_prompted"] else "No",
        })
    return pd.DataFrame(rows)


def build_prompts_df(report_name, parsed):
    rows = []
    for p in parsed["prompts"]:
        rows.append({
            "Report": report_name, "Parameter": p["parameter"], "Ref Query": p["ref_query"],
            "Options": ", ".join(p["options"]), "Default": p["default"], "Auto Submit": p["auto_submit"],
        })
    return pd.DataFrame(rows)


def build_drillthroughs_df(report_name, parsed):
    rows = []
    for d in parsed["drillthroughs"]:
        links = "; ".join(f"{l['source_parameter']} -> {l['target_parameter']}" for l in d["parameter_links"])
        rows.append({
            "Source Report": report_name, "Drill Name": d["drill_name"],
            "Target Report": d["target_report"], "Parameter Links": links,
            "Opens New Window": d["opens_new_window"],
        })
    return pd.DataFrame(rows)


def build_visualizations_df(report_name, parsed):
    rows = []
    for v in parsed["visualizations"]:
        for slot in v["slots"]:
            rows.append({
                "Report": report_name, "Visualization": v["name"], "Type": v["type"],
                "Data Store": slot["data_store"], "Slot": slot["slot"], "Columns": ", ".join(slot["columns"]),
            })
        if not v["slots"]:
            rows.append({
                "Report": report_name, "Visualization": v["name"], "Type": v["type"],
                "Data Store": "", "Slot": "", "Columns": "",
            })
    return pd.DataFrame(rows)


def build_custom_controls_df(report_name, parsed):
    rows = []
    for c in parsed["custom_controls"]:
        rows.append({
            "Report": report_name, "Control Name": c["name"], "Description": c["description"],
            "JS Path": c["path"], "Manual Review": "Yes - no Power BI equivalent",
        })
    return pd.DataFrame(rows)


def build_validation_df(report_name, issues):
    rows = []
    for issue in issues:
        rows.append({"Report": report_name, "Issue": issue})
    if not issues:
        rows.append({"Report": report_name, "Issue": "All fields verified against Data Module"})
    return pd.DataFrame(rows)


def build_excel_workbook(overview_df, fields_df, filters_df, prompts_df, drillthroughs_df, viz_df, controls_df, validation_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Overview", index=False)
        if not fields_df.empty:
            fields_df.to_excel(writer, sheet_name="Field Inventory", index=False)
        if not filters_df.empty:
            filters_df.to_excel(writer, sheet_name="Filters", index=False)
        if not prompts_df.empty:
            prompts_df.to_excel(writer, sheet_name="Prompts", index=False)
        if not drillthroughs_df.empty:
            drillthroughs_df.to_excel(writer, sheet_name="Drillthroughs", index=False)
        if not viz_df.empty:
            viz_df.to_excel(writer, sheet_name="Visualizations", index=False)
        if not controls_df.empty:
            controls_df.to_excel(writer, sheet_name="Custom Controls", index=False)
        if not validation_df.empty:
            validation_df.to_excel(writer, sheet_name="Validation", index=False)
    output.seek(0)
    return output


@st.fragment
def show_download_button(output_stream, file_name, key):
    st.download_button(
        label="Download Full Excel Report",
        data=output_stream,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
        use_container_width=True,
    )


def complexity_label(field_count, custom_control_count, validation_issue_count):
    """Same spirit as KashMap's complexity_label: a quick High/Medium/Low
    read on how much manual review a report will likely need."""
    score = field_count // 5 + custom_control_count * 3 + validation_issue_count * 2
    if score >= 12 or custom_control_count > 0 or validation_issue_count > 0:
        return "High", "high"
    if score >= 5:
        return "Medium", "medium"
    return "Low", "low"


if submit:
    if not uploaded_reports:
        st.error("Please upload at least one report XML file.")
        st.stop()

    data_module_schema = None
    if uploaded_data_module is not None:
        try:
            dm_payload = json.load(uploaded_data_module)
            data_module_schema = parse_cognos_data_module(dm_payload["data"])
            rel_count = len(data_module_schema.get("relationships", []))
            rel_note = f", {rel_count} relationship(s) found" if rel_count else ", no relationships declared (flat model)"
            st.success(f"Data Module loaded: {data_module_schema['module_name']} ({len(data_module_schema['tables'])} table(s)){rel_note}")
        except Exception as e:
            st.warning(f"Could not parse Data Module JSON: {e}")

    all_overview_rows = []
    all_fields_dfs = []
    all_filters_dfs = []
    all_prompts_dfs = []
    all_drillthroughs_dfs = []
    all_viz_dfs = []
    all_controls_dfs = []
    all_validation_dfs = []
    parsed_by_report = {}
    errors = []

    progress = st.progress(0)
    total = len(uploaded_reports)

    for idx, uploaded_file in enumerate(uploaded_reports, start=1):
        report_name = uploaded_file.name
        try:
            xml_text = uploaded_file.read().decode("utf-8", errors="replace")
            parsed = parse_cognos_report(xml_text)

            validation_issues = []
            if data_module_schema is not None:
                validation_issues = validate_report_against_model(parsed, data_module_schema)

            parsed_by_report[report_name] = (parsed, validation_issues)

            all_overview_rows.append(build_overview_row(report_name, parsed, validation_issues))
            all_fields_dfs.append(build_fields_df(report_name, parsed))
            all_filters_dfs.append(build_filters_df(report_name, parsed))
            all_prompts_dfs.append(build_prompts_df(report_name, parsed))
            all_drillthroughs_dfs.append(build_drillthroughs_df(report_name, parsed))
            all_viz_dfs.append(build_visualizations_df(report_name, parsed))
            all_controls_dfs.append(build_custom_controls_df(report_name, parsed))
            all_validation_dfs.append(build_validation_df(report_name, validation_issues))
        except Exception as e:
            errors.append(f"{report_name}: {e}")

        progress.progress(idx / total)

    overview_df = pd.DataFrame(all_overview_rows)
    fields_df = pd.concat(all_fields_dfs, ignore_index=True) if all_fields_dfs else pd.DataFrame()
    filters_df = pd.concat(all_filters_dfs, ignore_index=True) if all_filters_dfs else pd.DataFrame()
    prompts_df = pd.concat(all_prompts_dfs, ignore_index=True) if all_prompts_dfs else pd.DataFrame()
    drillthroughs_df = pd.concat(all_drillthroughs_dfs, ignore_index=True) if all_drillthroughs_dfs else pd.DataFrame()
    viz_df = pd.concat(all_viz_dfs, ignore_index=True) if all_viz_dfs else pd.DataFrame()
    controls_df = pd.concat(all_controls_dfs, ignore_index=True) if all_controls_dfs else pd.DataFrame()
    validation_df = pd.concat(all_validation_dfs, ignore_index=True) if all_validation_dfs else pd.DataFrame()

    workbook = build_excel_workbook(
        overview_df, fields_df, filters_df, prompts_df,
        drillthroughs_df, viz_df, controls_df, validation_df,
    )
    fn = output_name if output_name.lower().endswith(".xlsx") else f"{output_name}.xlsx"

    duplicate_records = build_cognos_duplicate_analysis(parsed_by_report)
    duplicate_df = build_duplicate_df(duplicate_records)

    st.session_state["cognosmap_result"] = {
        "overview_df": overview_df, "fields_df": fields_df, "filters_df": filters_df,
        "prompts_df": prompts_df, "drillthroughs_df": drillthroughs_df, "viz_df": viz_df,
        "controls_df": controls_df, "validation_df": validation_df,
        "parsed_by_report": parsed_by_report, "workbook": workbook, "fn": fn,
        "errors": errors, "data_module_schema": data_module_schema,
        "duplicate_df": duplicate_df, "duplicate_counts": cognos_duplicate_counts(duplicate_records),
    }

result = st.session_state.get("cognosmap_result")

if result:
    show_download_button(result["workbook"], result["fn"], "download_top")

    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    show_table_or_info(result["overview_df"], "No reports were analyzed.")
    dup_counts = result.get("duplicate_counts", {})
    if any(dup_counts.values()):
        st.markdown('<div class="section-label">Duplicate Analysis</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Exact Duplicates", dup_counts.get("exact", 0))
        d2.metric("Near Duplicates", dup_counts.get("near", 0))
        d3.metric("Same Data Source", dup_counts.get("same_source", 0))
        d4.metric("Unique", dup_counts.get("unique", 0))
        show_table_or_info(result.get("duplicate_df"), "No duplicate data available.")

    dm_schema = result.get("data_module_schema")
    if dm_schema is not None:
        relationships = dm_schema.get("relationships", [])
        with st.expander(f"Data Module Relationships ({len(relationships)} found)", expanded=False):
            if not relationships:
                st.info(
                    "No relationships/joins were declared in this Data Module - "
                    "each table stands alone with no connections between them. "
                    "This is common for models built from independently uploaded Excel/CSV files."
                )
            else:
                for rel in relationships:
                    st.write(f"**Found at:** `{rel['path']}`")
                    st.json(rel["value"])

    if result["errors"]:
        st.warning(f"{len(result['errors'])} file(s) had errors.")
        with st.expander("View Error Log"):
            for err in result["errors"]:
                st.text(err)

    st.markdown('<div class="section-label">Choose a report to inspect</div>', unsafe_allow_html=True)

    report_names = list(result["parsed_by_report"].keys())
    if report_names:
        selected_report = st.selectbox("Choose a report to inspect", report_names, label_visibility="collapsed")
        parsed, validation_issues = result["parsed_by_report"][selected_report]

        risk_label, risk_level = complexity_label(
            len(parsed["fields"]), len(parsed["custom_controls"]), len(validation_issues)
        )

        st.markdown(
            f"""
            <div class="report-header">
                <h2>{escape(selected_report)}</h2>
                <p><strong>Model path:</strong> <code>{escape(parsed['model_path'] or 'Not detected')}</code></p>
                <p><strong>Source tables:</strong> {escape(', '.join(parsed['source_tables']) or 'None detected')}</p>
                {badge(f'Risk: {risk_label}', risk_level)}
                {badge('Verified' if not validation_issues else f'{len(validation_issues)} validation issue(s)', 'ok' if not validation_issues else 'review')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            with st.container(border=True):
                st.metric("Fields", len(parsed["fields"]))
        with r2:
            with st.container(border=True):
                st.metric("Filters", len(parsed["filters"]))
        with r3:
            with st.container(border=True):
                st.metric("Prompts", len(parsed["prompts"]))
        with r4:
            with st.container(border=True):
                st.metric("Drillthroughs", len(parsed["drillthroughs"]))
        with r5:
            with st.container(border=True):
                st.metric("Custom Controls", len(parsed["custom_controls"]))

        tabs = st.tabs([
            "Fields", "Filters", "Prompts", "Drillthroughs",
            "Visualizations", "Custom Controls", "Validation", "AI Build Plan",
        ])

        with tabs[0]:
            report_fields_df = result["fields_df"][result["fields_df"]["Report"] == selected_report]
            show_table_or_info(
                report_fields_df.drop(columns=["Report"]) if not report_fields_df.empty else report_fields_df,
                "No fields found.", large=True,
            )

        with tabs[1]:
            report_filters_df = result["filters_df"][result["filters_df"]["Report"] == selected_report] if not result["filters_df"].empty else pd.DataFrame()
            show_table_or_info(
                report_filters_df.drop(columns=["Report"]) if not report_filters_df.empty else report_filters_df,
                "No filters found.",
            )

        with tabs[2]:
            report_prompts_df = result["prompts_df"][result["prompts_df"]["Report"] == selected_report] if not result["prompts_df"].empty else pd.DataFrame()
            show_table_or_info(
                report_prompts_df.drop(columns=["Report"]) if not report_prompts_df.empty else report_prompts_df,
                "No prompts found.",
            )

        with tabs[3]:
            report_drills_df = result["drillthroughs_df"][result["drillthroughs_df"]["Source Report"] == selected_report] if not result["drillthroughs_df"].empty else pd.DataFrame()
            show_table_or_info(report_drills_df, "No drillthroughs found.")

        with tabs[4]:
            report_viz_df = result["viz_df"][result["viz_df"]["Report"] == selected_report] if not result["viz_df"].empty else pd.DataFrame()
            show_table_or_info(
                report_viz_df.drop(columns=["Report"]) if not report_viz_df.empty else report_viz_df,
                "No visualizations found.",
            )

        with tabs[5]:
            report_controls_df = result["controls_df"][result["controls_df"]["Report"] == selected_report] if not result["controls_df"].empty else pd.DataFrame()
            if report_controls_df.empty:
                st.success("No custom controls - nothing needing manual JS review.")
            else:
                st.warning(f"{len(report_controls_df)} custom control(s) need manual review - no Power BI equivalent.")
                show_table_or_info(report_controls_df.drop(columns=["Report"]), "No custom controls found.")

        with tabs[6]:
            if not validation_issues:
                validation_summary_df = pd.DataFrame([{
                    "Report": selected_report,
                    "Status": "Verified",
                    "Issue Count": 0,
                    "Details": "All fields in this report verified against the Data Module (or no Data Module was uploaded).",
                }])
                show_table_or_info(validation_summary_df, "No validation data.")
            else:
                validation_issues_df = pd.DataFrame([
                    {"Report": selected_report, "Issue #": i + 1, "Issue": issue}
                    for i, issue in enumerate(validation_issues)
                ])
                st.error(f"{len(validation_issues)} field(s) could not be verified against the Data Module:")
                show_table_or_info(validation_issues_df, "No validation issues.")

        with tabs[7]:
            render_cognos_ai_build_plan(selected_report, parsed, validation_issues)