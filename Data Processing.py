#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from datetime import timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

# Install only the packages needed for this separate preparation step.
REQUIRED = {"pandas": "pandas", "requests": "requests"}
missing = [pip for module, pip in REQUIRED.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print("[SETUP] Installing missing packages:", ", ".join(missing), flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import pandas as pd
import requests


URLS = {
    "email_archive": (
        "https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz",
        "https://bebop.berkeley.edu/enron/enron_with_categories.tar.gz",
    ),
    "human_qrels": (
        "https://zenodo.org/records/8006820/files/repeated_qrels.txt?download=1",
        "https://raw.githubusercontent.com/JackMcKechnie/SARA/main/repeated_qrels.txt",
    ),
    "queries": (
        "https://zenodo.org/records/8006820/files/repeated_queries.tsv?download=1",
        "https://raw.githubusercontent.com/JackMcKechnie/SARA/main/repeated_queries.tsv",
    ),
    "needs": (
        "https://zenodo.org/records/8006820/files/information_needs.tsv?download=1",
        "https://raw.githubusercontent.com/JackMcKechnie/SARA/main/information_needs.tsv",
    ),
}

SOURCE_SHA256_REFERENCE = {
    "email_archive": "08625500ab4c032f99acfc1f8eed125b2cd5e9c4280c7ce2d7ce0bd28a190625",
    "human_qrels": "11abd7b0c465bd07c16848042dea8c160a3398c69e549fcf641f46d3b890c67b",
    "queries": "fdadebeba8f2c5b097636f4910ae83ffe2bad2e83b2fc891479eec8fbcdfdb7e",
    "needs": "0d0522fb77b46472e19326e550aeff855f53ecf41a5086b5f31a23c76e618f02",
}

SOURCE_SPECS = {
    "email_archive": ("enron_with_categories.tar.gz", 100_000),
    "human_qrels": ("repeated_qrels.txt", 100_000),
    "queries": ("repeated_queries.tsv", 1_000),
    "needs": ("information_needs.tsv", 1_000),
}


def log(message: str) -> None:
    print("[DATA-PROCESSING]", message, flush=True)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def obtain_source(key: str, supplied_path: str | None, raw_dir: Path,
                  no_download: bool) -> tuple[bytes, str]:
    filename, minimum_size = SOURCE_SPECS[key]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if supplied_path:
        path = Path(supplied_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found for {key}: {path}")
        data = path.read_bytes()
        source = f"local_file:{path.name}"
    else:
        path = raw_dir / filename
        if path.is_file() and path.stat().st_size >= minimum_size:
            data = path.read_bytes()
            source = f"local_cache:{filename}"
        elif no_download:
            raise FileNotFoundError(
                f"Required file is not present in {raw_dir}: {filename}. "
                "Provide its path or rerun without --no-download."
            )
        else:
            errors = []
            data = b""
            source = ""
            for url in URLS[key]:
                for attempt in range(2):
                    try:
                        log(f"Downloading {key} from {url.split('/')[2]}")
                        response = requests.get(
                            url, timeout=(20, 180),
                            headers={"User-Agent": "PAMR-SARA-Data-Preparation/1.0"}
                        )
                        response.raise_for_status()
                        candidate = response.content
                        if len(candidate) < minimum_size:
                            raise ValueError(
                                f"Downloaded file is unexpectedly small ({len(candidate)} bytes)"
                            )
                        if key == "email_archive" and candidate[:2] != b"\x1f\x8b":
                            raise ValueError("Downloaded email archive is not gzip-compressed")
                        data = candidate
                        source = url
                        path.write_bytes(data)
                        break
                    except Exception as exc:
                        errors.append(f"{url}: {type(exc).__name__}: {exc}")
                        time.sleep(0.5 + attempt)
                if data:
                    break
            if not data:
                raise RuntimeError(
                    f"Could not retrieve original {key}.\n" + "\n".join(errors)
                )

    if len(data) < minimum_size:
        raise ValueError(f"{key} is smaller than the expected minimum size")
    if key == "email_archive" and data[:2] != b"\x1f\x8b":
        raise ValueError(f"{key} is not a valid gzip-compressed archive")
    return data, source


def get_header(message, key: str) -> str:
    return " ".join(
        str(value).strip()
        for value in message.get_all(key, [])
        if str(value).strip()
    )


def date_parseable(value: str) -> bool:
    try:
        parsed = parsedate_to_datetime(value)
        if parsed is None:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return 1970 <= parsed.year <= 2026
    except Exception:
        return False


def email_body(message, raw: bytes) -> str:
    try:
        if message.is_multipart():
            parts = []
            for part in message.walk():
                if part.get_content_type() == "text/plain" and not part.get_filename():
                    parts.append(str(part.get_content() or ""))
            result = "\n".join(parts).strip()
        else:
            result = str(message.get_content() or "").strip()
        if result:
            return result
    except Exception:
        pass

    decoded = raw.decode("utf-8", errors="replace")
    separator = re.search(r"\r?\n\r?\n", decoded)
    return decoded[separator.end():].strip() if separator else ""


def parse_email_archive(blob: bytes) -> pd.DataFrame:
    rows = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".txt"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            if not raw or b"\x00" in raw[:256]:
                continue

            message = BytesParser(policy=policy.default).parsebytes(raw)
            values = {
                field: get_header(message, header)
                for field, header in (
                    ("from", "From"),
                    ("to", "To"),
                    ("subject", "Subject"),
                    ("date", "Date"),
                    ("message_id", "Message-ID"),
                )
            }
            if sum(bool(value) for value in values.values()) < 2:
                continue

            docno = re.sub(
                r"\.txt$", "", PurePosixPath(member.name).name, flags=re.IGNORECASE
            )
            if not docno:
                continue
            if docno in rows:
                raise ValueError(f"Duplicate email document identifier: {docno}")

            values["docno"] = docno
            values["body"] = email_body(message, raw)
            values["has_thread"] = int(bool(
                get_header(message, "In-Reply-To") or get_header(message, "References")
            ))
            for field in ("from", "to", "subject", "message_id"):
                values["has_" + field] = int(bool(values[field]))
            values["valid_date"] = int(date_parseable(values["date"]))
            values["completeness"] = sum((
                values["has_from"], values["has_to"], values["has_subject"],
                values["has_message_id"], values["valid_date"]
            )) / 5.0
            rows[docno] = values

    if not rows:
        raise ValueError("No email records were parsed from the Berkeley archive")

    return (
        pd.DataFrame(rows.values())
        .sort_values("docno")
        .reset_index(drop=True)
    )


def parse_benchmark(qbytes: bytes, rbytes: bytes, nbytes: bytes):
    queries = pd.read_csv(
        io.StringIO(qbytes.decode("utf-8-sig")),
        sep="\t", header=None, names=["qid", "query"], dtype=str
    )
    needs = pd.read_csv(
        io.StringIO(nbytes.decode("utf-8-sig")),
        sep="\t", header=None, names=["need_id", "need_description"], dtype=str
    )
    qrels = pd.read_csv(
        io.StringIO(rbytes.decode("utf-8-sig")),
        sep=r"\s+", header=None,
        names=["qid", "iteration", "docno", "grade"],
        engine="python",
        dtype={"qid": str, "iteration": str, "docno": str},
    )

    queries["qid"] = queries["qid"].astype(str)
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["grade"] = pd.to_numeric(qrels["grade"], errors="raise").astype(int)
    queries["qid_num"] = pd.to_numeric(queries["qid"], errors="raise").astype(int)

    if sorted(queries["qid_num"].tolist()) != list(range(1, 151)):
        raise ValueError("Expected the original 150 SARA query IDs numbered 1 through 150")
    queries["need_id"] = ((queries["qid_num"] - 1) // 3 + 1).astype(int)

    if len(needs) != 50:
        raise ValueError(f"Expected 50 information needs; found {len(needs)}")
    if queries.groupby("need_id").size().tolist() != [3] * 50:
        raise ValueError("Expected exactly three query formulations per information need")
    if len(qrels) != 34_413 or not qrels["grade"].isin([0, 1, 2]).all():
        raise ValueError("Expected 34,413 original human judgments with grades 0, 1, or 2")
    if qrels.duplicated(["qid", "docno"]).any():
        raise ValueError("Duplicate query-document judgments were found")
    if set(qrels["qid"]) != set(queries["qid"]):
        raise ValueError("The qrel query IDs do not match the query file")

    query_to_need = dict(zip(queries["qid"], queries["need_id"]))
    qrels["need_id"] = qrels["qid"].map(query_to_need).astype(int)

    grouped = qrels.groupby(["need_id", "docno"]).agg(
        repeated_rows=("grade", "size"),
        distinct_grades=("grade", "nunique"),
    )
    if not (grouped["repeated_rows"] == 3).all():
        raise ValueError("Each need-document judgment must be repeated exactly three times")
    if not (grouped["distinct_grades"] == 1).all():
        raise ValueError("Repeated judgments disagree for at least one need-document pair")
    if len(grouped) != 11_471:
        raise ValueError(
            f"Expected 11,471 unique need-document assessments; found {len(grouped)}"
        )

    return queries, qrels, needs


def validate_data(docs: pd.DataFrame, queries: pd.DataFrame,
                  qrels: pd.DataFrame, needs: pd.DataFrame) -> dict:
    expected = (1_702, 150, 34_413, 50)
    observed = (len(docs), len(queries), len(qrels), len(needs))
    if observed != expected:
        raise ValueError(
            "Unexpected original dataset dimensions. "
            f"Expected documents/queries/qrels/needs={expected}; found {observed}."
        )
    if docs["docno"].isna().any() or docs["docno"].duplicated().any():
        raise ValueError("Missing or duplicate email document identifiers")
    if docs["body"].fillna("").map(len).eq(0).all():
        raise ValueError("All parsed email bodies are empty")
    if not docs["completeness"].between(0, 1).all():
        raise ValueError("Metadata completeness values must be between 0 and 1")
    missing_doc_ids = sorted(set(qrels["docno"]) - set(docs["docno"]))
    if missing_doc_ids:
        raise ValueError(
            f"{len(missing_doc_ids)} judged document IDs are missing from the email corpus"
        )
    if qrels.duplicated(["qid", "docno"]).any():
        raise ValueError("Duplicate query-document pairs were found")

    unique_judgments = qrels.drop_duplicates(["need_id", "docno"])
    grade_counts = unique_judgments["grade"].value_counts().to_dict()
    metadata_fields = {
        "from": "has_from",
        "to": "has_to",
        "subject": "has_subject",
        "date_parseable": "valid_date",
        "message_id": "has_message_id",
        "thread_pointer": "has_thread",
    }
    metadata_coverage = {
        label: {
            "documents_present": int(docs[column].sum()),
            "coverage_percent": round(float(docs[column].mean() * 100), 4),
        }
        for label, column in metadata_fields.items()
    }

    return {
        "validation_status": "passed",
        "email_documents": int(len(docs)),
        "query_formulations": int(len(queries)),
        "independent_information_needs": int(queries["need_id"].nunique()),
        "repeated_human_judgment_rows": int(len(qrels)),
        "unique_need_document_assessments": int(len(unique_judgments)),
        "unique_judged_email_documents": int(qrels["docno"].nunique()),
        "emails_without_any_human_judgment": int(len(docs) - qrels["docno"].nunique()),
        "unique_assessments_by_grade": {
            str(grade): int(grade_counts.get(grade, 0)) for grade in (0, 1, 2)
        },
        "metadata_field_coverage": metadata_coverage,
        "document_ids_all_matched": True,
        "repeated_judgments_consistent": True,
        "human_judgments_modified_or_simulated": False,
        "metadata_completeness_is_authenticated_provenance": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and validate the original Berkeley Enron / SARA datasets."
    )
    parser.add_argument("--email-archive", help="Local path to enron_with_categories.tar.gz")
    parser.add_argument("--qrels", help="Local path to repeated_qrels.txt")
    parser.add_argument("--queries", help="Local path to repeated_queries.tsv")
    parser.add_argument("--needs", help="Local path to information_needs.tsv")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory for downloaded/cached original source files")
    parser.add_argument("--output-dir", default="processed_data",
                        help="Directory for processed CSV files and audit reports")
    parser.add_argument("--no-download", action="store_true",
                        help="Do not access the network; use supplied files or files in --raw-dir")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    supplied = {
        "email_archive": args.email_archive,
        "human_qrels": args.qrels,
        "queries": args.queries,
        "needs": args.needs,
    }
    source_bytes = {}
    source_manifest = {}

    for key in ("email_archive", "human_qrels", "queries", "needs"):
        data, source = obtain_source(key, supplied[key], raw_dir, args.no_download)
        digest = sha256(data)
        source_bytes[key] = data
        source_manifest[key] = {
            "source": source,
            "filename": SOURCE_SPECS[key][0],
            "size_bytes": len(data),
            "sha256": digest,
            "matches_reference_sha256": digest == SOURCE_SHA256_REFERENCE[key],
        }

    log("Parsing the Berkeley Enron email archive")
    docs = parse_email_archive(source_bytes["email_archive"])
    log("Parsing the original SARA queries, information needs, and human judgments")
    queries, qrels, needs = parse_benchmark(
        source_bytes["queries"], source_bytes["human_qrels"], source_bytes["needs"]
    )
    report = validate_data(docs, queries, qrels, needs)

    # A convenience table with one row per independent need-document assessment.
    unique_judgments = (
        qrels[["need_id", "docno", "grade"]]
        .drop_duplicates(["need_id", "docno"])
        .sort_values(["need_id", "docno"])
        .reset_index(drop=True)
    )

    docs.to_csv(output_dir / "emails_processed.csv", index=False, float_format="%.12g")
    queries.to_csv(output_dir / "queries_processed.csv", index=False)
    needs.to_csv(output_dir / "information_needs_processed.csv", index=False)
    qrels.to_csv(output_dir / "human_qrels_processed.csv", index=False)
    unique_judgments.to_csv(
        output_dir / "unique_need_document_judgments.csv", index=False
    )

    coverage = pd.DataFrame([
        {
            "field": field,
            "documents_present": values["documents_present"],
            "coverage_percent": values["coverage_percent"],
        }
        for field, values in report["metadata_field_coverage"].items()
    ])
    coverage.to_csv(output_dir / "metadata_field_coverage.csv", index=False)

    report["source_files"] = {
        key: {
            "filename": value["filename"],
            "size_bytes": value["size_bytes"],
            "sha256": value["sha256"],
            "matches_reference_sha256": value["matches_reference_sha256"],
        }
        for key, value in source_manifest.items()
    }
    report["output_files"] = [
        "emails_processed.csv",
        "queries_processed.csv",
        "information_needs_processed.csv",
        "human_qrels_processed.csv",
        "unique_need_document_judgments.csv",
        "metadata_field_coverage.csv",
        "data_processing_report.json",
        "source_manifest.json",
    ]
    report["data_handling_note"] = (
        "emails_processed.csv contains email body text and email header values. "
        "Keep generated data local unless reuse, licensing, and data-protection "
        "requirements have been reviewed."
    )

    (output_dir / "data_processing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )

    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    import zipfile
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{output_dir.name}/{path.name}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        failed = archive.testzip()
        if failed:
            raise IOError(f"Processed-data ZIP integrity check failed: {failed}")

    log("Validation passed.")
    log(f"Email documents: {report['email_documents']}")
    log(f"Query formulations: {report['query_formulations']}")
    log(f"Information needs: {report['independent_information_needs']}")
    log(f"Repeated human judgments: {report['repeated_human_judgment_rows']}")
    log(f"Unique need-document assessments: {report['unique_need_document_assessments']}")
    log(f"Processed files: {output_dir.resolve()}")
    log(f"Processed-data archive: {zip_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "\n[DATA-PROCESSING] FAILED — no validated processed dataset is claimed.\n"
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
