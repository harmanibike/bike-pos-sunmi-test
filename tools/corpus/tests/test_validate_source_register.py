import csv
import tempfile
import unittest
from pathlib import Path

from validate_source_register import validate_register


FIELDS = [
    "source_id", "title", "owner", "canonical_uri", "accessed_at",
    "declared_license", "license_evidence_uri", "training_permission",
    "commercial_use", "derivative_permission", "attribution_required",
    "redistribution_allowed", "privacy_screen", "reviewer", "decision",
    "decision_notes",
]


def row(**overrides):
    value = {
        "source_id": "source-001",
        "title": "Materi milik sendiri",
        "owner": "Tim Tata",
        "canonical_uri": "https://example.org/material",
        "accessed_at": "2026-11-28",
        "declared_license": "owned_or_explicitly_permitted",
        "license_evidence_uri": "https://example.org/permission",
        "training_permission": "yes",
        "commercial_use": "unclear",
        "derivative_permission": "yes",
        "attribution_required": "Atribusi bila dipublikasikan",
        "redistribution_allowed": "no",
        "privacy_screen": "Tidak ada data pribadi",
        "reviewer": "reviewer-internal",
        "decision": "approved",
        "decision_notes": "Izin tertulis mencakup pelatihan model lokal.",
    }
    value.update(overrides)
    return value


class SourceRegisterTests(unittest.TestCase):
    def write_csv(self, rows, fields=FIELDS):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "sources.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_accepts_explicitly_permitted_training_source(self):
        errors, stats = validate_register(self.write_csv([row()]))
        self.assertEqual([], errors)
        self.assertEqual(1, stats["approved"])

    def test_blocks_approved_source_without_training_permission(self):
        errors, _ = validate_register(self.write_csv([row(training_permission="unclear")]))
        self.assertTrue(any("training_permission bukan yes" in item for item in errors))

    def test_blocks_approved_source_without_evidence_or_reason(self):
        errors, _ = validate_register(self.write_csv([
            row(license_evidence_uri="", decision_notes="")
        ]))
        self.assertTrue(any("bukti lisensi/izin tidak ada" in item for item in errors))
        self.assertTrue(any("decision_notes wajib" in item for item in errors))

    def test_rejects_duplicate_source_ids(self):
        errors, _ = validate_register(self.write_csv([row(), row()]))
        self.assertTrue(any("source_id duplikat" in item for item in errors))

    def test_rejects_invalid_date_and_urls(self):
        errors, _ = validate_register(self.write_csv([
            row(accessed_at="28-11-2026", canonical_uri="ftp://example.org")
        ]))
        self.assertTrue(any("accessed_at" in item for item in errors))
        self.assertTrue(any("canonical_uri" in item for item in errors))

    def test_rejects_http_url_without_hostname(self):
        errors, _ = validate_register(self.write_csv([
            row(canonical_uri="https:///material")
        ]))
        self.assertTrue(any("canonical_uri" in item for item in errors))

    def test_rejects_embedded_url_credentials(self):
        errors, _ = validate_register(self.write_csv([
            row(license_evidence_uri="https://user:secret@example.org/license")
        ]))
        self.assertTrue(any("license_evidence_uri" in item for item in errors))

    def test_rejects_malformed_port(self):
        errors, _ = validate_register(self.write_csv([
            row(canonical_uri="https://example.org:99999/material")
        ]))
        self.assertTrue(any("canonical_uri" in item for item in errors))

    def test_pending_source_can_record_unclear_rights(self):
        errors, stats = validate_register(self.write_csv([
            row(decision="pending", training_permission="unclear")
        ]))
        self.assertEqual([], errors)
        self.assertEqual(1, stats["pending"])

    def test_empty_register_is_not_reported_as_ready(self):
        errors, stats = validate_register(self.write_csv([]))
        self.assertEqual(0, stats["sources"])
        self.assertTrue(any("belum berisi sumber" in item for item in errors))

    def test_rejects_duplicate_header_columns(self):
        fields = FIELDS + ["title"]
        errors, _ = validate_register(self.write_csv([row()], fields=fields))
        self.assertTrue(any("kolom duplikat" in item for item in errors))

    def test_rejects_extra_csv_cells(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "sources.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerow([row()[field] for field in FIELDS] + ["unexpected"])
        errors, _ = validate_register(path)
        self.assertTrue(any("melebihi header" in item for item in errors))

    def test_rejects_placeholder_privacy_review_for_approved_source(self):
        errors, _ = validate_register(self.write_csv([row(privacy_screen="TODO")]))
        self.assertTrue(any("privacy_screen" in item for item in errors))

    def test_rejects_placeholder_reviewer_for_approved_source(self):
        errors, _ = validate_register(self.write_csv([row(reviewer="unknown")]))
        self.assertTrue(any("reviewer" in item for item in errors))

    def test_requires_all_columns(self):
        errors, _ = validate_register(self.write_csv([], fields=["source_id", "title"]))
        self.assertTrue(any("Kolom wajib tidak ada" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
