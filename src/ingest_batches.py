import sys
import shutil
from pathlib import Path
from ingest_validate import ingest

# Validate every new labelled batch in data/new/ with the same schema as the original dataset.
# Valid batches are copied to data/validated/new/; if any batch is invalid the pipeline stops.
SCHEMA = "config/ingestion_schema.yaml"
NEW_DIR = Path("data/new")
OUT_DIR = Path("data/validated/new")
REPORT_DIR = Path("metrics/ingestion_batches")

# Start clean so a batch removed from data/new/ is also removed here
shutil.rmtree(OUT_DIR, ignore_errors=True)
shutil.rmtree(REPORT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True)
REPORT_DIR.mkdir(parents=True)

batches = sorted(NEW_DIR.glob("*.csv"))
failed = []
for batch in batches:
    code = ingest(batch, SCHEMA, OUT_DIR / batch.name, REPORT_DIR / f"{batch.stem}.json", require_labels=True)
    if code != 0:
        failed.append(batch.name)

print(f"Validated {len(batches) - len(failed)} of {len(batches)} new batches")
if failed:
    sys.exit(f"Invalid batches: {', '.join(failed)}. See {REPORT_DIR}/")