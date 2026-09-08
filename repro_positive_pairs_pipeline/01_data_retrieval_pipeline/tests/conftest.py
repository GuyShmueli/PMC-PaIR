from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


@dataclass(frozen=True)
class LocalSources:
    input_csv: Path
    oa_responses_dir: Path
    archives_dir: Path


def _image_bytes(
    image_format: str,
    *,
    mode: str = "RGB",
    color: tuple[int, ...] = (32, 96, 160),
) -> bytes:
    output = io.BytesIO()
    Image.new(mode, (3, 2), color).save(output, format=image_format)
    return output.getvalue()


def _tar_gz_bytes(members: dict[str, bytes]) -> bytes:
    """Create a byte-stable archive, including intentionally unsafe names."""
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gzip_file:
        with tarfile.open(fileobj=gzip_file, mode="w") as archive:
            for name, payload in sorted(members.items()):
                member = tarfile.TarInfo(name=name)
                member.size = len(payload)
                member.mtime = 0
                member.mode = 0o644
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _oa_response(pmc_id: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OA>
  <records returned-count="1">
    <record id="PMC{pmc_id}">
      <link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/PMC{pmc_id}.pdf"/>
      <link format="tgz" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/PMC{pmc_id}.tar.gz"/>
    </record>
  </records>
</OA>
""".encode("utf-8")


@pytest.fixture()
def local_sources(tmp_path: Path) -> LocalSources:
    source_dir = tmp_path / "offline sources"
    oa_responses_dir = source_dir / "oa responses"
    archives_dir = source_dir / "archives"
    oa_responses_dir.mkdir(parents=True)
    archives_dir.mkdir(parents=True)

    # PMC300 is genuinely multi-patient even though its second patient's
    # similarity mapping is empty. It must not become eligible merely because
    # empty-similarity rows are removed later.
    input_rows = [
        {
            "patient_uid": "100-1",
            "similar_patients": json.dumps(
                {"200-1": 0.9, "300-1": 0.7}, sort_keys=True
            ),
            "source_note": "alpha",
        },
        {
            "patient_uid": "200-1",
            "similar_patients": json.dumps({"100-1": 0.9}),
            "source_note": "beta",
        },
        {
            "patient_uid": "300-1",
            "similar_patients": json.dumps({"100-1": 0.7}),
            "source_note": "first patient in a multi-patient article",
        },
        {
            "patient_uid": "300-2",
            "similar_patients": "{}",
            "source_note": "second patient in a multi-patient article",
        },
        {
            "patient_uid": "400-1",
            "similar_patients": "{}",
            "source_note": "empty mapping",
        },
        {
            "patient_uid": "500-1",
            "similar_patients": None,
            "source_note": "missing mapping",
        },
        {
            "patient_uid": "600-1",
            "similar_patients": "['not', 'a', 'mapping']",
            "source_note": "non-mapping literal",
        },
    ]
    input_csv = source_dir / "patients input.csv"
    pd.DataFrame(input_rows).to_csv(input_csv, index=False)

    for pmc_id in ("100", "200"):
        (oa_responses_dir / f"PMC{pmc_id}.xml").write_bytes(
            _oa_response(pmc_id)
        )

    article_100 = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <body>
    <fig id="F1">
      <caption><p>Alpha <italic>case</italic> – café.</p></caption>
      <graphic xlink:href="fig.001?download=1"/>
    </fig>
    <fig id="F2">
      <graphic xlink:href="unused"/>
    </fig>
  </body>
</article>
""".encode("utf-8")
    article_200 = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <body>
    <fig id="F1">
      <caption><p>Beta <bold>case</bold> caption.</p></caption>
      <graphic href="images/case.jpg#panel-a"/>
    </fig>
  </body>
</article>
""".encode("utf-8")

    archive_100 = _tar_gz_bytes(
        {
            "PMC100/article-source.nxml": article_100,
            "PMC100/fig.001.PNG": _image_bytes(
                "PNG", mode="RGBA", color=(32, 96, 160, 128)
            ),
            # A vulnerable extractor would write this outside its destination.
            "../escape.jpg": _image_bytes("JPEG"),
        }
    )
    archive_200 = _tar_gz_bytes(
        {
            "PMC200/article-source.nxml": article_200,
            "PMC200/images/case.jpg": _image_bytes(
                "JPEG", color=(160, 96, 32)
            ),
        }
    )
    (archives_dir / "PMC100.tar.gz").write_bytes(archive_100)
    (archives_dir / "PMC200.tar.gz").write_bytes(archive_200)

    return LocalSources(
        input_csv=input_csv,
        oa_responses_dir=oa_responses_dir,
        archives_dir=archives_dir,
    )
