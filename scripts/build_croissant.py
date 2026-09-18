"""Build a Croissant 1.1 file for a dataset from its YAML config in configs/croissant/."""

import argparse
import hashlib
import json
from pathlib import Path

import mlcroissant as mlc
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def croissant_dtype(arrow_type):
    if pa.types.is_timestamp(arrow_type):
        return mlc.DataType.DATETIME
    if pa.types.is_boolean(arrow_type):
        return mlc.DataType.BOOL
    if pa.types.is_integer(arrow_type):
        return mlc.DataType.INTEGER
    if pa.types.is_floating(arrow_type):
        return mlc.DataType.FLOAT
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return mlc.DataType.TEXT
    raise ValueError(f"No Croissant type for arrow type {arrow_type}")


def field(rs, col, dtype, desc, file_id, references=None, separator=None):
    return mlc.Field(
        # mlcroissant rejects whitespace in @id, and the sales columns have spaces.
        id=f"{rs}/{col.replace(' ', '_')}",
        name=col,
        description=desc,
        data_types=dtype,
        source=mlc.Source(
            file_object=file_id,
            extract=mlc.Extract(column=col),
            transforms=[mlc.Transform(separator=separator)]
            if separator
            else [],
        ),
        references=mlc.Source(field=references) if references else None,
        repeated=bool(separator),
    )


def build_record_set(rs, rs_cfg, dataset_dir):
    path = dataset_dir / rs_cfg["file"]
    schema = pq.read_schema(path)
    columns = rs_cfg["columns"]
    if set(schema.names) != set(columns):
        raise ValueError(
            f"{path.name}: parquet and config columns differ: "
            f"{sorted(set(schema.names) ^ set(columns))}"
        )

    file_object = mlc.FileObject(
        id=path.stem,
        name=path.stem,
        content_url=path.name,
        description=rs_cfg["description"],
        encoding_formats=["application/x-parquet"],
        sha256=sha256(path),
    )
    fields = []
    for col in schema.names:
        col_cfg = columns[col]
        if isinstance(col_cfg, str):
            col_cfg = {"description": col_cfg}
        fields.append(
            field(
                rs,
                col,
                croissant_dtype(schema.field(col).type),
                col_cfg["description"],
                path.stem,
                col_cfg.get("references"),
                col_cfg.get("separator"),
            )
        )
    record_set = mlc.RecordSet(
        id=rs, name=rs, description=rs_cfg["description"], fields=fields
    )
    return file_object, record_set


def build_croissant(cfg):
    files, record_sets = [], []
    for rs, rs_cfg in cfg["record_sets"].items():
        file_object, record_set = build_record_set(
            rs, rs_cfg, Path(cfg["dataset_dir"])
        )
        files.append(file_object)
        record_sets.append(record_set)

    return mlc.Metadata(
        name=cfg["name"],
        description=cfg["description"],
        url="https://github.com/Aniket-Mishra/synthetic_data_generator",
        license="https://spdx.org/licenses/MIT",
        version=cfg["version"],
        conforms_to="http://mlcommons.org/croissant/1.1",
        cite_as=cfg["cite_as"],
        date_published=cfg["date_published"],
        distribution=files,
        record_sets=record_sets,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    for config_path in parser.parse_args().config:
        cfg = yaml.safe_load(Path(config_path).read_text())
        croissant_path = Path(cfg["dataset_dir"]) / "croissant.json"
        croissant_path.write_text(
            json.dumps(build_croissant(cfg).to_json(), indent=2)
        )
        # Reloading the written file is the fast validity check. Errors raise.
        issues = mlc.Dataset(jsonld=str(croissant_path)).metadata.issues
        print(f"Wrote {croissant_path}, warnings: {list(issues.warnings)}")


if __name__ == "__main__":
    main()
