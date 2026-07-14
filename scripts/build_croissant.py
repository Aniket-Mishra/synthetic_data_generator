import json, itertools
from pathlib import Path
import argparse
import pandas as pd
import hashlib
import datetime

import mlcroissant as mlc


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def field(rs, col, dtype, desc, file_id, references=None, separator=None):
    return mlc.Field(
        id=f"{rs}/{col}",
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


def build_croissant(base_path):

    # hardcoded path, will pass this to script later as argparse.
    stream_path = f"{base_path}/wind_turbine_stream_data_with_repair.parquet"
    events_path = f"{base_path}/wind_turbine_fault_events_with_repair.parquet"

    stream_id, events_id = "stream", "events"
    stream_name = Path(stream_path).name
    events_name = Path(events_path).name

    sha256(f"{base_path}/{stream_name}")
    sha256(f"{base_path}/{events_name}")

    stream_file = mlc.FileObject(
        id=stream_id,
        name=stream_id,
        content_url=stream_name,
        description="Per-turbine sensor stream at 10 minute resolution.",
        encoding_formats=["application/x-parquet"],
        sha256=sha256(f"{base_path}/{stream_name}"),
    )
    events_file = mlc.FileObject(
        id=events_id,
        name=events_id,
        content_url=events_name,
        description="One row per injected fault.",
        encoding_formats=["application/x-parquet"],
        sha256=sha256(f"{base_path}/{events_name}"),
    )

    # This one's from domain expertise, learnt from Ravi Nadageri.
    # More hardcoded stuff that can be passed as a separate info file.
    sensors_with_desc = [
        ("active_power", "Active power output in kW."),
        ("wind_speed", "Wind speed at the turbine in m/s."),
        ("air_density", "Air density in kg/m^3."),
        ("wind_direction", "Wind direction in degrees."),
        ("nacelle_direction", "Nacelle heading in degrees."),
        ("nacelle_position", "Nacelle position in degrees."),
        ("ambient_temp", "Ambient temperature in Celsius."),
        ("rotor_speed", "Rotor speed in rpm."),
        ("generator_speed", "Generator speed in rpm."),
        ("gearbox_oil_temp", "Gearbox oil temperature in Celsius."),
        ("generator_temp", "Generator temperature in Celsius."),
        ("bearing_temp", "Bearing temperature in Celsius."),
        ("converter_temp", "Converter temperature in Celsius."),
        ("pitch_blade_angle_1", "Pitch angle of blade 1 in degrees."),
        ("pitch_blade_angle_2", "Pitch angle of blade 2 in degrees."),
        ("pitch_blade_angle_3", "Pitch angle of blade 3 in degrees."),
    ]

    # fault_flags = [x for x in df.select_dtypes([float,int, object]).columns if ((x.startswith("fault_")) or x == "is_drifted")]
    fault_flags = [
        ("fault_temperature", "temperature"),
        ("fault_pitch_misalignment", "pitch misalignment"),
        ("fault_yaw_misalignment", "yaw misalignment"),
    ]

    measurements = mlc.RecordSet(
        id="measurements",
        name="measurements",
        description="One record per turbine per 10 minutes, with sensor readings and fault ground truth.",
        fields=[
            field(
                "measurements",
                "time",
                mlc.DataType.DATETIME,
                "Reading timestamp.",
                stream_id,
            ),
            field(
                "measurements",
                "device",
                mlc.DataType.TEXT,
                "Turbine id.",
                stream_id,
            ),
            *[
                field("measurements", col, mlc.DataType.FLOAT, desc, stream_id)
                for col, desc in sensors_with_desc
            ],
            field(
                "measurements",
                "is_drifted",
                mlc.DataType.INTEGER,
                "1 if any fault active, else 0.",
                stream_id,
            ),
            field(
                "measurements",
                "fault_labels",
                mlc.DataType.TEXT,
                "Active fault(s), pipe separated, else healthy.",
                stream_id,
                separator="|",
            ),
            field(
                "measurements",
                "fault_severity",
                mlc.DataType.TEXT,
                "none, low, medium, or high.",
                stream_id,
            ),
            field(
                "measurements",
                "drift_start_time",
                mlc.DataType.DATETIME,
                "First fault onset, else null.",
                stream_id,
            ),
            *[
                field(
                    "measurements",
                    col,
                    mlc.DataType.INTEGER,
                    f"1 if {label} fault active, else 0.",
                    stream_id,
                )
                for col, label in fault_flags
            ],
        ],
    )

    fault_events = mlc.RecordSet(
        id="fault_events",
        name="fault_events",
        description="One row per injected fault. device is a foreign key into measurements.",
        fields=[
            field(
                "fault_events",
                "device",
                mlc.DataType.TEXT,
                "Turbine the fault was injected on.",
                events_id,
                references="measurements/device",
            ),
            field(
                "fault_events",
                "fault_type",
                mlc.DataType.TEXT,
                "Fault type.",
                events_id,
            ),
            field(
                "fault_events",
                "start_time",
                mlc.DataType.DATETIME,
                "Fault start.",
                events_id,
            ),
            field(
                "fault_events",
                "end_time",
                mlc.DataType.DATETIME,
                "Fault end, null if never repaired.",
                events_id,
            ),
            field(
                "fault_events",
                "shape",
                mlc.DataType.TEXT,
                "Onset shape: linear, abrupt, intermittent.",
                events_id,
            ),
            field(
                "fault_events",
                "max_severity",
                mlc.DataType.FLOAT,
                "Peak severity.",
                events_id,
            ),
            field(
                "fault_events",
                "ramp_steps",
                mlc.DataType.FLOAT,
                "Onset ramp in steps, if given.",
                events_id,
            ),
            field(
                "fault_events",
                "ramp_days",
                mlc.DataType.FLOAT,
                "Onset ramp in days, if given.",
                events_id,
            ),
            field(
                "fault_events",
                "ramp_down_days",
                mlc.DataType.FLOAT,
                "Recovery ramp in days, if given.",
                events_id,
            ),
        ],
    )

    meta = mlc.Metadata(
        name="wind_farm_fault_telemetry",
        description="Physics-grounded synthetic wind farm telemetry with configurable fault injection and repair.",
        url="https://github.com/Aniket-Mishra/synthetic_data_generator",
        license="https://spdx.org/licenses/MIT",
        version="1.0.0",
        conforms_to="http://mlcommons.org/croissant/1.1",
        cite_as="Aniket Mishra, Wind Farm Fault Telemetry (synthetic), 2026.",
        date_published=datetime.date(2026, 7, 14),
        distribution=[stream_file, events_file],
        record_sets=[measurements, fault_events],
    )
    print("warnings:", list(meta.issues.warnings))

    croissant_path = Path(base_path) / "wind_croissant.json"
    json.dump(meta.to_json(), open(croissant_path, "w"), indent=2)

    # # This is super slow cuz it does join on all the records to return 2 rows
    # ds = mlc.Dataset(jsonld=str(croissant_path))
    # print("events:", list(itertools.islice(ds.records("fault_events"), 2)))
    # print("stream:", list(itertools.islice(ds.records("measurements"), 2)))

    # This one just checks if its a valid croissant, super fast

    ds = mlc.Dataset(jsonld=str(croissant_path))
    print("warnings:", list(ds.metadata.issues.warnings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    build_croissant(parser.parse_args().dataset_dir)


if __name__ == "__main__":
    main()
