#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict | str]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            status = response.getcode()
            raw = response.read().decode("utf-8")
            try:
                return status, json.loads(raw)
            except json.JSONDecodeError:
                return status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else str(exc)
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except urllib.error.URLError as exc:
        return 0, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Oxmaint API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002", help="API base URL")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    health_url = f"{base}/health"
    predict_url = f"{base}/predict"

    print(f"Testing: {base}")

    health_status, health_body = _request_json("GET", health_url)
    if health_status != 200:
        print("[FAIL] /health", health_status, health_body)
        return 1
    print("[OK] /health", health_body)

    predict_payload = {
        "pump_id": "PUMP-1",
        "sensor_reading": {
            "pump_id": "PUMP-1",
            "operational_hours": 1200,
            "flow_rate": 12.5,
            "vibration_level": 3.2,
            "temperature": 88.0,
            "pressure": 145.0,
            "rpm": 2200,
            "power_consumption": 35.0,
            "moisture_level": 0.2,
            "extra_fields": {
                "Vibration": 3.2,
                "Temperature": 88.0,
                "Pressure": 145.0,
                "Flow_Rate": 12.5,
                "Operational_Hours": 1200,
            },
        },
    }

    predict_status, predict_body = _request_json("POST", predict_url, payload=predict_payload)
    if predict_status != 200 or not isinstance(predict_body, dict):
        print("[FAIL] /predict", predict_status, predict_body)
        return 1

    summary = {
        "request_id": predict_body.get("request_id"),
        "pump_id": predict_body.get("pump_id"),
        "risk_level": predict_body.get("risk_level"),
        "failure_probability": predict_body.get("failure_probability"),
    }
    print("[OK] /predict", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())