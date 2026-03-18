"""
Investigator Node — checks all scheduled maintenance tasks for a pump and returns overdue tasks and hours overdue.
"""

from typing import Any
import logging


async def service_age_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Investigator Node — For each scheduled maintenance task for the pump:
      - Find the last completion in work_done_logs
      - Compare to interval in service_schedules
      - If overdue, return task and hours overdue

    Fallback behavior:
      - If DB rows are absent, read uploaded CSV paths from state.
    """
    logger = logging.getLogger("oxmaint.investigator")
    try:
        from app.db.session import async_session
        from sqlalchemy import text as sa_text
        import os
        import polars as pl

        pump_id = str(state.get("pump_id") or "").strip()
        pump_id_lc = pump_id.lower()
        current_total_hours = state.get("current_total_hours")
        logger.info(
            "[service_age_node] Running for pump_id=%s, current_total_hours=%s",
            pump_id,
            current_total_hours,
        )

        def _safe_float(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return default

        def _safe_int(v: Any, default: int = 0) -> int:
            try:
                return int(v)
            except Exception:
                return default

        # Try to get current_total_hours from historical_logs if missing.
        if current_total_hours is None:
            hist_path = state.get("historical_logs_path")
            if hist_path and os.path.exists(hist_path):
                try:
                    df = pl.read_csv(hist_path)
                    if "Operational_Hours" in df.columns:
                        current_total_hours = int(df["Operational_Hours"].max())
                        logger.info(
                            "[investigator] Got current_total_hours=%s from historical_logs.",
                            current_total_hours,
                        )
                except Exception as exc:
                    logger.warning("[investigator] Failed to read historical_logs: %s", exc)
            if current_total_hours is None:
                logger.warning("[service_age_node] Missing current_total_hours and cannot infer from historical_logs.")
                return {
                    "investigator_error": "Missing current_total_hours and cannot infer from historical_logs.",
                    "service_age_risk_score": 0.0,
                    "transactional_risk_score": 0.0,
                }

        overdue_tasks: list[dict[str, Any]] = []
        open_requests: list[dict[str, Any]] = []
        sched_rows: list[tuple[str, float]] = []
        db_last_done_by_task: dict[str, float] = {}

        wdl_path = state.get("work_done_logs_path")
        ss_path = state.get("service_schedules_path")
        mr_path = state.get("maintenance_requests_path")

        # Load CSV rows up front so node can still compute if DB is unavailable.
        csv_sched_rows: list[tuple[str, float]] = []
        csv_open_requests: list[dict[str, Any]] = []
        wdl_rows: list[dict[str, Any]] = []

        if ss_path and os.path.exists(ss_path):
            try:
                sched_df = pl.read_csv(ss_path)
                if {"pump_id", "task_name", "interval_hours"}.issubset(set(sched_df.columns)):
                    for row in sched_df.to_dicts():
                        if str(row.get("pump_id", "")).strip().lower() == pump_id_lc:
                            csv_sched_rows.append(
                                (
                                    str(row.get("task_name", "")).strip(),
                                    _safe_float(row.get("interval_hours"), 0.0),
                                )
                            )
            except Exception as exc:
                logger.warning("[service_age_node] service_schedules CSV parse failed: %s", exc)

        if mr_path and os.path.exists(mr_path):
            try:
                req_df = pl.read_csv(mr_path)
                if "pump_id" in req_df.columns:
                    for row in req_df.to_dicts():
                        if str(row.get("pump_id", "")).strip().lower() != pump_id_lc:
                            continue
                        status = str(row.get("status", "OPEN")).strip().upper()
                        if status != "OPEN":
                            continue
                        csv_open_requests.append(
                            {
                                "description": str(row.get("description", "")),
                                "priority": _safe_int(row.get("priority"), 1),
                                "status": status,
                            }
                        )
            except Exception as exc:
                logger.warning("[service_age_node] maintenance_requests CSV parse failed: %s", exc)

        if wdl_path and os.path.exists(wdl_path):
            try:
                wdl_df = pl.read_csv(wdl_path)
                if {"pump_id", "task_name", "hours_at_service"}.issubset(set(wdl_df.columns)):
                    wdl_rows = wdl_df.to_dicts()
            except Exception as exc:
                logger.warning("[service_age_node] work_done_logs CSV parse failed: %s", exc)

        # Prefer DB data when available, but do not fail the node if DB is unreachable.
        try:
            async with async_session() as session:
                sched_sql = sa_text(
                    """
                    SELECT task_name, interval_hours
                    FROM service_schedules
                    WHERE pump_id = :pump_id
                    """
                )
                db_sched_rows = (await session.execute(sched_sql, {"pump_id": pump_id})).fetchall()
                sched_rows.extend((str(r[0]), _safe_float(r[1], 0.0)) for r in db_sched_rows)

                open_req_sql = sa_text(
                    """
                    SELECT description, priority, status
                    FROM maintenance_requests
                    WHERE pump_id = :pump_id AND status = 'OPEN'
                    ORDER BY priority DESC
                    """
                )
                open_req_rows = (await session.execute(open_req_sql, {"pump_id": pump_id})).fetchall()
                for req in open_req_rows:
                    open_requests.append(
                        {
                            "description": req[0],
                            "priority": _safe_int(req[1], 1),
                            "status": str(req[2]),
                        }
                    )

                # Pull last done hours per task in one DB query.
                last_done_sql = sa_text(
                    """
                    SELECT task_name, MAX(hours_at_service) AS last_done
                    FROM work_done_logs
                    WHERE pump_id = :pump_id
                    GROUP BY task_name
                    """
                )
                last_done_rows = (await session.execute(last_done_sql, {"pump_id": pump_id})).fetchall()
                for row in last_done_rows:
                    db_last_done_by_task[str(row[0])] = _safe_float(row[1], 0.0)
        except Exception as exc:
            logger.warning("[service_age_node] DB unavailable, continuing with CSV fallback only: %s", exc)

        if not sched_rows:
            sched_rows = csv_sched_rows
        if not open_requests:
            open_requests = csv_open_requests

        for task_name, interval_hours in sched_rows:
            if task_name in db_last_done_by_task:
                last_done = db_last_done_by_task[task_name]
            else:
                csv_task_hours = [
                    _safe_float(r.get("hours_at_service"), 0.0)
                    for r in wdl_rows
                    if str(r.get("pump_id", "")).strip().lower() == pump_id_lc
                    and str(r.get("task_name", "")).strip() == str(task_name)
                ]
                last_done = max(csv_task_hours) if csv_task_hours else 0.0

            overdue_hours = (float(current_total_hours) - last_done) - float(interval_hours)
            if overdue_hours > 0:
                overdue_tasks.append(
                    {
                        "task_name": task_name,
                        "overdue_hours": round(overdue_hours, 2),
                        "interval_hours": float(interval_hours),
                        "last_done": float(last_done),
                        "current_total_hours": int(current_total_hours),
                    }
                )

        max_overdue_ratio = 0.0
        if overdue_tasks:
            max_overdue_ratio = max(
                float(t["overdue_hours"]) / max(float(t["interval_hours"]), 1.0)
                for t in overdue_tasks
            )

        service_age_risk_score = min(1.0, 0.15 * len(overdue_tasks) + 0.35 * max_overdue_ratio)
        transactional_risk_score = 0.0
        if open_requests:
            max_priority = max(_safe_int(req.get("priority"), 1) for req in open_requests)
            transactional_risk_score = min(1.0, 0.20 * len(open_requests) + 0.20 * max_priority)

        service_age_anomaly = ""
        if overdue_tasks:
            service_age_anomaly += f"{len(overdue_tasks)} overdue scheduled task(s)"
        if open_requests:
            if service_age_anomaly:
                service_age_anomaly += "; "
            service_age_anomaly += f"{len(open_requests)} open maintenance request(s)"

        logger.info("[service_age_node] Overdue tasks result: %s", overdue_tasks)
        logger.info("[service_age_node] Open maintenance requests result: %s", open_requests)

        prev_query = state.get("anomaly_query", "")
        appended_query = prev_query + (
            f" overdue_tasks: {overdue_tasks}; open_requests: {open_requests};"
            f" service_age_risk_score: {service_age_risk_score:.3f};"
            f" transactional_risk_score: {transactional_risk_score:.3f};"
        )

        return {
            "overdue_tasks": overdue_tasks,
            "open_requests": open_requests,
            "service_age_risk_score": round(service_age_risk_score, 4),
            "transactional_risk_score": round(transactional_risk_score, 4),
            "service_age_anomaly": service_age_anomaly,
            "anomaly_query": appended_query,
        }
    except Exception as exc:
        logger.warning("⚠️ Investigator node failed: %s. Skipping to next node.", exc)
        return state