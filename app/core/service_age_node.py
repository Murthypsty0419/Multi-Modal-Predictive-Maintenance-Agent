"""
Investigator Node — checks all scheduled maintenance tasks for a pump and returns overdue tasks and hours overdue.
"""
from typing import Any
from app.schemas import AgentState
import logging

async def service_age_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Investigator Node — For each scheduled maintenance task for the pump:
      - Find the last completion in work_done_logs
      - Compare to interval in service_schedules
      - If overdue, return task and hours overdue
    If current_total_hours is missing, try to get it from historical_logs (max Operational_Hours)
    """
    logger = logging.getLogger("oxmaint.investigator")
    try:
        from app.db.session import async_session
        from sqlalchemy import text as sa_text
        import os
        import polars as pl

        pump_id = state.get("pump_id")
        current_total_hours = state.get("current_total_hours")
        logger.info(f"[service_age_node] Running for pump_id={pump_id}, current_total_hours={current_total_hours}")

        # Try to get current_total_hours from historical_logs if missing
        if current_total_hours is None:
            hist_path = state.get("historical_logs_path")
            if hist_path and os.path.exists(hist_path):
                try:
                    df = pl.read_csv(hist_path)
                    if "Operational_Hours" in df.columns:
                        current_total_hours = int(df["Operational_Hours"].max())
                        logger.info(f"[investigator] Got current_total_hours={current_total_hours} from historical_logs.")
                except Exception as e:
                    logger.warning(f"[investigator] Failed to read historical_logs: {e}")
            if current_total_hours is None:
                logger.warning("[service_age_node] Missing current_total_hours and cannot infer from historical_logs.")
                return {"investigator_error": "Missing current_total_hours and cannot infer from historical_logs."}

        overdue_tasks = []
        open_requests = []

        async with async_session() as session:
            # Get all scheduled tasks for this pump
            sched_sql = sa_text("""
                SELECT task_name, interval_hours
                FROM service_schedules
                WHERE pump_id = :pump_id
            """)
            sched_rows = (await session.execute(sched_sql, {"pump_id": pump_id})).fetchall()
            print(f"[service_age_node] Scheduled tasks for pump_id={pump_id}: {sched_rows}")

            # Fetch open maintenance requests for this pump
            open_req_sql = sa_text("""
                SELECT description, priority, status
                FROM maintenance_requests
                WHERE pump_id = :pump_id AND status = 'OPEN'
                ORDER BY priority DESC
            """)
            open_req_rows = (await session.execute(open_req_sql, {"pump_id": pump_id})).fetchall()
            for req in open_req_rows:
                open_requests.append({
                    "description": req[0],
                    "priority": req[1],
                    "status": req[2]
                })
            print(f"[service_age_node] Open maintenance requests for pump_id={pump_id}: {open_requests}")

            for sched in sched_rows:
                task_name, interval_hours = sched
                # Get last completion for this task
                log_sql = sa_text("""
                    SELECT hours_at_service
                    FROM work_done_logs
                    WHERE pump_id = :pump_id AND task_name = :task_name
                    ORDER BY timestamp DESC, hours_at_service DESC
                    LIMIT 1
                """)
                log_row = (await session.execute(log_sql, {"pump_id": pump_id, "task_name": task_name})).first()
                if log_row:
                    print(f"[service_age_node] work_done_logs for pump_id={pump_id}, task_name={task_name}: {log_row[0]}")
                    last_done = log_row[0]
                else:
                    print(f"[service_age_node] work_done_logs for pump_id={pump_id}, task_name={task_name}: None")
                    last_done = 0  # Never done
                overdue_hours = (current_total_hours - last_done) - interval_hours
                if overdue_hours > 0:
                    overdue_tasks.append({
                        "task_name": task_name,
                        "overdue_hours": overdue_hours,
                        "interval_hours": interval_hours,
                        "last_done": last_done,
                        "current_total_hours": current_total_hours
                    })
        logger.info(f"[service_age_node] Overdue tasks result: {overdue_tasks}")
        logger.info(f"[service_age_node] Open maintenance requests result: {open_requests}")
        prev_query = state.get("anomaly_query", "")
        appended_query = prev_query + f" overdue_tasks: {overdue_tasks}; open_requests: {open_requests};"
        return {
            "overdue_tasks": overdue_tasks,
            "open_requests": open_requests,
            "anomaly_query": appended_query
        }
    except Exception as e:
        logger.warning(f"⚠️ Investigator node failed: {e}. Skipping to next node.")
        return state