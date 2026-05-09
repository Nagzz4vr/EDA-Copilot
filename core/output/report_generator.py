from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ReportConfig:
    output_dir:       str = "outputs/reports"
    include_markdown: bool = True
    top_k_signals:    int = 10
    indent:           int = 4



class ReportGenerator:

    def __init__(self,config: Optional[ReportConfig] = None,budget_controller=None,confidence_tracker=None,):
        self._cfg                 = config or ReportConfig()
        self._budget_controller   = budget_controller
        self._confidence_tracker  = confidence_tracker
    def generate(self,plan: Any,execution_result: Dict[str, Any],signal_bag: Any,confidence_score: float,job_id: str,validation_report: Optional[Any] = None,) -> str:

        now        = datetime.now(timezone.utc).isoformat()
        report     = self._build_report(
            plan, execution_result, signal_bag,
            confidence_score, job_id, now, validation_report
        )

        out_dir    = Path(self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path  = out_dir / f"{job_id}_report.json"
        self._write_json(report, json_path)

        if self._cfg.include_markdown:
            md_path = out_dir / f"{job_id}_report.md"
            self._write_markdown(report, md_path)

        return str(json_path)

    def _build_report(self,plan: Any,execution_result: Dict[str, Any],signal_bag: Any,confidence_score: float,job_id: str,generated_at: str,validation_report: Optional[Any],) -> Dict:
        report: Dict[str, Any] = {
            "schema_version":  "1.0.0",
            "job_id":          job_id,
            "generated_at":    generated_at,
        }

        report["execution_metrics"]   = self._execution_section(execution_result)
        report["decision_plan"]       = self._plan_section(plan)
        report["signal_summary"]      = self._signal_section(signal_bag)
        report["confidence_summary"]  = self._confidence_section(confidence_score)
        report["validation_summary"]  = self._validation_section(validation_report)

        if self._budget_controller:
            try:
                report["budget_summary"] = self._budget_controller.get_usage_summary(job_id)
            except KeyError:
                report["budget_summary"] = {"note": "budget record already released"}

        if self._confidence_tracker:
            try:
                report["confidence_timeline"] = self._confidence_tracker.get_timeline(job_id)
            except Exception:
                report["confidence_timeline"] = []

        return report

    def _execution_section(self, result: Dict[str, Any]) -> Dict:
        return {
            "rows_processed":       result.get("rows_processed"),
            "actions_applied":      result.get("actions_applied"),
            "columns_added":        result.get("columns_added", []),
            "columns_dropped":      result.get("columns_dropped", []),
            "columns_modified":     result.get("columns_modified", []),
            "execution_duration_s": result.get("duration_seconds"),
            "errors":               result.get("errors", []),
        }

    def _plan_section(self, plan: Any) -> Dict:
        if plan is None:
            return {"note": "no plan generated"}

        if hasattr(plan, "model_dump"):
            return plan.model_dump()

        if isinstance(plan, dict):
            return plan

        return {"raw": str(plan)}

    def _signal_section(self, signal_bag: Any) -> Dict:
        if signal_bag is None:
            return {"signals": [], "total": 0}

        signals = signal_bag if isinstance(signal_bag, list) else list(signal_bag)
        top_k   = signals[:self._cfg.top_k_signals]

        return {
            "total":          len(signals),
            "shown":          len(top_k),
            "top_k_signals":  [
                self._serialise_signal(s) for s in top_k
            ],
        }


    @staticmethod
    def _serialise_signal(signal: Any) -> Dict:
        if hasattr(signal, "model_dump"):
            return signal.model_dump()
        if isinstance(signal, dict):
            return signal
        return {"raw": str(signal)}

    def _confidence_section(self, final_score: float) -> Dict:
        label = (
            "HIGH"   if final_score >= 0.80 else
            "MEDIUM" if final_score >= 0.50 else
            "LOW"
        )
        return {
            "final_score":  round(final_score, 4),
            "label":        label,
        }

    @staticmethod
    def _validation_section(report: Optional[Any]) -> Dict:
        if report is None:
            return {"status": "not_run"}

        if hasattr(report, "passed"):
            return {
                "status":     "PASSED" if report.passed else "FAILED",
                "violations": getattr(report, "violations", []),
            }

        if isinstance(report, dict):
            return report

        return {"raw": str(report)}



    def _write_json(self, report: Dict, path: Path) -> None:
        path.write_text(
            json.dumps(report, indent=self._cfg.indent, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _write_markdown(report: Dict, path: Path) -> None:
        lines: List[str] = []

        def h(level: int, text: str) -> None:
            lines.append(f"\n{'#' * level} {text}\n")

        def kv(key: str, val: Any) -> None:
            lines.append(f"- **{key}:** {val}")

        h(1, f"dataprep_agent — Decision Report")
        kv("Job ID",       report.get("job_id"))
        kv("Generated at", report.get("generated_at"))
        kv("Schema",       report.get("schema_version"))

        em = report.get("execution_metrics", {})
        if em:
            h(2, "Execution Metrics")
            kv("Rows processed",   em.get("rows_processed"))
            kv("Actions applied",  em.get("actions_applied"))
            kv("Duration (s)",     em.get("execution_duration_s"))
            if em.get("columns_dropped"):
                kv("Columns dropped", ", ".join(em["columns_dropped"]))
            if em.get("columns_added"):
                kv("Columns added", ", ".join(em["columns_added"]))
            if em.get("errors"):
                h(3, "Errors")
                for err in em["errors"]:
                    lines.append(f"- `{err}`")
 
        # Confidence
        cs = report.get("confidence_summary", {})
        if cs:
            h(2, "Confidence")
            kv("Final score", cs.get("final_score"))
            kv("Label",       cs.get("label"))

        vs = report.get("validation_summary", {})
        if vs:
            h(2, "Validation")
            kv("Status", vs.get("status"))
            if vs.get("violations"):
                h(3, "Violations")
                for v in vs["violations"]:
                    lines.append(f"- {v}")

        ss = report.get("signal_summary", {})
        if ss:
            h(2, f"Top Signals (showing {ss.get('shown', 0)} of {ss.get('total', 0)})")
            for sig in ss.get("top_k_signals", []):
                col      = sig.get("column", "—")
                sig_type = sig.get("signal_type", sig.get("type", "—"))
                severity = sig.get("severity", "—")
                score    = sig.get("priority_score", sig.get("score", "—"))
                lines.append(f"- `{col}` · {sig_type} · severity={severity} · score={score}")

        bs = report.get("budget_summary", {})
        if bs and "total_tokens" in bs:
            h(2, "Budget Usage")
            kv("Total tokens", bs.get("total_tokens"))
            kv("Cost (USD)",   bs.get("cost_usd"))
            kv("Elapsed (s)",  bs.get("elapsed_seconds"))
            by_phase = bs.get("tokens_by_phase", {})
            if by_phase:
                h(3, "Tokens by Phase")
                for phase, tokens in sorted(by_phase.items()):
                    lines.append(f"- {phase}: {tokens:,}")


        dp = report.get("decision_plan", {})
        if dp and isinstance(dp, dict) and dp.get("actions"):
            h(2, "Decision Plan")
            for action in dp["actions"]:
                action_type = action.get("action_type", "—")
                column      = action.get("column", "—")
                rationale   = action.get("rationale", "")
                lines.append(f"- **{action_type}** on `{column}` — {rationale}")

        path.write_text("\n".join(lines), encoding="utf-8")