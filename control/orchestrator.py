from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, Field

class JobStatus(BaseModel):
    job_id:        str
    status:        str                  
    state:         Optional[str] = None  
    state_uuid:    Optional[str] = None
    created_at:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:    Optional[datetime] = None
    actions_applied: Optional[int] = None
    rows_processed:  Optional[int] = None
    error:         Optional[str]  = None
    progress:      float = 0.0          

class SubmitRequest(BaseModel):
    file_path: str
    base_dir: str
    target_column: Optional[str] = None
    job_id: Optional[str] = None
    max_iterations: int = 3
    budget_tokens: int = 30_000


class Orchestrator:
    def __init__(self,agent_factory: Callable[[Dict[str, Any]], Any],max_concurrent_jobs: int = 4,job_timeout_seconds: int = 1800,):
        self._agent_factory       = agent_factory
        self._job_timeout         = job_timeout_seconds
        self._semaphore           = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: Dict[str, JobStatus]          = {}
        self._tasks: Dict[str, asyncio.Task]      = {}
        self._started             = False


    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        pending = [t for t in self._tasks.values() if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._started = False

    
    async def submit_job(self, request: SubmitRequest) -> JobStatus:
        if not self._started:
            raise RuntimeError("Orchestrator has not been started — call await orchestrator.start()")
        job_id = request.job_id or str(uuid.uuid4())

        if job_id in self._jobs:
            raise ValueError(f"Job {job_id!r} already exists")
        status = JobStatus(job_id=job_id, status="QUEUED")
        self._jobs[job_id] = status

        job_context = {
            "job_id":         job_id,
            "file_path":      request.file_path,
            "target_column":  request.target_column,
            "max_iterations": request.max_iterations,
            "budget_tokens":  request.budget_tokens,
        }

        task = asyncio.create_task(
            self._run_job(job_id, job_context),
            name=f"job-{job_id}",
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda t: self._on_task_done(job_id, t))
        return status
    async def get_status(self, job_id: str) -> JobStatus:
        status = self._jobs.get(job_id)
        if status is None:
            raise KeyError(f"Unknown job: {job_id!r}")
        return status

    async def cancel_job(self, job_id: str) -> JobStatus:
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            self._update_status(job_id, status="CANCELLED")
        return self._jobs[job_id]
    async def list_jobs(self) -> list[JobStatus]:
        return list(self._jobs.values())
    
    async def _run_job(self, job_id: str, job_context: Dict[str, Any]) -> None:
        async with self._semaphore:
            self._update_status(job_id, status="RUNNING")
            agent = None  # Initialize to None

            try:
                agent = self._agent_factory(job_context)

                # Add callback for live state updates
                def update_state(state_name: str):
                    self._update_status(job_id, state=state_name)

                agent._update_orchestrator_callback = update_state

                # Run the agent (returns None)
                await asyncio.wait_for(
                    agent.run(),
                    timeout=self._job_timeout,
                )

                # Extract final state from agent after completion
                final_state = agent.state.name if agent else "UNKNOWN"

                # Update status based on final state
                self._update_status(
                    job_id,
                    status="COMPLETED" if final_state == "COMPLETED" else "FAILED",
                    state=final_state,
                    state_uuid=getattr(agent, 'state_uuid', None),
                    actions_applied=len(agent.optimized_plan.actions) if (agent and agent.optimized_plan) else 0,
                    rows_processed=agent.execution_result.get("rows_processed") if (agent and agent.execution_result) else None,
                    progress=1.0,
                )

            except asyncio.TimeoutError:
                # Get current state if available
                current_state = agent.state.name if agent else "UNKNOWN"
                self._update_status(
                    job_id,
                    status="FAILED",
                    state=current_state,
                    error=f"Job timed out after {self._job_timeout}s",
                )
            except asyncio.CancelledError:
                current_state = agent.state.name if agent else "CANCELLED"
                self._update_status(
                    job_id, 
                    status="CANCELLED",
                    state=current_state
                )
                raise
            except Exception as exc:
    
                current_state = agent.state.name if agent else "UNKNOWN"
                self._update_status(
                    job_id, 
                    status="FAILED",
                    state=current_state,
                    error=str(exc)
                )

    def _on_task_done(self, job_id: str, task: asyncio.Task) -> None:
        """Callback to catch any exception that slipped through _run_job."""
        if task.cancelled():
            self._update_status(job_id, status="CANCELLED")
        elif task.exception():
            self._update_status(job_id, status="FAILED", error=str(task.exception()))

    def _update_status(self, job_id: str, **fields) -> None:
        status = self._jobs.get(job_id)
        if status is None:
            return
        for key, val in fields.items():
            if val is not None:
                setattr(status, key, val)
        status.updated_at = datetime.now(timezone.utc)


    
