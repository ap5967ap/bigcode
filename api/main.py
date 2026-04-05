from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from routing.router import NightSafeRouter


class RouteRequest(BaseModel):
    origin: tuple[float, float] = Field(..., min_length=2, max_length=2)
    destination: tuple[float, float] = Field(..., min_length=2, max_length=2)
    travel_mode: str
    hour_of_day: int = Field(..., ge=0, le=23)
    is_female: bool = False
    destination_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading NightSafe graph, classifier, SHAP explanations, and RL agents")
    app.state.router = NightSafeRouter()
    yield
    logger.info("NightSafe API shutdown complete")


app = FastAPI(title="NightSafe Routes API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if request.url.path == "/route" and latency_ms > 500.0:
        logger.warning(f"Slow route request: path={request.url.path} latency_ms={latency_ms:.2f}")
    else:
        logger.info(
            f"Request complete method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={latency_ms:.2f}"
        )
    response.headers["X-Process-Time-ms"] = f"{latency_ms:.2f}"
    return response


@app.get("/health")
async def health() -> dict[str, Any]:
    router: NightSafeRouter = app.state.router
    return {
        "status": "ok",
        "graph_nodes": router.graph.number_of_nodes(),
        "graph_edges": router.graph.number_of_edges(),
        "service_area_bounds": router.service_area_bounds,
    }


@app.post("/route")
async def route(request: RouteRequest) -> dict[str, Any]:
    router: NightSafeRouter = app.state.router
    try:
        return router.route(
            origin_coords=list(request.origin),
            destination_coords=list(request.destination),
            user_context={
                "travel_mode": request.travel_mode,
                "hour_of_day": request.hour_of_day,
                "is_female": request.is_female,
                "destination_type": request.destination_type,
                "query_day_type": 0,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/segment/{edge_id}/explain")
async def explain_segment(edge_id: str) -> dict[str, Any]:
    router: NightSafeRouter = app.state.router
    try:
        return router.explain_segment(edge_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
