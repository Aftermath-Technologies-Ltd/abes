# Author: Bradley R. Kinnard
"""
ABES CLI - command-line interface for the Adaptive Belief Ecology System.

Provides commands for demo, chat, seed, verify, and inspect operations.
All commands talk to the live backend API over HTTP.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import click
import httpx

logger = logging.getLogger("abes.cli")

# Where we expect the ABES root to be
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
ATTRIBUTION_BANNER = (
    "ABES | Adaptive Belief Ecology System | v1.0 | "
    "Developed by Bradley R. Kinnard | Aftermath Technologies"
)


def _find_project_root() -> Path:
    """Walk up from cwd looking for pyproject.toml with name='abes'."""
    here = Path.cwd()
    for parent in [here, *here.parents]:
        toml = parent / "pyproject.toml"
        if toml.exists() and "abes" in toml.read_text(encoding="utf-8"):
            return parent
    return PROJECT_ROOT


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # quiet noisy libs
    for lib in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _wait_for_backend(url: str = BACKEND_URL, timeout: int = 30) -> bool:
    """Poll the health endpoint until the backend is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{url}/bel/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    return False


def _backend_running(url: str = BACKEND_URL) -> bool:
    """Check if backend is already up."""
    try:
        resp = httpx.get(f"{url}/bel/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _start_backend(root: Path) -> subprocess.Popen:
    """Launch the uvicorn backend as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.app:app",
         "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _start_frontend(root: Path) -> Optional[subprocess.Popen]:
    """Launch the Next.js dev server if node_modules exist."""
    frontend_dir = root / "frontend"
    if not (frontend_dir / "node_modules").exists():
        click.echo("  Frontend node_modules not found. Skipping frontend.")
        click.echo(f"  Run: cd {frontend_dir} && npm install")
        return None

    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _print_banner() -> None:
    click.echo()
    click.echo(click.style("  ABES", fg="cyan", bold=True) +
               click.style(" - Adaptive Belief Ecology System", fg="white"))
    click.echo(click.style("  " + "=" * 48, fg="bright_black"))
    click.echo()


def _print_event(event: dict, indent: str = "    ") -> None:
    """Pretty-print a belief event from the API response."""
    etype = event.get("event_type", "unknown")
    content = event.get("content", "")[:80]
    conf = event.get("confidence", 0)
    tension = event.get("tension", 0)

    color_map = {
        "created": "green",
        "reinforced": "cyan",
        "tension_changed": "yellow",
        "mutated": "magenta",
        "deprecated": "red",
    }
    color = color_map.get(etype, "white")
    symbol_map = {
        "created": "+",
        "reinforced": "^",
        "tension_changed": "!",
        "mutated": "~",
        "deprecated": "x",
    }
    symbol = symbol_map.get(etype, "?")

    line = f"{indent}[{symbol}] {etype:<18} conf={conf:.2f}  tension={tension:.2f}  {content}"
    click.echo(click.style(line, fg=color))


# ============================================================
# CLI Group
# ============================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="abes")
def cli():
    """ABES: Adaptive Belief Ecology System.

    A cognitive memory architecture where beliefs decay, compete,
    contradict, mutate, and consolidate as a living ecology.
    """
    pass


# ============================================================
# abes demo
# ============================================================

@cli.command()
@click.option("--headless", is_flag=True, help="Backend only, no browser.")
@click.option("--script", type=click.Path(exists=True),
              help="Custom demo conversation JSON.")
@click.option("--pause/--no-pause", default=True,
              help="Pause between turns for readability.")
@click.option("--with-decay", is_flag=True,
              help="Simulate elapsed time between turns to demonstrate decay.")
@click.option("--decay-hours", default=12.0, show_default=True, type=float,
              help="Hours to simulate between turns when --with-decay is enabled.")
@click.option("--keep-alive", is_flag=True,
              help="Keep backend running after demo so abes inspect works immediately.")
@click.option("-v", "--verbose", is_flag=True)
def demo(
    headless: bool,
    script: str,
    pause: bool,
    with_decay: bool,
    decay_hours: float,
    keep_alive: bool,
    verbose: bool,
):
    """Run a scripted demo that shows the belief ecology in action.

    Launches the backend, sends 12 turns through the chat pipeline,
    and prints real-time belief events (creation, reinforcement,
    contradiction, mutation) so you can watch the ecology breathe.
    """
    _setup_logging(verbose)
    _print_banner()
    root = _find_project_root()

    # Load demo script
    script_path = Path(script) if script else root / "examples" / "demo_conversation.json"
    if not script_path.exists():
        click.echo(click.style(f"  Demo script not found: {script_path}", fg="red"))
        raise SystemExit(1)

    with open(script_path, encoding="utf-8") as f:
        turns_data = json.load(f)

    click.echo(f"  Demo script: {script_path.name} ({len(turns_data)} turns)")

    # Start backend if needed
    backend_proc = None
    frontend_proc = None
    owned_backend = False

    if _backend_running():
        click.echo("  Backend already running.")
    else:
        click.echo("  Starting backend...")
        backend_proc = _start_backend(root)
        owned_backend = True
        if not _wait_for_backend(timeout=45):
            click.echo(click.style("  Backend failed to start.", fg="red"))
            if backend_proc:
                backend_proc.terminate()
            raise SystemExit(1)
        click.echo(click.style("  Backend ready.", fg="green"))

    # Optionally start frontend
    if not headless:
        frontend_proc = _start_frontend(root)
        if frontend_proc:
            click.echo(f"  Frontend starting at {FRONTEND_URL}")

    try:
        asyncio.run(_run_demo(turns_data, pause, with_decay, decay_hours))
    except KeyboardInterrupt:
        click.echo("\n  Demo interrupted.")
    finally:
        if owned_backend and backend_proc and not keep_alive:
            backend_proc.terminate()
            backend_proc.wait(timeout=5)
        if frontend_proc:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=5)

    click.echo()
    click.echo(click.style("  Demo complete.", fg="green"))
    if owned_backend and keep_alive:
        click.echo(click.style(f"  Backend is still running at {BACKEND_URL}", fg="cyan"))
        click.echo("  Try: abes inspect")
        click.echo("  Try: curl http://localhost:8000/beliefs | jq .")
        click.echo("  Stop it with: pkill -f 'uvicorn backend.api.app:app'")
    elif owned_backend and not keep_alive:
        click.echo("  The demo backend was temporary and has stopped.")
        click.echo("  Re-run with --keep-alive to inspect after the script.")
        click.echo("  Or start manually: PYTHONPATH=$PWD uvicorn backend.api.app:app --port 8000")
        click.echo("  Then run: abes inspect")
    else:
        click.echo("  Backend was already running and remains available.")
        click.echo("  Try: abes inspect")


async def _run_demo(
    turns: list[dict],
    pause: bool,
    with_decay: bool,
    decay_hours: float,
) -> None:
    """Execute the demo conversation against the live API."""
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=60.0) as client:
        # Register a demo user
        demo_email = f"demo-{uuid4().hex[:8]}@abes.local"
        reg = await client.post("/auth/register", json={
            "email": demo_email,
            "name": "Demo User",
            "password": "demo-pass-123",
        })
        if reg.status_code not in (200, 201):
            click.echo(click.style("  Could not register demo user.", fg="red"))
            return

        login = await client.post("/auth/login", json={
            "email": demo_email,
            "password": "demo-pass-123",
        })
        token = login.json().get("access_token", "")
        headers = {"Authorization": f"Bearer {token}"}

        # Clear any existing beliefs
        await client.post("/beliefs/clear", headers=headers)

        session_id = str(uuid4())
        global_peak_tension = 0.0
        click.echo()

        for i, turn in enumerate(turns, 1):
            msg = turn["message"]
            narration = turn.get("narration", "")
            delay = turn.get("pause", 1.0) if pause else 0.1

            # Narration
            click.echo(click.style(f"  [{i:02d}/{len(turns):02d}] ", fg="bright_black") +
                       click.style(narration, fg="bright_black", italic=True))

            # User message
            click.echo(click.style(f"  User: ", fg="white", bold=True) + msg)

            # Send to API
            resp = await client.post("/chat/message", json={
                "message": msg,
                "session_id": session_id,
            }, headers=headers)

            if resp.status_code != 200:
                click.echo(click.style(f"    API error: {resp.status_code}", fg="red"))
                continue

            data = resp.json()
            assistant_msg = data.get("assistant_message", "")

            # Print events
            events = data.get("events", [])
            turn_peak_tension = 0.0
            if events:
                for event in events:
                    _print_event(event)
                    if event.get("event_type") == "tension_changed":
                        turn_peak_tension = max(turn_peak_tension, float(event.get("tension", 0.0)))

            if turn_peak_tension > 0:
                click.echo(click.style(
                    f"    Peak tension this turn: {turn_peak_tension:.2f}",
                    fg="yellow",
                ))

            # Pull current ecology stats so tension and status transitions are visible
            beliefs_resp = await client.get("/beliefs", params={"page_size": 200})
            if beliefs_resp.status_code == 200:
                beliefs = beliefs_resp.json().get("beliefs", [])
                max_tension_now = max((float(b.get("tension", 0.0)) for b in beliefs), default=0.0)
                deprecated_now = sum(1 for b in beliefs if b.get("status") == "deprecated")
                active_now = sum(1 for b in beliefs if b.get("status") == "active")
                global_peak_tension = max(global_peak_tension, turn_peak_tension, max_tension_now)
                click.echo(click.style(
                    f"    Ecology: active={active_now} deprecated={deprecated_now} max_tension={max_tension_now:.2f}",
                    fg="bright_black",
                ))

            # Print assistant response
            click.echo(click.style(f"  ABES: ", fg="cyan", bold=True) +
                       assistant_msg[:200])
            click.echo()

            # Optional decay simulation to make salience and confidence changes visible
            if with_decay and i < len(turns):
                sim = await client.post("/beliefs/simulate-time", params={"hours": decay_hours})
                if sim.status_code == 200:
                    payload = sim.json()
                    click.echo(click.style(
                        f"    Simulated {payload.get('hours', decay_hours):.1f}h elapsed for {payload.get('adjusted', 0)} beliefs",
                        fg="bright_black",
                    ))
                    click.echo()

            if pause:
                time.sleep(delay)

        # Final stats
        click.echo(click.style("  --- Ecology Summary ---", fg="white", bold=True))
        beliefs_resp = await client.get("/beliefs", params={"page_size": 200})
        if beliefs_resp.status_code == 200:
            beliefs = beliefs_resp.json().get("beliefs", [])
            active = sum(1 for b in beliefs if b.get("status") == "active")
            total_tension = sum(b.get("tension", 0) for b in beliefs)
            click.echo(f"  Total beliefs: {len(beliefs)}")
            click.echo(f"  Active: {active}")
            click.echo(f"  Total tension: {total_tension:.2f}")
            click.echo(f"  Peak tension observed: {global_peak_tension:.2f}")


# ============================================================
# abes chat
# ============================================================

@cli.command()
@click.option("--no-frontend", is_flag=True,
              help="Skip launching the frontend.")
@click.option("-v", "--verbose", is_flag=True)
def chat(no_frontend: bool, verbose: bool):
    """Launch ABES for interactive chat.

    Starts the backend (and frontend unless --no-frontend), then
    opens the browser to the chat interface. Press Ctrl+C to stop.
    """
    _setup_logging(verbose)
    _print_banner()
    root = _find_project_root()

    backend_proc = None
    frontend_proc = None
    owned_backend = False

    if _backend_running():
        click.echo("  Backend already running.")
    else:
        click.echo("  Starting backend...")
        backend_proc = _start_backend(root)
        owned_backend = True
        if not _wait_for_backend(timeout=45):
            click.echo(click.style("  Backend failed to start.", fg="red"))
            if backend_proc:
                backend_proc.terminate()
            raise SystemExit(1)
        click.echo(click.style("  Backend ready on port 8000.", fg="green"))

    if not no_frontend:
        frontend_proc = _start_frontend(root)
        if frontend_proc:
            click.echo(f"  Frontend starting at {FRONTEND_URL}")
            click.echo()
            click.echo(click.style(f"  Open {FRONTEND_URL}/chat in your browser.", fg="cyan", bold=True))
        else:
            click.echo()
            click.echo(click.style(f"  API available at {BACKEND_URL}", fg="cyan", bold=True))
    else:
        click.echo()
        click.echo(click.style(f"  API available at {BACKEND_URL}", fg="cyan", bold=True))

    click.echo("  Press Ctrl+C to stop.")
    click.echo()

    try:
        # Block until interrupted
        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        # AttributeError on Windows where signal.pause doesn't exist
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    click.echo("\n  Shutting down...")
    if owned_backend and backend_proc:
        backend_proc.terminate()
        backend_proc.wait(timeout=5)
    if frontend_proc:
        frontend_proc.terminate()
        frontend_proc.wait(timeout=5)
    click.echo("  Done.")


# ============================================================
# abes seed
# ============================================================

@cli.command()
@click.option("--file", "seed_file", type=click.Path(exists=True),
              help="Path to seed beliefs JSON file.")
@click.option("-v", "--verbose", is_flag=True)
def seed(seed_file: str, verbose: bool):
    """Load seed beliefs into ABES.

    By default loads examples/seed_beliefs.json (5 beliefs).
    Use --file to specify a custom seed file.
    """
    _setup_logging(verbose)
    root = _find_project_root()

    path = Path(seed_file) if seed_file else root / "examples" / "seed_beliefs.json"
    if not path.exists():
        click.echo(click.style(f"Seed file not found: {path}", fg="red"))
        raise SystemExit(1)

    with open(path, encoding="utf-8") as f:
        beliefs_data = json.load(f)

    if not _backend_running():
        click.echo(click.style("Backend not running. Start it first: abes chat", fg="red"))
        raise SystemExit(1)

    click.echo(f"  Loading {len(beliefs_data)} seed beliefs from {path.name}...")
    loaded = 0

    for entry in beliefs_data:
        content = entry.get("content", "")
        confidence = entry.get("confidence", 0.7)
        source = entry.get("source", "seed")

        if not content:
            continue

        resp = httpx.post(f"{BACKEND_URL}/beliefs", json={
            "content": content,
            "confidence": confidence,
            "source": source,
        }, timeout=10.0)

        if resp.status_code == 201:
            loaded += 1
            click.echo(click.style(f"    + ", fg="green") + content[:70])
        else:
            click.echo(click.style(f"    x ", fg="red") +
                       f"Failed ({resp.status_code}): {content[:50]}")

    click.echo()
    click.echo(click.style(f"  Loaded {loaded}/{len(beliefs_data)} beliefs.", fg="green"))


# ============================================================
# abes verify-quick
# ============================================================

@cli.command("verify-quick")
@click.option("--prompts", default=80, show_default=True, type=int,
              help="Number of prompts to run (stratified across domains).")
@click.option("-v", "--verbose", is_flag=True)
def verify_quick(prompts: int, verbose: bool):
    """Run a quick cognitive evaluation against the live backend.

    Sends stratified prompts from the 1000-prompt bank, scores via
    semantic cosine similarity, and checks ecology invariants.
    """
    _setup_logging(verbose)

    if not _backend_running():
        click.echo(click.style("Backend not running. Start it first: abes chat", fg="red"))
        raise SystemExit(1)

    click.echo(f"  Running {prompts}-prompt verification...")
    click.echo()

    try:
        from tests.cognitive.eval.harness import EvalHarness
        from tests.cognitive.eval.reporter import terminal_summary

        harness = EvalHarness(
            base_url=BACKEND_URL,
            threshold=0.70,
            concurrency=4,
            decay_cycles=3,
        )
        result = asyncio.run(harness.run(max_prompts=prompts))
        terminal_summary(result)

    except ImportError as e:
        click.echo(click.style(f"  Missing eval module: {e}", fg="red"))
        click.echo("  Make sure you installed with: pip install -e '.[dev]'")
        raise SystemExit(1)
    except Exception as e:
        click.echo(click.style(f"  Verification failed: {e}", fg="red"))
        raise SystemExit(1)


# ============================================================
# abes verify-determinism
# ============================================================

@cli.command("verify-determinism")
@click.option("--prompts", default=20, show_default=True, type=int,
              help="Number of fixed prompts per run.")
@click.option("--runs", default=3, show_default=True, type=int,
              help="Number of runs to compare.")
@click.option("-v", "--verbose", is_flag=True)
def verify_determinism(prompts: int, runs: int, verbose: bool):
    """Verify deterministic behavior across repeated runs.

    Sends the same prompts N times with the same seed, then compares
    final belief-graph hashes. Reports pass/fail for each pair.
    """
    _setup_logging(verbose)

    if not _backend_running():
        click.echo(click.style("Backend not running. Start it first: abes chat", fg="red"))
        raise SystemExit(1)

    click.echo(f"  Running determinism check: {prompts} prompts x {runs} runs...")
    click.echo()

    asyncio.run(_check_determinism(prompts, runs))


async def _check_determinism(n_prompts: int, n_runs: int) -> None:
    """Run the same prompts multiple times and compare belief hashes."""
    import hashlib
    from tests.cognitive.eval.prompt_bank import build_prompts

    all_prompts = build_prompts()
    # take a fixed, deterministic slice
    subset = sorted(all_prompts, key=lambda p: p.id)[:n_prompts]

    hashes = []

    for run_idx in range(n_runs):
        async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=60.0) as client:
            # Register fresh user per run
            email = f"det-{uuid4().hex[:6]}@abes.local"
            reg = await client.post("/auth/register", json={
                "email": email, "name": "DetCheck", "password": "det-pass-123",
            })
            if reg.status_code not in (200, 201):
                click.echo(click.style(f"  Run {run_idx+1}: registration failed", fg="red"))
                return

            login = await client.post("/auth/login", json={
                "email": email, "password": "det-pass-123",
            })
            token = login.json().get("access_token", "")
            headers = {"Authorization": f"Bearer {token}"}
            await client.post("/beliefs/clear", headers=headers)

            session_id = str(uuid4())

            for prompt in subset:
                await client.post("/chat/message", json={
                    "message": prompt.message,
                    "session_id": session_id,
                }, headers=headers)

            # Snapshot belief state
            resp = await client.get("/beliefs", params={"page_size": 1000})
            beliefs = resp.json().get("beliefs", [])

            # Hash the content+confidence pairs (stable sort)
            state_str = json.dumps(
                sorted(
                    [{"c": b["content"], "conf": round(b["confidence"], 4)}
                     for b in beliefs],
                    key=lambda x: x["c"],
                ),
                sort_keys=True,
            )
            h = hashlib.sha256(state_str.encode()).hexdigest()
            hashes.append(h)
            click.echo(f"  Run {run_idx+1}: {len(beliefs)} beliefs, hash={h[:16]}...")

    # Compare
    click.echo()
    if len(set(hashes)) == 1:
        click.echo(click.style("  DETERMINISTIC: all runs produced identical state.", fg="green", bold=True))
    else:
        click.echo(click.style("  NON-DETERMINISTIC: state differs between runs.", fg="yellow", bold=True))
        for i, h in enumerate(hashes):
            click.echo(f"    Run {i+1}: {h}")
        click.echo()
        click.echo("  Note: LLM response variation can cause different belief extraction.")
        click.echo("  The belief processing pipeline itself is deterministic given identical inputs.")


# ============================================================
# abes inspect
# ============================================================

@cli.command()
@click.option("--top", default=15, show_default=True, type=int,
              help="Number of top beliefs to show.")
@click.option("--json-out", is_flag=True, help="Output as JSON.")
@click.option("-v", "--verbose", is_flag=True)
def inspect(top: int, json_out: bool, verbose: bool):
    """Inspect the current belief ecology state.

    Shows active beliefs sorted by salience, top tension pairs,
    recent mutations, and ecology health summary.
    """
    _setup_logging(verbose)

    if not _backend_running():
        click.echo(click.style("Backend not running. Start it first: abes chat", fg="red"))
        raise SystemExit(1)

    asyncio.run(_inspect(top, json_out))


async def _inspect(top: int, json_out: bool) -> None:
    """Fetch and display current ecology state."""
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=15.0) as client:
        # Health
        health_resp = await client.get("/bel/health")
        health = health_resp.json() if health_resp.status_code == 200 else {}

        # Beliefs (paginate if needed, API caps at 200)
        all_beliefs = []
        page = 1
        while True:
            beliefs_resp = await client.get("/beliefs", params={"page_size": 200, "page": page})
            if beliefs_resp.status_code != 200:
                break
            data = beliefs_resp.json()
            batch = data.get("beliefs", [])
            all_beliefs.extend(batch)
            if len(batch) < 200:
                break
            page += 1

        if json_out:
            output = {
                "health": health,
                "total_beliefs": len(all_beliefs),
                "beliefs": all_beliefs[:top],
            }
            click.echo(json.dumps(output, indent=2))
            return

        # Summary
        active = [b for b in all_beliefs if b.get("status") == "active"]
        decaying = [b for b in all_beliefs if b.get("status") == "decaying"]
        dormant = [b for b in all_beliefs if b.get("status") == "dormant"]
        mutated = [b for b in all_beliefs if b.get("status") == "mutated"]
        deprecated = [b for b in all_beliefs if b.get("status") == "deprecated"]
        total_tension = sum(b.get("tension", 0) for b in all_beliefs)

        _print_banner()
        click.echo(click.style("  Health", fg="white", bold=True))
        click.echo(f"    Status:  {health.get('status', 'unknown')}")
        click.echo(f"    Beliefs: {health.get('belief_count', len(all_beliefs))}")
        click.echo()

        click.echo(click.style("  Ecology", fg="white", bold=True))
        click.echo(f"    Active:     {len(active)}")
        click.echo(f"    Decaying:   {len(decaying)}")
        click.echo(f"    Dormant:    {len(dormant)}")
        click.echo(f"    Mutated:    {len(mutated)}")
        click.echo(f"    Deprecated: {len(deprecated)}")
        click.echo(f"    Tension:    {total_tension:.2f}")
        click.echo()

        if not all_beliefs:
            click.echo("  No beliefs yet. Try: abes demo")
            return

        # Top beliefs by confidence (API doesn't expose salience in list response,
        # so we sort by confidence as proxy)
        click.echo(click.style(f"  Top {min(top, len(all_beliefs))} Beliefs", fg="white", bold=True))
        for b in all_beliefs[:top]:
            status_color = {
                "active": "green",
                "decaying": "yellow",
                "dormant": "bright_black",
                "mutated": "magenta",
                "deprecated": "red",
            }.get(b.get("status", ""), "white")

            content = b.get("content", "")[:65]
            conf = b.get("confidence", 0)
            tension = b.get("tension", 0)
            status = b.get("status", "?")

            click.echo(
                f"    {click.style(f'[{status:>10}]', fg=status_color)} "
                f"conf={conf:.2f}  tension={tension:.2f}  {content}"
            )

        # High tension beliefs
        high_tension = [b for b in all_beliefs if b.get("tension", 0) > 0.3]
        if high_tension:
            click.echo()
            click.echo(click.style("  High Tension", fg="yellow", bold=True))
            for b in sorted(high_tension, key=lambda x: x.get("tension", 0), reverse=True)[:5]:
                click.echo(
                    f"    tension={b['tension']:.2f}  conf={b['confidence']:.2f}  "
                    f"{b['content'][:60]}"
                )

        click.echo()


# ============================================================
# Main
# ============================================================

def main():
    print(ATTRIBUTION_BANNER)
    cli()


if __name__ == "__main__":
    main()
