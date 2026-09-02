"""
Starting, stopping and inspecting the three applications - safely.

The safety rule this module exists to enforce:

    NEVER stop a process just because it has the pid we remember, and
    NEVER stop a process just because it occupies a port we wanted.

Pids get reused by the operating system. A port we expect may be held by
something entirely unrelated - another project, another user's server, a
different app that happened to grab 7860 first. So before this module
stops anything it checks four things against what it recorded when it
started the service:

    pid, process creation time, executable path, and command line.

All four must match. If any disagree, we refuse to act and say why. The
worst outcome for a beginner-friendly tool is killing something the user
cared about, so the failure mode here is always "do nothing and explain".

This module contains no user interface. manager.py prints things; this
decides things, which is what makes it testable without launching
anything real.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import config

# Creation time comes back as a float from the OS and can wobble slightly
# between reads; a second of tolerance is far tighter than the window in
# which a pid could realistically be recycled into a look-alike process.
CREATE_TIME_TOLERANCE_SECONDS = 1.0


class ProcessFacts(NamedTuple):
    """What we can observe about a running process, right now."""
    pid: int
    exe: str
    cmdline: str
    create_time: float


class ProbeUnavailable(RuntimeError):
    """psutil is missing, so we cannot verify process identity.

    We raise rather than guess: unverified stopping is exactly what this
    module is designed to prevent.
    """


class ProcessProbe:
    """Thin wrapper over psutil.

    Isolated behind one class so tests can substitute a fake and exercise
    every start/stop decision without a single real process.
    """

    def __init__(self):
        try:
            import psutil  # noqa: F401  (checked here, imported where used)
        except ImportError as error:
            raise ProbeUnavailable(
                "psutil is not installed for this Python, so process "
                "identity cannot be verified and it is not safe to stop "
                "anything automatically."
            ) from error

    def facts(self, pid):
        """Everything we know about one pid, or None if it isn't running."""
        import psutil

        try:
            proc = psutil.Process(pid)
            with proc.oneshot():
                return ProcessFacts(
                    pid=proc.pid,
                    exe=proc.exe() or "",
                    cmdline=" ".join(proc.cmdline() or []),
                    create_time=proc.create_time(),
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return None

    def children(self, pid):
        """Descendant pids. Empty if the process is gone."""
        import psutil

        try:
            return [child.pid for child in psutil.Process(pid).children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return []

    def listening_pid(self, port):
        """Which pid is listening on a port, if any. Used to EXPLAIN a
        conflict - never as grounds to stop anything."""
        import psutil

        try:
            for conn in psutil.net_connections(kind="inet"):
                if (conn.status == psutil.CONN_LISTEN
                        and conn.laddr
                        and conn.laddr.port == port):
                    return conn.pid
        except (psutil.AccessDenied, OSError):
            return None
        return None

    def listening_ports(self, pids):
        """Every port the given pids are listening on.

        This is how we report the port a service ACTUALLY bound, rather
        than the port we hoped it would take. Forge in particular moves to
        the next free port when its preferred one is busy.
        """
        import psutil

        wanted, found = set(pids), set()
        try:
            for conn in psutil.net_connections(kind="inet"):
                if (conn.status == psutil.CONN_LISTEN
                        and conn.pid in wanted
                        and conn.laddr):
                    found.add(conn.laddr.port)
        except (psutil.AccessDenied, OSError):
            return []
        return sorted(found)

    def port_occupied(self, host, port):
        """Is anything at all listening there? Works without admin rights,
        unlike asking who owns the socket - so this can say "busy" even
        when it cannot say by whom."""
        import socket

        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def http_healthy(self, url, timeout=None):
        """Does an HTTP endpoint answer sensibly? Used to recognise an app
        that was started by hand rather than by this manager."""
        try:
            import requests
            response = requests.get(url, timeout=timeout or config.HEALTH_TIMEOUT)
            return bool(response.ok)
        except Exception:
            return False

    def http_generates(self, url, timeout=None):
        """Can the model actually produce a token?

        This is the check that matters. The web server in front of the
        model can keep answering "I'm here" long after the model backend
        behind it has died - which is exactly what happened once: /v1/models
        returned 200 for hours while every generation returned 500.

        One token, short timeout. True = healthy, False = the server
        answered with an error, None = no answer at all.
        """
        try:
            import requests
            response = requests.post(
                url,
                json={"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                timeout=timeout or config.HEALTH_TIMEOUT * 3,
            )
            return bool(response.ok)
        except Exception:
            return None

    def stop_tree(self, pid, timeout, expected_create_time=None):
        """Ask a verified process and its children to exit, then insist.

        Re-checks the creation time at the moment of acting: the pid was
        verified a few milliseconds ago, but a few milliseconds is enough
        for it to have died and been handed to something else.

        Children first: killing a launcher before its child orphans the
        child, which is how a server ends up still holding a port with
        nothing tracking it.
        """
        import psutil

        try:
            parent = psutil.Process(pid)
            if (expected_create_time is not None and
                    abs(parent.create_time() - expected_create_time)
                    > CREATE_TIME_TOLERANCE_SECONDS):
                return False
        except (psutil.NoSuchProcess, OSError):
            return True

        family = []
        try:
            family = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
        family.append(parent)

        for proc in family:
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

        _, alive = psutil.wait_procs(family, timeout=timeout)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        _, still_alive = psutil.wait_procs(alive, timeout=5)
        return not still_alive


@dataclass(frozen=True)
class ServiceSpec:
    """How to start one application, and how to recognise it afterwards."""
    key: str
    label: str
    workdir: Path
    argv: list
    # A distinctive fragment that must appear in the running command line.
    # This is what stops us mistaking an unrelated python.exe for ours.
    marker: str
    # The process executable must live under this directory.
    exe_root: Path
    expected_ports: tuple = ()
    health_url: Optional[str] = None
    # An endpoint that exercises the real work (a one-token generation).
    # health_url says "the web server is there"; this says "it works".
    generate_url: Optional[str] = None
    notes: str = ""
    env: dict = field(default_factory=dict)

    def available(self):
        """Whether this service can be started on this machine at all."""
        if not self.workdir.exists():
            return False, f"{self.workdir} does not exist"
        if not self.argv:
            return False, "no launch command is configured (interpreter not found)"
        for part in self.argv[:2]:
            target = Path(part)
            if target.is_absolute() and not target.exists():
                return False, f"{target} does not exist"
        return True, ""


def _python_entry(interpreter, script):
    """argv for running a script under a specific interpreter - or nothing
    if that interpreter is missing, so available() can say so."""
    return [str(interpreter), str(script)] if interpreter else []


def build_registry():
    """The three services, described in terms of config, not fixed paths.

    Each app is launched through its own bundled Python, running its real
    entry point, rather than through the .bat wrapper it ships with. The
    wrapper would make the process we track a cmd.exe living in
    C:\\Windows - which fails the very executable-path check that keeps
    stopping safe. Launching the interpreter directly means the pid we
    record IS the server.

    Built by a function rather than at import time so that tests (and any
    future 'reload config' action) get a fresh view.
    """
    specs = {}

    specs["chat_llm"] = ServiceSpec(
        key="chat_llm",
        label="chat-llm (text-generation-webui + model API)",
        workdir=config.CHAT_LLM_DIR,
        argv=_python_entry(config.chat_llm_python(), config.CHAT_LLM_DIR / "server.py"),
        marker="server.py",
        exe_root=config.CHAT_LLM_DIR,
        expected_ports=(config.CHAT_LLM_UI_PORT, config.CHAT_LLM_API_PORT),
        health_url=config.MODEL_LIST_URL,
        generate_url=config.MODEL_CHAT_URL,
        notes="Serves both the chat UI and the API the learning app talks to.",
    )

    specs["image_gen"] = ServiceSpec(
        key="image_gen",
        label="image-gen (Stable Diffusion WebUI Forge)",
        workdir=config.IMAGE_GEN_DIR,
        argv=_python_entry(config.image_gen_python(), config.IMAGE_GEN_DIR / "launch.py"),
        marker="launch.py",
        exe_root=config.IMAGE_GEN_DIR,
        expected_ports=(config.IMAGE_GEN_PORT,),
        notes="Picks the next free port if its preferred one is taken.",
    )

    gradio_exe = config.gradio_python()
    specs["learning_web"] = ServiceSpec(
        key="learning_web",
        label="learning web UI",
        workdir=config.LEARNING_DIR,
        argv=[str(gradio_exe), str(config.LEARNING_DIR / "webui.py")] if gradio_exe else [],
        marker="webui.py",
        exe_root=Path(gradio_exe).parent if gradio_exe else config.LEARNING_DIR,
        expected_ports=(config.LEARNING_WEB_PORT,),
        notes="Needs an interpreter with gradio; the bundled apps have one.",
    )
    return specs


# --- Remembering what we started --------------------------------------

def read_state(path=None):
    path = Path(path or config.SERVICE_STATE_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # A missing or corrupt state file means "we are not tracking
        # anything", which is safe: we simply refuse to stop things.
        return {}


def write_state(state, path=None):
    path = Path(path or config.SERVICE_STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def record_for(facts, spec):
    """The fingerprint we store, and later demand a match against."""
    return {
        "pid": facts.pid,
        "create_time": facts.create_time,
        "exe": facts.exe,
        "cmdline": facts.cmdline,
        "marker": spec.marker,
        "exe_root": str(spec.exe_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


# --- The safety check -------------------------------------------------

class Ownership(NamedTuple):
    owned: bool
    reason: str


def verify_owned(record, facts):
    """Is this running process really the one we started?

    Every check has to pass. The reasons are written to be shown to a
    person, because when this says no, the manager has to explain itself.
    """
    if not record:
        return Ownership(False, "we have no record of starting it")
    if facts is None:
        return Ownership(False, "that process is no longer running")

    if facts.pid != record.get("pid"):
        return Ownership(False, "pid does not match our record")

    recorded_start = record.get("create_time")
    if recorded_start is None:
        return Ownership(False, "our record has no creation time to check")
    if abs(facts.create_time - recorded_start) > CREATE_TIME_TOLERANCE_SECONDS:
        # The strongest check: same pid, different start time means the
        # OS recycled the number for a different program entirely.
        return Ownership(
            False,
            "the pid was reused by a different process (start times differ)",
        )

    exe_root = record.get("exe_root")
    if exe_root:
        try:
            Path(facts.exe).resolve().relative_to(Path(exe_root).resolve())
        except (ValueError, OSError):
            return Ownership(False, f"its program is not inside {exe_root}")

    marker = record.get("marker")
    if marker and marker not in facts.cmdline:
        return Ownership(False, f"its command line no longer contains {marker!r}")

    return Ownership(True, "pid, start time, program path and command line all match")


# --- Status -----------------------------------------------------------

# The distinct situations a service can be in. `running` collapses these
# to a yes/no for display; `phase` keeps the distinction that matters for
# deciding whether we are allowed to act.
PHASE_STOPPED = "STOPPED"
PHASE_MANAGED_RUNNING = "MANAGED_RUNNING"      # we started it; it works; we may stop it
PHASE_MANAGED_STARTING = "MANAGED_STARTING"    # we started it; not answering yet
PHASE_MANAGED_DEGRADED = "MANAGED_DEGRADED"    # we started it; web server up, generation broken
PHASE_UNMANAGED_RUNNING = "UNMANAGED_RUNNING"  # started by hand; hands off
PHASE_UNMANAGED_DEGRADED = "UNMANAGED_DEGRADED"
PHASE_EXITED = "EXITED"                        # we started it; it has since died
PHASE_STALE_RECORD = "STALE_RECORD"            # our pid now belongs to something else


class ServiceStatus(NamedTuple):
    key: str
    label: str
    running: bool
    pid: Optional[int]
    ports: list
    urls: list
    detail: str
    tracked: bool
    conflicts: list
    phase: str = PHASE_STOPPED
    # Running is not the same as working. False here means the process is
    # up but the thing it exists to do is failing.
    healthy: bool = True


def check_health(spec, probe):
    """(healthy, detail) for a service that is known to be running.

    Two layers, in order: does the web server answer (health_url), and
    does the real work succeed (generate_url). A probe that cannot do the
    deep check (older fakes, or a spec without one) is trusted on the
    shallow one.
    """
    if spec.health_url and not probe.http_healthy(spec.health_url):
        return False, "still starting - the web server is not answering yet"

    deep = getattr(probe, "http_generates", None)
    if spec.generate_url and deep is not None:
        result = deep(spec.generate_url)
        if result is False:
            return False, ("the web server answers but generating fails - "
                           "the model backend behind it is not working")
        if result is None:
            return False, "the API stopped answering mid-check"
    return True, ""


def port_conflicts(spec, probe, our_pids=()):
    """Expected ports that are held by something that is not our service.

    Reported, never acted on. Something else owning 7860 is information
    for the user, not permission to kill it.
    """
    conflicts = []
    for port in spec.expected_ports:
        holder = probe.listening_pid(port)
        if holder is None or holder in our_pids:
            continue
        facts = probe.facts(holder)
        who = Path(facts.exe).name if facts and facts.exe else "an unknown program"
        conflicts.append(f"port {port} is already used by {who} (pid {holder})")
    return conflicts


def occupied_ports(spec, probe):
    """Expected ports that have SOMETHING listening, owner known or not."""
    return [port for port in spec.expected_ports
            if probe.port_occupied(config.LOOPBACK, port)]


def unmanaged_running(spec, probe):
    """Is the app up even though we did not start it?

    Someone starting chat-llm by hand is normal, and the manager should
    say RUNNING rather than "stopped" - but only when the app's own health
    check answers, because a busy port on its own could be anything.
    """
    if not spec.health_url:
        return False
    if not occupied_ports(spec, probe):
        return False
    return probe.http_healthy(spec.health_url)


def status(spec, probe, state=None):
    """What is this service doing right now?"""
    state = read_state() if state is None else state
    record = state.get(spec.key)
    facts = probe.facts(record["pid"]) if record and record.get("pid") else None
    ownership = verify_owned(record, facts)

    if ownership.owned:
        children = probe.children(facts.pid)
        ports = probe.listening_ports([facts.pid] + children)
        healthy, why = check_health(spec, probe)
        if healthy:
            phase, detail = PHASE_MANAGED_RUNNING, "running (started by this manager)"
        elif why.startswith("still starting"):
            phase, detail = PHASE_MANAGED_STARTING, why
        else:
            phase, detail = PHASE_MANAGED_DEGRADED, f"DEGRADED: {why}"
            if spec.generate_url and not children:
                # The tell-tale of the failure this was built for: the
                # server process is alive but the model process it spawned
                # has gone, so it can talk but cannot think.
                detail += " (its model backend process has exited - use Restart)"
        return ServiceStatus(
            key=spec.key, label=spec.label, running=True, pid=facts.pid,
            ports=ports, urls=[config.url_for(port) for port in ports],
            detail=detail, tracked=True, conflicts=[], phase=phase, healthy=healthy,
        )

    if unmanaged_running(spec, probe):
        ports = occupied_ports(spec, probe)
        healthy, why = check_health(spec, probe)
        detail = ("running, but started outside this manager - stop it "
                  "from wherever it was launched")
        phase = PHASE_UNMANAGED_RUNNING
        if not healthy:
            phase, detail = PHASE_UNMANAGED_DEGRADED, f"DEGRADED: {why} (started outside this manager)"
        return ServiceStatus(
            key=spec.key, label=spec.label, running=True, pid=None,
            ports=ports, urls=[config.url_for(port) for port in ports],
            detail=detail, tracked=False, conflicts=[], phase=phase, healthy=healthy,
        )

    # Not ours and not healthy - but something may still be on those
    # ports, which is the single most confusing situation for a beginner,
    # so name it plainly.
    conflicts = port_conflicts(spec, probe)
    if record and facts is None:
        detail, phase = "not running (it exited since we started it)", PHASE_EXITED
    elif record:
        detail, phase = f"not managed by us: {ownership.reason}", PHASE_STALE_RECORD
    else:
        detail, phase = "not running", PHASE_STOPPED
    return ServiceStatus(
        key=spec.key, label=spec.label, running=False, pid=None,
        ports=[], urls=[], detail=detail, tracked=bool(record),
        conflicts=conflicts, phase=phase,
    )


# --- Actions ----------------------------------------------------------

class ActionResult(NamedTuple):
    ok: bool
    message: str


def start(spec, probe, state_path=None):
    """Launch a service, unless it is already running or blocked."""
    state = read_state(state_path)
    current = status(spec, probe, state)

    if current.phase in (PHASE_UNMANAGED_RUNNING, PHASE_UNMANAGED_DEGRADED):
        return ActionResult(
            False,
            f"{spec.label} is already running, started outside this manager. "
            "Use it as it is, or stop it from wherever it was launched.",
        )
    if current.phase == PHASE_MANAGED_DEGRADED:
        return ActionResult(
            False,
            f"{spec.label} is already running (pid {current.pid}) but not working: "
            f"{current.detail}. Use Restart, not Start.",
        )
    if current.running:
        return ActionResult(False, f"{spec.label} is already running (pid {current.pid}).")

    ok, why_not = spec.available()
    if not ok:
        return ActionResult(False, f"Cannot start {spec.label}: {why_not}.")

    refusal = ("\nStop that program yourself, or change the port with an "
               "AIPLAY_* environment variable. This manager will not stop a "
               "process it did not start.")

    conflicts = port_conflicts(spec, probe)
    if conflicts:
        return ActionResult(
            False, f"Not starting {spec.label} because " + "; ".join(conflicts) + "." + refusal,
        )

    # A port can be busy even when nobody will tell us who owns it (that
    # needs admin rights on Windows). Busy is busy: refuse, and say that
    # the owner is unknown rather than guessing.
    busy = occupied_ports(spec, probe)
    if busy:
        return ActionResult(
            False,
            f"Not starting {spec.label}: port{'s' if len(busy) > 1 else ''} "
            f"{', '.join(map(str, busy))} already in use by an unknown program." + refusal,
        )

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / f"{spec.key}.log"

    child_env = os.environ.copy()
    child_env.update(spec.env)
    # Some environments set this, which stops cmd.exe resolving a .bat
    # from the working directory - and these launchers call each other by
    # bare name. Cleared for the child only.
    child_env.pop("NoDefaultCurrentDirectoryInExePath", None)

    creation_flags = 0
    if sys.platform == "win32":
        # Detached and in its own group, so the service outlives the
        # manager and a Ctrl+C here does not take the server down.
        creation_flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                          | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            spec.argv,
            cwd=str(spec.workdir),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=child_env,
            creationflags=creation_flags,
        )

    facts = probe.facts(proc.pid)
    if facts is None:
        return ActionResult(
            False,
            f"{spec.label} exited immediately. See {log_path} for why.",
        )

    state[spec.key] = record_for(facts, spec)
    write_state(state, state_path)
    return ActionResult(
        True,
        f"Starting {spec.label} (pid {proc.pid}). It may take a minute to "
        f"finish loading; log: {log_path}",
    )


def stop(spec, probe, state_path=None):
    """Stop a service - only if all four identity checks agree."""
    state = read_state(state_path)
    record = state.get(spec.key)

    if not record:
        return ActionResult(
            False,
            f"Not stopping anything: this manager has no record of starting "
            f"{spec.label}. If it is running, whatever launched it should "
            f"close it.",
        )

    facts = probe.facts(record.get("pid"))
    ownership = verify_owned(record, facts)
    if not ownership.owned:
        # Drop the stale record so status stops mentioning it, but do NOT
        # touch whatever is running now.
        state.pop(spec.key, None)
        write_state(state, state_path)
        return ActionResult(
            False,
            f"Not stopping anything: {ownership.reason}. Cleared our stale "
            f"record for {spec.label}.",
        )

    # The creation time travels with the request so the probe can check it
    # again at the instant it acts, not just when we looked a moment ago.
    stopped = probe.stop_tree(facts.pid, config.SHUTDOWN_TIMEOUT,
                              expected_create_time=record.get("create_time"))
    state.pop(spec.key, None)
    write_state(state, state_path)

    if stopped:
        return ActionResult(True, f"Stopped {spec.label} and its child processes.")
    return ActionResult(
        False,
        f"Did not finish stopping {spec.label}: either its pid changed hands "
        "at the last moment (so nothing was touched) or part of its process "
        "tree is still alive. Check Task Manager.",
    )


def restart(spec, probe, state_path=None, wait_seconds=3):
    stop_result = stop(spec, probe, state_path)
    # A service that was not running is fine to start; only a refusal to
    # act on someone else's process should block us.
    if not stop_result.ok and "Not stopping anything" not in stop_result.message:
        return stop_result
    time.sleep(wait_seconds)
    start_result = start(spec, probe, state_path)
    return ActionResult(
        start_result.ok,
        f"{stop_result.message}\n{start_result.message}",
    )


# --- Extras the manager displays --------------------------------------

def gpu_memory():
    """Used/total VRAM via nvidia-smi, or None when unavailable.

    Deliberately not WMI: Windows reports this GPU as 4 GB there because
    of a 32-bit overflow, while nvidia-smi reports the true 8 GB.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        name, used, total = (part.strip()
                             for part in out.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "used_mb": int(used), "total_mb": int(total)}
    except ValueError:
        return None


def loaded_model(status_obj, timeout=None):
    """Ask the model API which model is loaded. None if it isn't up."""
    if not status_obj.running:
        return None
    try:
        import requests
        response = requests.get(config.MODEL_LIST_URL,
                                timeout=timeout or config.HEALTH_TIMEOUT)
        response.raise_for_status()
        entries = response.json().get("data") or []
        return entries[0].get("id") if entries else None
    except Exception:
        # Any failure here just means "cannot tell", which the manager
        # renders as a dash. Never worth raising.
        return None
