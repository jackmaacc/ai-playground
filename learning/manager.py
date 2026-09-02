"""
The app manager: see what is running, and start or stop it.

This terminal manager is the authoritative control surface. It runs in its
own process, so it can stop any of the three applications without stopping
itself. A manager embedded in the learning web UI could never honestly
offer to restart the very server rendering its buttons, which is why that
one (when it exists) is read-only.

Run with: python manager.py
"""

import webbrowser

import config
import console
import services

console.use_utf8_output()

STATUS_SYMBOL = {True: "RUNNING", False: "stopped"}


def _probe_or_explain():
    """Get a process probe, or explain why management is unavailable."""
    try:
        return services.ProcessProbe(), None
    except services.ProbeUnavailable as error:
        return None, str(error)


def _snapshot(probe):
    """Current status of every service, reading shared state once."""
    registry = services.build_registry()
    state = services.read_state()
    return [(spec, services.status(spec, probe, state)) for spec in registry.values()]


def show_status():
    probe, problem = _probe_or_explain()
    if problem:
        print(f"\n{problem}")
        return

    warning = config.public_bind_warning()
    if warning:
        print(f"\n{warning}")

    print(f"\n{'app':<26}{'state':<10}{'ports':<16}{'model / notes'}")
    print("-" * 78)

    for spec, status in _snapshot(probe):
        ports = ", ".join(str(port) for port in status.ports) or "-"
        if status.running and not status.healthy:
            # Alive but broken deserves its own word: "RUNNING" here would
            # be the lie this check exists to stop.
            word, extra = "DEGRADED", status.detail
        elif spec.key == "chat_llm" and status.running:
            word = STATUS_SYMBOL[True]
            extra = services.loaded_model(status) or "(model still loading)"
            if status.phase == services.PHASE_UNMANAGED_RUNNING:
                extra += "  [started outside this manager]"
        elif status.phase == services.PHASE_UNMANAGED_RUNNING:
            word, extra = STATUS_SYMBOL[True], "[started outside this manager]"
        else:
            word = STATUS_SYMBOL[status.running]
            extra = "" if status.running else status.detail
        print(f"{spec.label[:25]:<26}{word:<10}{ports:<16}{extra}")

        for url in status.urls:
            print(f"{'':<26}open: {url}")
        for conflict in status.conflicts:
            print(f"{'':<26}! {conflict}")

    gpu = services.gpu_memory()
    if gpu:
        print(f"\nGPU: {gpu['name']} - {gpu['used_mb']} MB used of {gpu['total_mb']} MB")
        if gpu["used_mb"] > gpu["total_mb"] * 0.85:
            print("     Nearly full. Stop an app before starting another, or")
            print("     image generation may fail with a CUDA out-of-memory error.")
    else:
        print("\nGPU: not reported (nvidia-smi unavailable)")


def _choose_service(action_name):
    """Ask which app to act on. Returns a spec, or None if backed out."""
    registry = services.build_registry()
    specs = list(registry.values())
    print(f"\nWhich app do you want to {action_name}?")
    for number, spec in enumerate(specs, start=1):
        print(f"{number}) {spec.label}")
    print(f"{len(specs) + 1}) Back")

    choice = console.ask_int("choose", default=len(specs) + 1,
                             minimum=1, maximum=len(specs) + 1)
    if choice == len(specs) + 1:
        return None
    return specs[choice - 1]


def _act(action_name, function):
    probe, problem = _probe_or_explain()
    if problem:
        print(f"\n{problem}")
        return
    spec = _choose_service(action_name)
    if spec is None:
        return
    print(f"\n...{action_name}ing {spec.label}")
    result = function(spec, probe)
    print(f"\n{result.message}")
    if action_name == "start" and result.ok:
        print("Use 'Show status' in a moment to see when it has finished loading.")


def start_service():
    _act("start", services.start)


def stop_service():
    _act("stop", services.stop)


def restart_service():
    _act("restart", services.restart)


def open_service():
    probe, problem = _probe_or_explain()
    if problem:
        print(f"\n{problem}")
        return

    running = [(spec, status) for spec, status in _snapshot(probe)
               if status.running and status.urls]
    if not running:
        print("\nNothing is running that has a web page to open.")
        return

    print("\nWhich page do you want to open?")
    options = [(spec, url) for spec, status in running for url in status.urls]
    for number, (spec, url) in enumerate(options, start=1):
        print(f"{number}) {url}   ({spec.label})")
    print(f"{len(options) + 1}) Back")

    choice = console.ask_int("choose", default=len(options) + 1,
                             minimum=1, maximum=len(options) + 1)
    if choice == len(options) + 1:
        return
    url = options[choice - 1][1]
    print(f"opening {url}")
    webbrowser.open(url)


def show_settings():
    print("\n--- Where everything is configured ---")
    details = config.describe()
    print(f"repository:  {details['repo_root']}")
    print(f"bind host:   {details['bind_host']}"
          f"{'  (LOOPBACK ONLY - good)' if not details['public_bind'] else '  (PUBLIC)'}")
    print("default ports:")
    for name, port in details["ports"].items():
        print(f"  {name:<16}{port}")
    print(f"model API:   {details['model_api']}")
    print(f"chat-llm python:  {details['chat_llm_python'] or '(not found)'}")
    print(f"image-gen python: {details['image_gen_python'] or '(not found)'}")
    print("\nEvery value above can be overridden with an AIPLAY_* environment")
    print("variable, but nothing needs setting for normal local use.")


def run():
    console.run_menu(
        "App manager - start, stop and check the three applications",
        [
            ("Show status", show_status),
            ("Start an app", start_service),
            ("Stop an app", stop_service),
            ("Restart an app", restart_service),
            ("Open an app in the browser", open_service),
            ("Show configuration", show_settings),
        ],
        subtitle="this manager only ever stops apps that it started itself",
    )


if __name__ == "__main__":
    show_status()
    run()
