"""Service lifecycle tests, with the emphasis on NOT killing things.

Every case here runs against a fake process probe. Nothing is launched,
nothing is stopped, and no real pid is ever touched - which is the only
honest way to test code whose failure mode is "killed the wrong process".
"""

import sys
import tempfile
import unittest
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import config  # noqa: E402
import services  # noqa: E402


def make_facts(pid=4321, exe=None, cmdline="python.exe server.py", create_time=1000.0):
    exe = exe or str(config.CHAT_LLM_DIR / "installer_files" / "env" / "python.exe")
    return services.ProcessFacts(pid=pid, exe=exe, cmdline=cmdline, create_time=create_time)


def make_record(facts, marker="server.py", exe_root=None):
    return {
        "pid": facts.pid,
        "create_time": facts.create_time,
        "exe": facts.exe,
        "cmdline": facts.cmdline,
        "marker": marker,
        "exe_root": str(exe_root or config.CHAT_LLM_DIR),
        "started_at": "2026-09-01T20:36:54",
    }


class FakeProbe:
    """A process world we fully control, that records what was asked of it."""

    def __init__(self, facts_by_pid=None, listeners=None, ports_by_pid=None,
                 occupied=None, healthy=None):
        self.facts_by_pid = facts_by_pid or {}
        self.listeners = listeners or {}          # port -> pid
        self.ports_by_pid = ports_by_pid or {}    # pid -> [ports]
        self.stopped = []                         # every stop_tree call
        self.occupied = set(occupied or self.listeners)
        self.healthy = healthy or {}

    def facts(self, pid):
        return self.facts_by_pid.get(pid)

    def children(self, pid):
        return []

    def listening_pid(self, port):
        return self.listeners.get(port)

    def port_occupied(self, host, port):
        return port in self.occupied

    def http_healthy(self, url, timeout=None):
        return self.healthy.get(url, False)

    def listening_ports(self, pids):
        found = []
        for pid in pids:
            found.extend(self.ports_by_pid.get(pid, []))
        return sorted(set(found))

    def stop_tree(self, pid, timeout, expected_create_time=None):
        facts = self.facts(pid)
        if (expected_create_time is not None and facts is not None
                and facts.create_time != expected_create_time):
            return False
        self.stopped.append(pid)
        return True


class TempState(unittest.TestCase):
    """Every test gets its own state file; the real one is never touched."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self._dir.name) / "services.json"

    def tearDown(self):
        self._dir.cleanup()

    def write(self, state):
        services.write_state(state, self.state_path)


class VerifyOwnership(TempState):
    """The four identity checks, one failure mode at a time."""

    def test_all_four_matching_is_owned(self):
        facts = make_facts()
        verdict = services.verify_owned(make_record(facts), facts)
        self.assertTrue(verdict.owned)

    def test_no_record_is_not_owned(self):
        verdict = services.verify_owned(None, make_facts())
        self.assertFalse(verdict.owned)
        self.assertIn("no record", verdict.reason)

    def test_process_gone_is_not_owned(self):
        verdict = services.verify_owned(make_record(make_facts()), None)
        self.assertFalse(verdict.owned)
        self.assertIn("no longer running", verdict.reason)

    def test_pid_reuse_is_caught_by_creation_time(self):
        """The important one: same pid, different program."""
        original = make_facts(pid=4321, create_time=1000.0)
        impostor = make_facts(pid=4321, create_time=99999.0)
        verdict = services.verify_owned(make_record(original), impostor)
        self.assertFalse(verdict.owned)
        self.assertIn("reused", verdict.reason)

    def test_executable_outside_the_app_directory_is_rejected(self):
        record = make_record(make_facts())
        stranger = make_facts(exe=r"C:\Windows\System32\python.exe")
        verdict = services.verify_owned(record, stranger)
        self.assertFalse(verdict.owned)
        self.assertIn("not inside", verdict.reason)

    def test_command_line_without_the_marker_is_rejected(self):
        record = make_record(make_facts())
        other = make_facts(cmdline="python.exe something_else.py")
        verdict = services.verify_owned(record, other)
        self.assertFalse(verdict.owned)
        self.assertIn("command line", verdict.reason)

    def test_tiny_creation_time_wobble_is_tolerated(self):
        record = make_record(make_facts(create_time=1000.0))
        same = make_facts(create_time=1000.4)
        self.assertTrue(services.verify_owned(record, same).owned)


class StopRefusesUnlessVerified(TempState):
    """Nothing gets stopped without all four checks passing."""

    def setUp(self):
        super().setUp()
        self.spec = services.build_registry()["chat_llm"]

    def test_untracked_service_is_never_stopped(self):
        probe = FakeProbe(facts_by_pid={4321: make_facts()})
        result = services.stop(self.spec, probe, self.state_path)
        self.assertFalse(result.ok)
        self.assertEqual(probe.stopped, [], "must not stop an untracked process")
        self.assertIn("no record", result.message)

    def test_pid_reuse_does_not_get_stopped(self):
        facts = make_facts(create_time=1000.0)
        self.write({"chat_llm": make_record(facts)})
        impostor = make_facts(create_time=88888.0)
        probe = FakeProbe(facts_by_pid={facts.pid: impostor})

        result = services.stop(self.spec, probe, self.state_path)

        self.assertFalse(result.ok)
        self.assertEqual(probe.stopped, [], "must not stop a recycled pid")
        self.assertIn("reused", result.message)
        # The stale record is cleared so status stops reporting it.
        self.assertEqual(services.read_state(self.state_path), {})

    def test_foreign_process_on_our_port_is_never_stopped(self):
        probe = FakeProbe(
            facts_by_pid={999: make_facts(pid=999, exe=r"C:\other\python.exe",
                                          cmdline="python.exe someone_elses_app.py")},
            listeners={config.CHAT_LLM_UI_PORT: 999},
        )
        result = services.stop(self.spec, probe, self.state_path)
        self.assertFalse(result.ok)
        self.assertEqual(probe.stopped, [])

    def test_unknown_process_on_occupied_port_is_never_stopped(self):
        probe = FakeProbe(occupied={config.CHAT_LLM_UI_PORT})
        result = services.stop(self.spec, probe, self.state_path)
        self.assertFalse(result.ok)
        self.assertEqual(probe.stopped, [])

    def test_verified_process_is_stopped(self):
        facts = make_facts()
        self.write({"chat_llm": make_record(facts)})
        probe = FakeProbe(facts_by_pid={facts.pid: facts})

        result = services.stop(self.spec, probe, self.state_path)

        self.assertTrue(result.ok)
        self.assertEqual(probe.stopped, [facts.pid])
        self.assertEqual(services.read_state(self.state_path), {})


class StartGuards(TempState):
    def setUp(self):
        super().setUp()
        self.spec = services.build_registry()["chat_llm"]

    def test_refuses_to_start_a_duplicate(self):
        facts = make_facts()
        self.write({"chat_llm": make_record(facts)})
        probe = FakeProbe(facts_by_pid={facts.pid: facts},
                          ports_by_pid={facts.pid: [config.CHAT_LLM_UI_PORT]})

        result = services.start(self.spec, probe, self.state_path)

        self.assertFalse(result.ok)
        self.assertIn("already running", result.message)

    def test_refuses_to_start_when_a_stranger_holds_the_port(self):
        """Explains the conflict; does not offer to kill the stranger."""
        probe = FakeProbe(
            facts_by_pid={777: make_facts(pid=777, exe=r"C:\other\app.exe",
                                          cmdline="app.exe --serve")},
            listeners={config.CHAT_LLM_API_PORT: 777},
        )
        result = services.start(self.spec, probe, self.state_path)

        self.assertFalse(result.ok)
        self.assertIn(str(config.CHAT_LLM_API_PORT), result.message)
        self.assertIn("will not stop a process it did not start", result.message)
        self.assertEqual(probe.stopped, [])

    def test_refuses_when_port_is_occupied_but_owner_is_unknown(self):
        probe = FakeProbe(occupied={config.CHAT_LLM_API_PORT})
        result = services.start(self.spec, probe, self.state_path)
        self.assertFalse(result.ok)
        self.assertIn("unknown", result.message.lower())


class StatusReporting(TempState):
    def setUp(self):
        super().setUp()
        self.spec = services.build_registry()["chat_llm"]

    def test_reports_actual_bound_ports_not_expected_ones(self):
        """Forge moves to the next free port; we report what it really took."""
        facts = make_facts()
        state = {"chat_llm": make_record(facts)}
        probe = FakeProbe(facts_by_pid={facts.pid: facts},
                          ports_by_pid={facts.pid: [7899, 5000]})

        status = services.status(self.spec, probe, state)

        self.assertTrue(status.running)
        self.assertEqual(status.ports, [5000, 7899])
        self.assertIn("http://127.0.0.1:7899", status.urls)

    def test_exited_service_is_reported_not_running(self):
        facts = make_facts()
        state = {"chat_llm": make_record(facts)}
        probe = FakeProbe(facts_by_pid={})

        status = services.status(self.spec, probe, state)

        self.assertFalse(status.running)
        self.assertTrue(status.tracked)
        self.assertIn("exited", status.detail)

    def test_conflicts_are_surfaced_when_not_running(self):
        probe = FakeProbe(
            facts_by_pid={555: make_facts(pid=555, exe=r"C:\other\thing.exe",
                                          cmdline="thing.exe")},
            listeners={config.CHAT_LLM_UI_PORT: 555},
        )
        status = services.status(self.spec, probe, {})
        self.assertFalse(status.running)
        self.assertTrue(any("already used by" in c for c in status.conflicts))

    def test_manually_started_healthy_service_is_running_but_unmanaged(self):
        probe = FakeProbe(
            occupied=set(self.spec.expected_ports),
            healthy={self.spec.health_url: True},
        )
        status = services.status(self.spec, probe, {})
        self.assertTrue(status.running)
        self.assertFalse(status.tracked)
        self.assertEqual(status.phase, "UNMANAGED_RUNNING")


class StateFile(TempState):
    def test_round_trip(self):
        self.write({"chat_llm": {"pid": 1}})
        self.assertEqual(services.read_state(self.state_path), {"chat_llm": {"pid": 1}})

    def test_corrupt_state_reads_as_empty_rather_than_crashing(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(services.read_state(self.state_path), {})

    def test_missing_state_reads_as_empty(self):
        self.assertEqual(services.read_state(self.state_path / "nope.json"), {})

    def test_record_captures_all_four_identity_fields(self):
        spec = services.build_registry()["chat_llm"]
        record = services.record_for(make_facts(), spec)
        for field in ("pid", "create_time", "exe", "cmdline", "marker", "exe_root"):
            self.assertIn(field, record)


class Registry(unittest.TestCase):
    def test_three_services_are_described(self):
        registry = services.build_registry()
        self.assertEqual(set(registry), {"chat_llm", "image_gen", "learning_web"})

    def test_every_spec_has_a_marker_and_exe_root(self):
        for spec in services.build_registry().values():
            self.assertTrue(spec.marker, f"{spec.key} needs a command-line marker")
            self.assertTrue(str(spec.exe_root), f"{spec.key} needs an exe_root")

    def test_installed_apps_launch_their_python_entry_points_directly(self):
        registry = services.build_registry()
        for key, script in (("chat_llm", "server.py"),
                            ("image_gen", "launch.py"),
                            ("learning_web", "webui.py")):
            spec = registry[key]
            if spec.argv:
                self.assertTrue(str(spec.argv[0]).lower().endswith("python.exe"))
                self.assertEqual(Path(spec.argv[1]).name, script)

    def test_missing_launcher_is_reported_not_raised(self):
        spec = services.ServiceSpec(
            key="ghost", label="ghost", workdir=config.LEARNING_DIR,
            argv=[str(config.LEARNING_DIR / "definitely_not_here.bat")],
            marker="x", exe_root=config.LEARNING_DIR,
        )
        ok, why = spec.available()
        self.assertFalse(ok)
        self.assertIn("does not exist", why)


if __name__ == "__main__":
    unittest.main()
