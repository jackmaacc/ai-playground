"""Running is not the same as working.

The failure these tests pin down really happened: the model server's web
front answered /v1/models with 200 for a long time after the llama.cpp
process behind it had exited, so every generation returned 500 while the
manager said RUNNING. Health now means "can it generate".
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


def facts(pid=4321):
    return services.ProcessFacts(
        pid=pid,
        exe=str(config.CHAT_LLM_DIR / "installer_files" / "env" / "python.exe"),
        cmdline="python.exe server.py",
        create_time=1000.0,
    )


def record(f):
    return {"pid": f.pid, "create_time": f.create_time, "exe": f.exe,
            "cmdline": f.cmdline, "marker": "server.py",
            "exe_root": str(config.CHAT_LLM_DIR), "started_at": "t"}


class ShallowProbe:
    """A fake that only knows the shallow check - like the older fakes in
    test_services.py, which predate http_generates."""

    def __init__(self, shallow=True, children=(), facts_by_pid=None):
        self.shallow = shallow
        self._children = list(children)
        self.facts_by_pid = facts_by_pid or {}

    def facts(self, pid):
        return self.facts_by_pid.get(pid)

    def children(self, pid):
        return self._children

    def listening_pid(self, port):
        return None

    def listening_ports(self, pids):
        return [config.CHAT_LLM_API_PORT]

    def port_occupied(self, host, port):
        return True

    def http_healthy(self, url, timeout=None):
        return self.shallow

    def stop_tree(self, pid, timeout, expected_create_time=None):
        return True


class HealthProbe(ShallowProbe):
    """Shallow and deep answers set independently; counts deep calls."""

    def __init__(self, shallow=True, deep=True, children=(), facts_by_pid=None):
        super().__init__(shallow, children, facts_by_pid)
        self.deep = deep
        self.deep_calls = 0

    def http_generates(self, url, timeout=None):
        self.deep_calls += 1
        return self.deep


class ManagedService(unittest.TestCase):
    def setUp(self):
        self.spec = services.build_registry()["chat_llm"]
        self.f = facts()
        self.state = {"chat_llm": record(self.f)}

    def status_with(self, **probe_kwargs):
        probe = HealthProbe(facts_by_pid={self.f.pid: self.f}, **probe_kwargs)
        return services.status(self.spec, probe, self.state), probe

    def test_web_up_and_generating_is_running_and_healthy(self):
        status, probe = self.status_with(shallow=True, deep=True, children=[99])
        self.assertEqual(status.phase, services.PHASE_MANAGED_RUNNING)
        self.assertTrue(status.healthy)
        self.assertEqual(probe.deep_calls, 1, "the deep check must actually run")

    def test_web_up_but_generation_failing_is_degraded_not_running(self):
        """The real incident: /v1/models 200, chat completions 500."""
        status, _ = self.status_with(shallow=True, deep=False, children=[99])
        self.assertEqual(status.phase, services.PHASE_MANAGED_DEGRADED)
        self.assertTrue(status.running, "the process IS alive...")
        self.assertFalse(status.healthy, "...but it is not working")
        self.assertIn("DEGRADED", status.detail)
        self.assertIn("generating fails", status.detail)

    def test_missing_backend_child_is_named_in_the_diagnosis(self):
        status, _ = self.status_with(shallow=True, deep=False, children=[])
        self.assertIn("model backend process has exited", status.detail)
        self.assertIn("Restart", status.detail)

    def test_backend_child_present_does_not_claim_it_exited(self):
        status, _ = self.status_with(shallow=True, deep=False, children=[99])
        self.assertNotIn("has exited", status.detail)

    def test_web_not_answering_yet_is_starting(self):
        status, probe = self.status_with(shallow=False, deep=True)
        self.assertEqual(status.phase, services.PHASE_MANAGED_STARTING)
        self.assertFalse(status.healthy)
        self.assertEqual(probe.deep_calls, 0, "no point generating before the web server is up")

    def test_api_dying_mid_check_is_degraded(self):
        status, _ = self.status_with(shallow=True, deep=None)
        self.assertEqual(status.phase, services.PHASE_MANAGED_DEGRADED)
        self.assertIn("stopped answering", status.detail)

    def test_start_on_a_degraded_service_says_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            services.write_state(self.state, path)
            probe = HealthProbe(facts_by_pid={self.f.pid: self.f}, shallow=True, deep=False)
            result = services.start(self.spec, probe, path)
        self.assertFalse(result.ok)
        self.assertIn("Use Restart, not Start", result.message)


class UnmanagedService(unittest.TestCase):
    def test_hand_started_but_broken_is_unmanaged_degraded(self):
        spec = services.build_registry()["chat_llm"]
        probe = HealthProbe(shallow=True, deep=False)
        status = services.status(spec, probe, {})
        self.assertEqual(status.phase, services.PHASE_UNMANAGED_DEGRADED)
        self.assertTrue(status.running)
        self.assertFalse(status.healthy)
        self.assertFalse(status.tracked)


class ProbesWithoutDeepCheck(unittest.TestCase):
    def test_probe_lacking_http_generates_is_trusted_on_the_shallow_check(self):
        """Keeps the externally-maintained test_services fakes valid."""
        spec = services.build_registry()["chat_llm"]
        healthy, detail = services.check_health(spec, ShallowProbe(shallow=True))
        self.assertTrue(healthy)
        self.assertEqual(detail, "")


class SpecsAreWired(unittest.TestCase):
    def test_chat_llm_has_a_deep_health_url_and_others_do_not_pretend_to(self):
        registry = services.build_registry()
        self.assertEqual(registry["chat_llm"].generate_url, config.MODEL_CHAT_URL)
        self.assertIsNone(registry["image_gen"].generate_url)
        self.assertIsNone(registry["learning_web"].generate_url)


if __name__ == "__main__":
    unittest.main()
