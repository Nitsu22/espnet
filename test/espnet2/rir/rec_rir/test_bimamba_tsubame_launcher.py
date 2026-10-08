"""Queue-free checks of GPU time/point estimates and submission guards."""
import contextlib
import ast
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    'bimamba_launcher', ROOT / 'egs2/whamr/rir_2spk/local/launch_bimamba_ablations_tsubame.py'
)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_estimates_use_measurements_and_respect_24_hour_cap():
    fast = LAUNCHER.estimate(dict(steady_update_seconds=.0867,
                                 estimated_valid_step_seconds=.0093))
    slow = LAUNCHER.estimate(dict(steady_update_seconds=.2087,
                                 estimated_valid_step_seconds=.0203))
    assert fast['wall_hours'] == 20
    assert slow['wall_hours'] == 24
    assert slow['projected_hours_100_epochs'] > 24
    assert fast['expected_hours_55_epochs'] < slow['expected_hours_55_epochs']
    for value in (-1, float('nan'), float('inf')):
        with unittest.TestCase().assertRaises(ValueError):
            LAUNCHER.estimate(dict(steady_update_seconds=value,
                                  estimated_valid_step_seconds=.01))


def test_submission_reports_effective_resources_and_wall_seconds():
    with patch.object(LAUNCHER, 'remote', return_value='123456') as remote, \
            contextlib.redirect_stdout(io.StringIO()):
        result = LAUNCHER.submit('ffn_only', 24, False, 'abc123', 17.5)
    command = remote.call_args.args[0]
    assert 'h_rt=24:00:00' in command and 'false 86400' in command
    assert 'git rev-parse HEAD' in command
    assert result['resources'] == 'gpu_1=1' and result['priority'] == -5
    assert result['job_id'] == 123456 and not result['resume']
    assert abs(result['expected_points'] - 2.93) < 1e-10
    assert abs(result['upper_points'] - 3.84) < 1e-10
    with patch.object(LAUNCHER, 'remote', return_value='unexpected output'):
        with unittest.TestCase().assertRaises(RuntimeError):
            LAUNCHER.submit('bilstm', 20, False, 'abc123', 10)


def test_queue_inspection_parses_epoch_and_checks_exit_marker():
    captured = {}
    def fake_remote(code, payload):
        captured['code'] = code
        compile(code, '<remote-inspection>', 'exec')
        return '{"ffn_only":{"active":false,"exit_status":124,"resume_ready":true,"completed_epoch":50}}'
    with patch.object(LAUNCHER, 'remote_python', side_effect=fake_remote):
        info = LAUNCHER.inspect_jobs({'ffn_only': 123456})
    assert info['ffn_only']['completed_epoch'] == 50
    # Verify the generated remote regex, including its nested escaping.
    with patch('subprocess.run') as run, patch('sys.argv', ['check', '{}']):
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(captured['code'], namespace)
    expression = next(node.args[0].value for node in ast.walk(ast.parse(captured['code']))
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == 'findall')
    assert namespace['re'].findall(expression, '50epoch results:') == ['50']


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for name, value in list(globals().items()) if name.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
