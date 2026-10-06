import json
import os
import select
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import yaml


class CoV2VBridge(object):
    def __init__(self, config):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.debug = bool(self.config.get("debug", False))
        self.request_timeout_sec = float(self.config.get("request_timeout_sec", 30.0))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 85))
        self._process = None
        self._runtime_scene_dir = None
        self._worker_cwd = None
        self.driver_intents = {}

        if not self.enabled:
            return

        repo_root = Path(__file__).resolve().parents[3]
        self._worker_cwd = repo_root
        runtime_root = self.config.get("runtime_root")
        if runtime_root:
            runtime_root = Path(runtime_root)
        else:
            runtime_root = Path(tempfile.gettempdir()) / "cov2v_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        self._runtime_scene_dir = runtime_root / ("session_%s" % os.getpid())
        self._runtime_scene_dir.mkdir(parents=True, exist_ok=True)
        self._spawn_worker()

    def close(self):
        if self._process is None:
            return
        try:
            self._request({"type": "shutdown"}, timeout_sec=2.0)
        except Exception:
            pass
        try:
            self._process.terminate()
        except Exception:
            pass
        if self._runtime_scene_dir is not None:
            try:
                shutil.rmtree(str(self._runtime_scene_dir), ignore_errors=True)
            except Exception:
                pass
        self._process = None

    def negotiate(self, ego_data, baseline_controls, step, timestamp):
        if not self.enabled or self._process is None:
            return {}

        active_count = self._write_scene_snapshot(
            ego_data=ego_data,
            baseline_controls=baseline_controls,
            step=step,
            timestamp=timestamp,
        )
        if active_count <= 0:
            return {}

        payload = {
            "type": "negotiate",
            "scene_dir": str(self._runtime_scene_dir),
        }
        self._debug_print(
            "negotiate step=%d ts=%.2f active_vehicles=%d"
            % (step, timestamp, active_count)
        )
        response = self._request(payload, timeout_sec=None)
        if response.get("status") != "ok":
            raise RuntimeError(response.get("message", "unknown cov2v worker error"))

        result_map = {}
        for item in response.get("results", []):
            ego_id = str(item.get("ego_id", ""))
            if ego_id.startswith("veh_"):
                ego_id = ego_id.replace("veh_", "", 1)
            try:
                vehicle_num = int(ego_id)
            except Exception:
                continue
            result_map[vehicle_num] = item
        discussion_process = response.get("discussion_process")
        result_map["_discussion_process"] = discussion_process
        if isinstance(discussion_process, dict):
            result_map["_token_usage"] = discussion_process.get("token_usage", {})
            result_map["_token_usage_total"] = discussion_process.get("token_usage_total", {})
        return result_map

    def _spawn_worker(self):
        python_bin = self.config.get("python_bin") or shutil.which("python3.10") or "python3.10"
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        repo_root = str(self._worker_cwd)
        env["PYTHONPATH"] = repo_root if not pythonpath else repo_root + os.pathsep + pythonpath
        self._process = subprocess.Popen(
            [python_bin, "-m", "cov2v.worker_server"],
            cwd=str(self._worker_cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if self.debug else subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1,
        )
        driver_intents = {}
        if os.environ.get("COV2V_DISABLE_DRIVER_INTENTS", "").strip().lower() not in {"1", "true", "yes"}:
            driver_intents = dict(self.config.get("driver_intents") or {})
            driver_intents.update(self._load_scenario_driver_intents())
        self.driver_intents = driver_intents
        init_payload = {
            "type": "init",
            "gemini_config": self._build_gemini_config(),
            "driver_intents": driver_intents,
        }
        self._debug_print(
            "starting worker provider=%s model=%s mode=%s strategy=%s observations=%s"
            % (
                init_payload["gemini_config"].get("provider", "gemini"),
                init_payload["gemini_config"].get("model", "unknown"),
                init_payload["gemini_config"].get("negotiation_mode", "single"),
                init_payload["gemini_config"].get("use_strategy_knowledge", True),
                init_payload["gemini_config"].get("use_other_vehicle_observations", True),
            )
        )
        if driver_intents:
            self._debug_print("driver_intents loaded for: %s" % ", ".join(sorted(driver_intents)))
        response = self._request(init_payload, timeout_sec=self.request_timeout_sec)
        if response.get("status") != "ok":
            raise RuntimeError(response.get("message", "failed to initialize cov2v worker"))

    def _current_scenario_name(self):
        routes_dir = os.environ.get("ROUTES_DIR") or os.environ.get("ROUTES")
        if not routes_dir:
            return None
        return Path(routes_dir).name

    def _load_scenario_driver_intents(self):
        driver_intents_path = self.config.get("driver_intents_path")
        if not driver_intents_path:
            return {}
        scenario_name = self._current_scenario_name()
        if not scenario_name:
            return {}

        resolved_path = self._worker_cwd / driver_intents_path
        if not resolved_path.exists():
            return {}

        with open(str(resolved_path), "r") as file_obj:
            all_scenarios = yaml.safe_load(file_obj) or {}
        utterances = all_scenarios.get(scenario_name) or {}

        return {
            str(vehicle_id): {
                "intent_type": "passenger_declared",
                "justification": str(utterance).strip(),
            }
            for vehicle_id, utterance in utterances.items()
            if str(utterance).strip()
        }

    def _build_gemini_config(self):
        provider = str(self.config.get("provider", "gemini")).strip().lower() or "gemini"
        if provider == "qwen":
            api_key = self.config.get("qwen_api_key", "")
        else:
            api_key = self.config.get("gemini_api_key", "")

        system_prompt_path = self.config.get("system_prompt_path")
        if system_prompt_path:
            system_prompt_path = str((self._worker_cwd / system_prompt_path).resolve())

        return {
            "api_key": api_key,
            "provider": provider,
            "model": self.config.get("model", "gemini-2.5-flash"),
            "base_url": self.config.get("base_url"),
            "temperature": float(self.config.get("temperature", 0.0)),
            "cooperative_range_meters": float(self.config.get("cooperative_range_meters", 40.0)),
            "system_prompt_path": system_prompt_path,
            "negotiation_mode": self.config.get("negotiation_mode", "single"),
            "discussion_rounds": int(self.config.get("discussion_rounds", 2)),
            "use_strategy_knowledge": bool(self.config.get("use_strategy_knowledge", True)),
            "use_other_vehicle_observations": bool(self.config.get("use_other_vehicle_observations", True)),
            "debug": self.debug,
            "retry_wait_sec": float(self.config.get("retry_wait_sec", 10.0)),
            "retry_attempts": int(self.config.get("retry_attempts", 0)),
            "extra_body": self.config.get("extra_body", {}),
        }

    def _request(self, payload, timeout_sec):
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("cov2v worker is not running")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        ready, _, _ = select.select([self._process.stdout], [], [], timeout_sec)
        if not ready:
            raise RuntimeError("cov2v worker timed out after %.1fs" % timeout_sec)
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("cov2v worker exited unexpectedly")
        response = json.loads(line)
        return response

    def _write_scene_snapshot(self, ego_data, baseline_controls, step, timestamp):
        active_count = 0
        for ego_id, data in enumerate(ego_data):
            vehicle_dir = self._runtime_scene_dir / ("ego_vehicle_%d" % ego_id)

            if data is None:
                shutil.rmtree(str(vehicle_dir), ignore_errors=True)
                continue

            (vehicle_dir / "measurements").mkdir(parents=True, exist_ok=True)
            (vehicle_dir / "rgb_front").mkdir(parents=True, exist_ok=True)
            (vehicle_dir / "rgb_left").mkdir(parents=True, exist_ok=True)
            (vehicle_dir / "rgb_right").mkdir(parents=True, exist_ok=True)
            measurement_path = vehicle_dir / "measurements" / "current.json"
            image_paths = {
                image_name: vehicle_dir / image_name / "current.jpg"
                for image_name in ("rgb_front", "rgb_left", "rgb_right")
            }

            active_count += 1
            control = baseline_controls[ego_id]
            measurement_payload = {
                "frame": int(step),
                "ego_id": int(ego_id),
                "timestamp": float(timestamp),
                "measurements": {
                    "gps_x": float(data["measurements"]["gps_x"]),
                    "gps_y": float(data["measurements"]["gps_y"]),
                    "x": float(data["measurements"]["x"]),
                    "y": float(data["measurements"]["y"]),
                    "theta": float(data["measurements"]["theta"]),
                    "z": float(data["measurements"]["z"]),
                    "timestamp_sec": float(data["measurements"]["timestamp_sec"]),
                    "speed": float(data["measurements"]["speed"]),
                    "command": int(data["measurements"]["command"]),
                    "command_intent": str(
                        data["measurements"].get("command_intent", "UNKNOWN")
                    ),
                    "future_command_3s": int(
                        data["measurements"].get(
                            "future_command_3s",
                            data["measurements"]["command"],
                        )
                    ),
                    "future_command_3s_intent": str(
                        data["measurements"].get(
                            "future_command_3s_intent",
                            data["measurements"].get("command_intent", "UNKNOWN"),
                        )
                    ),
                },
                "control": {
                    "steer": float(control.steer if control is not None else 0.0),
                    "throttle": float(control.throttle if control is not None else 0.0),
                    "brake": float(control.brake if control is not None else 0.0),
                    "gear": int(control.gear if control is not None else 0),
                    "reverse": bool(control.reverse if control is not None else False),
                    "manual_gear_shift": bool(control.manual_gear_shift if control is not None else False),
                },
            }
            with open(str(measurement_path), "w") as file_obj:
                json.dump(measurement_payload, file_obj, indent=2)
            self._debug_print(
                "snapshot veh_%d speed=%.2f intent=%s future_3s=%s"
                % (
                    ego_id,
                    measurement_payload["measurements"]["speed"],
                    measurement_payload["measurements"]["command_intent"],
                    measurement_payload["measurements"]["future_command_3s_intent"],
                )
            )

            for image_name in ("rgb_front", "rgb_left", "rgb_right"):
                image = data.get(image_name)
                image_path = image_paths[image_name]
                if image is None:
                    if image_path.exists():
                        image_path.unlink()
                    continue
                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
        return active_count

    def _debug_print(self, message):
        if self.debug:
            print("[cov2v][bridge] %s" % message, flush=True)
