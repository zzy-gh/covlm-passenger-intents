import math
import json
import os
import pathlib

import carla
import cv2
import numpy as np
import yaml

from team_code.cov2v_bridge import CoV2VBridge
from team_code.utils.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from team_code.auto_pilot import AutoPilot, _numpy, _orientation, get_collision
from team_code.map_agent import get_nearby_lights
from team_code.pid_controller import PIDController
from agents.navigation.local_planner import RoadOption


def get_entry_point():
    return "CoVLMAgent"


class CoVLMAgent(AutoPilot):
    """
    AutoPilot-style rule-based controller with Alpamayo-compatible sensor I/O.

    The control stack follows AutoPilot:
    - dense near waypoint planner
    - sparse far waypoint planner
    - PID steering / speed control
    - hazard-based braking

    The perception-facing sensor declaration and processed tick output follow
    AlpamayoGrpcAgent so a VLM module can be plugged in later without changing
    the sensor interface.
    """

    def setup(self, path_to_conf_file, ego_vehicles_num):
        preload_config = {}
        if path_to_conf_file.endswith("yaml"):
            preload_config = yaml.load(
                open(path_to_conf_file, "r"), Loader=yaml.FullLoader
            )

        if not isinstance(preload_config, dict):
            preload_config = {}

        control_config = preload_config.get("control", {})
        camera_config = preload_config.get("camera", {})
        safety_config = preload_config.get("safety", {})
        route_config = preload_config.get("route", {})
        save_config = preload_config.get("save", {})
        debug_config = preload_config.get("debug", {})
        self.cov2v_config = preload_config.get("cov2v", {})

        self._wide_rgb_sensor_data = camera_config.get(
            "wide_rgb", {"width": 1920, "height": 1080, "fov": 120}
        )
        self._tele_rgb_sensor_data = camera_config.get(
            "tele_rgb", {"width": 1920, "height": 1080, "fov": 30}
        )
        self._topdown_rgb_sensor_data = camera_config.get(
            "topdown_rgb", {"width": 600, "height": 600, "fov": 80}
        )
        self._camera_pose_config = camera_config.get("poses", {})
        self._turn_pid_config = control_config.get(
            "turn_pid", {"K_P": 1.25, "K_I": 0.75, "K_D": 0.3, "n": 40}
        )
        self._speed_pid_config = control_config.get(
            "speed_pid", {"K_P": 5.0, "K_I": 0.5, "K_D": 1.0, "n": 40}
        )
        self.debug = bool(debug_config.get("enabled", False))
        self.debug_interval = int(debug_config.get("interval", 1))

        super().setup(path_to_conf_file, ego_vehicles_num)
        self.agent_name = "CoVLM"
        self.use_rsu = False
        self.change_rsu_frame = 0
        self.rsu_height = None
        self.rsu_lane_side = None
        self.rsu_distance = None
        self.max_speed = float(control_config.get("max_speed", self.max_speed))
        self.slow_speed = float(control_config.get("slow_speed", self.slow_speed))
        self.visibility = safety_config.get("visibility", self.visibility)
        self.destroy_hazard_actors = bool(
            safety_config.get("destroy_hazard_actors", self.destroy_hazard_actors)
        )
        self.weather_id = route_config.get("weather", self.weather_id)
        self.waypoint_disturb = float(
            route_config.get("waypoint_disturb", self.waypoint_disturb)
        )
        self.waypoint_disturb_seed = int(
            route_config.get("waypoint_disturb_seed", self.waypoint_disturb_seed)
        )
        self.save_skip_frames = int(
            save_config.get("save_skip_frames", self.save_skip_frames)
        )
        self._cov2v_enabled = bool(self.cov2v_config.get("enabled", False))
        self._cov2v_debug = bool(self.cov2v_config.get("debug", False))
        self._cov2v_interval = max(1, int(self.cov2v_config.get("interval", 1)))
        self._cov2v_driver_intents = self.cov2v_config.get("driver_intents") or {}
        self._cov2v_bridge = None
        self.latest_ego_data = [None for _ in range(self.ego_vehicles_num)]
        self.latest_target_speed_actions = ["TARGET_NORMAL" for _ in range(self.ego_vehicles_num)]
        self.latest_target_speed_action_payloads = [{} for _ in range(self.ego_vehicles_num)]
        self.latest_brake_hazards = [{} for _ in range(self.ego_vehicles_num)]
        self.latest_discussion_process = None
        self.vlm_token_usage = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "thoughts_token_count": 0,
            "total_token_count": 0,
            "call_count": 0,
        }
        self.save_frame = None
        self.birdview_producer = None
        self._ensure_covlm_save_dirs()
        if self._cov2v_enabled:
            self._cov2v_bridge = CoV2VBridge(self.cov2v_config)
            self._cov2v_driver_intents = self._cov2v_bridge.driver_intents or self._cov2v_driver_intents

    def _ensure_covlm_save_dirs(self):
        if self.save_path is None:
            return
        for ego_id in range(self.ego_vehicles_num):
            ego_root = self.save_path / pathlib.Path(f"ego_vehicle_{ego_id}")
            (ego_root / "measurements").mkdir(parents=True, exist_ok=True)
            (ego_root / "rgb_front").mkdir(parents=True, exist_ok=True)
            (ego_root / "rgb_left").mkdir(parents=True, exist_ok=True)
            (ego_root / "rgb_right").mkdir(parents=True, exist_ok=True)
            (ego_root / "rgb_front_tele").mkdir(parents=True, exist_ok=True)
            (ego_root / "rgb_topdown").mkdir(parents=True, exist_ok=True)
            (ego_root / "bev_result").mkdir(parents=True, exist_ok=True)

    def _make_transform(self, pose_config, default_pose):
        return carla.Transform(
            carla.Location(
                x=float(pose_config.get("x", default_pose.location.x)),
                y=float(pose_config.get("y", default_pose.location.y)),
                z=float(pose_config.get("z", default_pose.location.z)),
            ),
            carla.Rotation(
                roll=float(pose_config.get("roll", default_pose.rotation.roll)),
                pitch=float(pose_config.get("pitch", default_pose.rotation.pitch)),
                yaw=float(pose_config.get("yaw", default_pose.rotation.yaw)),
            ),
        )

    def pose_def(self):
        self.lidar_pose = carla.Transform(
            carla.Location(x=1.3, y=0.0, z=1.85),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=-90.0),
        )
        self.camera_front_pose = self._make_transform(
            self._camera_pose_config.get("front", {}),
            carla.Transform(
                carla.Location(x=1.3, y=0.0, z=2.3),
                carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
            ),
        )
        self.camera_left_pose = self._make_transform(
            self._camera_pose_config.get("left", {}),
            carla.Transform(
                carla.Location(x=1.3, y=0.0, z=2.3),
                carla.Rotation(roll=0.0, pitch=0.0, yaw=-90.0),
            ),
        )
        self.camera_right_pose = self._make_transform(
            self._camera_pose_config.get("right", {}),
            carla.Transform(
                carla.Location(x=1.3, y=0.0, z=2.3),
                carla.Rotation(roll=0.0, pitch=0.0, yaw=90.0),
            ),
        )
        self.camera_front_tele_pose = self._make_transform(
            self._camera_pose_config.get("front_tele", {}),
            carla.Transform(
                carla.Location(x=1.3, y=0.0, z=2.3),
                carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
            ),
        )
        self.camera_topdown_pose = self._make_transform(
            self._camera_pose_config.get("topdown", {}),
            carla.Transform(
                carla.Location(x=0.0, y=0.0, z=40.0),
                carla.Rotation(roll=0.0, pitch=-90.0, yaw=0.0),
            ),
        )

    def sensors(self):
        self.pose_def()
        sensors_list = [
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._wide_rgb_sensor_data["width"],
                "height": self._wide_rgb_sensor_data["height"],
                "fov": self._wide_rgb_sensor_data["fov"],
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._wide_rgb_sensor_data["width"],
                "height": self._wide_rgb_sensor_data["height"],
                "fov": self._wide_rgb_sensor_data["fov"],
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._wide_rgb_sensor_data["width"],
                "height": self._wide_rgb_sensor_data["height"],
                "fov": self._wide_rgb_sensor_data["fov"],
                "id": "rgb_right",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_front_tele_pose.location.x,
                "y": self.camera_front_tele_pose.location.y,
                "z": self.camera_front_tele_pose.location.z,
                "roll": self.camera_front_tele_pose.rotation.roll,
                "pitch": self.camera_front_tele_pose.rotation.pitch,
                "yaw": self.camera_front_tele_pose.rotation.yaw,
                "width": self._tele_rgb_sensor_data["width"],
                "height": self._tele_rgb_sensor_data["height"],
                "fov": self._tele_rgb_sensor_data["fov"],
                "id": "rgb_front_tele",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_topdown_pose.location.x,
                "y": self.camera_topdown_pose.location.y,
                "z": self.camera_topdown_pose.location.z,
                "roll": self.camera_topdown_pose.rotation.roll,
                "pitch": self.camera_topdown_pose.rotation.pitch,
                "yaw": self.camera_topdown_pose.rotation.yaw,
                "width": self._topdown_rgb_sensor_data["width"],
                "height": self._topdown_rgb_sensor_data["height"],
                "fov": self._topdown_rgb_sensor_data["fov"],
                "id": "rgb_topdown",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]
        return sensors_list

    def _init(self):
        super()._init()
        self._turn_controller = PIDController(**self._turn_pid_config)
        self._speed_controller = PIDController(**self._speed_pid_config)
        try:
            self.birdview_producer = BirdViewProducer(
                CarlaDataProvider.get_client(),
                target_size=PixelDimensions(width=400, height=400),
                pixels_per_meter=5,
                crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            )
        except Exception as exc:
            self.birdview_producer = None
            print("[covlm] birdview producer init failed: %s" % exc, flush=True)

    def _build_control_tick_data(self, input_data, vehicle_num):
        gps = input_data[f"gps_{vehicle_num}"][1][:2]
        move_state = input_data[f"speed_{vehicle_num}"][1]["move_state"]
        imu = input_data[f"imu_{vehicle_num}"][1][:]
        compass = input_data[f"imu_{vehicle_num}"][1][-1]
        if math.isnan(compass):
            compass = 0.0

        control_tick_data = {
            f"gps_{vehicle_num}": gps,
            f"move_state_{vehicle_num}": move_state,
            f"compass_{vehicle_num}": compass,
            f"imu_{vehicle_num}": imu,
        }
        if f"lidar_{vehicle_num}" in input_data:
            control_tick_data[f"lidar_{vehicle_num}"] = input_data[f"lidar_{vehicle_num}"][1]

        return control_tick_data

    def _command_to_intent(self, command):
        command_name = getattr(command, "name", "")
        intent_map = {
            "LEFT": "TURN_LEFT",
            "RIGHT": "TURN_RIGHT",
            "STRAIGHT": "GO_STRAIGHT",
            "LANEFOLLOW": "FOLLOW_LANE",
            "CHANGELANELEFT": "CHANGE_LANE_LEFT",
            "CHANGELANERIGHT": "CHANGE_LANE_RIGHT",
        }
        return intent_map.get(command_name, command_name or "UNKNOWN")

    def _estimate_future_command_3s(self, gps, vehicle_num, speed, fallback_command):
        route = getattr(self._command_planner, "route", [])
        if vehicle_num < 0 or vehicle_num >= len(route) or not route[vehicle_num]:
            return fallback_command

        lookahead_distance = max(3.0 * float(max(speed, 0.0)), 6.0)
        cumulative_distance = 0.0
        previous_position = gps
        for position, command in route[vehicle_num]:
            cumulative_distance += float(np.linalg.norm(position - previous_position))
            if cumulative_distance >= lookahead_distance:
                return command
            previous_position = position
        return route[vehicle_num][-1][1]

    def tick(
        self,
        input_data,
        vehicle_num,
        timestamp,
        near_node=None,
        near_command=None,
        future_command_3s=None,
    ):
        self._vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(
            self._vehicle, self._actors.filter("*traffic_light*")
        )
        self._stop_signs = get_nearby_lights(
            self._vehicle, self._actors.filter("*stop*")
        )

        control_tick_data = self._build_control_tick_data(input_data, vehicle_num)
        gps = self._get_position(control_tick_data, vehicle_num)
        speed = control_tick_data[f"move_state_{vehicle_num}"]["speed"]
        compass = control_tick_data[f"compass_{vehicle_num}"]

        if near_node is None or near_command is None:
            near_node, near_command = self._waypoint_planner.run_step(gps, vehicle_num)
        if future_command_3s is None:
            future_command_3s = near_command

        result = {
            "measurements": {
                "gps_x": float(gps[0]),
                "gps_y": float(gps[1]),
                "x": float(gps[1]),
                "y": float(-gps[0]),
                "theta": float(compass),
                "z": 0.0,
                "timestamp_sec": float(timestamp),
                "speed": float(speed),
                "command": int(near_command.value),
                "command_intent": self._command_to_intent(near_command),
                "future_command_3s": int(future_command_3s.value),
                "future_command_3s_intent": self._command_to_intent(future_command_3s),
            }
        }

        result["rgb_front"] = cv2.cvtColor(
            input_data[f"rgb_front_{vehicle_num}"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        result["rgb_left"] = cv2.cvtColor(
            input_data[f"rgb_left_{vehicle_num}"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        result["rgb_right"] = cv2.cvtColor(
            input_data[f"rgb_right_{vehicle_num}"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        result["rgb_front_tele"] = cv2.cvtColor(
            input_data[f"rgb_front_tele_{vehicle_num}"][1][:, :, :3],
            cv2.COLOR_BGR2RGB,
        )
        topdown_key = f"rgb_topdown_{vehicle_num}"
        if topdown_key in input_data:
            result["rgb_topdown"] = cv2.cvtColor(
                input_data[topdown_key][1][:, :, :3],
                cv2.COLOR_BGR2RGB,
            )

        return result

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        control_all = [[] for _ in range(self.ego_vehicles_num)]
        self.latest_ego_data = [None for _ in range(self.ego_vehicles_num)]
        near_nodes = [None for _ in range(self.ego_vehicles_num)]
        far_nodes = [None for _ in range(self.ego_vehicles_num)]
        target_speeds = [0.0 for _ in range(self.ego_vehicles_num)]
        junction_flags = [False for _ in range(self.ego_vehicles_num)]
        affected_light_ids = [-1 for _ in range(self.ego_vehicles_num)]
        control_tick_data_all = [None for _ in range(self.ego_vehicles_num)]
        near_commands = [None for _ in range(self.ego_vehicles_num)]
        far_commands = [None for _ in range(self.ego_vehicles_num)]

        for vehicle_num in range(self.ego_vehicles_num):
            self._vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if self._vehicle is None:
                self._clear_cov2v_vehicle_state(vehicle_num)
                continue
            if not self._has_required_input_data(input_data, vehicle_num):
                self._clear_cov2v_vehicle_state(vehicle_num)
                continue

            if self._vehicle.id not in self._vehicle_id:
                self._vehicle_id.append(self._vehicle.id)
            self.vehicle_num = vehicle_num

            control_tick_data = self._build_control_tick_data(input_data, vehicle_num)
            control_tick_data_all[vehicle_num] = control_tick_data
            gps = self._get_position(control_tick_data, vehicle_num)
            speed = control_tick_data[f"move_state_{vehicle_num}"]["speed"]
            near_node, near_command = self._waypoint_planner.run_step(gps, vehicle_num)
            far_node, far_command = self._command_planner.run_step(gps, vehicle_num)
            future_command_3s = self._estimate_future_command_3s(
                gps,
                vehicle_num,
                speed,
                far_command,
            )
            near_nodes[vehicle_num] = near_node
            far_nodes[vehicle_num] = far_node
            near_commands[vehicle_num] = near_command
            far_commands[vehicle_num] = far_command
            self.latest_ego_data[vehicle_num] = self.tick(
                input_data=input_data,
                vehicle_num=vehicle_num,
                timestamp=timestamp,
                near_node=near_node,
                near_command=near_command,
                future_command_3s=future_command_3s,
            )
            self.is_junction = self._map.get_waypoint(
                self._vehicle.get_location()
            ).is_junction
            junction_flags[vehicle_num] = self.is_junction

            light = self._find_closest_valid_traffic_light(
                self._vehicle.get_location(), min_dis=50
            )
            self.affected_light_id = light.id if light is not None else -1
            affected_light_ids[vehicle_num] = self.affected_light_id

        if (
            self._cov2v_enabled
            and self._cov2v_bridge is not None
            and self.step % self._cov2v_interval == 0
        ):
            try:
                negotiation_results = self._cov2v_bridge.negotiate(
                    ego_data=self.latest_ego_data,
                    baseline_controls=[None for _ in range(self.ego_vehicles_num)],
                    step=self.step,
                    timestamp=timestamp,
                )
            except Exception as exc:
                negotiation_results = {}
                print("[cov2v] negotiation failed: %s" % exc, flush=True)

            self.latest_discussion_process = negotiation_results.get("_discussion_process")
            self._update_vlm_token_usage(negotiation_results.get("_token_usage", {}))
            for vehicle_num in range(self.ego_vehicles_num):
                if self.latest_ego_data[vehicle_num] is None:
                    self._clear_cov2v_vehicle_state(vehicle_num)
                    continue
                result = negotiation_results.get(vehicle_num)
                if not result:
                    continue
                target_speed_action = str(result.get("target_speed_action", "TARGET_NORMAL"))
                self.latest_target_speed_actions[vehicle_num] = target_speed_action
                self.latest_target_speed_action_payloads[vehicle_num] = result

        for vehicle_num in range(self.ego_vehicles_num):
            self._vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if self._vehicle is None:
                continue
            self.vehicle_num = vehicle_num

            control_tick_data = control_tick_data_all[vehicle_num]
            near_node = near_nodes[vehicle_num]
            far_node = far_nodes[vehicle_num]
            near_command = near_commands[vehicle_num]
            far_command = far_commands[vehicle_num]
            if control_tick_data is None or near_node is None or far_node is None:
                continue

            steer, throttle, brake, target_speed = self._get_control_with_target_speed_action(
                near_node,
                far_node,
                near_command,
                far_command,
                control_tick_data,
                self.latest_target_speed_actions[vehicle_num],
            )

            control = carla.VehicleControl()
            control.steer = steer + 1e-2 * np.random.randn()
            control.throttle = throttle
            control.brake = float(brake)
            target_speeds[vehicle_num] = target_speed
            control_all[vehicle_num] = control

        if (
            self._cov2v_debug
            and self._cov2v_enabled
            and self._cov2v_bridge is not None
            and self.step % self._cov2v_interval == 0
        ):
            self._print_cov2v_summary(
                timestamp=timestamp,
                controls=control_all,
                target_speeds=target_speeds,
                discussion_process=self.latest_discussion_process,
                negotiated=True,
            )

        for vehicle_num in range(self.ego_vehicles_num):
            self._vehicle = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if self._vehicle is None:
                continue
            control = control_all[vehicle_num]
            if control in (None, []):
                continue

            if self.step % self.save_skip_frames == 0 and self.save_path is not None:
                self.save_frame = self._save_minimal_step(
                    ego_id=vehicle_num,
                    ego_data=self.latest_ego_data[vehicle_num],
                    near_node=near_nodes[vehicle_num],
                    far_node=far_nodes[vehicle_num],
                    control=control,
                    target_speed=target_speeds[vehicle_num],
                    timestamp=timestamp,
                    is_junction=junction_flags[vehicle_num],
                    affected_light_id=affected_light_ids[vehicle_num],
                )

            if self.destroy_hazard_actors:
                try:
                    self.loc_queue[vehicle_num].enqueue(self._vehicle.get_location())
                    self._detect_and_destroy_hazard_actors(vehicle_num)
                except Exception:
                    print("destroy hazard actors failed")

        return control_all

    def _update_vlm_token_usage(self, token_usage):
        if not isinstance(token_usage, dict):
            return
        updated = False
        for key in self.vlm_token_usage.keys():
            try:
                value = int(token_usage.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value:
                self.vlm_token_usage[key] += value
                updated = True
        if updated:
            self._write_vlm_token_usage()

    def _vlm_token_usage_output_paths(self):
        paths = []
        if self.save_path is not None:
            paths.append(pathlib.Path(self.save_path) / "vlm_token_usage.json")

        checkpoint_endpoint = os.environ.get("CHECKPOINT_ENDPOINT")
        if checkpoint_endpoint and not checkpoint_endpoint.startswith(("http:", "https:", "ftp:")):
            checkpoint_root = pathlib.Path(checkpoint_endpoint).resolve().parent
            paths.append(checkpoint_root / "vlm_token_usage.json")
            for ego_id in range(self.ego_vehicles_num):
                paths.append(checkpoint_root / ("ego_vehicle_%d" % ego_id) / "vlm_token_usage.json")
        return paths

    def _write_vlm_token_usage(self):
        payload = {
            "agent": self.agent_name,
            "step": int(self.step),
            "token_usage": dict(self.vlm_token_usage),
        }
        for output_path in self._vlm_token_usage_output_paths():
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(output_path), "w") as file_obj:
                    json.dump(payload, file_obj, indent=2, sort_keys=True)
            except Exception as exc:
                if self.debug:
                    print("[covlm] failed to write VLM token usage %s: %s" % (output_path, exc), flush=True)

    def _clear_cov2v_vehicle_state(self, vehicle_num):
        if vehicle_num < 0 or vehicle_num >= self.ego_vehicles_num:
            return
        self.latest_target_speed_actions[vehicle_num] = "TARGET_NORMAL"
        self.latest_target_speed_action_payloads[vehicle_num] = {}
        self.latest_brake_hazards[vehicle_num] = {}

    def _has_required_input_data(self, input_data, vehicle_num):
        required_keys = [
            "gps_%d" % vehicle_num,
            "imu_%d" % vehicle_num,
            "speed_%d" % vehicle_num,
            "rgb_front_%d" % vehicle_num,
            "rgb_left_%d" % vehicle_num,
            "rgb_right_%d" % vehicle_num,
            "rgb_front_tele_%d" % vehicle_num,
        ]
        for key in required_keys:
            if key not in input_data:
                return False
        return True

    def _get_control_with_target_speed_action(
        self,
        target,
        far_target,
        near_command,
        far_command,
        tick_data,
        target_speed_action,
    ):
        pos = self._get_position(tick_data, self.vehicle_num)
        theta = tick_data["compass_{}".format(self.vehicle_num)]
        speed = tick_data["move_state_{}".format(self.vehicle_num)]["speed"]
        if "lidar_{}".format(self.vehicle_num) in tick_data:
            lidar = tick_data["lidar_{}".format(self.vehicle_num)]
        else:
            lidar = None
        cur_lidar_pose = carla.Location(
            x=self.lidar_pose.location.x,
            y=self.lidar_pose.location.y,
            z=self.lidar_pose.location.z,
        )
        self._vehicle.get_transform().transform(cur_lidar_pose)

        view_data = {
            "theta": theta,
            "lidar": lidar,
            "cur_lidar_pose": cur_lidar_pose,
        }

        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        self.should_slow = should_slow
        target_speed = self.slow_speed if should_slow else self.max_speed
        brake = self._should_brake(near_command, view_data)
        self.latest_brake_hazards[self.vehicle_num] = self._collect_brake_hazards()
        if (
            brake
            and target_speed_action in ("TARGET_SLOW", "TARGET_NORMAL", "TARGET_FAST")
            and self._is_only_negotiated_vehicle_hazard(self.vehicle_num)
            and self._hazard_group_vehicles_are_yielding(self.vehicle_num)
        ):
            brake = False
        self.should_brake = brake

        if target_speed_action == "TARGET_0":
            target_speed = 0.0
            brake = True
        elif target_speed_action == "TARGET_SLOW":
            target_speed = 2.0
        elif target_speed_action == "TARGET_NORMAL":
            target_speed = target_speed
        elif target_speed_action == "TARGET_FAST":
            target_speed = target_speed * 1.5

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake==True:
            steer *= 0.5
            throttle = 0.0
        elif target_speed < speed:
            brake = 0.1

        return steer, throttle, brake, target_speed

    def _is_junction_vehicle_hazard(self, vehicle_list, command):
        res = []
        ego_transform = self._vehicle.get_transform()
        ego_location = self._vehicle.get_location()
        o1 = _orientation(ego_transform.rotation.yaw)
        x1 = self._vehicle.bounding_box.extent.x
        p1 = ego_location + x1 * ego_transform.get_forward_vector()
        w1 = self._map.get_waypoint(p1)
        s1 = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        if command == RoadOption.RIGHT:
            shift_angle = 25
        elif command == RoadOption.LEFT:
            shift_angle = -25
        else:
            shift_angle = 0
        # Use the original crossing-path junction check, but look farther ahead
        # so straight/turning conflicts are detected before vehicles get too close.
        v1 = (4 * s1 + 5) * _orientation(ego_transform.rotation.yaw + shift_angle)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            if not target_vehicle.is_alive:
                continue

            target_transform = target_vehicle.get_transform()
            o2 = _orientation(target_transform.rotation.yaw)
            o2_left = _orientation(target_transform.rotation.yaw - 15)
            o2_right = _orientation(target_transform.rotation.yaw + 15)
            x2 = target_vehicle.bounding_box.extent.x

            p2 = target_vehicle.get_location()
            p2_hat = p2 - (x2 + 2) * target_transform.get_forward_vector()
            w2 = self._map.get_waypoint(p2)
            s2 = np.linalg.norm(_numpy(target_vehicle.get_velocity()))

            v2 = (4 * s2 + 2 * x2 + 6) * o2
            v2_left = (4 * s2 + 2 * x2 + 6) * o2_left
            v2_right = (4 * s2 + 2 * x2 + 6) * o2_right

            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))
            distance = ego_location.distance(p2)

            if distance > 20:
                continue
            if w1.is_junction == False and w2.is_junction == False:
                continue

            if angle_between_heading < 15.0 or angle_between_heading > 165:
                continue
            collides, collision_point = get_collision(
                _numpy(p1), v1, _numpy(p2_hat), v2
            )
            if collides is None:
                collides, collision_point = get_collision(
                    _numpy(p1), v1, _numpy(p2_hat), v2_left
                )
            if collides is None:
                collides, collision_point = get_collision(
                    _numpy(p1), v1, _numpy(p2_hat), v2_right
                )

            light = self._find_closest_valid_traffic_light(
                target_vehicle.get_location(), min_dis=10
            )
            if (
                light is not None
                and (
                    self._vehicle.get_traffic_light_state()
                    == carla.libcarla.TrafficLightState.Yellow
                    or self._vehicle.get_traffic_light_state()
                    == carla.libcarla.TrafficLightState.Red
                )
            ):
                continue
            if collides:
                res.append(target_vehicle)
        return res

    def _collect_brake_hazards(self):
        hazard_attrs = [
            ("vehicle", "is_vehicle_present"),
            ("lane_vehicle", "is_lane_vehicle_present"),
            ("junction_vehicle", "is_junction_vehicle_present"),
            ("pedestrian", "is_pedestrian_present"),
            ("bike", "is_bike_present"),
            ("red_light", "is_red_light_present"),
            ("stop_sign", "is_stop_sign_present"),
        ]
        hazards = {}
        for label, attr_name in hazard_attrs:
            actor_ids = list(getattr(self, attr_name, []) or [])
            if actor_ids:
                hazards[label] = actor_ids
        return hazards

    def _overridable_brake_hazard_attr_names(self):
        return (
            "is_lane_vehicle_present",
            "is_junction_vehicle_present",
            "is_pedestrian_present",
            "is_bike_present",
            "is_red_light_present",
            "is_stop_sign_present",
        )

    def _actor_id_to_debug_name(self, actor_id):
        try:
            actor_id = int(actor_id)
        except (TypeError, ValueError):
            return str(actor_id)

        for vehicle_num in range(self.ego_vehicles_num):
            actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if actor is not None and actor.is_alive and actor.id == actor_id:
                return "veh_%d" % vehicle_num
        return "actor_%d" % actor_id

    def _format_brake_hazards(self, vehicle_num):
        if vehicle_num < 0 or vehicle_num >= len(self.latest_brake_hazards):
            return ""
        hazards = self.latest_brake_hazards[vehicle_num] or {}
        if not hazards:
            return ""

        parts = []
        for label in (
            "vehicle",
            "lane_vehicle",
            "junction_vehicle",
            "pedestrian",
            "bike",
            "red_light",
            "stop_sign",
        ):
            actor_ids = hazards.get(label, [])
            if not actor_ids:
                continue
            names = [self._actor_id_to_debug_name(actor_id) for actor_id in actor_ids]
            parts.append("%s=%s" % (label, ",".join(names)))
        return "; ".join(parts)

    def _is_only_negotiated_vehicle_hazard(self, ego_vehicle_num):
        if getattr(self, "is_vehicle_present", []) or []:
            return False

        negotiated_hazard_actor_ids = set()
        for attr_name in self._overridable_brake_hazard_attr_names():
            negotiated_hazard_actor_ids.update(getattr(self, attr_name, []) or [])
        if not negotiated_hazard_actor_ids:
            return False

        group_vehicle_nums = self._get_discussion_group_vehicle_nums(ego_vehicle_num)
        if not group_vehicle_nums:
            return False

        group_actor_ids = set()
        for vehicle_num in group_vehicle_nums:
            if vehicle_num == ego_vehicle_num:
                continue
            actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if actor is not None and actor.is_alive:
                group_actor_ids.add(actor.id)

        return bool(group_actor_ids) and negotiated_hazard_actor_ids.issubset(group_actor_ids)

    def _hazard_group_vehicles_are_yielding(self, ego_vehicle_num):
        hazard_actor_ids = set()
        for attr_name in self._overridable_brake_hazard_attr_names():
            hazard_actor_ids.update(getattr(self, attr_name, []) or [])
        if not hazard_actor_ids:
            return False

        final_actions = self._get_discussion_group_final_actions(ego_vehicle_num)
        if not final_actions:
            return False

        yielding_actions = {"TARGET_0"}
        for vehicle_num in self._get_discussion_group_vehicle_nums(ego_vehicle_num):
            if vehicle_num == ego_vehicle_num:
                continue
            actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if actor is None or not actor.is_alive or actor.id not in hazard_actor_ids:
                continue
            vehicle_id = "veh_%d" % vehicle_num
            if final_actions.get(vehicle_id) not in yielding_actions:
                return False
        return True

    def _get_discussion_group_final_actions(self, ego_vehicle_num):
        discussion_process = self.latest_discussion_process
        if not isinstance(discussion_process, dict):
            return {}

        ego_vehicle_id = "veh_%d" % ego_vehicle_num
        for group in discussion_process.get("group_discussions", []):
            group_ids = [str(vehicle_id) for vehicle_id in group.get("vehicle_ids", [])]
            if ego_vehicle_id not in group_ids:
                continue
            final_decision = group.get("final_decision", {})
            final_actions = {}
            for item in final_decision.get("final_actions", []):
                vehicle_id = str(item.get("vehicle_id", "")).strip()
                target_speed_action = str(item.get("target_speed_action", "")).strip()
                if vehicle_id and target_speed_action:
                    final_actions[vehicle_id] = target_speed_action
            return final_actions
        return {}

    def _get_discussion_group_vehicle_nums(self, ego_vehicle_num):
        discussion_process = self.latest_discussion_process
        if not isinstance(discussion_process, dict):
            return []

        ego_vehicle_id = "veh_%d" % ego_vehicle_num
        for group in discussion_process.get("group_discussions", []):
            group_ids = [str(vehicle_id) for vehicle_id in group.get("vehicle_ids", [])]
            if ego_vehicle_id not in group_ids:
                continue
            vehicle_nums = []
            for vehicle_id in group_ids:
                vehicle_num = self._vehicle_num_from_cov2v_id(vehicle_id)
                if vehicle_num is not None:
                    vehicle_nums.append(vehicle_num)
            return vehicle_nums
        return []

    def _print_cov2v_summary(self, timestamp, controls, target_speeds, discussion_process, negotiated):
        lines = [
            "[cov2v] step=%s ts=%.2f negotiated=%s" % (
                self.step,
                timestamp,
                negotiated,
            ),
        ]
        grouped_ids = set()
        if isinstance(discussion_process, dict):
            for group in discussion_process.get("group_discussions", []):
                group_ids = [str(vehicle_id) for vehicle_id in group.get("vehicle_ids", [])]
                grouped_ids.update(group_ids)
                source = str(group.get("final_decision_source", "unknown"))
                final_decision = group.get("final_decision", {})
                final_actions = [
                    "%s:%s" % (str(item.get("vehicle_id", "?")), str(item.get("target_speed_action", "?")))
                    for item in final_decision.get("final_actions", [])
                ]
                group_index = group.get("group_index", "?")
                lines.append(
                    "  discussion_group_%s | source=%s | vehicles=%s | final_actions=%s"
                    % (
                        str(group_index),
                        source,
                        ", ".join(group_ids) if group_ids else "n/a",
                        ", ".join(final_actions) if final_actions else "n/a",
                    )
                )
                for vehicle_id in group_ids:
                    vehicle_num = self._vehicle_num_from_cov2v_id(vehicle_id)
                    if vehicle_num is not None:
                        self._append_cov2v_vehicle_debug_line(lines, vehicle_num, controls, target_speeds, indent="    ")

            non_grouped_ids = [
                str(vehicle_id) for vehicle_id in discussion_process.get("non_grouped_vehicle_ids", [])
            ]
            if non_grouped_ids:
                lines.append("  non_discussion_vehicles | vehicles=%s" % ", ".join(non_grouped_ids))
                for vehicle_id in non_grouped_ids:
                    vehicle_num = self._vehicle_num_from_cov2v_id(vehicle_id)
                    if vehicle_num is not None:
                        self._append_cov2v_vehicle_debug_line(lines, vehicle_num, controls, target_speeds, indent="    ")

        shown_vehicle_nums = {
            vehicle_num for vehicle_num in (self._vehicle_num_from_cov2v_id(vehicle_id) for vehicle_id in grouped_ids)
            if vehicle_num is not None
        }
        for vehicle_num in range(self.ego_vehicles_num):
            if vehicle_num in shown_vehicle_nums:
                continue
            if not isinstance(discussion_process, dict):
                self._append_cov2v_vehicle_debug_line(lines, vehicle_num, controls, target_speeds, indent="  ")
        print("\n".join(lines), flush=True)

    def _vehicle_num_from_cov2v_id(self, vehicle_id):
        text = str(vehicle_id)
        if not text.startswith("veh_"):
            return None
        try:
            vehicle_num = int(text.split("_", 1)[1])
        except (TypeError, ValueError):
            return None
        if vehicle_num < 0 or vehicle_num >= self.ego_vehicles_num:
            return None
        return vehicle_num

    def _priority_for_vehicle(self, vehicle_num):
        discussion_process = self.latest_discussion_process
        if not isinstance(discussion_process, dict):
            return None
        vehicle_id = "veh_%d" % vehicle_num
        for group in discussion_process.get("group_discussions", []):
            for item in group.get("priorities", []):
                if str(item.get("vehicle_id", "")) == vehicle_id:
                    return item
        return None

    def _append_cov2v_vehicle_debug_line(self, lines, vehicle_num, controls, target_speeds, indent="  "):
        ego_data = self.latest_ego_data[vehicle_num]
        control = controls[vehicle_num] if vehicle_num < len(controls) else None
        if ego_data is None or control in (None, []):
            return
        measurements = ego_data["measurements"]
        result = self.latest_target_speed_action_payloads[vehicle_num]
        reason = str(result.get("reason", "")).strip()
        if len(reason) > 180:
            reason = reason[:177] + "..."
        observations = result.get("cooperative_observations", [])
        target_speed = target_speeds[vehicle_num] if vehicle_num < len(target_speeds) else 0.0
        priority = self._priority_for_vehicle(vehicle_num)
        priority_str = ""
        if priority is not None:
            priority_str = " | priority_score=%s conflict=%s" % (
                priority.get("priority_score", "?"),
                priority.get("conflict_detected", "?"),
            )
        lines.append(
            "%sveh_%d | pos=(%.1f,%.1f,%.2f) v=%.2f intent=%s | target_speed_action=%s target_speed=%.2f%s | control=(steer=%.2f, throttle=%.2f, brake=%.2f)"
            % (
                indent,
                vehicle_num,
                float(measurements["x"]),
                float(measurements["y"]),
                float(measurements["theta"]),
                float(measurements["speed"]),
                "%s->%s"
                % (
                    str(measurements.get("command_intent", "UNKNOWN")),
                    str(measurements.get("future_command_3s_intent", "UNKNOWN")),
                ),
                self.latest_target_speed_actions[vehicle_num],
                float(target_speed),
                priority_str,
                float(control.steer),
                float(control.throttle),
                float(control.brake),
            )
        )
        if reason:
            lines.append("%s  reason: %s" % (indent, reason))
        brake_hazards = self._format_brake_hazards(vehicle_num)
        if brake_hazards and float(control.brake) > 0.0:
            lines.append("%s  brake_hazards: %s" % (indent, brake_hazards))
        if observations:
            rel_parts = []
            for obs in observations:
                rel_parts.append(
                    "%s: %s, %s, v=%.1f, intent=%s->%s"
                    % (
                        str(obs.get("other_id", "?")),
                        str(obs.get("relation_to_ego", "relative position unknown")),
                        str(obs.get("heading_relation", "heading unknown")),
                        float(obs.get("speed_mps", 0.0)),
                        str(obs.get("route_intent", "UNKNOWN")),
                        str(obs.get("future_route_intent_3s", "UNKNOWN")),
                    )
                )
            lines.append("%s  neighbors: %s" % (indent, ", ".join(rel_parts)))

    def _save_minimal_step(
        self,
        ego_id,
        ego_data,
        near_node,
        far_node,
        control,
        target_speed,
        timestamp,
        is_junction,
        affected_light_id,
    ):
        if ego_data is None or self.save_path is None:
            return None

        frame = self.step // max(self.save_skip_frames, 1)
        ego_root = self.save_path / pathlib.Path(f"ego_vehicle_{ego_id}")
        for subdir in (
            "measurements",
            "rgb_front",
            "rgb_left",
            "rgb_right",
            "rgb_front_tele",
            "rgb_topdown",
            "bev_result",
        ):
            (ego_root / subdir).mkdir(parents=True, exist_ok=True)

        for image_name in (
            "rgb_front",
            "rgb_left",
            "rgb_right",
            "rgb_front_tele",
            "rgb_topdown",
        ):
            image = ego_data.get(image_name)
            if image is None:
                continue
            output_path = ego_root / image_name / ("%04d.jpg" % frame)
            try:
                ok = cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not ok and self.debug:
                    print("[covlm] cv2.imwrite returned false for %s" % output_path, flush=True)
            except Exception as exc:
                if self.debug:
                    print("[covlm] failed to save %s for veh_%d: %s" % (image_name, ego_id, exc), flush=True)

        measurements = ego_data["measurements"]
        try:
            self._save_bev_result(
                ego_id=ego_id,
                ego_root=ego_root,
                frame=frame,
                measurements=measurements,
                control=control,
                target_speed=target_speed,
                topdown_rgb=ego_data.get("rgb_topdown"),
            )
        except Exception as exc:
            if self.debug:
                print("[covlm] failed to save bev_result for veh_%d: %s" % (ego_id, exc), flush=True)
        measurement_payload = {
            "frame": int(frame),
            "timestamp_sec": float(timestamp),
            "gps_x": float(measurements["gps_x"]),
            "gps_y": float(measurements["gps_y"]),
            "x": float(measurements["x"]),
            "y": float(measurements["y"]),
            "theta": float(measurements["theta"]),
            "z": float(measurements["z"]),
            "speed": float(measurements["speed"]),
            "command": int(measurements["command"]),
            "command_intent": str(measurements.get("command_intent", "UNKNOWN")),
            "future_command_3s": int(measurements.get("future_command_3s", measurements["command"])),
            "future_command_3s_intent": str(
                measurements.get(
                    "future_command_3s_intent",
                    measurements.get("command_intent", "UNKNOWN"),
                )
            ),
            "target_speed": float(target_speed),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "target_speed_action": self.latest_target_speed_actions[ego_id],
            "target_speed_action_reason": self.latest_target_speed_action_payloads[ego_id].get("reason", ""),
            "target_speed_action_payload": self.latest_target_speed_action_payloads[ego_id],
            "discussion_process": self.latest_discussion_process,
            "driver_intent": (self._cov2v_driver_intents or {}).get("veh_%d" % ego_id),
            "near_node": [float(near_node[0]), float(near_node[1])],
            "far_node": [float(far_node[0]), float(far_node[1])],
            "is_junction": bool(is_junction),
            "affected_light_id": int(affected_light_id),
        }
        measurement_path = ego_root / "measurements" / ("%04d.json" % frame)
        with open(measurement_path, "w") as file_obj:
            json.dump(measurement_payload, file_obj, indent=4)

        return frame

    def _save_bev_result(
        self,
        ego_id,
        ego_root,
        frame,
        measurements,
        control,
        target_speed,
        topdown_rgb=None,
    ):
        bev_rgb = self._get_result_overhead_rgb(ego_id, topdown_rgb)
        if bev_rgb is None:
            return
        try:
            bev_rgb = self._draw_group_bev_overlay(bev_rgb.copy(), ego_id)
        except Exception as exc:
            if self.debug:
                print("[covlm] BEV overlay failed for veh_%d: %s" % (ego_id, exc), flush=True)

        panel = self._build_bev_text_panel(
            height=bev_rgb.shape[0],
            width=520,
            ego_id=ego_id,
            measurements=measurements,
            control=control,
            target_speed=target_speed,
        )
        result_rgb = np.concatenate([bev_rgb, panel], axis=1)
        output_path = ego_root / "bev_result" / ("%04d.jpg" % frame)
        ok = cv2.imwrite(str(output_path), cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
        if not ok and self.debug:
            print("[covlm] cv2.imwrite returned false for %s" % output_path, flush=True)

    def _bev_pixels_per_meter(self, bev_rgb):
        if (
            self.birdview_producer is not None
            and bev_rgb.shape[1] == self.birdview_producer.target_size.width
            and bev_rgb.shape[0] == self.birdview_producer.target_size.height
        ):
            return float(self.birdview_producer._pixels_per_meter)

        topdown_pose = self.camera_topdown_pose
        height = max(float(topdown_pose.location.z), 1.0)
        fov_rad = math.radians(float(self._topdown_rgb_sensor_data.get("fov", 80)))
        visible_width_m = 2.0 * height * math.tan(fov_rad * 0.5)
        return float(bev_rgb.shape[1]) / max(visible_width_m, 1.0)

    def _discussion_group_color_by_vehicle(self):
        palette = [
            (230, 76, 60),
            (52, 152, 219),
            (46, 204, 113),
            (241, 196, 15),
            (155, 89, 182),
            (230, 126, 34),
            (26, 188, 156),
        ]
        color_by_vehicle = {}
        discussion_process = self.latest_discussion_process
        if not isinstance(discussion_process, dict):
            return color_by_vehicle
        for group_offset, group in enumerate(discussion_process.get("group_discussions", [])):
            color = palette[group_offset % len(palette)]
            for vehicle_id in group.get("vehicle_ids", []):
                vehicle_num = self._vehicle_num_from_cov2v_id(vehicle_id)
                if vehicle_num is not None:
                    color_by_vehicle[vehicle_num] = color
        return color_by_vehicle

    def _world_point_to_bev_pixel(self, point, ego_actor, image_shape, pixels_per_meter):
        ego_transform = ego_actor.get_transform()
        ego_location = ego_transform.location
        yaw = math.radians(ego_transform.rotation.yaw)
        dx = float(point.x - ego_location.x)
        dy = float(point.y - ego_location.y)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        right = -dx * math.sin(yaw) + dy * math.cos(yaw)
        height, width = image_shape[:2]
        px = int(round(width * 0.5 + right * pixels_per_meter))
        py = int(round(height * 0.5 - forward * pixels_per_meter))
        return (px, py)

    def _vehicle_bev_polygon(self, actor, ego_actor, image_shape, pixels_per_meter):
        transform = actor.get_transform()
        extent = actor.bounding_box.extent
        local_corners = [
            carla.Location(x=extent.x, y=extent.y, z=0.0),
            carla.Location(x=extent.x, y=-extent.y, z=0.0),
            carla.Location(x=-extent.x, y=-extent.y, z=0.0),
            carla.Location(x=-extent.x, y=extent.y, z=0.0),
        ]
        world_corners = []
        for corner in local_corners:
            point = carla.Location(corner.x, corner.y, corner.z)
            transform.transform(point)
            world_corners.append(point)
        return np.array(
            [
                self._world_point_to_bev_pixel(point, ego_actor, image_shape, pixels_per_meter)
                for point in world_corners
            ],
            dtype=np.int32,
        )

    def _draw_label_with_halo(self, image, text, anchor, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(anchor[0] + 6)
        y = int(anchor[1] - 8)
        y = max(text_size[1] + 6, y)
        x = max(4, min(x, image.shape[1] - text_size[0] - 8))
        y = max(text_size[1] + 6, min(y, image.shape[0] - 6))
        cv2.rectangle(
            image,
            (x - 3, y - text_size[1] - 4),
            (x + text_size[0] + 3, y + baseline + 3),
            (18, 18, 18),
            -1,
        )
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_group_bev_overlay(self, bev_rgb, ego_id):
        ego_actor = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
        if ego_actor is None or not ego_actor.is_alive:
            return bev_rgb

        pixels_per_meter = self._bev_pixels_per_meter(bev_rgb)
        color_by_vehicle = self._discussion_group_color_by_vehicle()
        overlay = bev_rgb.copy()
        vehicle_draw_items = []

        for vehicle_num in range(self.ego_vehicles_num):
            actor = CarlaDataProvider.get_hero_actor(hero_id=vehicle_num)
            if actor is None or not actor.is_alive:
                continue

            group_color = color_by_vehicle.get(vehicle_num)
            label_color = group_color if group_color is not None else (255, 255, 255)
            center_px = self._world_point_to_bev_pixel(
                actor.get_location(),
                ego_actor,
                bev_rgb.shape,
                pixels_per_meter,
            )

            polygon = self._vehicle_bev_polygon(actor, ego_actor, bev_rgb.shape, pixels_per_meter)
            vehicle_draw_items.append((vehicle_num, center_px, polygon, group_color, label_color))

            if group_color is not None:
                cv2.fillPoly(overlay, [polygon], group_color)

        cv2.addWeighted(overlay, 0.32, bev_rgb, 0.68, 0.0, bev_rgb)

        for vehicle_num, center_px, polygon, group_color, label_color in vehicle_draw_items:
            if group_color is not None:
                cv2.polylines(bev_rgb, [polygon], True, group_color, 2, cv2.LINE_AA)
            self._draw_label_with_halo(bev_rgb, "veh_%d" % vehicle_num, center_px, label_color)

        return bev_rgb

    def _get_result_overhead_rgb(self, ego_id, topdown_rgb=None):
        if topdown_rgb is not None:
            return topdown_rgb
        if self.birdview_producer is None:
            return None
        vehicle = CarlaDataProvider.get_hero_actor(hero_id=ego_id)
        if vehicle is None:
            return None
        try:
            birdview = self.birdview_producer.produce(
                agent_vehicle=vehicle,
                actor_exist=True,
            )
            return BirdViewProducer.as_rgb(birdview)
        except Exception as exc:
            if self.debug:
                print(
                    "[covlm] save overhead result failed for veh_%d: %s" % (ego_id, exc),
                    flush=True,
                )
            return None

    def _build_bev_text_panel(self, height, width, ego_id, measurements, control, target_speed):
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        action = str(self.latest_target_speed_actions[ego_id])
        payload = self.latest_target_speed_action_payloads[ego_id] or {}
        reason = str(payload.get("reason", "")).strip() or "No VLM reason available."
        risks = payload.get("key_risks", [])
        priority = self._priority_for_vehicle(ego_id)
        priority_line = (
            "priority_score: %s (conflict=%s)" % (priority.get("priority_score", "?"), priority.get("conflict_detected", "?"))
            if priority is not None
            else "priority_score: n/a"
        )

        lines = [
            "CoVLM Decision",
            "vehicle: veh_%d" % ego_id,
            "target_speed_action: %s" % action,
            priority_line,
            "route_intent: %s -> %s"
            % (
                str(measurements.get("command_intent", "UNKNOWN")),
                str(measurements.get("future_command_3s_intent", "UNKNOWN")),
            ),
            "speed: %.2f m/s" % float(measurements.get("speed", 0.0)),
            "target_speed: %.2f m/s" % float(target_speed),
            "control: steer %.2f  throttle %.2f  brake %.2f"
            % (float(control.steer), float(control.throttle), float(control.brake)),
            "",
            "Reason:",
        ]
        lines.extend(self._wrap_text(reason, max_chars=44))
        if risks:
            lines.append("")
            lines.append("Key risks:")
            for risk in risks[:3]:
                lines.extend(self._wrap_text("- %s" % str(risk), max_chars=44))

        y = 32
        for idx, line in enumerate(lines):
            if y >= height - 18:
                break
            font_scale = 0.72 if idx == 0 else 0.52
            thickness = 2 if idx == 0 else 1
            color = (20, 20, 20) if idx != 2 else self._action_text_color(action)
            cv2.putText(
                panel,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += 30 if idx == 0 else 23
        return panel

    def _wrap_text(self, text, max_chars):
        words = str(text).split()
        if not words:
            return [""]
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _action_text_color(self, action):
        colors = {
            "TARGET_0": (200, 40, 40),
            "TARGET_SLOW": (220, 135, 20),
            "TARGET_NORMAL": (30, 120, 210),
            "TARGET_FAST": (30, 145, 60),
        }
        return colors.get(str(action), (20, 20, 20))

    def get_latest_ego_data(self):
        return self.latest_ego_data

    def destroy(self):
        if self._cov2v_bridge is not None:
            self._cov2v_bridge.close()
