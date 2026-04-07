import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import traci


class Old_Env:
    """SUMO multi-agent VSL environment.
    """

    DEMAND_TO_ROUTE = {
        "undersaturated": "freeflow.rou.xml",
        "saturated": "capacity.rou.xml",
        "oversaturated": "extreme.rou.xml",
    }

    def __init__(
        self,
        env_id: int = 0,
        demand: str = "freeflow",
        route_file: Optional[Union[str, Path]] = None,
        sumo_binary: str = "sumo",
        sumo_step_length: float = 0.2,
        control_interval: float = 1.0,
        emergencydecel_warning_threshold: float = 1.1,
    ):
        self.env_id = int(env_id)

        # Discrete speed levels (m/s): [80,70,60,50,40,30] km/h
        self.actions = [22.22, 19.44, 16.67, 13.89, 11.11, 8.33]

        # Physical simulation horizon (seconds)
        self.time = 0.0
        self.all_simulation_time = 7200.0  # 2 hours
        self.controlled = True

        # SUMO internal stepping vs. RL control stepping
        self.sumo_step_length = float(sumo_step_length)
        self.control_interval = float(control_interval)
        if self.sumo_step_length <= 0:
            raise ValueError("sumo_step_length must be > 0")
        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")

        ratio = self.control_interval / self.sumo_step_length
        rounded_ratio = round(ratio)
        if not np.isclose(ratio, rounded_ratio):
            raise ValueError(
                "control_interval must be an integer multiple of sumo_step_length. "
                f"Got control_interval={self.control_interval}, "
                f"sumo_step_length={self.sumo_step_length}."
            )
        self.steps_per_action = int(rounded_ratio)

        # Backward-compatible aliases used by some training scripts
        self.SIMULATION_TIME = self.control_interval
        self.emergencydecel_warning_threshold = float(emergencydecel_warning_threshold)

        # TRACI
        self.traci_label = f"sumo_env_{self.env_id}"
        self.conn = None

        # Resolve repo paths
        self.repo_dir = Path(__file__).resolve().parent.parent
        self.sumo_dir = self.repo_dir / "sumo"
        self.net_file = self.sumo_dir / "merage.net.xml"
        self.sumo_binary = str(sumo_binary)

        # Demand / route file
        self.demand = str(demand)
        self.route_file = None
        if route_file is not None:
            self.set_route_file(route_file)
        else:
            self.set_demand(self.demand)

        # Check SUMO_HOME for tools (optional, but common in SUMO workflows)
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            if tools not in sys.path:
                sys.path.append(tools)
        else:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'.")

        # Controlled edges (12)
        self.edges = [
            "kexueN1", "kexueN2", "kexueS1", "kexueS2",
            "tianhuE1", "tianhuW1",
            "HuangshanE1", "HuangshanE2", "HuangshanE3",
            "HuangshanW1", "HuangshanW2",
            "tianzhiN1",
        ]
        self.agent_num = len(self.edges)

        # Default speed (m/s)
        self.default_speed = 19.44  # 70 km/h
        self.edges_speed = {edge: self.default_speed for edge in self.edges}

        # Episode accumulators
        self._reset_episode_stats()

    # ---------------------------
    # Demand / route file control
    # ---------------------------

    def _resolve_route_from_demand(self, demand: str) -> Path:
        d = str(demand).strip().lower()
        if d not in self.DEMAND_TO_ROUTE:
            raise ValueError(
                f"Unknown demand='{demand}'. Supported: {list(self.DEMAND_TO_ROUTE.keys())}. "
                f"Or pass an explicit route_file path."
            )
        return (self.sumo_dir / self.DEMAND_TO_ROUTE[d]).resolve()

    def set_demand(self, demand: str):
        self.demand = str(demand).strip().lower()
        self.route_file = self._resolve_route_from_demand(self.demand)
        if not self.route_file.exists():
            raise FileNotFoundError(f"Route file not found: {self.route_file}")

    def set_route_file(self, route_file: Union[str, Path]):
        p = Path(route_file).expanduser()
        if not p.is_absolute():
            p = (self.sumo_dir / p).resolve()
        self.route_file = p
        if not self.route_file.exists():
            raise FileNotFoundError(f"Route file not found: {self.route_file}")

    def _build_sumo_cmd(self):
        """Build SUMO command based on current net + route + horizon."""
        if self.route_file is None:
            raise RuntimeError("route_file is not set. Call set_demand() or set_route_file() first.")
        if not self.net_file.exists():
            raise FileNotFoundError(f"Net file not found: {self.net_file}")

        return [
            self.sumo_binary,
            "--xml-validation=never",
            "-n", str(self.net_file),
            "-r", str(self.route_file),
            "--begin", "0",
            "--end", str(int(self.all_simulation_time)),
            "--step-length", str(self.sumo_step_length),
            "--step-method.ballistic", "true",
            "--default.action-step-length", str(self.sumo_step_length),
            "--emergencydecel.warning-threshold", str(self.emergencydecel_warning_threshold),
        ]

    # --------------
    # TraCI handling
    # --------------

    def _reset_episode_stats(self):
        self.ep_co2_kg = 0.0
        self.ep_travel_time_h = 0.0
        self.ep_ttc_violations = 0
        self.ep_ttc_penalty = 0.0
        self.ep_min_ttc = float("inf")

    def _close_conn(self):
        if self.conn is None:
            return
        try:
            self.conn.close()
        except Exception:
            try:
                traci.close()
            except Exception:
                pass
        self.conn = None

    def _ensure_clean_connection(self):
        self._close_conn()
        try:
            c = traci.getConnection(self.traci_label)
            try:
                c.close()
            except Exception:
                pass
        except Exception:
            pass

    def _start_sumo(self):
        self._ensure_clean_connection()
        sumo_cmd = self._build_sumo_cmd()

        try:
            traci.start(sumo_cmd, label=self.traci_label)
            self.conn = traci.getConnection(self.traci_label)
        except TypeError:
            traci.start(sumo_cmd)
            self.conn = traci

    # -----------------
    # Observation / act
    # -----------------

    def create_state_representation(self):
        occupancies = [self.conn.edge.getLastStepOccupancy(edge) for edge in self.edges]
        occupancies = np.asarray(occupancies, dtype=np.float32)

        obs = np.zeros((self.agent_num, 13), dtype=np.float32)
        for i in range(self.agent_num):
            rolled = np.roll(occupancies, -i)
            obs[i, :12] = rolled
            obs[i, 12] = (i / (self.agent_num - 1)) if self.agent_num > 1 else 0.0
        return obs

    def execute_action(self, edge, action_index):
        new_speed = float(self.actions[int(action_index)])
        self.conn.edge.setMaxSpeed(edge, new_speed)
        self.edges_speed[edge] = new_speed

    # -----------------
    # Objective metrics
    # -----------------

    def calculate_ttc(self, edge):
        veh_ids = self.conn.edge.getLastStepVehicleIDs(edge)
        min_ttc = float("inf")
        for veh in veh_ids:
            leader = self.conn.vehicle.getLeader(veh, dist=250.0)
            if leader is None:
                continue
            leader_id, distance_to_leader = leader[0], leader[1]
            speed_veh = self.conn.vehicle.getSpeed(veh)
            speed_leader = self.conn.vehicle.getSpeed(leader_id)
            relative_speed = speed_veh - speed_leader
            if relative_speed > 0:
                ttc = distance_to_leader / relative_speed
                if ttc < min_ttc:
                    min_ttc = ttc
        return min_ttc

    def calculate_reward_and_metrics(self):
        """Scalarized multi-objective reward + metrics for one env step.

        Since edge CO2 emission is reported as a rate, we multiply by the RL control
        interval so one env step corresponds to the accumulated emissions over the
        1.0 s decision interval.
        """
        emissions_kg = []
        for edge in self.edges:
            emissions_mg_per_s = float(self.conn.edge.getCO2Emission(edge))
            emissions_kg.append((emissions_mg_per_s * self.control_interval) / 1e6)
        total_emissions_kg = float(np.sum(emissions_kg))

        travel_times_h = []
        for edge in self.edges:
            tt = float(self.conn.edge.getTraveltime(edge))
            if not np.isfinite(tt) or tt < 0:
                tt = 0.0
            travel_times_h.append(tt / 3600.0)
        total_travel_h = float(np.sum(travel_times_h))

        TTC_THRESHOLD = 3.0
        ttc_penalty = 0.0
        ttc_violations = 0
        min_ttc = float("inf")
        for edge in self.edges:
            ttc = self.calculate_ttc(edge)
            if ttc < min_ttc:
                min_ttc = ttc
            if ttc < TTC_THRESHOLD:
                ttc_violations += 1
                ttc_penalty += (TTC_THRESHOLD - ttc) / TTC_THRESHOLD

        MAX_EMISSIONS_KG = 100.0
        MAX_TRAVEL_TIME_H = 10.0
        emissions_norm = min(total_emissions_kg / MAX_EMISSIONS_KG, 1.0)
        travel_norm = min(total_travel_h / MAX_TRAVEL_TIME_H, 1.0)
        ttc_norm = min(ttc_penalty / max(len(self.edges), 1), 1.0)

        W_CO2 = 0.5
        W_TRAVEL = 0.3
        W_TTC = 0.2
        reward = -float(W_CO2 * emissions_norm + W_TRAVEL * travel_norm + W_TTC * ttc_norm)

        metrics: Dict[str, Any] = {
            "step_co2_kg": total_emissions_kg,
            "step_travel_time_h": total_travel_h,
            "ttc_penalty": float(ttc_penalty),
            "ttc_violations": int(ttc_violations),
            "ttc_min": float(min_ttc) if np.isfinite(min_ttc) else float("inf"),
        }
        return reward, metrics

    # -----------------
    # Gym-like API
    # -----------------

    def reset(self):
        self._start_sumo()
        self.time = 0.0
        self._reset_episode_stats()

        for edge in self.edges:
            self.conn.edge.setMaxSpeed(edge, self.default_speed)
        self.edges_speed = {edge: self.default_speed for edge in self.edges}

        return self.create_state_representation()

    def step(self, actions):
        """Advance one RL environment step (= one control interval)."""
        if self.controlled:
            for i, edge in enumerate(self.edges):
                self.execute_action(edge, actions[i])
        else:
            for edge in self.edges:
                self.conn.edge.setMaxSpeed(edge, self.default_speed)
                self.edges_speed[edge] = self.default_speed

        for _ in range(self.steps_per_action):
            self.conn.simulationStep()
        self.time += self.control_interval

        reward, metrics = self.calculate_reward_and_metrics()
        state = self.create_state_representation()
        done = self.evaluate_terminal_condition()

        self.ep_co2_kg += metrics["step_co2_kg"]
        self.ep_travel_time_h += metrics["step_travel_time_h"]
        self.ep_ttc_violations += metrics["ttc_violations"]
        self.ep_ttc_penalty += metrics["ttc_penalty"]
        self.ep_min_ttc = min(self.ep_min_ttc, metrics["ttc_min"])

        info = dict(self.edges_speed)
        info.update(metrics)
        info.update({
            "ep_co2_kg": float(self.ep_co2_kg),
            "ep_travel_time_h": float(self.ep_travel_time_h),
            "ep_ttc_violations": int(self.ep_ttc_violations),
            "ep_ttc_penalty": float(self.ep_ttc_penalty),
            "ep_min_ttc": float(self.ep_min_ttc) if np.isfinite(self.ep_min_ttc) else float("inf"),
            "controlled": bool(self.controlled),
            "sim_time": float(self.time),
            "sumo_step_length": float(self.sumo_step_length),
            "control_interval": float(self.control_interval),
            "steps_per_action": int(self.steps_per_action),
            "demand": str(self.demand),
            "route_file": str(self.route_file) if self.route_file is not None else "",
            "emergencydecel_warning_threshold": float(self.emergencydecel_warning_threshold),
        })
        return state, reward, done, info

    def evaluate_terminal_condition(self):
        if self.time >= self.all_simulation_time:
            self._ensure_clean_connection()
            return True
        return False
