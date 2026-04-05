import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import traci


class Old_Env:
    """SUMO multi-agent VSL environment.

    Each agent controls the speed limit of one controlled edge (discrete 6 levels).

    This version exposes objective metrics in `info` so we can build the requested
    "training / comparison matrix":
      - emissions (CO2)
      - travel time
      - safety surrogate (TTC-based)

    It also supports switching traffic demand by selecting different route files:
      - undersaturated (free flow)
      - saturated (free flow)
      - oversaturated (free flow)

    Notes
    -----
    * SUMO is started via TraCI. If your SUMO version supports labeled connections,
      we use labels so multiple env instances don't collide.
    * Paths are resolved relative to this repository, not the current working dir.
    """

    DEMAND_TO_ROUTE = {
        "undersaturated": "undersaturated.rou.xml",
        "saturated": "saturated.rou.xml",
        # repo doesn't contain oversaturated.rou.xml; use the provided oversaturated_sine.rou.xml
        "oversaturated": "oversaturated.rou.xml",
    }

    def __init__(
        self,
        env_id: int = 0,
        demand: str = "saturated",
        route_file: Optional[Union[str, Path]] = None,
        sumo_binary: str = "sumo",
    ):
        self.env_id = int(env_id)

        # Discrete speed levels (m/s): [80,70,60,50,40,30] km/h
        self.actions = [22.22, 19.44, 16.67, 13.89, 11.11, 8.33]

        # Simulated time step per environment 'step' (SUMO seconds)
        self.SIMULATION_TIME = 1
        self.time = 0
        self.all_simulation_time = 7200  # 2 hours
        self.controlled = True
        self.control_interval = 1

        # TRACI
        self.traci_label = f"sumo_env_{self.env_id}"
        self.conn = None  # either traci.Connection or the traci module (fallback)

        # Resolve repo paths
        self.repo_dir = Path(__file__).resolve().parent.parent
        self.sumo_dir = self.repo_dir / "sumo"
        self.net_file = self.sumo_dir / "merage.net.xml"
        self.sumo_binary = str(sumo_binary)

        # Demand / route file
        self.demand = str(demand)
        self.route_file = None  # set below
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
            # Keep backward compatibility with your original code, but make the error explicit.
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

        # Episode accumulators (for logging/plotting)
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
        """Set traffic demand scenario by name and update the route file."""
        self.demand = str(demand).strip().lower()
        self.route_file = self._resolve_route_from_demand(self.demand)
        if not self.route_file.exists():
            raise FileNotFoundError(f"Route file not found: {self.route_file}")

    def set_route_file(self, route_file: Union[str, Path]):
        """Set an explicit route file path (overrides demand mapping)."""
        p = Path(route_file).expanduser()
        if not p.is_absolute():
            # interpret relative to the sumo directory
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
            # connection object
            self.conn.close()
        except Exception:
            try:
                # module-level fallback
                traci.close()
            except Exception:
                pass
        self.conn = None

    def _ensure_clean_connection(self):
        # close our stored connection first
        self._close_conn()
        # best-effort: also try close by label if exists
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

        # Try start with label (SUMO>=1.0 generally supports this).
        try:
            traci.start(sumo_cmd, label=self.traci_label)
            self.conn = traci.getConnection(self.traci_label)
        except TypeError:
            # Fallback: older SUMO versions without label parameter
            traci.start(sumo_cmd)
            self.conn = traci

    # -----------------
    # Observation / act
    # -----------------

    def create_state_representation(self):
        """obs_dim = 13.

        For each agent i:
          - occupancy vector is circularly shifted so obs[i][0] is always the *local* occupancy
          - obs[i][12] is agent_id_norm (0..1) to break symmetry under shared policy
        """
        occupancies = [self.conn.edge.getLastStepOccupancy(edge) for edge in self.edges]
        occupancies = np.asarray(occupancies, dtype=np.float32)  # (12,)

        obs = np.zeros((self.agent_num, 13), dtype=np.float32)
        for i in range(self.agent_num):
            rolled = np.roll(occupancies, -i)  # local occupancy goes to index 0
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
        """Scalarized multi-objective reward + metrics.

        Reward is only used for training. Metrics are used for evaluation/comparison.

        Returns
        -------
        reward: float
        metrics: dict
            step_co2_kg, step_travel_time_h, ttc_penalty, ttc_violations, ttc_min
        """
        # CO2 emission:
        # TraCI typically returns mg/s; with 1s step we treat it as mg and convert to kg: /1e6
        emissions_kg = []
        for edge in self.edges:
            emissions_kg.append(self.conn.edge.getCO2Emission(edge) / 1e6)
        total_emissions_kg = float(np.sum(emissions_kg))

        # Travel time (hours) proxy
        travel_times_h = []
        for edge in self.edges:
            tt = float(self.conn.edge.getTraveltime(edge))
            if not np.isfinite(tt) or tt < 0:
                tt = 0.0
            travel_times_h.append(tt / 3600.0)
        total_travel_h = float(np.sum(travel_times_h))

        # TTC penalty
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

        # Normalization (adjust as needed for your network size)
        MAX_EMISSIONS_KG = 100.0
        MAX_TRAVEL_TIME_H = 10.0
        emissions_norm = min(total_emissions_kg / MAX_EMISSIONS_KG, 1.0)
        travel_norm = min(total_travel_h / MAX_TRAVEL_TIME_H, 1.0)
        ttc_norm = min(ttc_penalty / max(len(self.edges), 1), 1.0)

        # weights
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
        self.time = 0
        self._reset_episode_stats()

        # Reset speeds on all edges
        for edge in self.edges:
            self.conn.edge.setMaxSpeed(edge, self.default_speed)
        self.edges_speed = {edge: self.default_speed for edge in self.edges}

        return self.create_state_representation()

    def step(self, actions):
        """Advance one environment step.

        Parameters
        ----------
        actions : list[int]
            length agent_num, each int in [0, len(self.actions)-1]
        """
        # Control update only at control interval
        if self.time % self.control_interval == 0:
            if self.controlled:
                for i, edge in enumerate(self.edges):
                    self.execute_action(edge, actions[i])
            else:
                for edge in self.edges:
                    self.conn.edge.setMaxSpeed(edge, self.default_speed)
                    self.edges_speed[edge] = self.default_speed

        # advance SUMO
        for _ in range(self.SIMULATION_TIME):
            self.conn.simulationStep()
        self.time += self.SIMULATION_TIME

        reward, metrics = self.calculate_reward_and_metrics()
        state = self.create_state_representation()
        done = self.evaluate_terminal_condition()

        # accumulate episode metrics
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
            "sim_time": int(self.time),
            "demand": str(self.demand),
            "route_file": str(self.route_file) if self.route_file is not None else "",
        })
        return state, reward, done, info

    def evaluate_terminal_condition(self):
        if self.time >= self.all_simulation_time:
            self._ensure_clean_connection()
            return True
        return False
