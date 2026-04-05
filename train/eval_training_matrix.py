import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from config import ...` works when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import pandas as pd
import torch

from config import get_config
from envs.env_wrappers import DummyVecEnv
from runner.shared.env_runner import EnvRunner as Runner


def _t2n(x):
    return x.detach().cpu().numpy()


def occupancy_rule_based_policy(obs: np.ndarray, action_dim: int) -> np.ndarray:
    """Simple occupancy-threshold VSL baseline.

    Assumes obs[i][0] is the *local* occupancy for agent i.

    Parameters
    ----------
    obs : np.ndarray
        shape (n_envs, n_agents, obs_dim)
    action_dim : int
        discrete action dimension (6)

    Returns
    -------
    np.ndarray
        shape (n_envs, n_agents, 1) discrete indices
    """
    occ = obs[:, :, 0]
    thr = [0.15, 0.30, 0.45, 0.60, 0.75]
    a = np.zeros_like(occ, dtype=np.int64)
    a[occ >= thr[0]] = 1
    a[occ >= thr[1]] = 2
    a[occ >= thr[2]] = 3
    a[occ >= thr[3]] = 4
    a[occ >= thr[4]] = action_dim - 1
    return a[:, :, None]


def set_envs_demand(envs: DummyVecEnv, demand: str):
    """Switch route file for all env instances (must be called before reset())."""
    for e in envs.env_list:
        e.set_demand(demand)


def safety_index_from_ttc_violations(ttc_violations_sum: float, episode_length: int, n_edges: int) -> float:
    """Safety index in [0,1] based on TTC violations (episode-level).

    In Old_Env, at each step we count the number of controlled edges whose min-TTC < 3s.
    So the maximum possible violations per step is `n_edges`.

    We define:
        safety_index = 1 - (violations / (episode_length * n_edges))

    1.0 => no TTC violations over the episode
    0.0 => all edges violate at every step
    """
    denom = float(max(int(episode_length) * int(n_edges), 1))
    frac = float(ttc_violations_sum) / denom
    frac = max(0.0, min(frac, 1.0))
    return 1.0 - frac


def safety_index_step(ttc_violations_step: float, n_edges: int) -> float:
    """Per-step safety index in [0,1]."""
    denom = float(max(int(n_edges), 1))
    frac = float(ttc_violations_step) / denom
    frac = max(0.0, min(frac, 1.0))
    return 1.0 - frac


def _aggregate_step_metrics(infos, n_edges: int):
    """Aggregate per-step metrics across n_eval_rollout_threads.

    Returns dict with:
        sim_time, step_co2_kg, step_travel_time_h, ttc_violations, ttc_penalty, ttc_min,
        safety_index_step
    """
    if infos is None or len(infos) == 0:
        return {
            "sim_time": None,
            "step_co2_kg": 0.0,
            "step_travel_time_h": 0.0,
            "ttc_violations": 0.0,
            "ttc_penalty": 0.0,
            "ttc_min": float("inf"),
            "safety_index_step": 1.0,
        }

    sim_time = infos[0].get("sim_time", None)

    step_co2 = float(np.mean([float(i.get("step_co2_kg", 0.0)) for i in infos]))
    step_tt_h = float(np.mean([float(i.get("step_travel_time_h", 0.0)) for i in infos]))
    ttc_viol = float(np.mean([float(i.get("ttc_violations", 0.0)) for i in infos]))
    ttc_pen = float(np.mean([float(i.get("ttc_penalty", 0.0)) for i in infos]))
    # For ttc_min, use the minimum across threads (worst-case)
    ttc_min = float(np.min([float(i.get("ttc_min", float("inf"))) for i in infos]))

    return {
        "sim_time": int(sim_time) if sim_time is not None else None,
        "step_co2_kg": step_co2,
        "step_travel_time_h": step_tt_h,
        "ttc_violations": ttc_viol,
        "ttc_penalty": ttc_pen,
        "ttc_min": ttc_min,
        "safety_index_step": float(safety_index_step(ttc_viol, n_edges=n_edges)),
    }


def rollout_controller(
    envs: DummyVecEnv,
    episode_length: int,
    controller: str,
    runner: Runner = None,
    collect_timeseries: bool = False,
    timeseries_stride: int = 1,
):
    """Run ONE rollout.

    controller in {"no_control","rule_based","mappo"}

    Returns
    -------
    ep_metrics: dict
    ts_df: pandas.DataFrame (only if collect_timeseries=True)
    """
    # Make sure env is in the correct controlled/uncontrolled mode *before* reset
    for e in envs.env_list:
        if controller == "no_control":
            e.set_controlled(False)
        else:
            e.set_controlled(True)

    obs = envs.reset()

    n_threads = envs.num_envs
    n_agents = envs.num_agent
    act_dim = envs.action_space[0].n
    n_edges = n_agents

    # init rnn state / masks for MAPPO
    if controller == "mappo":
        if runner is None:
            raise ValueError("runner is required for controller='mappo'")
        rnn_states = np.zeros((n_threads, n_agents, runner.recurrent_N, runner.hidden_size), dtype=np.float32)
        masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)

    # episode accumulators
    ep_reward = 0.0
    ep_co2_kg = 0.0
    ep_travel_time_h = 0.0
    ep_ttc_viol = 0.0
    ep_ttc_penalty = 0.0
    ep_min_ttc = float("inf")

    demand = None

    # time-series rows (optional)
    ts_rows = []
    stride = max(int(timeseries_stride), 1)

    if collect_timeseries:
        # Initial point at t=0 (before any simulation step)
        ts_rows.append({
            "time_s": 0,
            "step_co2_kg": 0.0,
            "co2_kg_cum": 0.0,
            "step_travel_time_h": 0.0,
            "travel_time_h_cum": 0.0,
            "ttc_violations": 0.0,
            "ttc_violations_cum": 0.0,
            "ttc_penalty": 0.0,
            "ttc_penalty_cum": 0.0,
            "ttc_min": float("inf"),
            "safety_index_step": 1.0,
            "safety_index_cum": 1.0,
        })

    for step in range(int(episode_length)):
        if controller == "no_control":
            actions_idx = np.zeros((n_threads, n_agents, 1), dtype=np.int64)
        elif controller == "rule_based":
            actions_idx = occupancy_rule_based_policy(obs, act_dim)
        elif controller == "mappo":
            runner.trainer.prep_rollout()
            action, rnn_states_new = runner.trainer.policy.act(
                np.concatenate(obs),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=True,
            )
            actions = np.array(np.split(_t2n(action), n_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states_new), n_threads))
            actions_idx = actions.astype(np.int64)
        else:
            raise ValueError(f"Unknown controller: {controller}")

        # convert to one-hot for env wrapper
        actions_env = np.squeeze(np.eye(act_dim)[actions_idx], 2)

        obs, rewards, dones, infos = envs.step(actions_env)

        ep_reward += float(np.mean(rewards))

        # demand label is included in info (Old_Env)
        if demand is None and infos and len(infos) > 0:
            demand = infos[0].get("demand", None)

        step_m = _aggregate_step_metrics(infos, n_edges=n_edges)

        # accumulate episode metrics
        ep_co2_kg += float(step_m["step_co2_kg"])
        ep_travel_time_h += float(step_m["step_travel_time_h"])
        ep_ttc_viol += float(step_m["ttc_violations"])
        ep_ttc_penalty += float(step_m["ttc_penalty"])
        ep_min_ttc = min(ep_min_ttc, float(step_m["ttc_min"]))

        # optional: save time-series row
        if collect_timeseries and ((step + 1) % stride == 0):
            t = step_m["sim_time"]
            if t is None:
                t = step + 1  # fallback
            # cumulative safety index up to time t
            safety_cum = safety_index_from_ttc_violations(ep_ttc_viol, episode_length=int(t), n_edges=n_edges)

            ts_rows.append({
                "time_s": int(t),
                "step_co2_kg": float(step_m["step_co2_kg"]),
                "co2_kg_cum": float(ep_co2_kg),
                "step_travel_time_h": float(step_m["step_travel_time_h"]),
                "travel_time_h_cum": float(ep_travel_time_h),
                "ttc_violations": float(step_m["ttc_violations"]),
                "ttc_violations_cum": float(ep_ttc_viol),
                "ttc_penalty": float(step_m["ttc_penalty"]),
                "ttc_penalty_cum": float(ep_ttc_penalty),
                "ttc_min": float(step_m["ttc_min"]),
                "safety_index_step": float(step_m["safety_index_step"]),
                "safety_index_cum": float(safety_cum),
            })

        if controller == "mappo":
            # reset rnn states if done
            dones = np.asarray(dones)
            if dones.ndim == 1:
                dones = np.repeat(dones[:, None], n_agents, axis=1)
            rnn_states[dones == True] = 0.0
            masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)
            masks[dones == True] = 0.0

    safety_index_ep = safety_index_from_ttc_violations(ep_ttc_viol, episode_length=int(episode_length), n_edges=n_edges)

    ep_metrics = {
        "reward_sum": float(ep_reward),
        "co2_kg_sum": float(ep_co2_kg),
        "travel_time_h_sum": float(ep_travel_time_h),
        "ttc_violations_sum": float(ep_ttc_viol),
        "ttc_penalty_sum": float(ep_ttc_penalty),
        "min_ttc": float(ep_min_ttc) if np.isfinite(ep_min_ttc) else float("inf"),
        "safety_index": float(safety_index_ep),
        "demand": str(demand) if demand is not None else "",
    }

    if not collect_timeseries:
        return ep_metrics

    ts_df = pd.DataFrame(ts_rows)
    return ep_metrics, ts_df


def main(argv):
    parser = get_config()

    # Extra eval args
    parser.add_argument("--out_dir", type=str, default="./results_eval/training_matrix", help="Where to save CSVs.")
    parser.add_argument("--n_episodes", type=int, default=10, help="Episodes per (demand, controller).")
    parser.add_argument(
        "--controllers",
        type=str,
        default="all",
        choices=["all", "mappo", "rule_based", "no_control"],
        help="Which controller(s) to evaluate.",
    )
    parser.add_argument(
        "--demands",
        type=str,
        default="all",
        help="Comma-separated list: undersaturated,saturated,oversaturated or 'all'.",
    )
    parser.add_argument(
        "--episode_length_eval",
        type=int,
        default=None,
        help="Override evaluation horizon. Default: --eval_episode_length (7200).",
    )

    # Time-series export
    parser.add_argument(
        "--save_timeseries",
        action="store_true",
        default=False,
        help="If set, save per-step time series CSVs (0..T seconds) for each (demand, controller, episode).",
    )
    parser.add_argument(
        "--timeseries_stride",
        type=int,
        default=1,
        help="Log every N steps. Default 1 means every second (SIMULATION_TIME=1).",
    )

    all_args = parser.parse_args(argv)

    out_dir = Path(all_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine demand list
    if str(all_args.demands).strip().lower() == "all":
        demands = ["undersaturated", "saturated", "oversaturated"]
    else:
        demands = [d.strip().lower() for d in str(all_args.demands).split(",") if d.strip()]

    # Determine controller list
    if all_args.controllers == "all":
        controllers = ["no_control", "rule_based", "mappo"]
    else:
        controllers = [all_args.controllers]

    # Build env (single-thread by default for eval)
    envs = DummyVecEnv(all_args)

    # Decide eval horizon
    episode_length = (
        int(all_args.episode_length_eval)
        if all_args.episode_length_eval is not None
        else int(getattr(all_args, "eval_episode_length", all_args.episode_length))
    )

    # Build runner (only if MAPPO is requested)
    runner = None
    if "mappo" in controllers:
        if all_args.model_dir is None:
            raise ValueError("--model_dir is required when evaluating controller='mappo'.")
        device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": envs,
            "num_agents": envs.num_agent,
            "device": device,
            "run_dir": out_dir,
        }
        runner = Runner(config)

    rows = []
    ts_all = []

    ts_dir = out_dir / "timeseries"
    if all_args.save_timeseries:
        ts_dir.mkdir(parents=True, exist_ok=True)

    for demand in demands:
        set_envs_demand(envs, demand)

        for ctrl in controllers:
            for ep in range(int(all_args.n_episodes)):
                if all_args.save_timeseries:
                    ep_metrics, ts_df = rollout_controller(
                        envs,
                        episode_length,
                        ctrl,
                        runner=runner,
                        collect_timeseries=True,
                        timeseries_stride=all_args.timeseries_stride,
                    )
                    # Add identifiers to ts_df
                    ts_df.insert(0, "episode", int(ep))
                    ts_df.insert(0, "controller", str(ctrl))
                    ts_df.insert(0, "demand", str(demand))

                    ts_path = ts_dir / f"timeseries_{demand}_{ctrl}_ep{int(ep)}.csv"
                    ts_df.to_csv(ts_path, index=False, encoding="utf-8-sig")
                    ts_all.append(ts_df)
                else:
                    ep_metrics = rollout_controller(envs, episode_length, ctrl, runner=runner)

                ep_metrics.update({
                    "controller": ctrl,
                    "episode": int(ep),
                    "demand": demand,
                    "episode_length": int(episode_length),
                })
                rows.append(ep_metrics)
                print(f"[{demand} | {ctrl}] ep={ep} metrics={ep_metrics}")

    df = pd.DataFrame(rows)

    # Raw per-episode results
    raw_path = out_dir / "training_matrix_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    # Grouped mean/std (tidy)
    metrics_cols = [
        "co2_kg_sum",
        "travel_time_h_sum",
        "safety_index",
        "ttc_violations_sum",
        "ttc_penalty_sum",
        "min_ttc",
        "reward_sum",
    ]

    g = df.groupby(["demand", "controller"], as_index=False)[metrics_cols].agg(["mean", "std"]).reset_index()
    # flatten columns
    g.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in g.columns
    ]

    agg_path = out_dir / "training_matrix_mean_std.csv"
    g.to_csv(agg_path, index=False, encoding="utf-8-sig")

    # A compact "matrix" view (mean only) for the three target metrics
    mean_df = df.groupby(["controller", "demand"], as_index=False).agg(
        co2_kg_sum_mean=("co2_kg_sum", "mean"),
        travel_time_h_sum_mean=("travel_time_h_sum", "mean"),
        safety_index_mean=("safety_index", "mean"),
    )
    mean_path = out_dir / "training_matrix_mean.csv"
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")

    # Save concatenated time-series (long format)
    ts_long_path = None
    if all_args.save_timeseries and len(ts_all) > 0:
        ts_long = pd.concat(ts_all, axis=0, ignore_index=True)
        ts_long_path = out_dir / "timeseries_long.csv"
        ts_long.to_csv(ts_long_path, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {agg_path}")
    print(f"  {mean_path}")
    if ts_long_path is not None:
        print(f"  {ts_long_path}")
        print(f"  {ts_dir}\n")
    else:
        print("")


if __name__ == "__main__":
    main(sys.argv[1:])
