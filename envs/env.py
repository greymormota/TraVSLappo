import numpy as np

from envs.old_sumo_env import Old_Env


class Env(object):
    """Environment interface wrapping the SUMO multi-agent environment.

    Each 'agent' corresponds to one controlled edge.

    This wrapper also forwards:
      - set_controlled(True/False)
      - set_demand('undersaturated'|'saturated'|'oversaturated')
      - set_route_file(path)

    so evaluation scripts can produce the requested 3x3 (demand x controller) matrix.
    """

    def __init__(self, i, all_args=None):
        """Create one env instance.

        Parameters
        ----------
        i : int
            index of this env instance (used for unique TraCI labels).
        all_args : argparse.Namespace | None
            If provided, may contain `demand` and/or `route_file`.
        """
        self.agent_num = 12      # number of agents
        self.obs_dim = 13        # dimension of observation (per agent)
        self.action_dim = 6      # discrete: 6 speed levels

        demand = getattr(all_args, "demand", "saturated") if all_args is not None else "saturated"
        route_file = getattr(all_args, "route_file", None) if all_args is not None else None

        # Pass env_id=i so Old_Env uses a unique TraCI label
        self.sumo_env = Old_Env(env_id=i, demand=demand, route_file=route_file)

    def reset(self):
        """Returns a NumPy array of shape (agent_num, obs_dim)."""
        return self.sumo_env.reset()

    def step(self, actions):
        """Step the environment.

        MAPPO may supply either:
          - (agent_num, 1) discrete indices
          - (agent_num, action_dim) one-hot
          - (agent_num,) indices

        We convert to a list[int] for Old_Env.
        """
        actions_arr = np.asarray(actions)

        # (agent_num, 1) => index
        if actions_arr.ndim == 2 and actions_arr.shape[0] == self.agent_num and actions_arr.shape[1] == 1:
            action_idx = actions_arr.squeeze(1)

        # (agent_num, action_dim) => one-hot => argmax
        elif actions_arr.ndim == 2 and actions_arr.shape[0] == self.agent_num and actions_arr.shape[1] == self.action_dim:
            action_idx = actions_arr.argmax(axis=1)

        # (agent_num,) => index
        elif actions_arr.ndim == 1 and actions_arr.shape[0] == self.agent_num:
            action_idx = actions_arr

        else:
            raise ValueError(
                f"Unexpected actions shape {actions_arr.shape}. "
                f"Expected (agent_num,1) or (agent_num,action_dim) or (agent_num,)."
            )

        a = [int(x) for x in action_idx.tolist()]
        state, reward, done, info = self.sumo_env.step(a)
        return [state, reward, done, info]

    # -----------------
    # Control utilities
    # -----------------

    def set_controlled(self, controlled: bool):
        """Toggle VSL control on/off (used for the 'no control' baseline)."""
        self.sumo_env.controlled = bool(controlled)

    def set_demand(self, demand: str):
        """Switch demand scenario (route file) before calling reset()."""
        self.sumo_env.set_demand(demand)

    def set_route_file(self, route_file: str):
        """Set an explicit route file path before calling reset()."""
        self.sumo_env.set_route_file(route_file)
