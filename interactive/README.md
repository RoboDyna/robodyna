# Human experiment logs

Per-participant records written by `interactive/experiment_gui.py`.

## Launch

```bash
conda activate robodyna
python interactive/experiment_gui.py
```

## Layout

```
data/exp_logs/<user_slug>/user.json
```

The slug is the participant name, lowercased, with spaces turned into underscores.

Each `user.json` stores:

- `user_name` / `user_id`
- pre-experiment answers (`experience`): games, controllers, simulators, teleop, 3D apps, mouse hand
- post-experiment answers (`post_survey`): overall difficulty, clarity, task ranking (most difficult first), hardest aspect per controller (control vs event prediction), tasks where the gripper view was useful (multi-select over the assigned tasks, or `none`), preferred controller, policy outlook. The **Post-experiment Questionnaire** card stays gray until every assigned task is finished with both controllers; after submit it stays locked. Questionnaire question boxes use a white background with `#1A1A1A` text.
- `completed_keys` — finished items that stay gray in the task GUIs
- `plays` — every attempt, including:

  - task, suite, scenario (base only)
  - controller (`keyboard` or `robot`)
  - seed, result (`SUCCESS` / `FAILURE` / `closed` / `stopped` / `error`)
  - metrics (success, total_time_sim_s, wall_s, steps, option_label for base, manipulation score, …)
  - time: `total_time_sim_s`, `wall_s`, `steps` (legacy aliases `wall_clock_s` / `simulation_s` / `simulation_steps` still written)
  - `option_label` / `scenario` — base catalog tag (`default` / `opt1` / `opt2` / `opt1+2`)

A task/scenario is locked (gray, not selectable) only after a terminal **SUCCESS** or **FAILURE**. Closing the viewer early or pressing Stop does not consume the slot, so the participant can retry.

## Protocol config

Edit `interactive/experiment.yml` to set:

- `record_data` / `save_video` — `true` / `false`, or a scenario subset (`opt1+2`, `{opt1+2: true}`, `[opt1, opt2]`)
- `controller` (`robot` or `keyboard`)
- `base_tasks` / `household_tasks` — 1-based card numbers (see the tables in that file)

Fixed evaluation seeds are defined separately in
[`task_config/eval_seeds.yml`](../task_config/eval_seeds.yml). The human GUI
and policy evaluators use this same task/scenario seed list.

Those controls are locked in the task GUIs during an experiment session. The standalone base / household launchers are unchanged.
