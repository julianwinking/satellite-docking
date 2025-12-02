import pprint
from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.log_visualisation import plot_player_log
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_animation import create_animation
from dg_commons.sim.utils import run_simulation
from collections import defaultdict

from satellite_docking.simulation.perf_metrics import ex13_metrics
from satellite_docking.simulation.utils_config import sim_context_from_yaml
from satellite_docking.simulation.get_config import get_config
from satellite_docking.simulation.utils_output import out_dir
import os
import matplotlib.pyplot as plt


def ex13_evaluation(sim_context: SimContext) -> Tuple[str, float]:
    # run simulation
    run_simulation(sim_context)
    # visualisation
    _ex13_vis(sim_context=sim_context)
    # compute metrics
    avg_player_metrics, _ = ex13_metrics(sim_context)
    # report evaluation
    score: float = avg_player_metrics.reduce_to_score()
    print(f"EpisodeEvaluation:\n{pprint.pformat(avg_player_metrics)}")
    print(f"OverallScore: {score:.2f}")
    return (sim_context.description, score)


def _ex13_vis(sim_context: SimContext) -> None:
    # Save animation directly to output folder
    out_path = out_dir("out")
    os.makedirs(out_path, exist_ok=True)
    anim_file = os.path.join(out_path, f"animation_{sim_context.description}.mp4")
    
    import dg_commons.sim.simulator_visualisation as sim_vis_module
    original_plot_spaceship = sim_vis_module.plot_spaceship

    def patched_plot_spaceship(ax, player_name, *args, **kwargs):
        display_name = player_name
        return original_plot_spaceship(ax, display_name, *args, **kwargs)

    sim_vis_module.plot_spaceship = patched_plot_spaceship

    try:
        create_animation(
            file_path=anim_file,
            sim_context=sim_context,
            figsize=(16, 16),
            dt=50,
            dpi=120,
            plot_limits=[[-12, 27], [-12, 12]],
        )
    finally:
        sim_vis_module.plot_spaceship = original_plot_spaceship
        
    print(f"Animation saved to: {anim_file}")

    # state/commands plots
    for pn in sim_context.log.keys():
        if pn not in sim_context.missions:
            continue
        
        # Save to file
        fig = plt.figure(figsize=(20, 15))
        plot_player_log(log=sim_context.log[pn], fig=fig)
        log_file = os.path.join(out_path, f"log_{pn}_{sim_context.description}.png")
        fig.savefig(log_file)
        plt.close(fig)
        print(f"Log plot saved to: {log_file}")
