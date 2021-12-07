import math

from pytorch_lightning.callbacks import Callback

from custom_blocks import get_modules, ValveBlock
from logging_utils import rank


class ValveDecayCallback(Callback):
    def __init__(self, valves_config, valves_steps, initial_value=1.0):
        self.valves_config = valves_config
        self.valves_steps = valves_steps
        self.initial_value = initial_value
        self.valves_modules = {valve_tag: ([], []) for valve_tag in self.valves_config.keys()}

    def setup(self, trainer, pl_module, stage=None) -> None:
        for valve_tag, valve_config in self.valves_config.items():
            anti_valve_tag = valve_config['anti']
            valve_modules = get_modules(pl_module, ValveBlock, [valve_tag])
            anti_valve_modules = get_modules(pl_module, ValveBlock, [anti_valve_tag])
            self.valves_modules[valve_tag][0].extend(valve_modules)
            self.valves_modules[valve_tag][1].extend(anti_valve_modules)
        self._update_valve_modules(pl_module)

    def on_batch_end(self, trainer, pl_module) -> None:
        should_step_valves = pl_module.global_step % self.valves_steps == 0
        if should_step_valves:
            self._update_valve_modules(pl_module)

    def _get_ratio(self, global_step, valve_tag):
        valve_config = self.valves_config[valve_tag]
        valve_limit = valve_config['limit']
        valve_start = valve_config['start']
        if valve_limit < global_step:
            ratio = 0
        elif valve_start > global_step:
            ratio = 1
        else:
            ratio = math.pow(valve_config['decay'], global_step - valve_start)
        return ratio

    def _get_anti_ratio(self, global_step, valve_tag):
        valve_config = self.valves_config[valve_tag]
        valve_limit = valve_config['limit']
        valve_start = valve_config['start']
        if valve_limit < global_step:
            ratio = 0
        elif valve_start > global_step:
            ratio = 1
        else:
            ratio = math.pow(valve_config['anti_decay'], global_step - valve_start)
        return 1 - ratio

    def _update_valve_modules(self, pl_module):
        sw = pl_module.logger.experiment
        for valve_tag, all_valve_modules in self.valves_modules.items():
            valve_modules, anti_valves_modules = all_valve_modules
            ratio = self._get_ratio(pl_module.global_step, valve_tag) * self.initial_value
            anti_ratio = self._get_anti_ratio(pl_module.global_step, valve_tag) * self.initial_value
            for valve_module in valve_modules:
                valve_module.ratio = ratio
            for anti_valve_module in anti_valves_modules:
                anti_valve_module.ratio = anti_ratio
            anti_valve_tag = self.valves_config[valve_tag]['anti']
            sw.add_scalar(rank(f'params/valves/{valve_tag}'), ratio, pl_module.global_step)
            sw.add_scalar(rank(f'params/valves/{anti_valve_tag}'), anti_ratio, pl_module.global_step)
