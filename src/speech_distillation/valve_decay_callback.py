import math

from pytorch_lightning.callbacks import Callback

from custom_blocks import get_modules, ValveBlock


class ValveDecayCallback(Callback):
    def __init__(self, valves_config, valves_steps):
        self.valves_config = valves_config
        self.valves_steps = valves_steps
        self.valves_modules = {valve_tag: ([], []) for valve_tag in self.valves_config.keys()}

    def setup(self, trainer, pl_module, stage=None) -> None:
        for module in pl_module.children():
            for valve_tag, valve_config in self.valves_config.items():
                anti_valve_tag = valve_config['anti']
                valve_modules = get_modules(module, ValveBlock, [valve_tag])
                anti_valve_modules = get_modules(module, ValveBlock, [anti_valve_tag])
                self.valves_modules[valve_tag][0].extend(valve_modules)
                self.valves_modules[valve_tag][1].extend(anti_valve_modules)

    def on_batch_end(self, trainer, pl_module) -> None:
        should_step_valves = pl_module.global_step % self.valves_steps == 0
        if should_step_valves:
            for valve_tag, all_valve_modules in self.valves_modules.items():
                valve_modules, anti_valves_modules = all_valve_modules
                valve_config = self.valves_config[valve_tag]
                valve_limit = valve_config['limit']
                if valve_limit < pl_module.global_step:
                    pow_decay = 0
                    anti_pow_decay = 0
                else:
                    valve_decay = valve_config['decay']
                    pow_decay = math.pow(valve_decay, self.valves_steps)

                    anti_valve_decay = valve_config['anti_decay']
                    anti_pow_decay = math.pow(anti_valve_decay, self.valves_steps)
                for valve_module in valve_modules:
                    valve_module.ratio *= pow_decay
                for anti_valve_module in anti_valves_modules:
                    anti_valve_module.ratio = (1 - (1 - anti_valve_module.ratio) * anti_pow_decay)
