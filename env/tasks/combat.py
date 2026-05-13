from __future__ import annotations

from .base import TaskBase
from env.tasks.utils.obs_check import *
from env.tasks.utils.init_task import *
import time


class Combat(TaskBase):
    def __init__(self, llm_description: str, object: str, quantity: int, tool: str, save: str, init_commands: list,
                 evaluator: str, difficulty: str):
        super().__init__(llm_description, object, quantity, tool, save, init_commands, evaluator, difficulty)
        self.baseline_known = False
        self.baseline_reset_confirmed = False
        self.evaluation_diagnostics = []
        self.persistent_evaluation_diagnostics = []

    @staticmethod
    def _append_diagnostic(diag_list, diagnostic_type: str, source: str, detail: str) -> None:
        candidate = {
            "type": str(diagnostic_type or "").strip(),
            "source": str(source or "").strip(),
            "detail": str(detail or "").strip(),
        }
        if not candidate["type"]:
            return
        if candidate not in diag_list:
            diag_list.append(candidate)

    def _build_evaluation_diagnostics(self):
        merged = list(self.persistent_evaluation_diagnostics)
        for item in self.evaluation_diagnostics:
            if item not in merged:
                merged.append(item)
        return merged

    def _clear_persistent_diagnostic(self, diagnostic_type: str, source: str | None = None) -> None:
        normalized_type = str(diagnostic_type or "").strip()
        normalized_source = str(source or "").strip()
        kept = []
        for item in self.persistent_evaluation_diagnostics:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").strip()
            item_source = str(item.get("source", "") or "").strip()
            if item_type != normalized_type:
                kept.append(item)
                continue
            if normalized_source and item_source != normalized_source:
                kept.append(item)
                continue
        self.persistent_evaluation_diagnostics = kept

    def init_task(self, proxy):
        super().init_task(proxy)
        self.last_count = 0
        self.baseline_known = False
        self.baseline_reset_confirmed = False
        self.evaluation_diagnostics = []
        self.persistent_evaluation_diagnostics = []
        if self.evaluator == "kill" and self.object:
            reset_value = None
            try:
                reset_value = proxy.set_monster_stat(self.object, 0)
                time.sleep(0.5)
            except Exception as exc:
                self._append_diagnostic(
                    self.persistent_evaluation_diagnostics,
                    "combat_evaluator_unavailable",
                    "set_monster_stat",
                    str(exc),
                )

            confirm_value = proxy.get_monster_kills(self.object, retries=5, retry_sleep_s=0.5)
            if confirm_value == 0:
                self.last_count = 0
                self.baseline_known = True
                self.baseline_reset_confirmed = True
            elif confirm_value is None:
                self._append_diagnostic(
                    self.persistent_evaluation_diagnostics,
                    "combat_baseline_unconfirmed",
                    "get_monster_kills",
                    f"confirm_reset_unavailable:{self.object}:reset_ack={reset_value!r}",
                )
            else:
                self._append_diagnostic(
                    self.persistent_evaluation_diagnostics,
                    "combat_baseline_unconfirmed",
                    "get_monster_kills",
                    f"confirm_reset_nonzero:{confirm_value}:{self.object}:reset_ack={reset_value!r}",
                )

    def evaluate(self, obs, proxy: InitTaskProxy) -> dict:
        self.evaluation_diagnostics = []
        if self.last_obs is None:
            self.last_obs = obs
            if self.evaluator == "kill":
                if not self.baseline_known:
                    kill_stat = proxy.get_monster_kills(
                        self.object,
                        retries=2,
                        retry_sleep_s=0.2,
                    )
                    if kill_stat is None:
                        self._append_diagnostic(
                            self.evaluation_diagnostics,
                            "combat_evaluator_unavailable",
                            "get_monster_kills",
                            f"baseline_unavailable:{self.object}",
                        )
                        self.last_count = 0
                        self.baseline_known = False
                    else:
                        if not self.baseline_reset_confirmed:
                            self._append_diagnostic(
                                self.persistent_evaluation_diagnostics,
                                "combat_baseline_unconfirmed",
                                "get_monster_kills",
                                f"baseline_established_without_confirmed_reset:{kill_stat}:{self.object}",
                            )
                        if kill_stat == 0:
                            self.baseline_reset_confirmed = True
                            self._clear_persistent_diagnostic("combat_baseline_unconfirmed")
                        self.last_count = kill_stat
                        self.baseline_known = True
            return {
                "completed": False,
                "quantity": 0,
                "evaluation_diagnostics": self._build_evaluation_diagnostics(),
                "baseline_known": self.baseline_known,
                "baseline_reset_confirmed": self.baseline_reset_confirmed,
            }

        if self.evaluator == "kill":
            kill_stat = proxy.get_monster_kills(
                self.object,
                retries=2,
                retry_sleep_s=0.2,
            )
            if kill_stat is None:
                self._append_diagnostic(
                    self.evaluation_diagnostics,
                    "combat_progress_probe_missed" if self.baseline_known else "combat_evaluator_unavailable",
                    "get_monster_kills",
                    f"progress_unavailable:{self.object}",
                )
                self.quantity_change = 0
            elif not self.baseline_known:
                self.last_count = kill_stat
                self.baseline_known = True
                if not self.baseline_reset_confirmed:
                    self._append_diagnostic(
                        self.persistent_evaluation_diagnostics,
                        "combat_baseline_unconfirmed",
                        "get_monster_kills",
                        f"progress_untrusted_after_unconfirmed_reset:{kill_stat}:{self.object}",
                    )
                if kill_stat == 0:
                    self.baseline_reset_confirmed = True
                    self._clear_persistent_diagnostic("combat_baseline_unconfirmed")
                self.quantity_change = 0
            else:
                self.quantity_change = max(0, kill_stat - self.last_count)
                self.last_count = kill_stat

        elif self.evaluator == "skill":
            self.quantity_change = obs["player"]["skills"]["combat"] - self.last_obs["player"]["skills"]["combat"]

        elif self.evaluator == "profession":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["player"]["professions"],
                                               self.last_obs["player"]["professions"], False,
                                               "profession")

        self.last_obs = obs
        self.current_quantity += self.quantity_change
        if self.current_quantity >= self.quantity:
            completed = True
        else:
            completed = False

        return {
            "completed": completed,
            "quantity": self.current_quantity,
            "evaluation_diagnostics": self._build_evaluation_diagnostics(),
            "baseline_known": self.baseline_known,
            "baseline_reset_confirmed": self.baseline_reset_confirmed,
        }
