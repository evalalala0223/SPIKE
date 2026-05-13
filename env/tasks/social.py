from .base import TaskBase
from env.tasks.utils.obs_check import *


class Social(TaskBase):
    def evaluate(self, obs, proxy) -> dict:
        if self.last_obs is None:
            self.last_obs = obs
            return {
                "completed": False,
                "quantity": 0,
            }

        if self.evaluator == "sell":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               True, "object_at_inventory")
        elif self.evaluator == "retrieve":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               False, "object_at_inventory")

        elif self.evaluator == "purchase":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               False, "object_at_inventory")

        elif self.evaluator == "harvest":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               False, "object_at_inventory")

        elif self.evaluator == "upgrade_tool":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               False, "object_at_inventory")

        elif self.evaluator == "break":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               True, "object_at_inventory")

        elif self.evaluator == "jojamart":
            progression = obs.get("Progression") or {}
            last_progression = self.last_obs.get("Progression") or {}
            if self.object == "Joja Membership":
                self.quantity_change = (int(progression.get("JojaMembership", 0)) -
                                        int(last_progression.get("JojaMembership", 0)))
            elif self.object == "Joja Community Development Form":
                self.quantity_change = count_check(self.check_list, progression.get("Repairs", []),
                                                   last_progression.get("Repairs", []), False,
                                                   "repair_all")
            elif self.object == "Movie Theater":
                self.quantity_change = (int(progression.get('MovieTheater', 0)) -
                                        int(last_progression.get('MovieTheater', 0)))
            else:
                self.check_list = [self.object]
                self.quantity_change = count_check(self.check_list, progression.get("Repairs", []),
                                                   last_progression.get("Repairs", []), False,
                                                   "repair")

        elif self.evaluator == "backpack":
            self.quantity_change = len(obs["inventory"]) - len(self.last_obs["inventory"])

        elif self.evaluator == "purchase_animal" or self.evaluator == "sell_animal":
            if self.object == "Chicken":
                self.check_list = ["White Chicken", "Brown Chicken"]
            elif self.object == "Cow":
                self.check_list = ["White Cow", "Brown Cow"]
            else:
                self.check_list = [self.object]

            if self.evaluator == "purchase_animal":
                self.quantity_change = count_check(self.check_list, obs["farm"]["animals"],
                                                   self.last_obs["farm"]["animals"], False,
                                                   "animal")
            else:
                self.quantity_change = count_check(self.check_list, obs["farm"]["animals"],
                                                   self.last_obs["farm"]["animals"], True,
                                                   "animal")

        elif self.evaluator == "build":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["farm"]["buildings"],
                                               self.last_obs["farm"]["buildings"], False, "building")

        elif self.evaluator == "move":
            self.check_list = [self.object]
            self.quantity_change = building_moving_check(self.check_list, obs["farm"]["buildings"],
                                                         self.last_obs["farm"]["buildings"],
                                                         True)

        elif self.evaluator == "demolish":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["farm"]["buildings"],
                                               self.last_obs["farm"]["buildings"], True, "building")

        elif self.evaluator == "upgrade_farmhouse":
            self.quantity_change = count_check(self.check_list, obs["farm"]["buildings"],
                                               self.last_obs["farm"]["buildings"], False,
                                               "farmhouse")

        elif self.evaluator == "talk":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["npcs"], self.last_obs["npcs"], False,
                                               "talk")
        elif self.evaluator == "gift":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["npcs"], self.last_obs["npcs"], False,
                                               "gift")

        elif self.evaluator == "date" or self.evaluator == "breakup":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["player"]["datingpartners"],
                                               self.last_obs["player"]["datingpartners"], self.evaluator == "breakup",
                                               "date")

        elif self.evaluator == "propose":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["player"]["spouse"],
                                               self.last_obs["player"]["spouse"], False, "propose")

        elif self.evaluator == "friendship":
            self.check_list = [self.object]
            self.current_quantity = count(self.check_list, obs["npcs"], "npc_friendship")

        self.last_obs = obs
        self.current_quantity += self.quantity_change
        if self.current_quantity >= self.quantity:
            completed = True
        else:
            completed = False

        return {
            "completed": completed,
            "quantity": self.current_quantity,
        }
