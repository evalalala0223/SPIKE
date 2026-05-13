from .base import TaskBase
from env.tasks.utils.obs_check import *


class Exploration(TaskBase):
    @staticmethod
    def _extract_tile_position(value):
        if isinstance(value, dict):
            x = value.get("x", value.get("X"))
            y = value.get("y", value.get("Y"))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return float(x), float(y)
            return None
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x, y = value[0], value[1]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return float(x), float(y)
        return None

    @classmethod
    def _is_player_on_bed(cls, obs) -> bool:
        player = obs.get("player") or {}
        location = str(player.get("location") or "")
        if "farmhouse" not in location.lower():
            return False

        player_pos = cls._extract_tile_position(player.get("position"))
        if player_pos is None:
            return False
        px, py = player_pos

        furnitures = obs.get("furnitures") or []
        for furniture in furnitures:
            if not isinstance(furniture, dict):
                continue
            name = str(furniture.get("name") or "")
            if "bed" not in name.lower():
                continue

            bounding_box = furniture.get("boundingbox", furniture.get("boundingBox"))
            if isinstance(bounding_box, dict):
                left = bounding_box.get("left", bounding_box.get("Left"))
                right = bounding_box.get("right", bounding_box.get("Right"))
                top = bounding_box.get("top", bounding_box.get("Top"))
                bottom = bounding_box.get("bottom", bounding_box.get("Bottom"))
                if all(isinstance(v, (int, float)) for v in (left, right, top, bottom)):
                    if float(left) <= px <= float(right) and float(top) <= py <= float(bottom):
                        return True

            bed_pos = cls._extract_tile_position(furniture.get("position"))
            if bed_pos is not None:
                bx, by = bed_pos
                if abs(px - bx) <= 1.0 and abs(py - by) <= 1.0:
                    return True

        return False

    def evaluate(self, obs, proxy) -> dict:
        if self.last_obs is None:
            self.last_obs = obs
            return {
                "completed": False,
                "quantity": 0,
            }

        self.quantity_change = 0

        progression = obs.get("Progression") or {}
        last_progression = self.last_obs.get("Progression") or {}

        def _progress_items(key):
            current_items = progression.get(key)
            previous_items = last_progression.get(key)
            if current_items is None:
                current_items = []
            if previous_items is None:
                previous_items = []
            return current_items, previous_items

        if self.evaluator == "harvest":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               False, "object_at_inventory")

        elif self.evaluator == "silo":
            self.quantity_change = count_check(self.check_list, obs["farm"]["buildings"],
                                               self.last_obs["farm"]["buildings"], False, "silo")

        elif self.evaluator == "skill":
            if self.object == "Foraging Skill":
                check_object = "foraging"
            elif self.object == "Mining Skill":
                check_object = "mining"
            else:
                check_object = "fishing"

            self.quantity_change = (obs["player"]["skills"][check_object] -
                                    self.last_obs["player"]["skills"][check_object])

        elif self.evaluator == "profession":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["player"]["professions"],
                                               self.last_obs["player"]["professions"], False,
                                               "profession")
        elif self.evaluator == "bundle":
            self.check_list = [self.object]
            bundles, last_bundles = _progress_items("Bundles")
            self.quantity_change = count_check(self.check_list, bundles, last_bundles, False, "bundle")
        elif self.evaluator == "museum":
            museum, last_museum = _progress_items("Museum")
            if self.object == "Item":
                self.quantity_change = len(museum) - len(last_museum)
            else:
                self.check_list = [self.object]
                self.quantity_change = count_check(self.check_list, museum, last_museum, False, "museum")

        elif self.evaluator == "repair":
            self.check_list = [self.object]
            repairs, last_repairs = _progress_items("Repairs")
            self.quantity_change = count_check(self.check_list, repairs, last_repairs, False, "repair")

        elif self.evaluator == "location":
            if obs['player']['location'] == self.object:
                self.current_quantity = 1

        elif self.evaluator == "accept":
            quests, last_quests = _progress_items("Quests")
            self.quantity_change = count_check(self.check_list, quests, last_quests, False, "help_quest")

        elif self.evaluator == "quit":
            quests, last_quests = _progress_items("Quests")
            self.quantity_change = len(last_quests) - len(quests)

        elif self.evaluator == "reward":
            quests, last_quests = _progress_items("Quests")
            self.quantity_change = count_check(self.check_list, quests, last_quests, True, "completed_quest")

        elif self.evaluator == "complete_help":
            quests, last_quests = _progress_items("Quests")
            self.quantity_change = count_check(self.check_list, quests, last_quests, False, "completed_quest")

        elif self.evaluator == "exchange":
            self.check_list = [self.object]
            self.quantity_change = count_check(self.check_list, obs["inventory"], self.last_obs["inventory"],
                                               False, "object_at_inventory")

        elif self.evaluator == "complete_story":
            self.check_list = [self.object]
            quests, last_quests = _progress_items("Quests")
            self.quantity_change = count_check(self.check_list, quests, last_quests, True, "story_quest")

        elif self.evaluator == 'watch':
            curMenuData = obs['CurrentMenuData']
            if curMenuData is not None and 'dialogues' in curMenuData and curMenuData['dialogues'] is not None and len(
                    curMenuData['dialogues']) > 0 and 'forecast' in curMenuData['dialogues'][0]:
                self.current_quantity = 1

        elif self.evaluator == 'read':
            curMenuData = obs['CurrentMenuData']
            if curMenuData is not None and 'type' in curMenuData and curMenuData['type'] == 'Letter':
                self.current_quantity = 1

        elif self.evaluator == 'sleep':
            day_started = False
            if 'callbackdata' in obs and 'ondaystarted' in obs['callbackdata']:
                dayStartedTimes = obs['callbackdata']['ondaystarted']
                if dayStartedTimes > 1:
                    day_started = True
            if day_started or self._is_player_on_bed(obs):
                self.current_quantity = 1

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
