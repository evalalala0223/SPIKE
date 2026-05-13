import re
import time
from stardojo.config import Config
from stardojo.log import Logger
from stardojo.gameio.io_env import IOEnvironment
from stardojo.environment.stardew.skill_registry import register_skill

PORT = 10783

config = Config()
logger = Logger()
io_env = IOEnvironment()
actionproxy = None


def _require_actionproxy():
    if actionproxy is None:
        raise RuntimeError(
            "actionproxy not injected. SkillExecutor must set basic_skills.actionproxy before calling skills."
        )


@register_skill("move")
def move(x, y):
    """
    Move to the position (x, y). Call template: move(x = ..., y = ...)
    For example:
        - call move(x=0, y=1) to move to (0, 1)
        - call move(x=1, y=0) to move to (1, 0)

    Parameters:
     - x: The X-coordinate of the destination of move action.
     - y: The Y-coordinate of the destination of move action.
    """
    _require_actionproxy()
    return actionproxy.move(x, y)


# @register_skill("move_step")
# def move_step(direction):
#     """
#     Move to the direciton
#
#     Parameters:
#      - direction: a integer, 0 for stay still, 1 for move up, 2 for move right, 3 for move down, 4 for move left
#     """
#     actionproxy.move_step(direction=direction)

# @register_skill("turn")
# def turn(direction):
#     """
#     Turn to the direction, Call template: turn(direction = ...)

#     Parameters:
#      - direction: a string, up, right, down, and left.
#     """
#     direction_int = 0
#     if direction == "up":
#         direction_int = 0
#     elif direction == "right":
#         direction_int = 1
#     elif direction == "down":
#         direction_int = 2
#     elif direction == "left":
#         direction_int = 3
#     actionproxy.turn(direction_int)


@register_skill("craft")
def craft(item):
    """
    Craft an item based on its name. Call template: craft(item = ...)
    For example:
        - call craft("chest") to craft chest

    Parameters:
     - item: The name of the item to craft. A string
    """
    _require_actionproxy()
    message = f"craft%{item}"
    return actionproxy._post_message(message)
    
# @register_skill("open_map")
# def open_map():
#     """
#     Open the map. Call template: open_map()
#     """
#     message = f"open_map"
#     actionproxy._post_message(message)

# @register_skill("exit_menu")
# def exit_menu():
#     """
#     Exit the current menu. Call template: exit_menu()
#     """
#     message = f"exit_menu"
#     actionproxy._post_message(message)


@register_skill("use")
def use(direction):
    """
    Use an item you choose
    You must move to the target with the right direction before using the tool
    Do any action by "Use"
    Call template: use(direction = ...)
    For example:
        - call use(direction="up") to use against (0,-1)
        - call use(direction="right") to use against (1,0)
        - call use(direction="down") to use against (0,1)
        - call use(direction="left") to use against (-1,0)

    Parameters:
     - direction: a string, up, right, down, and left.
    """
    direction_int = 0
    if direction == "up":
        direction_int = 0
    elif direction == "right":
        direction_int = 1
    elif direction == "down":
        direction_int = 2
    elif direction == "left":
        direction_int = 3
    _require_actionproxy()
    actionproxy.use(direction_int)


@register_skill("choose_item")
def choose_item(slot_index):
    """
    Choose the item in the slot. Call template: choose_item(slot_index = ...)
    For example:
        - call choose_item(slot_index=0) to choose the item in the first slot

    Parameters:
     - slot_index: The index of the inventory slot (0-35). This is an integer
    """
    _require_actionproxy()
    actionproxy.choose_item(slot_index)


# @register_skill("interact")
# def interact():
#     """
#     Interact with an object or NPC in a specific direction. Call template: interact()
#     """
#     actionproxy.interact()
    
@register_skill("interact")
def interact(direction):
    """
    Interact with an object or NPC in a specific direction. Also you can call interact to harvest crops. Call template: interact(direction = ...)
    For example:
        - call interact(direction="up") to interact with (0,-1)
        - call interact(direction="right") to interact with (1,0)
        - call interact(direction="down") to interact with (0,1)
        - call interact(direction="left") to interact with (-1,0)


    Parameters:
     - direction: a string, up, right, down, and left.
    """
    direction_int = 0
    if direction == "up":
        direction_int = 0
    elif direction == "right":
        direction_int = 1
    elif direction == "down":
        direction_int = 2
    elif direction == "left":
        direction_int = 3
    _require_actionproxy()
    actionproxy.interact(direction_int)




@register_skill("choose_option")
def choose_option(option_index, quantity=None, direction=None):
    """
    Choose an option from a list of options presented by an NPC or object, with optional parameters for buying. Index starts from 1, 0 to close the menu. Call template: choose_option(option_index = ...)
    For example:
        choose_option(0,0) to close the menu
        choose_option(1,0) to choose the first option or continue the chat when there is no option
        choose_option(2,1) to choose the second option with quantity 1
        choose_option(1,1,"in") to choose the first option with quantity 1 and direction in
        choose_option(1,1,"out") to choose the first option with quantity 1 and direction out

    Parameters:
     - option_index: The index of the option to choose. This is an integer. 0 to close the menu, 1 to continue the chat.
     - quantity: An optinal integer, the quantity of items to buy if interacting with a shop menu, default is None.
     - direction: A string, in, out, indicating the direction of the option, default is None. Sell or put to a box or a bin option is out, buy or take from a box or a bin option is in.
    """
    direction_int = 0
    if direction == "in":
        direction_int = 0
    elif direction == "out":
        direction_int = 1
    _require_actionproxy()
    actionproxy.choose_option(option_index, quantity, direction_int)

# @register_skill("sell_current_item")
# def sell_current_item():
#     """
#     Sell the current item in the inventory to the current shop. Call template: sell_current_item()
#     """
#     actionproxy.sell_current_item()

# @register_skill("take_from_chest")
# def take_from_chest(index, quantity):
#     """
#     Take the item from chest at index to inventory, with the given quantity. Call template: take_from_chest(0, 1)
#     Only call when the chest menu is open.

#     Parameters:
#      - index: The index of the item in the chest. An integer
#      - quantity: The quantity of the item to take. An integer
#     """
#     actionproxy.take_from_chest(index, quantity)

# @register_skill("put_to_chest")
# def put_to_chest(index, quantity):
#     """
#     Put the item from inventory at index to chest, with the given quantity. Call template: put_to_chest(0, 1).
#     Only call when the chest menu is open.
#     Parameters:
#      - index: The index of the item in the inventory. An integer
#      - quantity: The quantity of the item to put. An integer
#     """
#     actionproxy.put_to_chest(index, quantity)



@register_skill("attach_item")
def attach_item(slot_index):
    """
    Attach the item to the slot. Call template: attach_item(slot_index = ...)
    For example:
        - call attach_item(slot_index=0) to attach the item in the first slot to the current tool

    Parameters:
     - slot_index: The index of the inventory slot (0-35). This is an integer
    """
    _require_actionproxy()
    actionproxy.attach_item(slot_index=slot_index)



@register_skill("unattach_item")
def unattach_item():
    """
    Unattach the item from the current tool. Call template: unattach_item()
    """
    _require_actionproxy()
    actionproxy.unattach_item()


@register_skill("menu")
def menu(option, menu_name="current_menu"):
    """
    Open or close a certain menu. Call template: menu(option = ..., menu_name = ...)
    For example:
        - call menu("open", "map") to open the map
        - call menu("open", "inventory") to open the inventory
        - call menu("open", "crafting") to open the inventory/crafting menu
        - call menu("close", "current_menu") to close the current menu

    Parameters:
     - option: A string, open or close.
     - menu_name: A string, the name of the menu. Defaults to current_menu when closing.
       Supported values: current_menu, inventory, crafting, map
    """
    normalized_option = str(option or "").strip().lower()
    normalized_menu = str(menu_name or "").strip().lower()
    if normalized_option in {"inventory", "open_inventory"}:
        normalized_option = "open"
        normalized_menu = "inventory"
    elif normalized_option in {"craft", "crafting", "open_crafting"}:
        normalized_option = "open"
        normalized_menu = "crafting"
    elif normalized_option in {"exit"}:
        normalized_option = "close"

    if normalized_option == "close":
        # Escape is the most reliable universal dismiss in Stardew:
        # it closes regular menus and also clears object-dialogue popups
        # that do not populate activeClickableMenu.
        return io_env.key_press("esc")
    else:
        if normalized_menu == "map":
            _require_actionproxy()
            actionproxy.open_map()
            return True
        elif normalized_menu in {"current_menu", "inventory", "crafting"}:
            # Stardew's inventory/crafting screen is opened via the inventory hotkey.
            return io_env.key_press("e")

    return False


def mouse_check_do_action(x, y, duration=0.1):
    io_env.mouse_move(x, y)
    io_env.mouse_click_button("right mouse button", duration=duration)


def do_action():
    io_env.key_press("x")
    return True


def use_tool():
    io_env.key_press("c")
    return True


def move_up(duration=0.1):
    io_env.key_press("w", duration)
    return True


def move_down(duration=0.1):
    io_env.key_press("s", duration)
    return True


def move_left(duration=0.1):
    io_env.key_press("a", duration)
    return True


def move_right(duration=0.1):
    io_env.key_press("d", duration)
    return True


def select_tool(key):
    regex_pattern = r"[0-9\-+]"
    if re.match(regex_pattern, str(key)):
        io_env.key_press(str(key))
        return True
    raise ValueError("Invalid key in select_tool. Key must be in the range [0-9,-,+]")


#@register_skill("navigate")
#def navigate(name):
#     """
#     Navigate to a certain location. Call template: navigate(name = ...)
#     For example:
#         - call navigate("farm") to navigate to farm
#
#     Parameters:
#      - name: The name of the location to navigate to.
#     """
#     actionproxy.navigate(name)




__all__ = [
    "move",
    "craft",
    "use",
    "choose_item",
    "interact",
    "choose_option",
    "attach_item",
    "unattach_item",
    "menu",
    #"navigate"
]

