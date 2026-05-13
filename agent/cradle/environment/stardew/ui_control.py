import time

from PIL import Image
import mss

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio import gui_utils  # Use gui_utils for proper keyboard input
from cradle import constants
from cradle.environment import UIControl


config = Config()
logger = Logger()
io_env = IOEnvironment()

class StardewUIControl(UIControl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @staticmethod
    def _is_non_fatal_activate_error(e: Exception) -> bool:
        error_text = str(e)
        return (
            "Error code from Windows: 0" in error_text
            or "Error code from Windows: 122" in error_text
        )


    def pause_game(self, env_name: str, ide_name: str) -> None:
        """Pause the game by switching focus away from game window.

        Stardew Valley auto-pauses when it loses focus (single-player mode).
        We use Alt+Tab to switch away from the game window, which reliably pauses the game
        without the state synchronization issues of ESC menu toggle.
        """
        # logger.write(f"[DEBUG] pause_game called: env_name={env_name}, ide_name='{ide_name}'")

        if ide_name:
            # logger.write(f"[DEBUG] pause_game: Using ide_name='{ide_name}' to switch window")
            ide_window = io_env.get_windows_by_name(ide_name)[0]
            ide_window.activate()
            ide_window.show()
        else:
            # Use Alt+Tab to switch away from game window
            # This reliably pauses Stardew Valley without ESC state sync issues
            # logger.write(f"[DEBUG] pause_game: No ide_name, using Alt+Tab to lose focus")
            try:
                # First ensure game window is active
                game_window = io_env.get_windows_by_name(config.env_name)[0]
                try:
                    game_window.activate()
                    time.sleep(0.2)
                except Exception as e:
                    # logger.write(f"[DEBUG] pause_game: Window activate warning: {e}")
                    pass

                # Use Alt+Tab to switch to previous window (loses focus = pauses game)
                gui_utils.key_down('alt')
                time.sleep(0.05)
                gui_utils.key_down('tab')
                time.sleep(0.05)
                gui_utils.key_up('tab')
                time.sleep(0.05)
                gui_utils.key_up('alt')
                # logger.write("[DEBUG] pause_game: Pressed Alt+Tab to lose focus (pause game)")
                time.sleep(0.3)

            except Exception as e:
                logger.warn(f"Failed to pause game via Alt+Tab: {e}")
        time.sleep(0.5)


    def unpause_game(self, env_name: str, ide_name: str) -> None:
        """Unpause the game by activating game window.

        Since we pause by losing focus (Alt+Tab), we unpause by regaining focus.
        Simply activating the game window resumes the game in Stardew Valley.
        """
        # logger.write(f"[DEBUG] unpause_game called: env_name={env_name}, ide_name='{ide_name}'")

        target_window = io_env.get_windows_by_name(config.env_name)[0]

        # Activate the game window to resume
        try:
            target_window.activate()
            time.sleep(0.3)  # Wait for window to become active
            # logger.write("[DEBUG] unpause_game: Activated game window (game should resume)")
        except Exception as e:
            if not self._is_non_fatal_activate_error(e):
                logger.warn(f"Failed to activate game window: {e}")

        # Move mouse to game area to ensure focus
        io_env.mouse_move(960, 540)  # Center of 1920x1080
        time.sleep(0.3)


    def switch_to_game(self, env_name: str, ide_name: str) -> None:

        target_window = io_env.get_windows_by_name(config.env_name)[0]
        try:
            target_window.activate()
        except Exception as e:
            if self._is_non_fatal_activate_error(e):
                # Handle pygetwindow exception
                pass
            else:
                raise e
        time.sleep(1)


    def exit_back_to_pause(self, env_name: str, ide_name: str) -> None:
        max_steps = 10

        back_steps = 0
        while not self.is_env_paused() and back_steps < max_steps:
            back_steps += 1
            self.pause_game(env_name, ide_name)
            time.sleep(constants.PAUSE_SCREEN_WAIT)

        if back_steps >= max_steps:
            logger.warn("The environment fails to pause!")


    def exit_back_to_game(self, env_name: str, ide_name: str) -> None:

        self.exit_back_to_pause(env_name, ide_name)

        # Unpause the game, to keep the rest of the agent flow consistent
        self.unpause_game(env_name, ide_name)


    def is_env_paused(self) -> bool:
        target_window = io_env.get_windows_by_name(config.env_name)[0]
        is_active = target_window.is_active()
        return not is_active


    def take_screenshot(self,
                        tid: float,
                        screen_region: tuple[int, int, int, int] = None) -> str:

        if screen_region is None:
            screen_region = config.env_region

        region = screen_region
        region = {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3],
        }

        output_dir = config.work_dir

        # Save screenshots
        screen_image_filename = output_dir + "/screen_" + str(tid) + ".jpg"

        with mss.mss() as sct:
            screen_image = sct.grab(region)
            image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
            image.save(screen_image_filename)

        return screen_image_filename
