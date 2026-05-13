using System;
using System.Collections.Generic;
using System.Reflection;
using StardewModdingAPI;
using StardewValley;
using StardewValley.Menus;
using StardewValley.Objects;
using StardewValley.Tools;
using Microsoft.Xna.Framework;

namespace testUtils
{
    public static class TestUtils
    {
        // Item name -> ID mapping for common items
        public static readonly Dictionary<string, int> ItemIdMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            {"parsnip_seeds", 472}, {"potato_seeds", 475}, {"cauliflower_seeds", 474},
            {"bean_starter", 473}, {"kale_seeds", 477}, {"melon_seeds", 479},
            {"tomato_seeds", 480}, {"blueberry_seeds", 481}, {"corn_seeds", 487},
            {"wheat_seeds", 483}, {"pumpkin_seeds", 490}, {"parsnip", 24},
            {"potato", 192}, {"cauliflower", 190}, {"melon", 254},
            {"wood", 388}, {"stone", 390}, {"fiber", 771},
            {"coal", 382}, {"iron_ore", 380}, {"gold_ore", 384},
            {"copper_ore", 378}, {"hay", 178}, {"sap", 92},
        };

        public static void enterLoadGameMenu(Mod mod, Action onComplete)
        {
            Game1.activeClickableMenu = new TitleMenu();
            TitleMenu.subMenu = new LoadGameMenu();
            onComplete?.Invoke();
        }

        public static void loadGame(string which, Mod mod, Action onComplete)
        {
            try
            {
                SaveGame.Load(which);
            }
            catch (Exception ex)
            {
                mod.Monitor.Log($"Error loading game '{which}': {ex.Message}", LogLevel.Error);
            }
            onComplete?.Invoke();
        }

        public static void exitGameToTitle()
        {
            Game1.ExitToTitle();
        }

        public static bool callTryToPurchaseItem(ShopMenu shopMenu, ISalable item, ISalable? held_item, int count)
        {
            // Use reflection since tryToPurchaseItem may be private/internal in SV 1.6
            try
            {
                var method = typeof(ShopMenu).GetMethod("tryToPurchaseItem",
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (method != null)
                {
                    var parameters = method.GetParameters();
                    object? result;
                    if (parameters.Length == 5)
                        result = method.Invoke(shopMenu, new object?[] { item, held_item, count, 0, 0 });
                    else if (parameters.Length == 4)
                        result = method.Invoke(shopMenu, new object?[] { item, held_item, count, 0 });
                    else if (parameters.Length == 3)
                        result = method.Invoke(shopMenu, new object?[] { item, held_item, count });
                    else
                        result = method.Invoke(shopMenu, new object?[] { item, held_item, count, 0, 0 });

                    return result is bool b && b;
                }
            }
            catch { }
            return false;
        }

        public static NPC? GetNearestNPC(Farmer player)
        {
            float minDist = float.MaxValue;
            NPC? nearest = null;
            var playerTile = player.Tile;

            foreach (NPC npc in player.currentLocation.characters)
            {
                if (!npc.IsVillager && !npc.isVillager())
                    continue;

                float dist = Vector2.Distance(playerTile, npc.Tile);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearest = npc;
                }
            }

            return nearest;
        }

        public static void print_mouse_pos(Mod mod)
        {
            var mouseState = Game1.getMousePosition();
            var tileX = (Game1.viewport.X + mouseState.X) / Game1.tileSize;
            var tileY = (Game1.viewport.Y + mouseState.Y) / Game1.tileSize;
            mod.Monitor.Log($"Mouse pos: pixel=({mouseState.X},{mouseState.Y}), tile=({tileX},{tileY})", LogLevel.Info);
        }

        public static void give_money(int amount)
        {
            Game1.player.Money += amount;
        }

        public static void set_time(Mod mod)
        {
            Game1.timeOfDay = 1200;
            mod.Monitor.Log("Time set to 12:00", LogLevel.Info);
        }

        public static void give_items(string itemId, int count)
        {
            var item = ItemRegistry.Create(itemId, count);
            if (item != null)
            {
                Game1.player.addItemToInventory(item);
            }
        }

        public static void tp_player(string locationName, Mod mod)
        {
            var location = Game1.getLocationFromName(locationName);
            if (location != null)
            {
                Game1.warpFarmer(locationName, 0, 0, false);
                mod.Monitor.Log($"Teleported to {locationName}", LogLevel.Info);
            }
            else
            {
                mod.Monitor.Log($"Location not found: {locationName}", LogLevel.Warn);
            }
        }

        public static void give_tool(string toolName, Mod mod)
        {
            Tool? tool = toolName.ToLower() switch
            {
                "axe" => new Axe(),
                "pickaxe" => new Pickaxe(),
                "hoe" => new Hoe(),
                "wateringcan" or "watering_can" => new WateringCan(),
                "fishingrod" or "fishing_rod" => new FishingRod(),
                "scythe" => new MeleeWeapon("47"),
                _ => null
            };

            if (tool != null)
            {
                Game1.player.addItemToInventory(tool);
                mod.Monitor.Log($"Gave tool: {toolName}", LogLevel.Info);
            }
            else
            {
                mod.Monitor.Log($"Unknown tool: {toolName}", LogLevel.Warn);
            }
        }

        public static void add_chest(int x, int y, Color color, Mod mod)
        {
            var location = Game1.player.currentLocation;
            var tilePos = new Vector2(x, y);
            var chest = new Chest(true, tilePos);
            chest.playerChoiceColor.Value = color;
            location.objects.Add(tilePos, chest);
            mod.Monitor.Log($"Added chest at ({x},{y})", LogLevel.Info);
        }
    }
}
