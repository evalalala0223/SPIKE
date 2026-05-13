using System.Threading;
using HarmonyLib;
using StardewValley;

namespace ActionSpace.patches
{
    /// <summary>
    /// Prevents game time from advancing while any agent port is paused.
    /// Uses a reference counter so parallel ports don't interfere with each other:
    ///   - Each pause() increments the counter
    ///   - Each resume() decrements the counter
    ///   - Time is frozen whenever counter > 0
    /// </summary>
    [HarmonyPatch(typeof(Game1), "shouldTimePass")]
    public static class TimePassPatch
    {
        private static int _pauseCount = 0;

        public static int CurrentCount => Volatile.Read(ref _pauseCount);

        /// <summary>
        /// Convenience property for backward compatibility checks.
        /// </summary>
        public static bool AgentPaused => CurrentCount > 0;

        public static int Pause()
        {
            return Interlocked.Increment(ref _pauseCount);
        }

        public static int Resume()
        {
            int newValue = Interlocked.Decrement(ref _pauseCount);
            if (newValue < 0)
            {
                Interlocked.Exchange(ref _pauseCount, 0);
                return 0;
            }

            return newValue;
        }

        public static void Reset()
        {
            Interlocked.Exchange(ref _pauseCount, 0);
        }

        public static bool Prefix(ref bool __result)
        {
            if (AgentPaused)
            {
                __result = false;
                return false; // Skip original method
            }
            return true; // Run original method
        }
    }
}
