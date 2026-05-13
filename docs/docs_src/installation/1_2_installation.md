### Step 1: Add Environment Variables

Create or edit the file at `StarDojo/env/.env` and include the following keys:

```bash
# Required
STARDEW_APP_PATH=/path/to/StardewModdingAPI

# Optional (if using external LLM services)
OPENAI_API_KEY=<your-openai-api-key>
DASHSCOPE_API_KEY=<your-dashscope-api-key>
GEMINI_API_KEY=<your-gemini-api-key>
```

* Make sure `STARDEW_APP_PATH` correctly points to the **file path** of `StardewModdingAPI.exe` on Windows.

### Step 2: Initialize Environment

```powershell
cd StarDojo
.\setup.ps1
```

This command installs dependencies and prepares the shell environment for easy agent launching.
