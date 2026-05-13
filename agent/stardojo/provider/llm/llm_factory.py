from stardojo.utils import Singleton


class LLMFactory(metaclass=Singleton):

    def __init__(self):
        self._builders = {}


    def create(self, llm_provider_config_path, embed_provider_config_path, **kwargs):

        llm_provider = None
        embed_provider = None

        key = str(llm_provider_config_path).lower()

        if "opensrc" in key or "qwen" in key:
            from stardojo.provider.llm.openai import OpenAIProvider

            llm_provider = OpenAIProvider(is_opensource=True)
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(
                embed_provider_config_path,
                embedding_only=True,
            )

        elif "openai" in key:
            from stardojo.provider.llm.openai import OpenAIProvider

            llm_provider = OpenAIProvider()
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = llm_provider
        elif "claude" in key:
            from stardojo.provider.llm.openai import OpenAIProvider
            from stardojo.provider.llm.claude import ClaudeProvider

            llm_provider = ClaudeProvider()
            llm_provider.init_provider(llm_provider_config_path)
            #logger.warn(f"Claude do not support embedding, use OpenAI instead.")
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(
                embed_provider_config_path,
                embedding_only=True,
            )
        elif "gemini" in key:
            from stardojo.provider.llm.openai import OpenAIProvider
            from stardojo.provider.llm.gemini import GeminiProvider

            llm_provider = GeminiProvider()
            llm_provider.init_provider(llm_provider_config_path)
            # logger.warn(f"Claude do not support embedding, use OpenAI instead.")
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(
                embed_provider_config_path,
                embedding_only=True,
            )

        if not llm_provider or not embed_provider:
            raise ValueError(key)

        return llm_provider, embed_provider
