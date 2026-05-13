from cradle.provider.llm.openai import OpenAIProvider
from cradle.utils import Singleton
from cradle.log import Logger

logger = Logger()


class LLMFactory(metaclass=Singleton):

    def __init__(self):
        self._builders = {}


    def create(self, llm_provider_config_path, embed_provider_config_path, **kwargs):
        """
        Create LLM and embedding providers.

        Supports both legacy providers and new LangChain-based providers.
        Priority: enhanced_config.yaml features.use_langchain > config JSON > kwargs
        """
        llm_provider = None
        embed_provider = None

        key = str(llm_provider_config_path).lower()

        # Priority 1: Check enhanced_config.yaml
        use_langchain = False
        try:
            from cradle.config.enhanced_config import EnhancedConfig
            enhanced_cfg = EnhancedConfig()
            use_langchain = enhanced_cfg._raw_config.get('features', {}).get('use_langchain', False)
            if use_langchain:
                logger.write("[Config] use_langchain=true (from enhanced_config.yaml features)")
        except Exception as e:
            logger.debug(f"Could not read enhanced_config: {e}")
        
        # Priority 2: Check config JSON file (backward compatibility)
        if not use_langchain and isinstance(llm_provider_config_path, str):
            try:
                from cradle.utils.json_utils import load_json
                from cradle.utils.file_utils import assemble_project_path
                config_path = assemble_project_path(llm_provider_config_path)
                config_dict = load_json(config_path)
                use_langchain = config_dict.get("use_langchain", False)
                if use_langchain:
                    logger.write("[Config] use_langchain=true (from config JSON, legacy)")
            except Exception:
                pass  # If loading fails, use legacy provider
        
        # Priority 3: Check kwargs (lowest priority)
        if not use_langchain:
            use_langchain = kwargs.get("use_langchain", False)
            if use_langchain:
                logger.write("[Config] use_langchain=true (from kwargs)")

        if use_langchain:
            logger.write("Using LangChain-based LLM provider (Phase 0)")
            from cradle.provider.langchain import LangChainLLMProvider

            llm_provider = LangChainLLMProvider()
            llm_provider.init_provider(llm_provider_config_path)

            # For embedding, still use OpenAI for now
            # (will be upgraded in Phase 2 with Mem0)
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(
                embed_provider_config_path,
                embedding_only=True,
            )

        elif "openai" in key or "qwen" in key or "opensrc" in key:
            llm_provider = OpenAIProvider()
            llm_provider.init_provider(llm_provider_config_path)
            embed_provider = llm_provider
        elif "claude" in key:
            from cradle.provider.llm.restful_claude import RestfulClaudeProvider

            llm_provider = RestfulClaudeProvider()
            llm_provider.init_provider(llm_provider_config_path)
            #logger.warn(f"Claude do not support embedding, use OpenAI instead.")
            embed_provider = OpenAIProvider()
            embed_provider.init_provider(
                embed_provider_config_path,
                embedding_only=True,
            )

        if not llm_provider or not embed_provider:
            raise ValueError(key)

        return llm_provider, embed_provider
