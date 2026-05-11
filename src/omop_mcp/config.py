# Supported LLM Providers
LLM_PROVIDERS = (
    "azure_openai",
    "openai",
    "anthropic",
    "gemini",
    "ollama",
    "openrouter",
    "groq",
    "huggingface",
)


LLM_API_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
)


SYSTEM_ENV_VARS = ("PATH", "SYSTEMROOT", "HOME")
ENV_VARS = frozenset(LLM_API_KEY_VARS + ("OMOPHUB_API_KEY",) + SYSTEM_ENV_VARS)
