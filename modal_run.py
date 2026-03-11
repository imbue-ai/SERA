#!/usr/bin/env python3
"""Run the SERA pipeline with Modal Sandboxes instead of local Docker.

This script monkey-patches the DistillRunner to inject Modal deployment
flags into the SWE-agent CLI commands. Everything runs locally (generate,
eval, postprocess, LLM API calls) except the SWE-agent container
environments, which run as Modal Sandboxes in the cloud.

This solves the x86_64 Linux image incompatibility on Apple Silicon.

Usage:
    # Same interface as sera/main.py, just use modal_run.py instead:
    python modal_run.py --config-name=specialization_django_anthropic \\
        distill.model.name=openai/qwen/qwen3.5-397b-a17b \\
        distill.model.url=https://openrouter.ai/api/v1 \\
        name=test_modal

    # Override Modal-specific settings via env vars:
    MODAL_RUNTIME_TIMEOUT=300 MODAL_DEPLOYMENT_TIMEOUT=7200 \\
        python modal_run.py --config-name=specialization_django ...
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def validate_modal_setup():
    """Check that Modal is installed and authenticated."""
    try:
        import modal
    except ImportError:
        print("Error: modal is not installed. Run: uv pip install modal")
        sys.exit(1)

    try:
        modal.App.lookup("swe-rex", create_if_missing=True)
    except modal.exception.AuthError:
        print("Error: Modal is not authenticated. Run: modal token new")
        sys.exit(1)


def patch_distill_runner():
    """Monkey-patch DistillRunner to inject Modal deployment flags."""
    import sera.datagen.data.distill.distill as distill_module

    OriginalDistillRunner = distill_module.DistillRunner

    # Modal config from env vars with sensible defaults
    runtime_timeout = os.environ.get("MODAL_RUNTIME_TIMEOUT", "300")
    deployment_timeout = os.environ.get("MODAL_DEPLOYMENT_TIMEOUT", "3600")
    startup_timeout = os.environ.get("MODAL_STARTUP_TIMEOUT", "180")

    modal_flags = (
        " --instances.deployment.type modal"
        f" --instances.deployment.startup_timeout {startup_timeout}"
        f" --instances.deployment.runtime_timeout {runtime_timeout}"
        f" --instances.deployment.deployment_timeout {deployment_timeout}"
    )

    # Tell SWE-agent to read the API key from env var instead of the
    # config's hardcoded "not-needed" placeholder. The "$OPENAI_API_KEY"
    # syntax is handled by SWE-agent's get_api_keys() method.
    api_key_flag = ""
    if os.environ.get("OPENAI_API_KEY"):
        api_key_flag = " --agent.model.api_key $OPENAI_API_KEY"

    class ModalDistillRunner(OriginalDistillRunner):

        def _build_sweagent_cmd(self, *args, **kwargs):
            cmd = super()._build_sweagent_cmd(*args, **kwargs)
            return cmd + modal_flags + api_key_flag

        def _build_mini_swe_agent_cmd(self, *args, **kwargs):
            cmd = super()._build_mini_swe_agent_cmd(*args, **kwargs)
            return cmd + " --environment-class swerex_modal"

    distill_module.DistillRunner = ModalDistillRunner


def main():
    validate_modal_setup()
    patch_distill_runner()

    # Hydra resolves config_path relative to the decorated function's module.
    # Since we call sera.main.main() from here, Hydra still resolves relative
    # to sera/main.py, which requires sera/configs/ to exist as a package.
    # Ensure the configs dir has an __init__.py so Hydra finds it.
    configs_dir = os.path.join(os.path.dirname(__file__), "sera", "configs")
    init_py = os.path.join(configs_dir, "__init__.py")
    if not os.path.exists(init_py):
        open(init_py, "w").close()

    from sera.main import main as sera_main

    sera_main()


if __name__ == "__main__":
    main()
