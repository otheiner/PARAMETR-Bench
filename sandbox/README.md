# Sandbox

Agentic evaluation allows running python. To make this safe, framework provides Dockerfile to build a self-contained sandbox environment. The sandbox has no access to the Internet, its memory is capped to 512 MB, and it doesn't allow writing `.pyc` files.

To make the comparison of LLM agentic capabilities fair there are only some libraries allowed on top of the standard python libraries. This list is specified in `requirements.txt` in this folder.

Docker container needs to be built locally before starting the agentic evaluation. This is done by calling following command from the top-level of this repository.

```bash
docker build -t benchmark-sandbox sandbox/
```

If you are running agentic evaluation, make sure Docker daemon is running on your machine!