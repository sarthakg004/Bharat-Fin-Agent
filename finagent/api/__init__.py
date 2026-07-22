"""FastAPI layer: request models, SSE streaming endpoints, agent service.

The server is stateless — the client owns the conversation and replays recent
turns with each request.
"""
