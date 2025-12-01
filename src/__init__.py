# Persona Chatbot with RAG + Multi-Agent Orchestration
"""Project package initializer with protobuf compatibility helpers."""

try:
    # Protobuf 5.0 removed MessageFactory.GetPrototype; re-add for older call sites.
    from google.protobuf import message_factory
except Exception:  # pragma: no cover - protobuf is optional in some envs.
    message_factory = None
else:
    if (
        message_factory is not None
        and not hasattr(message_factory.MessageFactory, "GetPrototype")
        and hasattr(message_factory.MessageFactory, "GetMessageClass")
    ):
        setattr(
            message_factory.MessageFactory,
            "GetPrototype",
            message_factory.MessageFactory.GetMessageClass,
        )

