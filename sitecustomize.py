"""Global Python interpreter tweaks for the project.

This module is auto-imported (if present on sys.path) before the rest of the
application starts. We use it to smooth over breaking changes in dependencies.
"""

try:
    from google.protobuf import message_factory, symbol_database
except Exception:  # pragma: no cover - protobuf might be optional.
    message_factory = None
    symbol_database = None
else:
    needs_factory_patch = (
        message_factory is not None
        and not hasattr(message_factory.MessageFactory, "GetPrototype")
        and hasattr(message_factory.MessageFactory, "GetMessageClass")
    )
    if needs_factory_patch:
        # Protobuf 5.0 removed MessageFactory.GetPrototype, but many callers (us
        # included) still rely on it. Map it to the modern API so legacy code and
        # third-party libraries continue to work without noisy warnings.
        setattr(
            message_factory.MessageFactory,
            "GetPrototype",
            message_factory.MessageFactory.GetMessageClass,
        )

    if (
        symbol_database is not None
        and hasattr(symbol_database, "_SymbolDatabase")
        and not hasattr(symbol_database._SymbolDatabase, "GetPrototype")
        and hasattr(symbol_database._SymbolDatabase, "GetMessageClass")
    ):
        # Some call sites go through the default symbol database. Provide the same
        # fallback there to ensure compatibility regardless of import order.
        setattr(
            symbol_database._SymbolDatabase,
            "GetPrototype",
            symbol_database._SymbolDatabase.GetMessageClass,
        )
