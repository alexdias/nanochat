import pytest

from nanochat.storage import (
    ConversationNotFoundError,
    ConversationStore,
)


def test_create_and_fetch_conversation_messages():
    with ConversationStore(":memory:") as store:
        conversation_id = store.create_conversation("My chat")
        store.append_message(conversation_id, "user", "Hello")
        store.append_message(conversation_id, "assistant", "Hi there")

        messages = store.get_conversation(conversation_id)

    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there"


def test_list_conversations_orders_by_recent_activity():
    with ConversationStore(":memory:") as store:
        first = store.create_conversation("First")
        second = store.create_conversation("Second")

        store.append_message(first, "user", "Hi")
        store.append_message(second, "user", "Hey")
        store.append_message(first, "assistant", "Hello again")

        summaries = store.list_conversations()

    assert [summary.id for summary in summaries] == [first, second]
    assert summaries[0].last_message_preview == "Hello again"


def test_soft_delete_hides_conversation_from_default_listing():
    with ConversationStore(":memory:") as store:
        conversation_id = store.create_conversation("Temp")
        store.append_message(conversation_id, "user", "test")

        store.soft_delete_conversation(conversation_id)
        default_results = store.list_conversations()
        deleted_results = store.list_conversations(include_deleted=True)

    assert default_results == []
    assert [summary.id for summary in deleted_results] == [conversation_id]


def test_append_message_requires_existing_conversation():
    with ConversationStore(":memory:") as store:
        with pytest.raises(ConversationNotFoundError):
            store.append_message(9999, "user", "nope")
