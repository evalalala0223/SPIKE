from types import SimpleNamespace

from cradle.memory.local_memory import LocalMemory, normalize_sakg_experience


def test_normalize_sakg_experience_handles_raw_objects():
    state = SimpleNamespace(description="Standing near the farmhouse entrance")
    action = SimpleNamespace(action="go_through_door()")

    normalized = normalize_sakg_experience(
        {
            "state": state,
            "action": action,
            "similarity": 0.93,
            "success_rate": 0.75,
        }
    )

    assert normalized["state_description"] == "Standing near the farmhouse entrance"
    assert normalized["action"] == "go_through_door()"
    assert normalized["state_node"] is state
    assert normalized["action_edge"] is action
    assert normalized["similarity"] == 0.93
    assert normalized["success_rate"] == 0.75


def test_retrieve_similar_experiences_respects_disabled_sakg():
    memory = object.__new__(LocalMemory)
    memory.sa_kg = SimpleNamespace(enabled=False)

    assert memory.retrieve_similar_experiences("current state", top_k=3) == []


def test_retrieve_similar_experiences_normalizes_sakg_hits():
    state = SimpleNamespace(description="At Pierre's counter with shop menu open")
    action = SimpleNamespace(action="buy_item('Parsnip Seeds', 1)")

    class DummySAKG:
        enabled = True

        def __init__(self):
            self.calls = []

        def retrieve_similar_states(self, state_description, top_k):
            self.calls.append((state_description, top_k))
            return [
                {
                    "state": state,
                    "action": action,
                    "similarity": 0.91,
                    "success_rate": 1.0,
                }
            ]

    sakg = DummySAKG()
    memory = object.__new__(LocalMemory)
    memory.sa_kg = sakg

    results = memory.retrieve_similar_experiences("shop context", top_k=2)

    assert sakg.calls == [("shop context", 2)]
    assert results == [
        {
            "state": state,
            "action": "buy_item('Parsnip Seeds', 1)",
            "similarity": 0.91,
            "success_rate": 1.0,
            "state_description": "At Pierre's counter with shop menu open",
            "state_node": state,
            "action_edge": action,
        }
    ]
