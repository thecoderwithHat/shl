from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def chat(messages):
    response = client.post("/chat", json={"messages": messages})
    assert response.status_code == 200
    return response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_leadership_flow_asks_then_commits():
    first = chat([{"role": "user", "content": "We need a solution for senior leadership."}])
    assert first["recommendations"] is None
    assert "who is this meant for" in first["reply"].lower() or "selection against a leadership benchmark" in first["reply"].lower()

    second = chat([
        {"role": "user", "content": "We need a solution for senior leadership."},
        {"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."},
        {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."},
        {"role": "user", "content": "Perfect, that's what we need."},
    ])
    names = [item["name"] for item in second["recommendations"]]
    assert "Occupational Personality Questionnaire OPQ32r" in names
    assert "OPQ Leadership Report" in names
    assert second["end_of_conversation"] is True


def test_technical_flow_with_updates():
    reply = chat([
        {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"},
        {"role": "user", "content": "Yes, go ahead. Should I also add a cognitive test for this level?"},
    ])
    names = [item["name"] for item in reply["recommendations"]]
    assert "SHL Verify Interactive G+" in names
    assert "Occupational Personality Questionnaire OPQ32r" in names
    assert reply["end_of_conversation"] is False


def test_contact_center_language_clarification():
    first = chat([{"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}])
    assert first["recommendations"] is None
    assert "what language are the calls in" in first["reply"].lower()

    second = chat([
        {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"},
        {"role": "user", "content": "English."},
        {"role": "user", "content": "US."},
    ])
    names = [item["name"] for item in second["recommendations"]]
    assert "SVAR - Spoken English (US) (New)" in names
    assert "Contact Center Call Simulation (New)" in names


def test_compare_explains_without_recommendations():
    result = chat([
        {"role": "user", "content": "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?"}
    ])
    assert result["recommendations"] is None
    assert "distinct catalog items" in result["reply"].lower()


def test_safety_and_compliance_refusal():
    result = chat([
        {"role": "user", "content": "Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?"}
    ])
    assert result["recommendations"] is None
    assert "legal or compliance obligations" in result["reply"].lower()


def test_graduate_battery_and_removal():
    initial = chat([
        {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."},
    ])
    initial_names = [item["name"] for item in initial["recommendations"]]
    assert "SHL Verify Interactive G+" in initial_names
    assert "Graduate Scenarios" in initial_names

    removed = chat([
        {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."},
        {"role": "user", "content": "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long."},
    ])
    assert removed["recommendations"] is not None
    assert all(item["name"] != "Occupational Personality Questionnaire OPQ32r" for item in removed["recommendations"])

    final = chat([
        {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."},
        {"role": "user", "content": "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long."},
        {"role": "user", "content": "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."},
    ])
    final_names = [item["name"] for item in final["recommendations"]]
    assert "SHL Verify Interactive G+" in final_names
    assert "Graduate Scenarios" in final_names
    assert final["end_of_conversation"] is True


def test_off_topic_refusal():
    result = chat([
        {"role": "user", "content": "What's the weather in Seattle?"}
    ])
    assert result["recommendations"] is None
    assert "only help with shl assessment selection" in result["reply"].lower()

