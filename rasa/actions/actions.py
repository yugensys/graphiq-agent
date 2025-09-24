# rasa/actions/actions.py
import os, json, requests
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

BACKEND = os.environ.get("GRAPHIQ_BACKEND_URL", "http://backend:8000")

class ActionReceiveMDL(Action):
    def name(self) -> Text:
        return "action_receive_mdl"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        metadata = tracker.latest_message.get("metadata") or {}
        mdl = metadata.get("mdl") or tracker.latest_message.get("text")

        if not mdl:
            dispatcher.utter_message(text="I didn't receive the MDL. Please send it again.")
            return []

        # store MDL
        dispatcher.utter_message(text="MDL received. Generating SQL...")
        # forward to backend
        try:
            resp = requests.post(f"{BACKEND}/api/agent_grapher/generate_sql", json={
                "user_id": tracker.sender_id,
                "mdl": mdl
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            dispatcher.utter_message(text="Sorry, couldn't contact the analysis backend. Try again later.")
            return []

        sql = data.get("sql")
        if not sql:
            dispatcher.utter_message(text="Backend couldn't produce SQL from MDL.")
            return []

        # Save SQL in slot and send it to frontend for execution
        dispatcher.utter_message(text="I generated SQL. Run this on your local dataset and send me the result.")
        dispatcher.utter_message(json_message={
            "type": "graphiq_sql",
            "sql": sql,
            "note": "Run this SQL on your local file and send back the resulting table (small JSON or upload to backend)."
        })
        return [SlotSet("mdl", json.dumps(mdl)), SlotSet("last_sql", sql)]


class ActionReceiveTable(Action):
    def name(self) -> Text:
        return "action_receive_table"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        metadata = tracker.latest_message.get("metadata") or {}
        table_json = metadata.get("table_json")           # small table inline
        table_id = metadata.get("table_id")              # or id if already uploaded to backend

        if not table_json and not table_id:
            dispatcher.utter_message(text="I didn't receive the result table. Please send it or upload it so I can process it.")
            return []

        # If frontend uploaded table to backend and returned table_id
        if table_id:
            payload = {"user_id": tracker.sender_id, "table_id": table_id, "sql": tracker.get_slot("last_sql")}
        else:
            # forward the inline table to backend
            payload = {"user_id": tracker.sender_id, "table": table_json, "sql": tracker.get_slot("last_sql")}

        dispatcher.utter_message(text="Received the table — generating chart and insights now...")

        try:
            resp = requests.post(f"{BACKEND}/api/agent_grapher/process_table", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except Exception:
            dispatcher.utter_message(text="Processing failed. Please try again later.")
            return []

        chart_url = result.get("chart_url")
        insights = result.get("insights")

        text_fallback = f"I generated the chart and insights.\n\nInsights: {insights or 'none'}"
        dispatcher.utter_message(text=text_fallback)
        dispatcher.utter_message(json_message={
            "type": "graphiq_result",
            "chart_url": chart_url,
            "insights": insights,
            "sql": tracker.get_slot("last_sql"),
            "meta": result.get("meta", {})
        })
        return [SlotSet("table_id", result.get("table_id"))]
