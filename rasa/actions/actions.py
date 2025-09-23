# rasa/actions/actions.py
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType

import os
import uuid

# NOTE: Replace the placeholder implementations below with your real
# charting / data-processing code (reading CSV/Excel, using pandas + matplotlib/plotly),
# upload chart image to a hosting/location and return a URL.

def save_placeholder_chart() -> str:
    """
    Placeholder: generate or copy a sample image and return an accessible URL/path.
    Replace this with real chart generation code.
    """
    # For demo, we'll return a dummy path. In production, return remote URL or serve static file.
    return "https://placehold.co/600x400?text=Chart+Preview"

class ActionHandleFileUpload(Action):
    def name(self) -> Text:
        return "action_handle_file_upload"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        # Rasa doesn't natively receive binary files via REST webhook.
        # Your streamlit/HF space should call your backend to upload the file and then
        # send a message like `{"sender":"user1","message":"I uploaded data.csv","metadata":{"file_url":"https://..."}}`.
        metadata = tracker.latest_message.get("metadata", {}) or {}
        file_url = metadata.get("file_url") or metadata.get("uploaded_file") or None

        # store file url or filename in slot
        if file_url:
            dispatcher.utter_message(text=f"File saved: {file_url}")
            return [SlotSet("uploaded_file", file_url), SlotSet("file_name", os.path.basename(file_url))]
        else:
            dispatcher.utter_message(text="I couldn't find a file URL in your message. Please upload or provide a link.")
            return []

class ActionDatasetSummary(Action):
    def name(self) -> Text:
        return "action_dataset_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        file_url = tracker.get_slot("uploaded_file")
        if not file_url:
            dispatcher.utter_message(text="I don't see a dataset. Please upload a CSV/Excel file first.")
            return []

        # TODO: implement real summary logic (download file_url, pandas.read_csv/read_excel, compute stats)
        # We'll return a placeholder summary
        dispatcher.utter_message(text=f"Summary for file: {file_url}\n- Rows: (demo)\n- Columns: (demo)\n- Example stats: mean(sales)=123.4")
        return []

class ActionGenerateChart(Action):
    def name(self) -> Text:
        return "action_generate_chart"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        file_url = tracker.get_slot("uploaded_file")
        chart_type = tracker.get_slot("chart_type")
        chart_columns = tracker.get_slot("chart_columns") or []

        if not file_url:
            dispatcher.utter_message(text="Please upload a dataset first.")
            return []

        # Here you would:
        # 1. Download file_url
        # 2. Load into pandas
        # 3. Generate plot with matplotlib/plotly based on chart_type & columns
        # 4. Save and serve the image (or return base64)
        # 5. Return an image message (URL or binary) to the user

        # For demo, return placeholder chart URL
        chart_url = save_placeholder_chart()
        dispatcher.utter_message(text="Generating chart... please wait.")
        dispatcher.utter_message(text=f"Chart URL: {chart_url}")
        # Optionally you can send a custom payload for UI to render image:
        dispatcher.utter_message(json_message={"type": "chart", "url": chart_url})

        # Save the last used slots for follow-ups
        events = [
            SlotSet("chart_type", chart_type),
            SlotSet("chart_columns", chart_columns),
            SlotSet("last_user_message", tracker.latest_message.get("text"))
        ]
        return events

class ActionClearSession(Action):
    def name(self) -> Text:
        return "action_clear_session"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
        dispatcher.utter_message(text="Clearing session and uploaded files.")
        # Clear relevant slots
        return [
            SlotSet("uploaded_file", None),
            SlotSet("file_name", None),
            SlotSet("chart_type", None),
            SlotSet("chart_columns", None),
            SlotSet("aggregation", None),
            SlotSet("last_user_message", None)
        ]
