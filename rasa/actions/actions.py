# rasa/actions/actions.py
import os, json, requests
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

BACKEND = os.environ.get("GRAPHIQ_BACKEND_URL", "http://localhost:8000")


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
import aiohttp
from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Tracker
from typing import Dict, Any, List
import asyncio

class ActionProcessMDL(Action):
    def name(self) -> str:
        return "action_process_mdl"

    async def run(
        self, 
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        import logging
        import socket
        from datetime import datetime
        
        # Configure logging
        logger = logging.getLogger('action_process_mdl')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        logger.info("=" * 50)
        logger.info("STARTING BACKEND CONNECTION TEST")
        logger.info("=" * 50)
        
        test_url = "http://backend:8080/api/v1/chat/chat"
        test_payload = {"message": "test connection"}
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        
        # Log test details
        logger.info(f"Test URL: {test_url}")
        logger.info(f"Test payload: {test_payload}")
        logger.info(f"Current time: {datetime.utcnow().isoformat()}")
        
        try:
            # Test DNS resolution first
            backend_host = 'backend'
            logger.info(f"Resolving DNS for: {backend_host}")
            try:
                ip = socket.gethostbyname(backend_host)
                logger.info(f"Resolved {backend_host} to IP: {ip}")
            except socket.gaierror as e:
                error_msg = f"DNS resolution failed for {backend_host}: {str(e)}"
                logger.error(error_msg)
                dispatcher.utter_message(text=error_msg)
                return []
            
            # Make the HTTP request with timeout
            logger.info("Sending HTTP request...")
            start_time = datetime.utcnow()
            
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        test_url,
                        json=test_payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        response_time = (datetime.utcnow() - start_time).total_seconds()
                        status = response.status
                        response_text = await response.text()
                        
                        logger.info(f"Response time: {response_time:.2f}s")
                        logger.info(f"Status: {status}")
                        logger.info(f"Response: {response_text}")
                        
                        if status == 200:
                            msg = f"✅ Backend connection successful! (Status: {status})"
                            logger.info(msg)
                            dispatcher.utter_message(text=msg)
                            logger.warning(msg)
                            dispatcher.utter_message(text=msg)
                            
            except asyncio.TimeoutError:
                error_msg = "⌛ Request timed out (10s) - Backend is not responding"
                logger.error(error_msg)
                dispatcher.utter_message(text=error_msg)
                return []
            except aiohttp.ClientError as e:
                error_msg = f"❌ HTTP client error: {str(e)}"
                logger.error(error_msg)
                dispatcher.utter_message(text=error_msg)
                return []
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                logger.error(error_msg, exc_info=True)
                dispatcher.utter_message(text=error_msg)
                return []
            
            # Process successful response
            try:
                response_data = await response.json()
                if "message" in response_data:
                    dispatcher.utter_message(text=response_data["message"])
                if "sql" in response_data:
                    dispatcher.utter_message(json_message={
                        "type": "graphiq_sql",
                        "sql": response_data["sql"],
                        "note": "SQL generated from MDL"
                    })
                    return [SlotSet("last_sql", response_data["sql"])]
            except Exception as e:
                error_msg = f"Error processing response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                dispatcher.utter_message(text=error_msg)
                return []
            
        except aiohttp.ClientError as e:
            error_msg = f"Error connecting to the backend: {str(e)}"
            logger.error(error_msg, exc_info=True)
            dispatcher.utter_message(text=error_msg)
            return []
            
        except Exception as e:
            error_msg = f"Error processing MDL: {str(e)}"
            logger.error(error_msg, exc_info=True)
            dispatcher.utter_message(text=error_msg)
            return []
            
        return []
        return []
