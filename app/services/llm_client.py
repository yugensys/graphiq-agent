"""LLM client for interacting with Deepseek API."""
import httpx
import time
import json
import logging
from typing import Optional, Dict, Any
from app.config import settings

# Get the root logger
logger = logging.getLogger()

class LLMClient:
    """Client for interacting with LLM services."""
    
    def __init__(self, 
                 api_url: Optional[str] = settings.DEEPSEEK_API_URL,
                 api_key: Optional[str] = settings.DEEPSEEK_API_KEY):
        self.api_url = api_url
        self.api_key = api_key

    async def summarize(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
        """
        Call LLM to produce a summary. Returns text.
        Raises:
            ValueError: If API key is not configured
            httpx.HTTPStatusError: If the API request fails
        """
        # Log API key status (masking the actual key for security)
        if not self.api_key:
            error_msg = "Deepseek API key is not configured. Please set DEEPSEEK_API_KEY in your environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        api_key_display = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        
        # Log request details
        logger.info("=== LLM API Request ===")
        # logger.info(f"API Endpoint: {self.api_url}")
        # logger.info(f"API Key: {api_key_display}")
        # logger.info("System Prompt:")
        # logger.info(system_prompt)
        logger.info("\nUser Prompt:")
        # logger.info(user_prompt)
        # logger.info(f"Max Tokens: {max_tokens}")
        
        # Record start time for performance tracking
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": "deepseek-chat",  
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        # Prepare request data
        request_data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        # Log full request payload
        logger.info("\nRequest Payload:")
        logger.info(json.dumps(request_data, indent=2, ensure_ascii=False))
        logger.info("\nSending request...")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Make the API request
                resp = await client.post(
                    self.api_url,
                    json=request_data,
                    headers=headers,
                    timeout=60.0
                )
                
                # Calculate request duration
                duration = time.time() - start_time
                
                # Log response status and timing
                logger.info(f"\n=== LLM API Response ===")
                logger.info(f"Status Code: {resp.status_code}")
                logger.info(f"Response Time: {duration:.2f} seconds")
                
                # Parse response
                resp.raise_for_status()
                data = resp.json()
                
                # Log full response
                logger.info("\nResponse Headers:")
                for header, value in resp.headers.items():
                    logger.info(f"  {header}: {value}")
                
                logger.info("\nResponse Body:")
                logger.info(json.dumps(data, indent=2, ensure_ascii=False))
                
                # Extract and log content
                content = None
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    logger.info("\nGenerated Content:")
                    logger.info(content)
                elif "text" in data:
                    content = data["text"]
                elif "output" in data:
                    content = data["output"]
                else:
                    # If we get here, the response format is unexpected
                    error_msg = f"Unexpected API response format: {data}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Log token usage if available
                # if "usage" in data:
                #     usage = data["usage"]
                #     logger.info("\nToken Usage:")
                #     logger.info(f"Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
                #     logger.info(f"Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
                #     logger.info(f"Total Tokens: {usage.get('total_tokens', 'N/A')}")
                
                logger.info("=" * 50)  # End of request/response log
                
                if content is None:
                    raise ValueError("No content found in the response")
                    
                return content
                
        except httpx.HTTPStatusError as e:
            duration = time.time() - start_time
            error_msg = f"API request failed with status {e.response.status_code} after {duration:.2f}s"
            logger.error(error_msg)
            try:
                error_data = e.response.json()
                logger.error("Error details: %s", json.dumps(error_data, indent=2))
            except:
                logger.error("Response text: %s", e.response.text)
            raise
            
        except httpx.RequestError as e:
            duration = time.time() - start_time
            error_msg = f"Failed to connect to the API after {duration:.2f}s: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(error_msg) from e
            
        except json.JSONDecodeError as e:
            duration = time.time() - start_time
            error_msg = f"Failed to parse API response after {duration:.2f}s: {str(e)}"
            logger.error(error_msg)
            logger.error("Response text: %s", getattr(resp, 'text', 'No response content'))
            raise ValueError("Invalid JSON response from API") from e
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error after {duration:.2f}s during API call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

# Singleton instance
llm = LLMClient()
