"""
Apify Client Wrapper for Social Data Scraping.
Provides robust token management, rate-limiting retry handling,
actor execution, dataset fetching, and diagnostic reporting.
Supports both official apify_client SDK and direct zero-dependency Apify REST API.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import requests

from config.settings import APIFY_API_TOKEN, REQUEST_TIMEOUT_SECONDS, MAX_RETRIES

logger = logging.getLogger("hengseng.apify")

APIFY_BASE_URL = "https://api.apify.com/v2"

class ApifyScraperClient:
    """Wrapper around Apify with automated retry, SDK support, and REST fallback."""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or APIFY_API_TOKEN
        self._sdk_client = None
        self.is_authenticated = False
        self._init_client()

    def _init_client(self) -> None:
        """Initialize credentials check via SDK or direct REST API."""
        if not self.api_token or self.api_token.strip() == "" or "your_" in self.api_token:
            logger.info("No valid APIFY_API_TOKEN configured. Scrapers will operate in cached/direct-parse mode.")
            self.is_authenticated = False
            return

        # 1. Try SDK initialization
        try:
            from apify_client import ApifyClient
            self._sdk_client = ApifyClient(self.api_token)
            user_info = self._sdk_client.user().get()
            username = user_info.get("username", "user") if user_info else "user"
            logger.info(f"Apify SDK authenticated successfully as: {username}")
            self.is_authenticated = True
            return
        except ImportError:
            logger.info("apify_client SDK not installed; using Apify REST API directly.")
        except Exception as e:
            logger.warning(f"Apify SDK auth check returned: {e}")

        # 2. Try Direct REST endpoint authentication
        try:
            resp = requests.get(
                f"{APIFY_BASE_URL}/users/me?token={self.api_token}",
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            if resp.status_code == 200:
                user_data = resp.json().get("data", {})
                username = user_data.get("username", "user")
                logger.info(f"Apify REST API verified successfully as: {username}")
                self.is_authenticated = True
            else:
                logger.warning(f"Apify token check failed with HTTP {resp.status_code}")
                self.is_authenticated = False
        except Exception as e:
            logger.warning(f"Apify connection check error: {e}")
            self.is_authenticated = False

    def run_actor(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
        timeout_secs: int = REQUEST_TIMEOUT_SECONDS,
        memory_mbytes: Optional[int] = 512
    ) -> List[Dict[str, Any]]:
        """
        Executes an Apify actor and waits for dataset results.
        Returns empty list on error or unauthenticated state.
        """
        if not self.is_authenticated:
            logger.info(f"Apify not authenticated. Skipping live actor run: {actor_id}")
            return []

        # 1. Execute via SDK if loaded
        if self._sdk_client is not None:
            try:
                run = self._sdk_client.actor(actor_id).call(
                    run_input=run_input,
                    timeout_secs=timeout_secs,
                    memory_mbytes=memory_mbytes
                )
                if run and run.get("defaultDatasetId"):
                    items = list(self._sdk_client.dataset(run["defaultDatasetId"]).iterate_items())
                    return items
            except Exception as e:
                logger.error(f"Apify SDK execution error for {actor_id}: {e}")

        # 2. Execute via Direct REST API
        safe_actor_id = actor_id.replace("/", "~")
        run_url = f"{APIFY_BASE_URL}/acts/{safe_actor_id}/runs?token={self.api_token}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Triggering Apify actor '{actor_id}' via REST (attempt {attempt}/{MAX_RETRIES})...")
                post_resp = requests.post(
                    run_url,
                    json=run_input,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    headers={"Content-Type": "application/json"}
                )

                if post_resp.status_code not in (200, 201):
                    logger.warning(f"Apify run creation failed: HTTP {post_resp.status_code} - {post_resp.text[:200]}")
                    return []

                run_data = post_resp.json().get("data", {})
                run_id = run_data.get("id")
                dataset_id = run_data.get("defaultDatasetId")

                if not run_id:
                    return []

                # Poll run completion
                max_wait = min(timeout_secs, 60)
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    time.sleep(3)
                    poll_resp = requests.get(
                        f"{APIFY_BASE_URL}/actor-runs/{run_id}?token={self.api_token}",
                        timeout=REQUEST_TIMEOUT_SECONDS
                    )
                    if poll_resp.status_code == 200:
                        status = poll_resp.json().get("data", {}).get("status")
                        if status == "SUCCEEDED":
                            break
                        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                            logger.warning(f"Actor run ended with status: {status}")
                            return []

                # Fetch dataset items
                if dataset_id:
                    ds_resp = requests.get(
                        f"{APIFY_BASE_URL}/datasets/{dataset_id}/items?token={self.api_token}&format=json",
                        timeout=REQUEST_TIMEOUT_SECONDS
                    )
                    if ds_resp.status_code == 200:
                        items = ds_resp.json()
                        logger.info(f"Actor '{actor_id}' completed. Retrieved {len(items)} items from dataset.")
                        return items

            except Exception as e:
                logger.error(f"Apify REST run error on attempt {attempt}: {e}")
                time.sleep(attempt * 2)

        return []

    def get_status(self) -> Dict[str, Any]:
        """Diagnostic state of the Apify integration."""
        return {
            "authenticated": self.is_authenticated,
            "has_token": bool(self.api_token and "your_" not in self.api_token),
            "sdk_client_loaded": self._sdk_client is not None,
            "mode": "SDK" if self._sdk_client else "REST"
        }
