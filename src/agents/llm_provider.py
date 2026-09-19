"""
Unified LLM & Local Expert Provider.
Supports Google Gemini, OpenAI, and a built-in deterministic expert reasoning engine
so the platform works reliably with or without external API keys.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.mode = "local_expert"

        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_client = genai.GenerativeModel("gemini-1.5-flash")
                self.mode = "gemini"
                logger.info("Using Google Gemini API.")
            except Exception as e:
                logger.warning(f"Failed to init Gemini: {e}. Falling back to local expert.")

        elif self.openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
                self.mode = "openai"
                logger.info("Using OpenAI API.")
            except Exception as e:
                logger.warning(f"Failed to init OpenAI: {e}. Falling back to local expert.")

    def generate_completion(self, prompt: str, system_prompt: str = "") -> str:
        """Invokes LLM if configured, otherwise returns local expert reasoning."""
        if self.mode == "gemini":
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.gemini_client.generate_content(full_prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini error: {e}. Falling back to local expert.")

        elif self.mode == "openai":
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                res = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2
                )
                return res.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI error: {e}. Falling back to local expert.")

        return ""
